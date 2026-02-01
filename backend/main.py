"""
ASRS Human Error Pattern Analysis — FastAPI Backend v2
========================================================
Fixes: EDA distribution auto-select, timeline parsing, ML target detection
"""

import os, re, warnings, json, io
from collections import Counter
from typing import Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk

warnings.filterwarnings("ignore")

for pkg in ["punkt", "stopwords", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="ASRS Human Error Analysis API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ─── State ────────────────────────────────────────────────────────────────────
_df: Optional[pd.DataFrame] = None
_stop_words = set(stopwords.words("english")) | set(stopwords.words("spanish"))
_stop_words.update(["the","and","was","for","are","but","not","you","all","can",
    "had","her","one","our","out","has","que","los","las","del","con","por","una",
    "fue","est","como","sin","sobre","entre","durante","cuando","donde","porque",
    "mientras","también","report","reported","incident","aircraft","pilot","crew"])


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"[^a-záéíóúñüà-ÿ\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _stop_words and len(t) > 2]
    return " ".join(tokens)

def _get_df() -> pd.DataFrame:
    if _df is None:
        raise HTTPException(status_code=400, detail="Sin datos. Sube el CSV primero via /upload.")
    return _df

def _get_object_columns(df):
    """Return only object/string columns — useful for distribution dropdowns."""
    return df.select_dtypes(include="object").columns.tolist()

def _get_text_column(df):
    """Auto-detect the richest text column (longest avg string)."""
    obj_cols = _get_object_columns(df)
    if not obj_cols: return None
    return max(obj_cols, key=lambda c: df[c].astype(str).str.len().mean())


# ═══════════════════════════════════════════════════════════════════════════════
# UPLOAD & HEALTH
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/health")
def health():
    return {"status": "ok", "loaded": _df is not None, "rows": len(_df) if _df is not None else 0}

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    global _df
    try:
        content = await file.read()
        raw_df = None
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                raw_df = pd.read_csv(io.StringIO(content.decode(enc)), header=None)
                break
            except Exception:
                continue
        if raw_df is None:
            raise HTTPException(status_code=400, detail="No se pudo decodificar el CSV.")

        # Detectar multi-header ASRS:
        # Fila 0 = grupo general (Time, Time, Place, Place...)
        # Fila 1 = sub-nombres reales (ACN, Date, Local Time Of Day...)
        # Fila 2+ = datos reales
        # Si la fila 1 tiene mayoria de strings descriptivos, es sub-header
        row1 = raw_df.iloc[1].fillna("").astype(str)
        looks_like_subheader = row1.str.match(r"^[A-Za-z]").sum() > len(row1) * 0.4

        if looks_like_subheader:
            # Usar fila 1 como nombres de columna, datos desde fila 2
            _df = raw_df.iloc[2:].copy()
            _df.columns = raw_df.iloc[1].fillna("").astype(str).str.strip()
            _df = _df.reset_index(drop=True)
        else:
            # CSV normal con una sola fila de header
            _df = raw_df.iloc[1:].copy()
            _df.columns = raw_df.iloc[0].fillna("").astype(str).str.strip()
            _df = _df.reset_index(drop=True)

        # Normalizar nombres: lowercase, espacios a guión bajo
        _df.columns = _df.columns.str.strip().str.lower().str.replace(" ", "_")

        # Deduplicar nombres de columna (el CSV ASRS tiene Aircraft 1/2, Person 1/2 etc
        # que tras normalizar quedan iguales → df[col] retorna DataFrame en lugar de Series)
        cols = list(_df.columns)
        seen = {}
        new_cols = []
        for c in cols:
            if c in seen:
                seen[c] += 1
                new_cols.append(f"{c}.{seen[c]}")
            else:
                seen[c] = 0
                new_cols.append(c)
        _df.columns = new_cols

        # Convertir columnas numéricas que quedaron como string
        for col in _df.columns:
            try:
                _df[col] = pd.to_numeric(_df[col])
            except (ValueError, TypeError):
                pass

        # Eliminar columna unnamed si existe (índice original del CSV)
        unnamed_cols = [c for c in _df.columns if "unnamed" in c]
        _df.drop(columns=unnamed_cols, inplace=True, errors="ignore")

        return {"status": "success", "rows": len(_df), "columns": list(_df.columns),
                "message": f"Dataset cargado: {len(_df)} registros, {len(_df.columns)} columnas."}
    except HTTPException: raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/dashboard/overview")
def dashboard_overview():
    df = _get_df()
    obj_cols = _get_object_columns(df)

    overview = {
        "total_incidents": len(df),
        "total_columns": len(df.columns),
        "object_columns": obj_cols,  # ← send categorical cols to frontend
        "date_range": None,
        "top_type": None
    }

    # Try to find a meaningful categorical for "top type"
    # ASRS uses columns like: report_1, events.1, assessments, component, person_1 etc.
    type_candidates = [c for c in obj_cols if any(k in c for k in
        ["report", "event", "assessment", "component", "person", "place", "environment"])]
    if type_candidates:
        tc = type_candidates[0]
        top = df[tc].value_counts().head(3)
        overview["top_type"] = {"column": tc, "data": {str(k): int(v) for k, v in top.items()}}

    # Date range: YYYYMM values (ej: 201601) viven en la columna 'time'
    # pandas lee el header agrupado 'Time' y lo convierte en 'time'
    date_col = next((c for c in df.columns if c == "time"), None)
    if date_col is None:
        date_col = next((c for c in df.columns if "date" in c or "time" in c), None)
    if date_col:
        try:
            sample = df[date_col].dropna()
            if sample.dtype in ["int64","float64"]:
                vals = sample.astype(int)
                # YYYYMM format: 6 digits, between 200001 and 203012
                if vals.min() > 200001 and vals.max() < 203012:
                    year_min = vals.min() // 100
                    month_min = vals.min() % 100
                    year_max = vals.max() // 100
                    month_max = vals.max() % 100
                    overview["date_range"] = {
                        "from": f"{year_min}-{month_min:02d}",
                        "to": f"{year_max}-{month_max:02d}"
                    }
        except Exception:
            pass

    return overview


# ═══════════════════════════════════════════════════════════════════════════════
# EDA
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/eda/columns")
def eda_columns():
    """Return ONLY object/categorical columns for the distribution dropdown.
       Also exclude columns that contain numeric-like data (time codes, IDs)."""
    df = _get_df()
    cols = _get_object_columns(df)
    # Exclude columns with too many unique values (likely IDs) or too few (boring)
    filtered = []
    for c in cols:
        nunique = df[c].nunique()
        if nunique < 3 or nunique > 500:
            continue
        # Skip columns whose values look like numbers (time codes like 201601, franja horaria)
        sample = df[c].dropna().head(20).astype(str)
        numeric_ratio = sample.str.match(r"^\d+$").mean()
        if numeric_ratio > 0.5:
            continue
        filtered.append(c)
    return {"columns": filtered if filtered else cols}

@app.get("/eda/distribution/{column}")
def eda_distribution(column: str, top_n: int = 15):
    df = _get_df()
    if column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Columna '{column}' no encontrada.")
    counts = df[column].value_counts().head(top_n)
    return {
        "column": column,
        "data": [{"label": str(k), "value": int(v)} for k, v in counts.items()],
        "total_unique": int(df[column].nunique())
    }

@app.get("/eda/timeline")
def eda_timeline():
    """
    Timeline from 'date' column which has YYYYMM format (e.g. 201601 = Jan 2016).
    Parses into proper year-month labels and groups by month.
    """
    df = _get_df()

    # Find the column with YYYYMM dates — in this CSV pandas names it 'time'
    date_col = next((c for c in df.columns if c == "time"), None)
    if date_col is None:
        date_col = next((c for c in df.columns if "date" in c or "time" in c), None)

    if date_col and df[date_col].dtype in ["int64", "float64"]:
        try:
            vals = df[date_col].dropna().astype(int)
            # Confirm YYYYMM format (6-digit, range 200001–203012)
            if vals.min() > 200001 and vals.max() < 203012:
                # Convert YYYYMM → "YYYY-MM" string
                df["_date_label"] = vals.apply(lambda v: f"{v // 100}-{v % 100:02d}")
                counts = df["_date_label"].value_counts().sort_index()
                df.drop(columns=["_date_label"], inplace=True, errors="ignore")
                return {
                    "column_used": date_col,
                    "type": "monthly",
                    "data": [{"period": str(k), "count": int(v)} for k, v in counts.items()]
                }
        except Exception:
            pass

    # Fallback: try parsing any date/time column
    date_candidates = [c for c in df.columns if any(k in c for k in ["date", "time"])]
    for dc in date_candidates:
        try:
            parsed = pd.to_datetime(df[dc], errors="coerce")
            valid = parsed.dropna()
            if len(valid) > 10:
                df["_ym"] = parsed.dt.to_period("M").astype(str)
                counts = df["_ym"].value_counts().sort_index()
                df.drop(columns=["_ym"], inplace=True, errors="ignore")
                return {
                    "column_used": dc,
                    "type": "monthly",
                    "data": [{"period": str(k), "count": int(v)} for k, v in counts.items()]
                }
        except Exception:
            continue

    # Last fallback: first categorical
    obj_cols = _get_object_columns(df)
    if obj_cols:
        fc = obj_cols[0]
        counts = df[fc].value_counts().head(20).sort_index()
        return {
            "column_used": fc,
            "type": "categorical",
            "data": [{"period": str(k), "count": int(v)} for k, v in counts.items()]
        }

    return {"error": "No se encontró columna de tiempo.", "columns": list(df.columns)}


# ═══════════════════════════════════════════════════════════════════════════════
# NLP
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/nlp/top-keywords")
def nlp_top_keywords(text_column: Optional[str] = None, top_n: int = 30):
    df = _get_df()
    if text_column is None:
        text_column = _get_text_column(df)
    if text_column is None or text_column not in df.columns:
        raise HTTPException(status_code=404, detail="No se encontró columna de texto.")

    texts = df[text_column].dropna().astype(str).apply(_clean_text)
    texts = texts[texts.str.len() > 0]
    if len(texts) == 0:
        return {"error": "No hay texto procesable."}

    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)
    scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
    terms = vectorizer.get_feature_names_out()
    top_indices = scores.argsort()[-top_n:][::-1]
    return {
        "text_column": text_column,
        "keywords": [{"term": terms[i], "score": round(float(scores[i]), 4)} for i in top_indices]
    }

@app.get("/nlp/bigrams")
def nlp_bigrams(text_column: Optional[str] = None, top_n: int = 20):
    df = _get_df()
    if text_column is None:
        text_column = _get_text_column(df)
    if text_column is None or text_column not in df.columns:
        raise HTTPException(status_code=404, detail="No se encontró columna de texto.")

    texts = df[text_column].dropna().astype(str).apply(_clean_text)
    all_tokens = []
    for t in texts:
        tokens = t.split()
        all_tokens.extend(zip(tokens[:-1], tokens[1:]))
    bigram_counts = Counter(all_tokens).most_common(top_n)
    return {
        "text_column": text_column,
        "bigrams": [{"term": f"{a} {b}", "count": c} for (a, b), c in bigram_counts]
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/clustering/kmeans")
def clustering_kmeans(n_clusters: int = 5, text_column: Optional[str] = None):
    df = _get_df()
    if text_column is None:
        text_column = _get_text_column(df)
    if text_column is None or text_column not in df.columns:
        raise HTTPException(status_code=404, detail="No se encontró columna de texto para clustering.")

    texts = df[text_column].dropna().astype(str).apply(_clean_text)
    texts = texts[texts.str.len() > 0]
    if len(texts) < n_clusters:
        raise HTTPException(status_code=400, detail=f"Se necesitan al menos {n_clusters} documentos.")

    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(tfidf)

    clusters_info = []
    for i in range(n_clusters):
        center = km.cluster_centers_[i]
        top_idx = center.argsort()[-8:][::-1]
        clusters_info.append({
            "cluster_id": i,
            "size": int((labels == i).sum()),
            "top_terms": [{"term": terms[j], "weight": round(float(center[j]), 4)} for j in top_idx]
        })

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(tfidf.toarray())
    sample_size = min(len(coords), 800)
    sample_idx = np.random.choice(len(coords), sample_size, replace=False)
    scatter = [
        {"x": round(float(coords[i][0]), 3), "y": round(float(coords[i][1]), 3), "cluster": int(labels[i])}
        for i in sample_idx
    ]
    return {
        "n_clusters": n_clusters, "text_column": text_column,
        "clusters": clusters_info, "scatter": scatter,
        "explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_]
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ML — RISK PROFILES (fixed target detection for ASRS column names)
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/ml/risk-profiles")
def ml_risk_profiles(target_column: Optional[str] = None):
    df = _get_df()

    # ASRS columns are like: report_1, report_1.1, events.1, assessments, component, person_1...
    # Strategy: find the best categorical target — moderate unique count, good coverage
    obj_cols = _get_object_columns(df)

    if target_column and target_column in df.columns:
        target_col = target_column
    else:
        # Score each object column: prefer 3-30 unique values with >80% coverage
        best_col, best_score = None, -1
        for c in obj_cols:
            nunique = df[c].nunique()
            coverage = df[c].notna().mean()
            if 2 <= nunique <= 50 and coverage > 0.5:
                # Prefer columns whose name suggests classification relevance
                name_bonus = 0
                for kw in ["report", "event", "assessment", "component", "person", "environment", "place"]:
                    if kw in c:
                        name_bonus = 5
                        break
                score = coverage * 10 - abs(nunique - 10) * 0.3 + name_bonus
                if score > best_score:
                    best_score = score
                    best_col = c
        target_col = best_col

    if target_col is None:
        raise HTTPException(status_code=400, detail="No se encontró columna objetivo adecuada para ML.")

    df_clean = df.dropna(subset=[target_col]).copy()
    value_counts = df_clean[target_col].value_counts()
    valid_classes = value_counts[value_counts >= 5].index.tolist()  # lowered threshold
    df_clean = df_clean[df_clean[target_col].isin(valid_classes)]

    if len(df_clean) < 30:
        raise HTTPException(status_code=400, detail="Insuficientes muestras para modelado ML.")

    le_target = LabelEncoder()
    y = le_target.fit_transform(df_clean[target_col].astype(str))

    # FILTRO CLAVE: eliminar columnas con >90% NaN antes de usar como features
    # En ASRS la mayoría de columnas (aircraft_2, person_2, component, etc) están casi vacías
    feature_cols = [c for c in df_clean.columns if c != target_col]
    feature_cols = [c for c in feature_cols if df_clean[c].notna().mean() > 0.10]

    X_parts, feature_names = [], []

    for col in feature_cols:
        if df_clean[col].dtype in ["int64", "float64"]:
            X_parts.append(df_clean[col].fillna(0).values.reshape(-1, 1))
            feature_names.append(col)
        elif df_clean[col].dtype == "object":
            # Solo usar si tiene más de 1 valor real (no solo UNKNOWN)
            real_vals = df_clean[col].dropna().nunique()
            if real_vals < 2:
                continue
            le = LabelEncoder()
            try:
                encoded = le.fit_transform(df_clean[col].fillna("UNKNOWN").astype(str)).reshape(-1, 1)
                X_parts.append(encoded)
                feature_names.append(col)
            except Exception:
                continue

    if not X_parts:
        raise HTTPException(status_code=400, detail="No se encontraron características útiles.")

    X = np.hstack(X_parts)

    # If only 1 class after filtering, can't classify
    if len(np.unique(y)) < 2:
        raise HTTPException(status_code=400, detail="Solo hay una clase en el objetivo. No se puede clasificar.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    importances = rf.feature_importances_
    top_feat_idx = importances.argsort()[-10:][::-1]
    feature_importance = [
        {"feature": feature_names[i], "importance": round(float(importances[i]), 4)}
        for i in top_feat_idx
    ]

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_clean = {}
    for k, v in report.items():
        if isinstance(v, dict):
            report_clean[str(k)] = {kk: round(float(vv), 3) for kk, vv in v.items()}
        else:
            report_clean[str(k)] = round(float(v), 3) if isinstance(v, (int, float)) else v

    return {
        "target_column": target_col,
        "classes": le_target.classes_.tolist(),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "feature_importance": feature_importance,
        "classification_report": report_clean,
        "accuracy": round(float((y_pred == y_test).mean()), 4)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MITIGATION
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/mitigation/strategies")
def mitigation_strategies():
    df = _get_df()
    strategies = []
    text_col = _get_text_column(df)

    if text_col:
        all_text = " ".join(df[text_col].dropna().astype(str).str.lower())

        checks = [
            (["communication","comunicación","misunderstanding","confusion","unclear","ambiguous","misread","not informed"],
             {"id":1,"category":"Comunicación","title":"Protocolos de Comunicación Estandarizados",
              "description":"Implementar y reforzar fraseología estandarizada y listas de verificación de comunicación para todas las interacciones ATC-tripulación, especialmente durante fases críticas del vuelo.",
              "evidence_strength":"Alta","priority":1}),
            (["fatigue","fatiga","tired","exhausted","long duty","sleep"],
             {"id":2,"category":"Gestión de Fatiga","title":"Sistema de Monitoreo de Fatiga de Tripulación",
              "description":"Desplegar sistemas de gestión de riesgo por fatiga (FRMS) con limitaciones de tiempo de servicio, períodos de descanso obligatorios y evaluaciones previas al vuelo.",
              "evidence_strength":"Alta","priority":2}),
            (["workload","carga de trabajo","overloaded","too many","task saturation"],
             {"id":3,"category":"Gestión de Carga de Trabajo","title":"Redistribución de Carga en Cabina",
              "description":"Rediseñar protocolos de asignación de tareas para equilibrar la carga cognitiva entre los miembros de la tripulación durante fases de alta complejidad.",
              "evidence_strength":"Media","priority":3}),
            (["training","entrenamiento","skill","competency","unfamiliar","inexperienced"],
             {"id":4,"category":"Mejora de Entrenamiento","title":"Entrenamiento Basado en Competencias",
              "description":"Diseñar programas de entrenamiento basados en escenarios que aborden los patrones de error más frecuentes identificados en los datos ASRS.",
              "evidence_strength":"Alta","priority":4}),
            (["equipment","equipo","malfunction","failure","instrument","system failure"],
             {"id":5,"category":"Equipos y Sistemas","title":"Mejora de Interfaz Humano-Máquina",
              "description":"Revisar y mejorar los paneles de instrumentos y sistemas de alerta en la cabina para reducir la confusión del operador y mejorar la conciencia situacional.",
              "evidence_strength":"Media","priority":5}),
        ]
        for keywords, strategy in checks:
            if any(kw in all_text for kw in keywords):
                strategies.append(strategy)

    strategies.append({
        "id": 6, "category": "Procedimiento y Política",
        "title": "Refuerzo de Cultura de Reporte de Errores",
        "description": "Fortalecer la cultura de reporte no punitivo estableciendo ciclos de retroalimentación regulares desde el análisis ASRS hacia las unidades operacionales.",
        "evidence_strength": "Media", "priority": 6
    })
    return {"strategies": strategies, "total": len(strategies)}


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
