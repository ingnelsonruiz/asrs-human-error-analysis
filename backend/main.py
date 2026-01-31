"""
ASRS Human Error Pattern Analysis — FastAPI Backend
=====================================================
Objective: Develop an analytical model for identification and characterization
of human error patterns in aeronautical operations using data mining and NLP
on the ASRS database.
"""

import os
import re
import warnings
import json
from collections import Counter
from typing import Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk

# ─── Suppress warnings ───────────────────────────────────────────────────────
warnings.filterwarnings("ignore")

# ─── NLTK downloads (silent) ─────────────────────────────────────────────────
for pkg in ["punkt", "stopwords", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ─── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="ASRS Human Error Analysis API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global state ─────────────────────────────────────────────────────────────
_df: Optional[pd.DataFrame] = None
_stop_words = set(stopwords.words("english")) | set(stopwords.words("spanish"))
_stop_words.update(["the","and","was","for","are","but","not","you","all","can",
                    "had","her","one","our","out","has","que","los","las","del",
                    "con","por","una","fue","est","como","sin","sobre","entre",
                    "durante","cuando","donde","porque","mientras","también",
                    "report","reported","incident","aircraft","pilot","crew"])


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _clean_text(text: str) -> str:
    """Basic text cleaning for NLP processing."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-záéíóúñüà-ÿ\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _stop_words and len(t) > 2]
    return " ".join(tokens)


def _get_df() -> pd.DataFrame:
    if _df is None:
        raise HTTPException(status_code=400, detail="No data loaded. Upload CSV first via /upload.")
    return _df


# ═══════════════════════════════════════════════════════════════════════════════
# UPLOAD & HEALTH
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "loaded": _df is not None, "rows": len(_df) if _df is not None else 0}


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload the ASRS CSV dataset.
    Expected path on local: C:\\Fuerza_Aerea\\ASRS_DBOnline.csv
    Upload this file to initialize the analysis engine.
    """
    global _df
    try:
        import io
        content = await file.read()
        # Try multiple encodings
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                _df = pd.read_csv(io.StringIO(content.decode(enc)))
                break
            except (UnicodeDecodeError, Exception):
                continue
        if _df is None:
            raise HTTPException(status_code=400, detail="Could not decode CSV file.")

        # Normalize column names
        _df.columns = _df.columns.str.strip().str.lower().str.replace(" ", "_")

        return {
            "status": "success",
            "rows": len(_df),
            "columns": list(_df.columns),
            "message": f"Dataset loaded: {len(_df)} records, {len(_df.columns)} columns."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. EDA — EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/eda/summary")
def eda_summary():
    """General dataset summary: shape, dtypes, nulls, sample."""
    df = _get_df()
    summary = {}
    for col in df.columns:
        summary[col] = {
            "dtype": str(df[col].dtype),
            "nulls": int(df[col].isnull().sum()),
            "unique": int(df[col].nunique()) if df[col].dtype == "object" else None,
            "sample": str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else None
        }
    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": summary
    }


@app.get("/eda/distribution/{column}")
def eda_distribution(column: str, top_n: int = 15):
    """Value distribution for a given categorical column."""
    df = _get_df()
    if column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found.")
    counts = df[column].value_counts().head(top_n)
    return {
        "column": column,
        "data": [{"label": str(k), "value": int(v)} for k, v in counts.items()],
        "total_unique": int(df[column].nunique())
    }


@app.get("/eda/timeline")
def eda_timeline():
    """
    Incident count over time. Tries common date columns.
    """
    df = _get_df()
    date_candidates = [c for c in df.columns if any(k in c for k in ["date","time","year","month"])]
    if not date_candidates:
        return {"error": "No date column found", "columns": list(df.columns)}

    date_col = date_candidates[0]
    try:
        df["_parsed_date"] = pd.to_datetime(df[date_col], errors="coerce")
        df["_year_month"] = df["_parsed_date"].dt.to_period("M").astype(str)
        timeline = df["_year_month"].value_counts().sort_index()
        return {
            "column_used": date_col,
            "data": [{"period": str(k), "count": int(v)} for k, v in timeline.items()]
        }
    except Exception as e:
        return {"error": str(e), "column_used": date_col}


@app.get("/eda/columns")
def eda_columns():
    """Return all column names for frontend dropdowns."""
    df = _get_df()
    return {"columns": list(df.columns)}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. NLP — TEXT ANALYSIS & PATTERN EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/nlp/top-keywords")
def nlp_top_keywords(text_column: Optional[str] = None, top_n: int = 30):
    """
    Extract top keywords from narrative/text columns using TF-IDF.
    If text_column not specified, auto-detect the most text-rich column.
    """
    df = _get_df()

    if text_column is None:
        # Auto-detect: pick the object column with longest average text
        obj_cols = df.select_dtypes(include="object").columns.tolist()
        if not obj_cols:
            raise HTTPException(status_code=400, detail="No text columns found.")
        text_column = max(obj_cols, key=lambda c: df[c].astype(str).str.len().mean())

    if text_column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{text_column}' not found.")

    texts = df[text_column].dropna().astype(str).apply(_clean_text)
    texts = texts[texts.str.len() > 0]

    if len(texts) == 0:
        return {"error": "No processable text found after cleaning."}

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
    """Extract top bigrams from narrative text."""
    df = _get_df()

    if text_column is None:
        obj_cols = df.select_dtypes(include="object").columns.tolist()
        text_column = max(obj_cols, key=lambda c: df[c].astype(str).str.len().mean()) if obj_cols else None

    if text_column is None or text_column not in df.columns:
        raise HTTPException(status_code=404, detail="No valid text column.")

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
# 3. CLUSTERING — UNSUPERVISED PATTERN DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/clustering/kmeans")
def clustering_kmeans(n_clusters: int = 5, text_column: Optional[str] = None):
    """
    K-Means clustering on TF-IDF vectors of incident narratives.
    Returns cluster labels + top terms per cluster + PCA 2D projection.
    """
    df = _get_df()

    if text_column is None:
        obj_cols = df.select_dtypes(include="object").columns.tolist()
        text_column = max(obj_cols, key=lambda c: df[c].astype(str).str.len().mean()) if obj_cols else None

    if text_column is None or text_column not in df.columns:
        raise HTTPException(status_code=404, detail="No valid text column for clustering.")

    texts = df[text_column].dropna().astype(str).apply(_clean_text)
    valid_idx = texts[texts.str.len() > 0].index
    texts = texts.loc[valid_idx]

    if len(texts) < n_clusters:
        raise HTTPException(status_code=400, detail=f"Need at least {n_clusters} documents.")

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()

    # K-Means
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(tfidf)

    # Top terms per cluster
    clusters_info = []
    for i in range(n_clusters):
        center = km.cluster_centers_[i]
        top_idx = center.argsort()[-8:][::-1]
        clusters_info.append({
            "cluster_id": i,
            "size": int((labels == i).sum()),
            "top_terms": [{"term": terms[j], "weight": round(float(center[j]), 4)} for j in top_idx]
        })

    # PCA 2D for visualization
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(tfidf.toarray())

    # Sample max 800 points for performance
    sample_size = min(len(coords), 800)
    sample_idx = np.random.choice(len(coords), sample_size, replace=False)

    scatter = [
        {"x": round(float(coords[i][0]), 3), "y": round(float(coords[i][1]), 3), "cluster": int(labels[i])}
        for i in sample_idx
    ]

    return {
        "n_clusters": n_clusters,
        "text_column": text_column,
        "clusters": clusters_info,
        "scatter": scatter,
        "explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_]
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ML — RISK PROFILE SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/ml/risk-profiles")
def ml_risk_profiles(target_column: Optional[str] = None):
    """
    Train a Random Forest to segment incidents into risk profiles.
    Auto-selects target (severity/type column) and numeric+encoded features.
    Returns feature importances + classification report.
    """
    df = _get_df()

    # Auto-detect target column (severity, type, category...)
    target_candidates = [c for c in df.columns if any(k in c for k in
                         ["severity", "type", "category", "phase", "event_type", "report_type"])]
    if target_column:
        target_candidates = [target_column] if target_column in df.columns else []

    if not target_candidates:
        raise HTTPException(status_code=400, detail="No suitable target column found. Provide target_column param.")

    target_col = target_candidates[0]
    df_clean = df.dropna(subset=[target_col]).copy()

    # Filter target to top classes (min 20 samples)
    value_counts = df_clean[target_col].value_counts()
    valid_classes = value_counts[value_counts >= 20].index.tolist()
    df_clean = df_clean[df_clean[target_col].isin(valid_classes)]

    if len(df_clean) < 50:
        raise HTTPException(status_code=400, detail="Not enough samples for ML modeling.")

    # Encode target
    le_target = LabelEncoder()
    y = le_target.fit_transform(df_clean[target_col].astype(str))

    # Feature engineering: encode categoricals, use numerics
    feature_cols = [c for c in df_clean.columns if c != target_col]
    X_parts = []
    feature_names = []

    for col in feature_cols:
        if df_clean[col].dtype in ["int64", "float64"]:
            vals = df_clean[col].fillna(0).values.reshape(-1, 1)
            X_parts.append(vals)
            feature_names.append(col)
        elif df_clean[col].dtype == "object":
            le = LabelEncoder()
            try:
                encoded = le.fit_transform(df_clean[col].fillna("UNKNOWN").astype(str)).reshape(-1, 1)
                X_parts.append(encoded)
                feature_names.append(col)
            except Exception:
                continue

    if not X_parts:
        raise HTTPException(status_code=400, detail="No usable features found.")

    X = np.hstack(X_parts)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Results
    importances = rf.feature_importances_
    top_feat_idx = importances.argsort()[-10:][::-1]
    feature_importance = [
        {"feature": feature_names[i], "importance": round(float(importances[i]), 4)}
        for i in top_feat_idx
    ]

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    # Convert to JSON-safe format
    report_clean = {}
    for k, v in report.items():
        if isinstance(v, dict):
            report_clean[str(k)] = {kk: round(float(vv), 3) for kk, vv in v.items()}
        else:
            report_clean[str(k)] = round(float(v), 3) if isinstance(v, (int, float)) else v

    class_labels = le_target.classes_.tolist()

    return {
        "target_column": target_col,
        "classes": class_labels,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_importance": feature_importance,
        "classification_report": report_clean,
        "accuracy": round(float((y_pred == y_test).mean()), 4)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MITIGATION STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/mitigation/strategies")
def mitigation_strategies():
    """
    Generate evidence-based mitigation strategies derived from the identified
    patterns and risk profiles in the ASRS data.
    """
    df = _get_df()

    # Gather pattern signals from the data
    strategies = []

    # Detect communication-related patterns
    text_cols = df.select_dtypes(include="object").columns.tolist()
    text_col = max(text_cols, key=lambda c: df[c].astype(str).str.len().mean()) if text_cols else None

    if text_col:
        all_text = " ".join(df[text_col].dropna().astype(str).str.lower())
        comm_keywords = ["communication", "comunicación", "misunderstanding", "confusion",
                         "unclear", "ambiguous", "misread", "not informed"]
        if any(kw in all_text for kw in comm_keywords):
            strategies.append({
                "id": 1,
                "category": "Communication",
                "title": "Standardized Communication Protocols",
                "description": "Implement and reinforce standardized phraseology and communication checklists "
                               "for all ATC-crew interactions, especially during critical phases of flight.",
                "evidence_strength": "High",
                "priority": 1
            })

        fatigue_kw = ["fatigue", "fatiga", "tired", "exhausted", "long duty", "sleep"]
        if any(kw in all_text for kw in fatigue_kw):
            strategies.append({
                "id": 2,
                "category": "Fatigue Management",
                "title": "Crew Fatigue Monitoring System",
                "description": "Deploy fatigue risk management systems (FRMS) with duty-time limitations, "
                               "mandatory rest periods, and pre-flight fatigue assessments.",
                "evidence_strength": "High",
                "priority": 2
            })

        workload_kw = ["workload", "carga de trabajo", "overloaded", "too many", "task saturation"]
        if any(kw in all_text for kw in workload_kw):
            strategies.append({
                "id": 3,
                "category": "Workload Management",
                "title": "Cockpit Workload Redistribution",
                "description": "Redesign task allocation protocols to balance cognitive workload between "
                               "flight crew members during high-complexity phases.",
                "evidence_strength": "Medium",
                "priority": 3
            })

        training_kw = ["training", "entrenamiento", "skill", "competency", "unfamiliar", "inexperienced"]
        if any(kw in all_text for kw in training_kw):
            strategies.append({
                "id": 4,
                "category": "Training Enhancement",
                "title": "Targeted Competency-Based Training",
                "description": "Design scenario-based training programs targeting the specific error patterns "
                               "most frequently identified in the ASRS data.",
                "evidence_strength": "High",
                "priority": 4
            })

        equip_kw = ["equipment", "equipo", "malfunction", "failure", "instrument", "system failure"]
        if any(kw in all_text for kw in equip_kw):
            strategies.append({
                "id": 5,
                "category": "Equipment & Systems",
                "title": "Human-Machine Interface Improvement",
                "description": "Review and upgrade cockpit instrument displays and alerting systems to reduce "
                               "operator confusion and improve situation awareness.",
                "evidence_strength": "Medium",
                "priority": 5
            })

    # Always include procedural strategy
    strategies.append({
        "id": 6,
        "category": "Procedure & Policy",
        "title": "Error Reporting Culture Reinforcement",
        "description": "Strengthen the non-punitive reporting culture by establishing regular feedback loops "
                       "from ASRS analysis back to operational units.",
        "evidence_strength": "Medium",
        "priority": 6
    })

    return {"strategies": strategies, "total": len(strategies)}


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD OVERVIEW (single-call summary for the frontend)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/dashboard/overview")
def dashboard_overview():
    """
    Quick stats for the main dashboard cards.
    """
    df = _get_df()
    obj_cols = df.select_dtypes(include="object").columns.tolist()

    # Detect key columns
    type_col = next((c for c in df.columns if any(k in c for k in
                     ["type","category","event_type"])), None)
    sev_col = next((c for c in df.columns if "severity" in c), None)
    phase_col = next((c for c in df.columns if "phase" in c), None)

    overview = {
        "total_incidents": len(df),
        "total_columns": len(df.columns),
        "date_range": None,
        "top_type": None,
        "top_severity": None,
        "top_phase": None
    }

    # Date range
    date_col = next((c for c in df.columns if any(k in c for k in ["date","time","year"])), None)
    if date_col:
        try:
            dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
            if not dates.empty:
                overview["date_range"] = {
                    "from": str(dates.min().date()),
                    "to": str(dates.max().date())
                }
        except Exception:
            pass

    if type_col:
        overview["top_type"] = df[type_col].value_counts().head(3).to_dict()
        overview["top_type"] = {str(k): int(v) for k, v in overview["top_type"].items()}

    if sev_col:
        overview["top_severity"] = df[sev_col].value_counts().head(3).to_dict()
        overview["top_severity"] = {str(k): int(v) for k, v in overview["top_severity"].items()}

    if phase_col:
        overview["top_phase"] = df[phase_col].value_counts().head(3).to_dict()
        overview["top_phase"] = {str(k): int(v) for k, v in overview["top_phase"].items()}

    return overview


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
