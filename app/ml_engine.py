"""
ml_engine.py
Core AI module for A World Away — Exoplanet Classifier
------------------------------------------------------
🪐 Responsibilities:
- Load model + scaler
- Apply teammate’s data cleaning pipeline
- Predict exoplanet classification (single or batch)
- Optimize model hyperparameters via Optuna
- Generate SHAP explainability summaries
"""

import pandas as pd
import numpy as np
import joblib
import shap
import optuna
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

# Load pre-trained artifacts
MODEL_PATH = "models/exoplanet_classifier.pkl"
SCALER_PATH = "models/scaler.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("✅ Loaded pretrained model and scaler.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model, scaler = None, None

# 🧹 Placeholder for teammate's data cleaning function
def clean_and_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    🔧 Apply teammate's custom preprocessing steps here:
    - Handle mission-based features
    - Impute missing values
    - Encode categorical variables
    - Align with model’s expected schema
    """
    # TODO: Insert teammate's data cleaning code
    return df

# 🧠 Prediction function
def predict(input_data: pd.DataFrame):
    df = clean_and_preprocess(input_data.copy())
    df_scaled = scaler.transform(df)
    preds = model.predict(df_scaled)
    probs = model.predict_proba(df_scaled)[:, 1]
    return pd.DataFrame({
        "Prediction": preds,
        "Probability": probs
    })

# ⚙️ Optuna optimization function
def objective(trial, X, y):
    from xgboost import XGBClassifier
    n_estimators = trial.suggest_int("n_estimators", 200, 600)
    max_depth = trial.suggest_int("max_depth", 4, 10)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1)
    subsample = trial.suggest_float("subsample", 0.6, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
    
    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pr_aucs = []
    for train_idx, val_idx in cv.split(X, y):
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_proba = clf.predict_proba(X.iloc[val_idx])[:, 1]
        pr_aucs.append(average_precision_score(y.iloc[val_idx], y_proba))
    return np.mean(pr_aucs)

def run_optuna(X, y, n_trials=20):
    study = optuna.create_study(direction="maximize", study_name="exoplanet_pr_auc")
    study.optimize(lambda t: objective(t, X, y), n_trials=n_trials)
    logger.info(f"🏆 Best trial PR-AUC: {study.best_value:.4f}")
    return study.best_params, study.best_value

# 🧩 SHAP explainability
def explain(sample: pd.DataFrame):
    df = clean_and_preprocess(sample.copy())
    df_scaled = scaler.transform(df)
    explainer = shap.Explainer(model)
    shap_values = explainer(df_scaled)
    return shap_values
