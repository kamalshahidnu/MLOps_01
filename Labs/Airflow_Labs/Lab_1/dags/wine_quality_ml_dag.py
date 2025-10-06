from __future__ import annotations

import os
import json
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import joblib

from airflow import DAG
from airflow.operators.python import PythonOperator


ARTIFACTS_DIR = "/opt/airflow/artifacts"
# Switch to white wine dataset to differentiate from the reference implementation
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
CSV_SEP = ";"


def ensure_dirs() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def download_data(**context) -> str:
    ensure_dirs()
    df = pd.read_csv(DATA_URL, sep=CSV_SEP)
    raw_path = os.path.join(ARTIFACTS_DIR, "winequality-white.csv")
    df.to_csv(raw_path, index=False)
    context["ti"].xcom_push(key="raw_path", value=raw_path)
    return raw_path


def split_and_save(**context) -> dict:
    raw_path = context["ti"].xcom_pull(key="raw_path")
    df = pd.read_csv(raw_path)

    # Binary classification: quality >= 7 as good wine
    df["label"] = (df["quality"] >= 7).astype(int)
    X = df.drop(columns=["quality", "label"])  # keep only features
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    paths = {
        "X_train": os.path.join(ARTIFACTS_DIR, "X_train.csv"),
        "X_test": os.path.join(ARTIFACTS_DIR, "X_test.csv"),
        "y_train": os.path.join(ARTIFACTS_DIR, "y_train.csv"),
        "y_test": os.path.join(ARTIFACTS_DIR, "y_test.csv"),
    }

    X_train.to_csv(paths["X_train"], index=False)
    X_test.to_csv(paths["X_test"], index=False)
    y_train.to_csv(paths["y_train"], index=False)
    y_test.to_csv(paths["y_test"], index=False)

    context["ti"].xcom_push(key="data_paths", value=paths)
    return paths


def train_model(**context) -> str:
    paths = context["ti"].xcom_pull(key="data_paths")
    X_train = pd.read_csv(paths["X_train"])
    y_train = pd.read_csv(paths["y_train"]).squeeze("columns")

    # Use a RandomForest classifier without scaling to add variety and robustness
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    model_path = os.path.join(ARTIFACTS_DIR, "wine_quality_rf.joblib")
    joblib.dump(model, model_path)
    context["ti"].xcom_push(key="model_path", value=model_path)
    return model_path


def evaluate_model(**context) -> dict:
    paths = context["ti"].xcom_pull(key="data_paths")
    model_path = context["ti"].xcom_pull(key="model_path")

    X_test = pd.read_csv(paths["X_test"])
    y_test = pd.read_csv(paths["y_test"]).squeeze("columns")

    model = joblib.load(model_path)
    preds = model.predict(X_test)
    # For ROC-AUC we need probabilities or decision function
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback to predicted labels if probabilities are unavailable
        y_scores = preds

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, y_scores)),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    report = classification_report(y_test, preds, output_dict=True)

    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(ARTIFACTS_DIR, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Persist feature importances to help interpretability
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_.tolist()
        with open(os.path.join(ARTIFACTS_DIR, "feature_importances.json"), "w") as f:
            json.dump({
                "columns": X_test.columns.tolist(),
                "importances": feature_importances
            }, f, indent=2)

    return metrics


def predict_sample(**context) -> dict:
    model_path = context["ti"].xcom_pull(key="model_path")
    model = joblib.load(model_path)

    # Use mean feature values from training set for a demo prediction
    paths = context["ti"].xcom_pull(key="data_paths")
    X_train = pd.read_csv(paths["X_train"])
    sample = X_train.mean().to_frame().T
    pred = int(model.predict(sample)[0])

    with open(os.path.join(ARTIFACTS_DIR, "sample_prediction.json"), "w") as f:
        json.dump({"prediction": pred}, f, indent=2)

    return {"prediction": pred}


def make_dag() -> DAG:
    with DAG(
        dag_id="wine_quality_ml",
        default_args={"owner": "mlops"},
        schedule_interval=None,
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=["ml", "example", "custom"],
    ) as dag:
        t_download = PythonOperator(task_id="download_data", python_callable=download_data)
        t_split = PythonOperator(task_id="split_data", python_callable=split_and_save)
        t_train = PythonOperator(task_id="train_model", python_callable=train_model)
        t_eval = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)
        t_predict = PythonOperator(task_id="predict_sample", python_callable=predict_sample)

        t_download >> t_split >> t_train >> t_eval >> t_predict

    return dag


dag = make_dag()
