from __future__ import annotations

import os
import json
from datetime import datetime
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from airflow import DAG
from airflow.operators.python import PythonOperator

ARTIFACTS_DIR = "/opt/airflow/artifacts"

def ensure_dirs() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def prepare_data(**context):
    ensure_dirs()
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    paths = {
        "X_train": os.path.join(ARTIFACTS_DIR, "iris_X_train.csv"),
        "X_test": os.path.join(ARTIFACTS_DIR, "iris_X_test.csv"),
        "y_train": os.path.join(ARTIFACTS_DIR, "iris_y_train.csv"),
        "y_test": os.path.join(ARTIFACTS_DIR, "iris_y_test.csv"),
    }
    X_train.to_csv(paths["X_train"], index=False)
    X_test.to_csv(paths["X_test"], index=False)
    pd.Series(y_train).to_csv(paths["y_train"], index=False)
    pd.Series(y_test).to_csv(paths["y_test"], index=False)
    context["ti"].xcom_push(key="paths", value=paths)


def train(**context):
    paths = context["ti"].xcom_pull(key="paths")
    X_train = pd.read_csv(paths["X_train"])
    y_train = pd.read_csv(paths["y_train"]).squeeze("columns")
    # Use a RandomForest classifier to diversify from baseline
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    model_path = os.path.join(ARTIFACTS_DIR, "iris_rf.joblib")
    joblib.dump(model, model_path)
    context["ti"].xcom_push(key="model_path", value=model_path)


def evaluate(**context):
    paths = context["ti"].xcom_pull(key="paths")
    model_path = context["ti"].xcom_pull(key="model_path")
    X_test = pd.read_csv(paths["X_test"])
    y_test = pd.read_csv(paths["y_test"]).squeeze("columns")
    model = joblib.load(model_path)
    preds = model.predict(X_test)
    # Probabilities for macro ROC-AUC across classes
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        macro_roc_auc = float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))
    else:
        macro_roc_auc = None
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro")),
        "macro_roc_auc": macro_roc_auc,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(ARTIFACTS_DIR, "iris_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(ARTIFACTS_DIR, "iris_classification_report.json"), "w") as f:
        json.dump(classification_report(y_test, preds, output_dict=True), f, indent=2)
    # Persist confusion matrix and feature importances
    cm = confusion_matrix(y_test, preds)
    with open(os.path.join(ARTIFACTS_DIR, "iris_confusion_matrix.json"), "w") as f:
        json.dump(cm.tolist(), f, indent=2)
    if hasattr(model, "feature_importances_"):
        with open(os.path.join(ARTIFACTS_DIR, "iris_feature_importances.json"), "w") as f:
            json.dump({
                "columns": X_test.columns.tolist(),
                "importances": model.feature_importances_.tolist()
            }, f, indent=2)


def make_dag():
    with DAG(
        dag_id="iris_classification",
        schedule_interval=None,
        start_date=datetime(2024,1,1),
        catchup=False,
        tags=["ml","lab2"],
        default_args={"owner":"mlops"}
    ) as dag:
        t_prep = PythonOperator(task_id="prepare_data", python_callable=prepare_data)
        t_train = PythonOperator(task_id="train", python_callable=train)
        t_eval = PythonOperator(task_id="evaluate", python_callable=evaluate)
        t_prep >> t_train >> t_eval
    return dag


dag = make_dag()
