from __future__ import annotations

import os
import json
from datetime import datetime
import joblib
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

from airflow import DAG
from airflow.operators.python import PythonOperator

ARTIFACTS_DIR = "/opt/airflow/artifacts"

def ensure_dirs() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def prepare_data(**context):
    ensure_dirs()
    data = load_diabetes(as_frame=True)
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    paths = {
        "X_train": os.path.join(ARTIFACTS_DIR, "diabetes_X_train.csv"),
        "X_test": os.path.join(ARTIFACTS_DIR, "diabetes_X_test.csv"),
        "y_train": os.path.join(ARTIFACTS_DIR, "diabetes_y_train.csv"),
        "y_test": os.path.join(ARTIFACTS_DIR, "diabetes_y_test.csv"),
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
    # Switch to RandomForestRegressor for non-linear modeling and importance
    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    model_path = os.path.join(ARTIFACTS_DIR, "diabetes_rf.joblib")
    joblib.dump(model, model_path)
    context["ti"].xcom_push(key="model_path", value=model_path)


def evaluate(**context):
    paths = context["ti"].xcom_pull(key="paths")
    model_path = context["ti"].xcom_pull(key="model_path")
    X_test = pd.read_csv(paths["X_test"])
    y_test = pd.read_csv(paths["y_test"]).squeeze("columns")
    model = joblib.load(model_path)
    preds = model.predict(X_test)
    metrics = {
        "r2": float(r2_score(y_test, preds)),
        "rmse": float(mean_squared_error(y_test, preds, squared=False)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(ARTIFACTS_DIR, "diabetes_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    # Persist feature importances for interpretability
    if hasattr(model, "feature_importances_"):
        with open(os.path.join(ARTIFACTS_DIR, "diabetes_feature_importances.json"), "w") as f:
            json.dump({
                "columns": X_test.columns.tolist(),
                "importances": model.feature_importances_.tolist()
            }, f, indent=2)


def make_dag():
    with DAG(
        dag_id="diabetes_regression",
        schedule_interval=None,
        start_date=datetime(2024,1,1),
        catchup=False,
        tags=["ml","lab3"],
        default_args={"owner":"mlops"}
    ) as dag:
        t_prep = PythonOperator(task_id="prepare_data", python_callable=prepare_data)
        t_train = PythonOperator(task_id="train", python_callable=train)
        t_eval = PythonOperator(task_id="evaluate", python_callable=evaluate)
        t_prep >> t_train >> t_eval
    return dag


dag = make_dag()
