### Airflow_Labs (Custom) — Three Labs

Reference (followed and adapted): https://github.com/raminmohammadi/MLOps/tree/main/Labs/Airflow_Labs

#### Structure (mirrors instructional repo labs)
```
Airflow_Labs/
  .env
  docker-compose.yaml
  airflow/
    dags/
      lab1/
        wine_quality_ml_dag.py
      lab2/
        iris_classification_dag.py
      lab3/
        diabetes_regression_dag.py
    logs/
    plugins/
  artifacts/
  requirements/
    requirements.txt
```

#### Lab 1 — Wine Quality (binary classification)
- Dataset: UCI Wine Quality (red) downloaded at runtime
- Model: StandardScaler + LogisticRegression
- Outputs: `wine_quality_lr.joblib`, `metrics.json`, `classification_report.json`, `sample_prediction.json`
- DAG ID: `wine_quality_ml`

#### Lab 2 — Iris Classification
- Dataset: sklearn Iris (bundled)
- Model: StandardScaler + LogisticRegression
- Outputs: `iris_X_train.csv`, `iris_X_test.csv`, `iris_y_train.csv`, `iris_y_test.csv`, `iris_lr.joblib`, `iris_metrics.json`, `iris_classification_report.json`
- DAG ID: `iris_classification`

#### Lab 3 — Diabetes Regression
- Dataset: sklearn Diabetes (bundled)
- Model: StandardScaler + Ridge
- Outputs: `diabetes_*` CSVs, `diabetes_ridge.joblib`, `diabetes_metrics.json`
- DAG ID: `diabetes_regression`

#### Run
```bash
cd /Users/shahidkamal/Documents/MLOps_01/Labs/Airflow_Labs
docker compose up airflow-init
docker compose up -d webserver scheduler
open http://localhost:8080  # login: admin/admin
```
Trigger DAGs from UI or CLI, e.g.:
```bash
docker compose exec scheduler airflow dags trigger wine_quality_ml
docker compose exec scheduler airflow dags trigger iris_classification
docker compose exec scheduler airflow dags trigger diabetes_regression
```

#### Clean up
```bash
docker compose down -v
rm -rf artifacts airflow/logs
```
