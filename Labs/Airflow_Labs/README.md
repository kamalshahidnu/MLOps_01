### Airflow_Labs (Custom) — Three Labs

Reference (followed and adapted): https://github.com/raminmohammadi/MLOps/tree/main/Labs/Airflow_Labs

#### Structure (mirrors instructional repo labs)
```
Airflow_Labs/
  .env
  docker-compose.yaml
  airflow/
    dags/
      Lab_1/
        airflow.py
        wine_quality_ml_dag.py
      Lab_2/
        airflow.py
        iris_classification_dag.py
      Lab_3/
        airflow.py
        diabetes_regression_dag.py
    logs/
    plugins/
  requirements/
    requirements.txt
```

#### Lab 1 — Wine Quality (binary classification)
- Dataset: UCI Wine Quality (white) downloaded at runtime
- Model: RandomForestClassifier (no scaling needed)
- Outputs: `wine_quality_rf.joblib`, `metrics.json` (includes ROC-AUC), `classification_report.json`, `feature_importances.json`, `sample_prediction.json`
- DAG ID: `wine_quality_ml`

#### Lab 2 — Iris Classification
- Dataset: sklearn Iris (bundled)
- Model: RandomForestClassifier
- Outputs: `iris_X_train.csv`, `iris_X_test.csv`, `iris_y_train.csv`, `iris_y_test.csv`, `iris_rf.joblib`, `iris_metrics.json` (includes macro ROC-AUC), `iris_classification_report.json`, `iris_confusion_matrix.json`, `iris_feature_importances.json`
- DAG ID: `iris_classification`

#### Lab 3 — Diabetes Regression
- Dataset: sklearn Diabetes (bundled)
- Model: RandomForestRegressor
- Outputs: `diabetes_*` CSVs, `diabetes_rf.joblib`, `diabetes_metrics.json` (includes MAE), `diabetes_feature_importances.json`
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
rm -rf airflow/logs
```
