# Basic MLflow Experiment Tracking (Breast Cancer)

This lab replicates an easy MLflow tracking workflow inspired by the Experiment_Tracking_Labs and modifies it by:
- Using the Breast Cancer dataset (from scikit-learn)
- Adding richer metrics (accuracy, precision, recall, f1, ROC AUC)
- Logging artifacts: confusion matrix, ROC curve, calibration curve
- Including a simple Streamlit dashboard to browse runs and artifacts

Reference: `https://github.com/raminmohammadi/MLOps/tree/main/Labs/Experiment_Tracking_Labs`

## Structure

```
Labs/Experiment_Tracking_Labs/Basic_MLflow_BreastCancer/
  README.md
  requirements.txt
  train.py
  dashboard/
    app.py
```

## Setup

```bash
# From repo root
python3 -m venv .venv
source .venv/bin/activate
pip install -r Labs/Experiment_Tracking_Labs/Basic_MLflow_BreastCancer/requirements.txt
```

Optionally, set a remote/local MLflow tracking URI (otherwise local `mlruns/` is used):
```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

## Run Training

```bash
python Labs/Experiment_Tracking_Labs/Basic_MLflow_BreastCancer/train.py \
  --model logreg \
  --test_size 0.2 \
  --random_state 42 \
  --C 1.0 --penalty l2 --max_iter 1000
```

RandomForest example:
```bash
python Labs/Experiment_Tracking_Labs/Basic_MLflow_BreastCancer/train.py \
  --model random_forest \
  --n_estimators 300 --max_depth 8 --min_samples_split 4
```

## View MLflow UI

Local file store UI:
```bash
mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5000
```

If you set `MLFLOW_TRACKING_URI`, ensure the server matches it.

## Streamlit Dashboard

```bash
streamlit run Labs/Experiment_Tracking_Labs/Basic_MLflow_BreastCancer/dashboard/app.py
```

Use the left sidebar to set the tracking URI (if any), pick an experiment (`breast_cancer_tracking` by default), select a run, and view parameters, metrics, and plots.

## Notes
- Artifacts are saved under the run's `plots` artifact path.
- The model is logged via `mlflow.sklearn.log_model`.
- This lab is intentionally minimal to get you productive quickly.


