import argparse
import os
import tempfile
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and track models with MLflow on Breast Cancer dataset")
    parser.add_argument("--model", type=str, default="logreg", choices=["logreg", "random_forest"], help="Model type")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    # Logistic Regression params
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength for LogisticRegression")
    parser.add_argument("--penalty", type=str, default="l2", choices=["l2", "none"], help="Penalty for LogisticRegression")
    parser.add_argument("--max_iter", type=int, default=1000, help="Max iterations for LogisticRegression")
    # RandomForest params
    parser.add_argument("--n_estimators", type=int, default=200, help="Number of trees for RandomForest")
    parser.add_argument("--max_depth", type=int, default=None, help="Max depth for RandomForest (None=unlimited)")
    parser.add_argument("--min_samples_split", type=int, default=2, help="Min samples split for RandomForest")
    parser.add_argument("--experiment_name", type=str, default="breast_cancer_tracking", help="MLflow experiment name")
    parser.add_argument("--tracking_uri", type=str, default=os.getenv("MLFLOW_TRACKING_URI", ""), help="MLflow tracking URI")
    return parser.parse_args()


def load_data(random_state: int, test_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, feature_names


def build_model(model_type: str, args: argparse.Namespace) -> Pipeline:
    if model_type == "logreg":
        clf = LogisticRegression(
            C=args.C,
            penalty=None if args.penalty == "none" else args.penalty,
            max_iter=args.max_iter,
            solver="lbfgs",
            n_jobs=None,
        )
        steps = [("scaler", StandardScaler()), ("clf", clf)]
    else:
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=args.random_state,
            n_jobs=-1,
        )
        steps = [("clf", clf)]
    return Pipeline(steps)


def evaluate_and_log(
    model: Pipeline,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    artifact_dir: str,
) -> Dict[str, float]:
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba") and callable(getattr(model, "predict_proba")):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback: use decision_function if available, else hard predictions
        if hasattr(model, "decision_function"):
            raw = model.decision_function(X_test)
            y_proba = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
        else:
            y_proba = y_pred.astype(float)

    metrics_dict = {
        "accuracy": metrics.accuracy_score(y_test, y_pred),
        "precision": metrics.precision_score(y_test, y_pred, zero_division=0),
        "recall": metrics.recall_score(y_test, y_pred, zero_division=0),
        "f1": metrics.f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": metrics.roc_auc_score(y_test, y_proba),
    }

    # Confusion matrix plot
    fig_cm, ax_cm = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues", ax=ax_cm)
    ax_cm.set_title("Confusion Matrix")
    cm_path = os.path.join(artifact_dir, "confusion_matrix.png")
    fig_cm.tight_layout()
    fig_cm.savefig(cm_path)
    plt.close(fig_cm)

    # ROC curve plot
    fpr, tpr, _ = metrics.roc_curve(y_test, y_proba)
    roc_auc = metrics.auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots(figsize=(5, 5))
    ax_roc.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(loc="lower right")
    roc_path = os.path.join(artifact_dir, "roc_curve.png")
    fig_roc.tight_layout()
    fig_roc.savefig(roc_path)
    plt.close(fig_roc)

    mlflow.log_artifact(cm_path, artifact_path="plots")
    mlflow.log_artifact(roc_path, artifact_path="plots")

    # Calibration plot (optional but useful)
    try:
        prob_true, prob_pred = metrics.calibration_curve(y_test, y_proba, n_bins=10)
        fig_cal, ax_cal = plt.subplots(figsize=(5, 5))
        ax_cal.plot(prob_pred, prob_true, marker="o")
        ax_cal.plot([0, 1], [0, 1], "k--")
        ax_cal.set_title("Calibration Curve")
        ax_cal.set_xlabel("Predicted probability")
        ax_cal.set_ylabel("True probability in bin")
        cal_path = os.path.join(artifact_dir, "calibration_curve.png")
        fig_cal.tight_layout()
        fig_cal.savefig(cal_path)
        plt.close(fig_cal)
        mlflow.log_artifact(cal_path, artifact_path="plots")
    except Exception:
        # Ignore calibration issues for models without good probability outputs
        pass

    return metrics_dict


def main() -> None:
    args = parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    mlflow.set_experiment(args.experiment_name)

    X_train, X_test, y_train, y_test, feature_names = load_data(args.random_state, args.test_size)
    model = build_model(args.model, args)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_type", args.model)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        if args.model == "logreg":
            mlflow.log_param("C", args.C)
            mlflow.log_param("penalty", args.penalty)
            mlflow.log_param("max_iter", args.max_iter)
        else:
            mlflow.log_param("n_estimators", args.n_estimators)
            mlflow.log_param("max_depth", args.max_depth)
            mlflow.log_param("min_samples_split", args.min_samples_split)

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_dict = evaluate_and_log(model, X_train, X_test, y_train, y_test, tmpdir)

        # Log metrics
        for k, v in metrics_dict.items():
            mlflow.log_metric(k, float(v))

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Tag run with useful metadata
        mlflow.set_tags({
            "dataset": "sklearn_breast_cancer",
            "framework": "scikit-learn",
            "stage": "dev",
        })

        print("Logged metrics:")
        for k, v in metrics_dict.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()


