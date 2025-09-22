from pathlib import Path
import joblib
from typing import Tuple
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "iris_model.pkl"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please run train.py first.")
    return joblib.load(MODEL_PATH)


def make_features(petal_length: float, sepal_length: float, petal_width: float, sepal_width: float):
    return [[sepal_length, sepal_width, petal_length, petal_width]]


def predict_class(model, features) -> Tuple[int, str, float]:
    pred = model.predict(features)
    proba = None
    class_name = None
    try:
        probas = model.predict_proba(features)
        class_idx = int(pred[0])
        proba = float(np.max(probas[0]))
    except Exception:
        proba = 1.0

    # Attempt to map to iris target names if available
    try:
        from sklearn.datasets import load_iris

        iris = load_iris()
        class_name = iris.target_names[int(pred[0])]
    except Exception:
        class_name = str(int(pred[0]))

    return int(pred[0]), class_name, proba
