import os
from typing import Dict, List

import mlflow
import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient


def get_client(tracking_uri: str | None) -> MlflowClient:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()


def list_experiments(client: MlflowClient) -> Dict[str, str]:
    exps = client.search_experiments()
    return {exp.name: exp.experiment_id for exp in exps}


def get_runs_df(client: MlflowClient, experiment_id: str) -> pd.DataFrame:
    runs = client.search_runs([experiment_id], order_by=["attributes.start_time DESC"])
    records: List[Dict] = []
    for r in runs:
        rec: Dict = {
            "run_id": r.info.run_id,
            "status": r.info.status,
            "start_time": r.info.start_time,
            "end_time": r.info.end_time,
        }
        rec.update({f"param_{k}": v for k, v in r.data.params.items()})
        rec.update({f"metric_{k}": v for k, v in r.data.metrics.items()})
        records.append(rec)
    return pd.DataFrame.from_records(records)


def show_artifacts(client: MlflowClient, run_id: str) -> None:
    st.subheader("Artifacts")
    imgs = []
    for art in client.list_artifacts(run_id, path="plots"):
        if art.path.endswith(".png"):
            imgs.append(art.path)
    for img_path in imgs:
        uri = client.download_artifacts(run_id, img_path)
        st.image(uri, caption=img_path)


def main() -> None:
    st.set_page_config(page_title="MLflow Dashboard", layout="wide")
    st.title("MLflow Experiment Dashboard")

    tracking_uri = st.sidebar.text_input("MLflow Tracking URI", os.getenv("MLFLOW_TRACKING_URI", ""))
    client = get_client(tracking_uri if tracking_uri.strip() else None)

    exp_map = list_experiments(client)
    if not exp_map:
        st.info("No experiments found. Run a training script first.")
        return

    exp_name = st.sidebar.selectbox("Experiment", sorted(exp_map.keys()))
    exp_id = exp_map[exp_name]

    df = get_runs_df(client, exp_id)
    if df.empty:
        st.info("No runs yet. Trigger a run and refresh.")
        return

    st.subheader("Runs")
    st.dataframe(df, use_container_width=True, hide_index=True)

    run_id = st.sidebar.selectbox("Select Run", list(df["run_id"]))
    if run_id:
        st.subheader(f"Run: {run_id}")
        run = client.get_run(run_id)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Parameters**")
            st.json(run.data.params)
        with col2:
            st.markdown("**Metrics**")
            st.json(run.data.metrics)

        show_artifacts(client, run_id)


if __name__ == "__main__":
    main()


