from datetime import datetime
import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="wide")

API_URL = os.getenv("API_URL", "http://api:8000/predict")
API_BASE_URL = API_URL.rsplit("/predict", 1)[0]
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
REGISTERED_MODEL_NAME = "iris-rf-classifier"

PRESETS = {
    "Setosa": (5.1, 3.5, 1.4, 0.2),
    "Versicolor": (5.9, 2.8, 4.3, 1.3),
    "Virginica": (6.7, 3.1, 5.6, 2.4),
}

DEFAULTS = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
    "history": [],
    "last_result": None,
}

for key, default_val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_val


def apply_preset(values: tuple[float, float, float, float]) -> None:
    st.session_state["sepal_length"] = values[0]
    st.session_state["sepal_width"] = values[1]
    st.session_state["petal_length"] = values[2]
    st.session_state["petal_width"] = values[3]


def clear_history() -> None:
    st.session_state["history"] = []


def check_api_health() -> bool:
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=2)
        return response.ok
    except requests.exceptions.RequestException:
        return False


def fetch_mlflow_model_info() -> dict | None:
    try:
        model_resp = requests.get(
            f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/registered-models/get",
            params={"name": REGISTERED_MODEL_NAME},
            timeout=2,
        )
        if not model_resp.ok:
            return None
        latest_versions = model_resp.json().get("registered_model", {}).get("latest_versions", [])
        if not latest_versions:
            return None
        latest = latest_versions[0]
        run_id = latest.get("run_id")
        metrics = {}
        if run_id:
            run_resp = requests.get(
                f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/runs/get",
                params={"run_id": run_id},
                timeout=2,
            )
            if run_resp.ok:
                raw_metrics = run_resp.json().get("run", {}).get("data", {}).get("metrics", [])
                metrics = {m["key"]: m["value"] for m in raw_metrics}
        return {
            "name": REGISTERED_MODEL_NAME,
            "version": latest.get("version"),
            "metrics": metrics,
        }
    except (requests.exceptions.RequestException, ValueError, KeyError):
        return None


with st.sidebar:
    st.header("System Status")
    if check_api_health():
        st.success("API: Online")
    else:
        st.error("API: Offline")

    st.divider()
    st.subheader("MLflow Model")
    model_info = fetch_mlflow_model_info()
    if model_info:
        st.write(f"**{model_info['name']}** (v{model_info['version']})")
        col_acc, col_f1 = st.columns(2)
        accuracy = model_info["metrics"].get("accuracy")
        f1_score = model_info["metrics"].get("f1_score")
        with col_acc:
            st.metric("Accuracy", f"{accuracy:.2%}" if accuracy is not None else "N/A")
        with col_f1:
            st.metric("F1-score", f"{f1_score:.2%}" if f1_score is not None else "N/A")
    else:
        st.caption("MLflow registry info is currently unavailable.")

    st.divider()
    st.subheader("Links")
    st.markdown("- [MLflow UI](http://localhost:5000)")
    st.markdown("- [FastAPI Docs](http://localhost:8000/docs)")
    st.caption(f"API: `{API_URL}`")


st.title("🌸 Iris Flower Classifier")
st.write("Adjust flower measurements or choose a preset to predict the Iris species and inspect class probabilities.")

left_col, right_col = st.columns([1.1, 1], gap="large")

with left_col:
    st.subheader("Input Features")
    st.caption("Quick presets")
    preset_cols = st.columns(len(PRESETS))
    for col, (label, values) in zip(preset_cols, PRESETS.items()):
        with col:
            st.button(
                label,
                key=f"preset_{label}",
                on_click=apply_preset,
                args=(values,),
                use_container_width=True,
            )

    with st.form("prediction_form"):
        sepal_length = st.slider("Sepal length (cm)", min_value=4.0, max_value=8.0, step=0.1, key="sepal_length")
        sepal_width = st.slider("Sepal width (cm)", min_value=2.0, max_value=4.5, step=0.1, key="sepal_width")
        petal_length = st.slider("Petal length (cm)", min_value=1.0, max_value=7.0, step=0.1, key="petal_length")
        petal_width = st.slider("Petal width (cm)", min_value=0.1, max_value=2.5, step=0.1, key="petal_width")
        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        if response.ok:
            result = response.json()
            st.session_state["last_result"] = result
            probs = result.get("probabilities", {})
            top_prob = probs.get(result["class_name"])
            st.session_state["history"].insert(
                0,
                {
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Sepal L": sepal_length,
                    "Sepal W": sepal_width,
                    "Petal L": petal_length,
                    "Petal W": petal_width,
                    "Species": result["class_name"].capitalize(),
                    "Confidence": f"{top_prob * 100:.1f}%" if top_prob is not None else "N/A",
                },
            )
        else:
            st.error(f"API error {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not reach the API at {API_URL}: {exc}")

with right_col:
    st.subheader("Prediction Result")
    last_result = st.session_state["last_result"]
    if last_result:
        probs = last_result.get("probabilities", {})
        species = last_result["class_name"].capitalize()
        top_prob = probs.get(last_result["class_name"])

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Predicted Species", species)
        with metric_col2:
            st.metric(
                "Confidence",
                f"{top_prob * 100:.1f}%" if top_prob is not None else f"Class #{last_result['class_id']}",
            )

        if probs:
            st.write("**Class Probabilities**")
            chart_df = pd.DataFrame(
                {"Probability": list(probs.values())},
                index=[name.capitalize() for name in probs.keys()],
            )
            st.bar_chart(chart_df)

            for name, prob in probs.items():
                st.progress(min(max(float(prob), 0.0), 1.0), text=f"{name.capitalize()}: {prob * 100:.1f}%")
    else:
        st.info("Adjust the measurements and click Predict to see the classification result and class probabilities.")

if st.session_state["history"]:
    st.divider()
    hist_header_col, hist_btn_col = st.columns([5, 1])
    with hist_header_col:
        st.subheader("Prediction History")
    with hist_btn_col:
        st.button("Clear history", on_click=clear_history, use_container_width=True)
    st.dataframe(pd.DataFrame(st.session_state["history"]), use_container_width=True, hide_index=True)
