from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="NYC Taxi Fare Predictor", layout="wide")
st.title("NYC Taxi Fare Prediction")
st.caption("Predict expected total fare and inspect local feature contributions.")

base_dir = Path(__file__).resolve().parent
artifact_path = base_dir / "outputs" / "models" / "best_fare_model.joblib"

if not artifact_path.exists():
    st.error("Model artifact not found. Run: python src/train.py --download-if-missing")
    st.stop()

artifact = joblib.load(artifact_path)
model = artifact["model"]
feature_columns = artifact["feature_columns"]
model_name = artifact["model_name"]

st.subheader(f"Loaded model: {model_name}")

with st.sidebar:
    st.header("Input Features")
    trip_distance = st.number_input("trip_distance", min_value=0.1, max_value=100.0, value=3.0, step=0.1)
    trip_minutes = st.number_input("trip_minutes", min_value=1.0, max_value=300.0, value=15.0, step=1.0)
    pickup_hour = st.slider("pickup_hour", min_value=0, max_value=23, value=18)
    pickup_dayofweek = st.slider("pickup_dayofweek (0=Mon, 6=Sun)", min_value=0, max_value=6, value=4)
    pickup_day = st.slider("pickup_day", min_value=1, max_value=31, value=15)
    pickup_month = st.slider("pickup_month", min_value=1, max_value=12, value=1)
    is_weekend = st.selectbox("is_weekend", options=[0, 1], index=0)
    passenger_count = st.number_input("passenger_count", min_value=1.0, max_value=8.0, value=1.0, step=1.0)
    pulocation_id = st.number_input("PULocationID", min_value=1, max_value=265, value=138)
    dolocation_id = st.number_input("DOLocationID", min_value=1, max_value=265, value=161)
    cbd_congestion_fee = st.number_input("cbd_congestion_fee", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
    payment_type = st.selectbox("payment_type", options=[1, 2, 3, 4], index=0)


def build_row() -> pd.DataFrame:
    row = pd.DataFrame([[0.0] * len(feature_columns)], columns=feature_columns)
    base_values = {
        "trip_distance": float(trip_distance),
        "trip_minutes": float(trip_minutes),
        "pickup_hour": float(pickup_hour),
        "pickup_dayofweek": float(pickup_dayofweek),
        "pickup_day": float(pickup_day),
        "pickup_month": float(pickup_month),
        "is_weekend": float(is_weekend),
        "passenger_count": float(passenger_count),
        "PULocationID": float(pulocation_id),
        "DOLocationID": float(dolocation_id),
        "cbd_congestion_fee": float(cbd_congestion_fee),
    }
    for key, value in base_values.items():
        if key in row.columns:
            row.loc[0, key] = value

    payment_dummy = f"payment_type_{payment_type}"
    if payment_dummy in row.columns:
        row.loc[0, payment_dummy] = 1.0

    return row.astype(float)


x_input = build_row()
prediction = float(model.predict(x_input)[0])

col1, col2 = st.columns([1, 1])
with col1:
    st.metric("Predicted total_amount", f"${prediction:,.2f}")
    st.write("Input row used for prediction:")
    st.dataframe(x_input)

with col2:
    st.write("Local explanation (optional SHAP)")
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_input)
        if isinstance(shap_values, list):
            shap_vector = np.array(shap_values[0]).ravel()
        else:
            shap_vector = np.array(shap_values).ravel()

        contribution_df = pd.DataFrame(
            {
                "feature": x_input.columns,
                "shap_value": shap_vector,
                "abs_shap": np.abs(shap_vector),
            }
        ).sort_values("abs_shap", ascending=False)

        st.dataframe(contribution_df.head(12)[["feature", "shap_value"]])
        st.bar_chart(contribution_df.head(12).set_index("feature")["shap_value"])
    except Exception as error:
        st.info(f"SHAP explanation unavailable in app runtime: {error}")
