# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load model
with open("model.pkl", "rb") as f:
    model, feature_names = pickle.load(f)

# Page config
st.set_page_config(page_title="GreenML Dashboard", layout="wide")

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<h1>💡 GreenML Dashboard</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Choose Mode")
mode = st.sidebar.radio("", ["Single Prediction", "Batch Upload (CSV)"])

# Input ranges
ranges = {
    "T1": (10, 35), "RH_1": (10, 90), "T2": (10, 35), "RH_2": (10, 90),
    "T3": (10, 35), "RH_3": (10, 90), "T4": (10, 35), "RH_4": (10, 90),
    "T5": (10, 35), "RH_5": (10, 90), "T6": (10, 35), "RH_6": (10, 90),
    "T7": (10, 35), "RH_7": (10, 90), "T8": (10, 35), "RH_8": (10, 90),
    "T9": (10, 35), "RH_9": (10, 90), "T_out": (-10, 40), "Press_mm_hg": (700, 800),
    "RH_out": (10, 100), "Windspeed": (0, 15), "Visibility": (20, 70),
    "Tdewpoint": (-10, 25), "rv1": (0, 50), "rv2": (0, 50)
}

# ------------------ Mode: Single Prediction ------------------ #
if mode == "Single Prediction":
    st.markdown("<h3>🎯 Enter Values for Prediction</h3>", unsafe_allow_html=True)
    cols = st.columns(3)
    input_data = []

    for idx, feature in enumerate(feature_names):
        col = cols[idx % 3]
        min_val, max_val = ranges.get(feature, (0, 100))
        val = col.slider(feature, float(min_val), float(max_val), (min_val + max_val) / 2)
        input_data.append(val)

    if st.button("🔍 Predict Energy Usage"):
        prediction = model.predict([input_data])[0]
        st.success(f"⚡ Predicted Appliance Energy Usage: **{prediction:.2f} Wh**")

# ------------------ Mode: Batch Upload ------------------ #
elif mode == "Batch Upload (CSV)":
    st.markdown("### 📂 Upload a CSV file with input features:")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
        missing = [f for f in feature_names if f not in input_df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            preds = model.predict(input_df[feature_names])
            input_df["Predicted Energy (Wh)"] = preds
            st.dataframe(input_df)
            csv = input_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")

