# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 17:10:22 2026

@author: User
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# =========================
# FILES (same folder)
# =========================
MODEL_PATH = "decision_tree_model.joblib"
FEATURE_COLS_PATH = "feature_cols.json"
MEAN_STD_PATH = "mean_std_values.csv"
SHAP_BEESWARM_PATH = "SHAP_summary_beeswarm.png"
LOGO_PATH = "LJMU Logo.png"

# =========================
# RANGES (your values)
# =========================
RANGES = {
    "Confining_pressure (kPa)": (40.0, 300.0),
    "CSR": (0.09, 40.0),
    "N_cycles": (1.0, 50000.0),
    "Natural_density (g/cm3)": (1.51, 2.01),
    "Water_content (%)": (15.28, 57.5),
    "Liquid_limit (%)": (21.72, 64.0),
    "Plastic_limit (%)": (13.21, 28.0),
    "Plasticity_index": (8.51, 36.0),
}

# =========================
# Page setup
# =========================
st.set_page_config(
    page_title="Cumulative Strain Predictor (LJMU)",
    layout="wide",
)

# Optional: slight styling
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 1.2rem; }
      .stButton>button { padding: 0.55rem 1.1rem; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Load artifacts (cached)
# =========================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_feature_cols():
    with open(FEATURE_COLS_PATH, "r") as f:
        return json.load(f)

@st.cache_data
def load_mean_std():
    ms = pd.read_csv(MEAN_STD_PATH, index_col=0)
    if not {"mean", "std"}.issubset(ms.columns):
        raise ValueError("mean_std_values.csv must have columns: mean, std")
    return ms

model = load_model()
feature_cols = load_feature_cols()
mean_std = load_mean_std()

# =========================
# Header (like your Tkinter)
# =========================
h1, h2 = st.columns([6, 1])

with h1:
    st.title("Cumulative Strain Prediction App for Soil Liquefaction")
    st.markdown(
        "**Developed by Liverpool John Moores University (LJMU), School of Engineering and Built Environment**"
    )
    st.markdown(
        "**Developers:** Delbaz Samadian; Maria Ferentinou; Michaela Gkantou; Georgios Nikitas"
    )
    st.write(
        "This app predicts cumulative strain based on soil index properties and loading characteristics."
    )

with h2:
    try:
        st.image(LOGO_PATH, use_container_width=True)
    except Exception:
        st.caption("Logo file not found.")

st.divider()

# =========================
# Main layout: left inputs / right SHAP
# =========================
left, right = st.columns([1.15, 1.0], gap="large")

with left:
    st.subheader("Enter the values for the following features")

    # Make two columns of inputs to mimic your GUI layout
    c1, c2 = st.columns(2, gap="large")
    half = (len(feature_cols) + 1) // 2
    left_feats = feature_cols[:half]
    right_feats = feature_cols[half:]

    inputs = {}

    def add_inputs(cols, container):
        for f in cols:
            lo, hi = RANGES.get(f, (None, None))
            label = f"{f}"
            if lo is not None and hi is not None:
                label = f"{f}  —  Range: ({lo} to {hi})"
                # Choose a sensible default (clipped mean if you want)
                default = float(np.clip(lo, lo, hi))
                val = container.number_input(
                    label,
                    min_value=float(lo),
                    max_value=float(hi),
                    value=float(lo),
                    step=(1.0 if f == "N_cycles" else 0.01),
                    format="%.6f" if f != "N_cycles" else "%.0f",
                )
            else:
                val = container.number_input(label, value=0.0)
            inputs[f] = val

    add_inputs(left_feats, c1)
    add_inputs(right_feats, c2)

    st.markdown("### Cumulative Strain Prediction using the trained Decision Tree model")

    if st.button("Predict"):
        user_df = pd.DataFrame([inputs], columns=feature_cols)

        # Standardize X using mean/std from original features
        means = mean_std.loc[feature_cols, "mean"].values
        stds = mean_std.loc[feature_cols, "std"].values

        if np.any(stds == 0):
            bad = [feature_cols[i] for i, s in enumerate(stds) if s == 0]
            st.error(f"Some features have std=0 in mean_std_values.csv: {bad}")
        else:
            X_std = (user_df.values - means) / stds
            pred = float(model.predict(X_std)[0])

            st.success(f"Predicted Cumulative_strain: {pred:.6f}")

with right:
    st.subheader("SHAP Summary (Beeswarm)")
    try:
        st.image(SHAP_BEESWARM_PATH, use_container_width=True)
    except Exception:
        st.warning("SHAP_summary_beeswarm.png not found in the app folder.")
