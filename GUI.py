# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 14:27:11 2026

@author: User
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import json
import os
from PIL import Image, ImageTk

# =========================
# FILES (same folder)
# =========================
MODEL_PATH = "xgboost_model.joblib"
FEATURE_COLS_PATH = "feature_cols.json"
SHAP_BEESWARM_PATH = "SHAP_summary_beeswarm.png"
LOGO_PATH = "LJMU Logo.png"

# =========================
# THEME / COLORS
# =========================
THEME = {
    # window + panels
    "app_bg": "white",
    "header_bg": "white",
    "main_bg": "white",
    "left_bg": "white",
    "right_bg": "white",

    # text
    "title_fg": "#111111",
    "subtitle_fg": "#333333",
    "desc_fg": "#444444",
    "section_fg": "#111111",
    "label_fg": "#333333",
    "shap_title_fg": "#111111",

    # header title background
    "title_bg": "white",

    # input widgets
    "input_bg": "#efefef",
    "input_fg": "#111111",
    "input_block_bg": "#00FFFF",

    # buttons
    "predict_btn_bg": "#BBF90F",
    "predict_btn_fg": "black",

    # prediction result band
    "result_bg": "#dff3ea",
    "result_fg": "#1d7a3b",

    # SHAP area
    "canvas_bg": "white",

    # misc
    "error_fg": "red",
}

# =========================
# Logo sizing
# =========================
LOGO_MAX_W = 240
LOGO_MAX_H = 240

# =========================
# Ranges shown under each input
# Replace these with your NEW dataset ranges
# Keys must exactly match feature_cols.json
# =========================
RANGES = {
    "H (m)": "Range: (0-120)",
    "𝛾": "Range: (0-0.01)",
    "W (%)": "Range: (15-40)",
    "𝜌 (k𝑔/m^3 )": "Range: (1730-2230)",
}

# =========================
# Optional display names for GUI labels
# This only changes what the GUI shows, not the model input names
# Keys must exactly match feature_cols.json
# =========================
DISPLAY_NAMES = {
    "H (m)": "H (m)",
    "𝛾": "γ",
    "W (%)": "w (%)",
    "𝜌 (k𝑔/m^3 )": "ρ_b (kg/m³)",
}

# =========================
# Load artifacts
# =========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

if not os.path.exists(FEATURE_COLS_PATH):
    raise FileNotFoundError(f"Missing feature columns file: {FEATURE_COLS_PATH}")

model = joblib.load(MODEL_PATH)

with open(FEATURE_COLS_PATH, "r", encoding="utf-8") as f:
    feature_cols = json.load(f)

# =========================
# Prediction
# =========================
def predict_output():
    try:
        user_raw = {}

        for feat in feature_cols:
            val = pd.to_numeric(input_widgets[feat].get(), errors="coerce")
            user_raw[feat] = val

        user_df = pd.DataFrame([user_raw], columns=feature_cols)

        if user_df.isna().any().any():
            bad = user_df.columns[user_df.isna().any()].tolist()
            raise ValueError(f"Non-numeric or missing inputs in: {bad}")

        # XGBoost model was trained on raw features
        pred = float(model.predict(user_df)[0])

        pred_value_label.config(text=f"Predicted G_raw: {pred:.6f}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# =========================
# Tkinter App
# =========================
app = tk.Tk()
app.title("XGBoost Predictor")
app.geometry("1500x900")
app.configure(bg=THEME["app_bg"])

# Make main grid expand
app.grid_rowconfigure(1, weight=1)
app.grid_columnconfigure(0, weight=1)

# =========================
# Header
# =========================
header = tk.Frame(app, bg=THEME["header_bg"])
header.grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 6))
header.grid_columnconfigure(0, weight=1)
header.grid_columnconfigure(1, weight=0)

title = tk.Label(
    header,
    text="Prediction of Dynamic Properties of Fine-Grained Offshore Marine Sediments",
    font=("Times New Roman", 24, "bold"),
    bg=THEME["title_bg"],
    fg=THEME["title_fg"]
)
title.grid(row=0, column=0, sticky="w")

# Right: logo
logo_label = tk.Label(header, bg=THEME["header_bg"])
logo_label.grid(row=0, column=1, rowspan=4, sticky="e", padx=(20, 0))

if os.path.exists(LOGO_PATH):
    logo_img = Image.open(LOGO_PATH)
    logo_img.thumbnail((LOGO_MAX_W, LOGO_MAX_H), Image.LANCZOS)
    logo_photo = ImageTk.PhotoImage(logo_img)
    logo_label.configure(image=logo_photo)
    logo_label.image = logo_photo
else:
    logo_label.configure(
        text="(Logo missing)",
        fg=THEME["error_fg"],
        font=("Times New Roman", 12)
    )

subtitle = tk.Label(
    header,
    text="Developed by Liverpool John Moores University (LJMU), School of Engineering and Built Environment",
    font=("Times New Roman", 15, "bold"),
    bg=THEME["header_bg"],
    fg=THEME["subtitle_fg"]
)
subtitle.grid(row=1, column=0, sticky="w", pady=(6, 0))

developers = tk.Label(
    header,
    text="Developers: Delbaz Samadian; Maria Ferentinou, Michaela Gkantou, Georgios Nikitas",
    font=("Times New Roman", 13, "bold"),
    bg=THEME["header_bg"],
    fg=THEME["subtitle_fg"]
)
developers.grid(row=2, column=0, sticky="w", pady=(2, 0))

desc = tk.Label(
    header,
    text="This app predicts shear modulus based on soil index properties and loading characteristics using the trained XGBoost model.",
    font=("Times New Roman", 12),
    bg=THEME["header_bg"],
    fg=THEME["desc_fg"]
)
desc.grid(row=3, column=0, sticky="w", pady=(6, 0))

# Separation line
sep = ttk.Separator(app, orient="horizontal")
sep.grid(row=0, column=0, sticky="ew", padx=18, pady=(220, 0))

# =========================
# Main area
# =========================
main = tk.Frame(app, bg=THEME["main_bg"])
main.grid(row=1, column=0, sticky="nsew", padx=18, pady=12)
main.grid_columnconfigure(0, weight=1)
main.grid_columnconfigure(1, weight=1)
main.grid_rowconfigure(0, weight=1)

# =========================
# Left panel
# =========================
left = tk.Frame(main, bg=THEME["left_bg"])
left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
left.grid_columnconfigure(0, weight=1)

section_title = tk.Label(
    left,
    text="Enter the values for the following features",
    font=("Times New Roman", 18, "bold"),
    bg=THEME["left_bg"],
    fg=THEME["section_fg"]
)
section_title.grid(row=0, column=0, sticky="w", pady=(0, 10))

# Inputs grid (2 columns)
inputs_grid = tk.Frame(left, bg=THEME["left_bg"])
inputs_grid.grid(row=1, column=0, sticky="nw")
inputs_grid.grid_columnconfigure(0, weight=1)
inputs_grid.grid_columnconfigure(1, weight=1)

half = (len(feature_cols) + 1) // 2
left_feats = feature_cols[:half]
right_feats = feature_cols[half:]

input_widgets = {}

def make_feature_block(parent, feat, r, c):
    block = tk.Frame(parent, bg=THEME["input_block_bg"])
    block.grid(row=r, column=c, sticky="ew", padx=14, pady=10)

    shown_name = DISPLAY_NAMES.get(feat, feat)
    range_txt = RANGES.get(feat, "")
    text = f"{shown_name}   -   {range_txt}" if range_txt else shown_name

    lbl = tk.Label(
        block,
        text=text,
        font=("Times New Roman", 12),
        bg=THEME["input_block_bg"],
        fg=THEME["label_fg"]
    )
    lbl.pack(anchor="w")

    sb = tk.Spinbox(
        block,
        from_=-1e12,
        to=1e12,
        increment=0.1,
        width=28,
        justify="left",
        font=("Times New Roman", 12),
        bg=THEME["input_bg"],
        fg=THEME["input_fg"],
        insertbackground=THEME["input_fg"],
        relief="flat"
    )
    sb.pack(fill="x", pady=(4, 0))

    input_widgets[feat] = sb

for i, feat in enumerate(left_feats):
    make_feature_block(inputs_grid, feat, i, 0)

for i, feat in enumerate(right_feats):
    make_feature_block(inputs_grid, feat, i, 1)

# =========================
# Prediction section
# =========================
pred_section = tk.Frame(left, bg=THEME["left_bg"])
pred_section.grid(row=2, column=0, sticky="ew", pady=(18, 0))
pred_section.grid_columnconfigure(0, weight=1)

pred_title = tk.Label(
    pred_section,
    text="Shear molusus prediction for soils using the trained XGBoost model",
    font=("Times New Roman", 16, "bold"),
    bg=THEME["left_bg"],
    fg=THEME["section_fg"]
)
pred_title.grid(row=0, column=0, sticky="w", pady=(0, 10))

predict_btn = tk.Button(
    pred_section,
    text="Predict",
    command=predict_output,
    font=("Times New Roman", 13, "bold"),
    bg=THEME["predict_btn_bg"],
    fg=THEME["predict_btn_fg"],
    relief="flat",
    padx=16,
    pady=6
)
predict_btn.grid(row=1, column=0, sticky="w")

pred_box = tk.Frame(pred_section, bg=THEME["result_bg"], bd=0)
pred_box.grid(row=2, column=0, sticky="ew", pady=(14, 0))
pred_box.grid_columnconfigure(0, weight=1)

pred_value_label = tk.Label(
    pred_box,
    text="Predicted value:",
    font=("Times New Roman", 13, "bold"),
    bg=THEME["result_bg"],
    fg=THEME["result_fg"]
)
pred_value_label.grid(row=0, column=0, sticky="w", padx=12, pady=12)

# =========================
# Right panel: SHAP image
# =========================
right = tk.Frame(main, bg=THEME["right_bg"])
right.grid(row=0, column=1, sticky="nsew", padx=(14, 0))
right.grid_rowconfigure(1, weight=1)
right.grid_columnconfigure(0, weight=1)

shap_title = tk.Label(
    right,
    text="SHAP Summary (Beeswarm)",
    font=("Times New Roman", 18, "bold"),
    bg=THEME["right_bg"],
    fg=THEME["shap_title_fg"]
)
shap_title.grid(row=0, column=0, sticky="ew", pady=(0, 10))
shap_title.configure(anchor="center")

canvas = tk.Canvas(right, bg=THEME["canvas_bg"], highlightthickness=0)
canvas.grid(row=1, column=0, sticky="nsew")

scroll = ttk.Scrollbar(right, orient="vertical", command=canvas.yview)
scroll.grid(row=1, column=1, sticky="ns")
canvas.configure(yscrollcommand=scroll.set)

img_container = tk.Frame(canvas, bg=THEME["canvas_bg"])
canvas.create_window((-12, 0), window=img_container, anchor="nw")

def show_shap():
    for w in img_container.winfo_children():
        w.destroy()

    if not os.path.exists(SHAP_BEESWARM_PATH):
        tk.Label(
            img_container,
            text=f"Missing: {SHAP_BEESWARM_PATH}",
            fg=THEME["error_fg"],
            bg=THEME["canvas_bg"],
            font=("Times New Roman", 14)
        ).pack(pady=20)
        return

    img = Image.open(SHAP_BEESWARM_PATH)

    max_w = 720
    if img.width > max_w:
        ratio = max_w / img.width
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)

    photo = ImageTk.PhotoImage(img)
    lbl = tk.Label(img_container, image=photo, bg=THEME["canvas_bg"])
    lbl.image = photo
    lbl.pack(anchor="n")

show_shap()

def _update_scrollregion(_):
    canvas.configure(scrollregion=canvas.bbox("all"))

img_container.bind("<Configure>", _update_scrollregion)

app.mainloop()