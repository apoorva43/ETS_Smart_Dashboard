# app.py
"""
Streamlit application for exploring PISA score distributions.

This app provides an interactive dashboard for visualizing weighted PISA student score distributions by country, subject, and selected 
contextual groupings. It uses PISA sampling weights and plausible values to compute weighted percentile curves and weighted mean scores.

Notes
-----
The app expects the sample PISA dataset to be available at:

    data/raw/sampledat.csv
    
Examples
--------
Run the app from the project root with:

    streamlit run src/app.py
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pisa_stats import weighted_percentiles_pv, weighted_mean_pv
from config import KEEP_COLS


@st.cache_data
def load_data():
    """
    Load the sample PISA dataset for the dashboard.

    The function reads only the columns defined in ``KEEP_COLS`` that are
    present in the raw CSV file. If a ``YEAR`` column is not available in the
    dataset, it is added manually and set to 2022.

    Returns
    -------
    pandas.DataFrame
        Filtered PISA dataset containing selected identifier, score, weight,
        equity, and contextual variables.
    """
    filepath = "data/raw/sampledat.csv"

    # Read just the header first to see what columns actually exist in the sample
    available_cols = pd.read_csv(filepath, nrows=0).columns.tolist()
    keep = [c for c in KEEP_COLS if c in available_cols]

    # Check if YEAR is in the file, if so load it, otherwise we add it later
    usecols = keep + ["YEAR"] if "YEAR" in available_cols else keep

    df = pd.read_csv(filepath, usecols=usecols, low_memory=False)

    if "YEAR" not in df.columns:
        df["YEAR"] = 2022

    return df


df = load_data()

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.header("Filters")

country = st.sidebar.selectbox(
    "Country", sorted(df["CNT"].unique()), index=0
)
subject = st.sidebar.selectbox(
    "Subject", ["MATH", "READ", "SCIE"],
    format_func=lambda x: {"MATH": "Mathematics",
                           "READ": "Reading", "SCIE": "Science"}[x]
)
group_by = st.sidebar.selectbox(
    "Break down by",
    ["None", "Gender", "Immigration status", "SES quartile"]
)

# ── Main panel ────────────────────────────────────────────────────────────────
st.title("PISA Dashboard – Score Distributions")
st.markdown(f"Showing **{subject}** results for **{country}**")

# Map group_by selection to actual column + values
GROUP_MAP = {
    "None":               (None, None),
    "Gender":             ("ST004D01T", {1.0: "Female", 2.0: "Male"}),
    "Immigration status": ("IMMIG", {1.0: "Native", 2.0: "2nd-gen", 3.0: "1st-gen"}),
    # Added placeholder for SES quartile using the ESCS index
    "SES quartile":       ("ESCS", {1.0: "Q1 (Lowest)", 4.0: "Q4 (Highest)"})
}

group_col, group_vals = GROUP_MAP.get(group_by, (None, None))

# Compute and plot
subset = df[df["CNT"] == country]
PERCS = [10, 25, 50, 75, 90]

fig, ax = plt.subplots(figsize=(9, 5))

if group_col and group_vals:
    for code, label in group_vals.items():
        g_data = subset[subset[group_col] == code]
        if len(g_data) < 30:
            continue
        percs = weighted_percentiles_pv(g_data, subject, PERCS)
        ax.plot(PERCS, percs, lw=2.5, marker="o", ms=5, label=label)
else:
    percs = weighted_percentiles_pv(subset, subject, PERCS)
    ax.plot(PERCS, percs, lw=2.5, marker="o",
            ms=5, color="#185FA5", label=country)


ax.set_xticks(PERCS)
ax.set_xlabel("Percentile")
ax.set_ylabel("Score")
ax.legend()
st.pyplot(fig)

# Dynamic text below the chart
mean_score = weighted_mean_pv(subset, subject)

# Handle the case where the group is too small and returns NaN
if np.isnan(mean_score):
    mean_text = "unavailable due to insufficient data"
else:
    mean_text = f"**{mean_score:.0f}**"

st.markdown(f"""
**How to read this chart:** Each point shows the score at that percentile 
for students in **{country}**. The weighted mean score is {mean_text}.
""")
