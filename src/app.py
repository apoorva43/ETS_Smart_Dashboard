# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assuming pisa_stats.py is in a 'src' folder next to app.py
from src.pisa_stats import weighted_percentiles_pv, weighted_mean_pv
from src.config import KEEP_COLS


@st.cache_data
def load_data():
    """Cache the data so it doesn't reload on every interaction."""
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

# Commented out as it is not defined in pisa_stats.py
# add_oecd_reference(ax, df, subject, PERCS)

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
