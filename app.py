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

    PYTHONPATH=. streamlit run src/app.py
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from src.data_loader import load_all_years, load_sample_csv

from src.pisa_stats import weighted_percentiles_pv, weighted_mean_pv
from src.config import SUBJECTS, GROUP_OPTIONS
from src.plotting import (plot_country_distributions,
                          plot_group_comparison,
                          plot_gender_percentile_line,
                          plot_escs_gap,
                          plot_naep_time_comparison)
from src.text_generator import (country_distribution_text,
                                ses_gap_text,
                                gender_gap_text)

st.set_page_config(page_title="PISA Dashboard", layout="wide")

# Helper function to load data with caching
@st.cache_data
def get_data():
    """
    Load all processed PISA years if available; otherwise use sample CSV.

    Returns
    -------
    pandas.DataFrame
        PISA data loaded from processed parquet files or sample CSV.
    """

    available = [
        y for y in [2015, 2018, 2022]
        if Path(f"data/processed/pisa_{y}.parquet").exists()
    ]

    if not available:
        st.warning("No parquet files found -- using sample CSV")
        return load_sample_csv("data/raw/sampledat.csv")

    # Set PISA_PROFILE_MEMORY=1 in your shell to see memory savings on load
    profile = os.environ.get("PISA_PROFILE_MEMORY", "0") == "1"
    return load_all_years(optimize_memory=True, profile_memory=profile)


def check_group_sizes(df, group_col, group_vals, cnt, year=None):
    """
    The function counts valid observations for each group category within a
    selected country and optional year. Groups with fewer than 30 observations
    are flagged because weighted percentile estimates may be unstable or
    suppressed in the dashboard.

    Parameters
    ----------
    df : pandas.DataFrame
        PISA dataset containing country, year, weight, and grouping columns.
    group_col : str
        Column name used to define subgroup categories.
    group_vals : dict
        Mapping from raw category codes to human-readable group labels.
    cnt : str
        Country code used to filter the dataset, such as ``"CAN"`` or ``"USA"``.
    year : int, optional
        PISA cycle year used to filter the dataset. If ``None``, all available
        years are included.

    Returns
    -------
    list of str
        Warning messages for groups with fewer than 30 valid observations.
        Returns an empty list if all groups meet the minimum size threshold.
    """
    subset = df[df["CNT"] == cnt]
    if year is not None and "YEAR" in df.columns:
        subset = subset[subset["YEAR"] == year]
    warnings = []
    for code, label in group_vals.items():
        n = len(subset[subset[group_col] == code].dropna(subset=["W_FSTUWT"]))
        if n < 30:
            warnings.append(f"{label}: only {n} students — results suppressed")
    return warnings


# ── Load data and derive country lists ────────────────────────────────────────
df = get_data()

available_years = sorted(df["YEAR"].unique().tolist())
all_countries = sorted(df["CNT"].unique().tolist())
oecd_countries = sorted(df[df["OECD"] == 1]["CNT"].unique().tolist())
partner_countries = sorted(df[df["OECD"] == 0]["CNT"].unique().tolist())

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.header("Filters")

country_group = st.sidebar.radio(
    "Country group", ["All", "OECD members", "Partner countries"]
)
if country_group == "OECD members":
    country_pool = oecd_countries
elif country_group == "Partner countries":
    country_pool = partner_countries
else:
    country_pool = all_countries

selected_countries = st.sidebar.multiselect(
    "Countries", country_pool,
    default=country_pool[:2]
)

# prevent multiselect from returning empty list if user clears it
if not selected_countries:
    st.warning("Please select at least one country.")
    st.stop()

subject = st.sidebar.selectbox(
    "Subject", list(SUBJECTS.keys()),
    format_func=lambda x: SUBJECTS[x]
)

# Chart type selector
chart_type = st.sidebar.radio(
    "View",
    ["Score distribution", "Gender gap", "SES gap",
     "Group comparison", "Change over time"]
)

# Year selector -- only relevant for single-country charts
if len(available_years) > 1:
    year_mode = st.sidebar.radio(
        "Year", ["Latest (2022)", "All years"]
    )
    selected_year = 2022 if year_mode == "Latest (2022)" else None
else:
    selected_year = available_years[0]

# Primary country for single-country charts
primary_country = selected_countries[0]

# ── Main panel ────────────────────────────────────────────────────────────────
st.title("PISA Score Distribution Dashboard")
st.caption(f"Data: PISA {', '.join(str(y) for y in available_years)}  |  "
           f"{len(all_countries)} countries  |  "
           f"{len(df):,} students")

if chart_type == "Score distribution":
    fig = plot_country_distributions(
        df, subject, selected_countries, year=selected_year
    )
    st.pyplot(fig)
    st.markdown(country_distribution_text(
        df, subject, selected_countries, year=selected_year
    ))

elif chart_type == "Gender gap":
    if len(selected_countries) > 1:
        st.info(
            f"Gender gap shows one country at a time — displaying {primary_country}.")
    fig = plot_gender_percentile_line(
        df, subject, primary_country, year=selected_year)
    st.pyplot(fig)
    st.markdown(gender_gap_text(
        df, subject, primary_country, year=selected_year))
    st.info("The x-axis shows Female scores as the reference group. "
            "Where the Male line sits above the diagonal, males score higher "
            "at that point in the distribution.")

elif chart_type == "SES gap":
    if len(selected_countries) > 1:
        st.info(
            f"SES gap shows one country at a time — displaying {primary_country}.")
    fig = plot_escs_gap(df, subject, primary_country, year=selected_year)
    st.pyplot(fig)
    st.markdown(ses_gap_text(df, subject, primary_country, year=selected_year))
    st.info("Students are split into four equal groups by socioeconomic status "
            "(ESCS index). Q1 = lowest SES, Q4 = highest.")

elif chart_type == "Group comparison":
    group_name = st.sidebar.selectbox(
        "Break down by", list(GROUP_OPTIONS.keys())
    )
    group_col, group_vals = GROUP_OPTIONS[group_name]

    if len(selected_countries) > 1:
        st.info(
            f"Group comparison shows one country at a time — displaying {primary_country}.")

    warns = check_group_sizes(df, group_col, group_vals,
                              primary_country, year=selected_year)
    for w in warns:
        st.warning(w)

    fig = plot_group_comparison(
        df, subject, group_col, group_vals,
        cnt=primary_country, year=selected_year,
        title=f"{SUBJECTS[subject]} by {group_name} — {primary_country}"
    )
    st.pyplot(fig)
    st.info(
        f"Score distribution broken down by {group_name} for {primary_country}.")

elif chart_type == "Change over time":
    if len(available_years) < 2:
        st.warning(
            "Only one year of data loaded. Run `make data` to add more years.")
    else:
        if len(selected_countries) > 1:
            st.info(
                f"Time comparison shows one country at a time — displaying {primary_country}.")
        fig = plot_naep_time_comparison(
            df,
            subject=subject,
            cnt=primary_country,
            reference_year=max(available_years),
            comparison_years=[
                y for y in available_years if y != max(available_years)]
        )
        st.pyplot(fig)
        st.info(
            f"X-axis shows {max(available_years)} scores as the reference. "
            "Points above the diagonal indicate improvement relative to the reference year. "
            "Points below indicate decline."
        )

