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
from src.data_loader import (load_all_years,
                             load_all_years_s3,
                             load_sample_csv)

from src.pisa_stats import weighted_percentiles_pv, weighted_mean_pv
from src.config import SUBJECTS, GROUP_OPTIONS
from src.plotting_plotly import (plot_country_distributions,
                          plot_group_comparison,
                          plot_gender_percentile_line,
                          plot_escs_gap,
                          plot_naep_time_comparison,
                          plot_year_diff_percentile,
                          plot_weighted_interval_distribution,
                          plot_gender_diff_percentile,
                          plot_belonging_by_immigration,
                          plot_immigration_score_distribution,
                          plot_school_location_boxplot,
                          plot_school_type_distribution,
                          plot_resource_scatter)
from src.text_generator import (country_distribution_text,
                                ses_gap_text,
                                gender_gap_text,
                                scatter_correlation_text)

st.set_page_config(page_title="PISA Dashboard", layout="wide")

CHART_TYPES = [
    "Score distribution",
    "Interval distribution",
    "Change over time",
    "Gender gap",
    "SES gap",
    "Belonging by Immigration",
    "Group comparison",
    "Country Scatterplot"
]

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

    # Try local parquet files first
    available = [
        y for y in [2015, 2018, 2022]
        if Path(f"data/processed/pisa_{y}.parquet").exists()
    ]
    # Set PISA_PROFILE_MEMORY=1 in your shell to see memory savings on load
    if available:
        profile = os.environ.get("PISA_PROFILE_MEMORY", "0") == "1"
        return load_all_years(optimize_memory=True, profile_memory=profile)

    # Try public S3 (Posit Cloud)
    try:
        return load_all_years_s3(years=[2022])
    except RuntimeError:
        pass

    # Last resort: local sample CSV
    local_sample = Path("data/raw/sampledat.csv")
    if local_sample.exists():
        st.warning("No parquet files found -- using sample CSV")
        return load_sample_csv(local_sample)

    st.error("No data source available. Upload parquets to S3 or add sampledat.csv.")
    st.stop()


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
            warnings.append(f"Results suppressed for {label} (only {n} students)")
    return warnings

def check_missing_countries(df, required_cols, countries, year=None, min_n=30):
    """
    Checks which countries lack sufficient data for specific columns 
    and returns a list of the missing country codes.
    """
    missing = []
    for cnt in countries:
        subset = df[df["CNT"] == cnt]
        if year is not None and "YEAR" in df.columns:
            subset = subset[subset["YEAR"] == year]
        
        # Check if there are enough valid rows after dropping NaNs
        if len(subset.dropna(subset=required_cols + ["W_FSTUWT"])) < min_n:
            missing.append(cnt)
    return missing

CHART_HELP_TEXT = {
    "Score distribution": """
**How to read this chart:**
* **The Curves:** Each line represents the full range of student scores for a country. 
* **The X-Axis (Percentiles):** Shows the ranking of students from lowest performing (P10) to highest performing (P90).
* **The Slope:** A steeper, wider curve means there is a larger gap between the lowest and highest achievers (higher inequality).
    """,
    "Box Plot": """
**How to read this box plot:**
* **The Box (Middle 50%):** The colored rectangle represents the core of the student population. The bottom edge is the 25th percentile and the top edge is the 75th percentile.
* **The Center Line (Median):** Half the students scored above this thick line, and half scored below.
* **The Whiskers (The Tails):** The lines extending from the box show the 10th and 90th percentiles.
* **The Dots (Jitter):** A weighted sample of up to 1,000 students, showing exactly how individual scores are clustered.
    """,
    "Gender gap": """
**How to read this gap chart:**
* **The Diagonal Line:** This represents perfect equality. If boys and girls scored exactly the same, the colored line would sit perfectly on this dotted line.
* **Above the Line:** If the colored curve goes *above* the dotted line, Male students are scoring higher at that specific percentile.
* **Below the Line:** If the colored curve dips *below*, Female students are scoring higher.
    """,
    "Country Scatterplot": """
**How to read this scatterplot:**
* **The Dots:** Each dot represents an entire country's national average.
* **The Trend:** If the dots generally slope upwards from left to right, it means higher levels of the resource (like Belonging or SES) correlate with higher test scores globally.
    """
}

def render_chart_help(chart_type, group_key=None):
    """Renders a collapsible help section above complex charts."""
    
    # Determine which help text to grab
    help_key = None
    if chart_type == "Score distribution":
        help_key = "Score distribution"
    elif chart_type == "Gender gap":
        help_key = "Gender gap"
    elif chart_type == "Country Scatterplot":
        help_key = "Country Scatterplot"
    elif chart_type == "Group comparison" and group_key == "School location":
        help_key = "Box Plot"

    # Render it if it exists
    if help_key and help_key in CHART_HELP_TEXT:
        with st.expander(f"How to read the {chart_type.lower()} chart"):
            st.markdown(CHART_HELP_TEXT[help_key])


def render_chart(df, chart_type, subject, selected_countries,
                 selected_year, available_years,
                 primary_country, ref_year=None, comp_year=None,
                 group_key=None):
    """
    Render a single chart panel and its accompanying text/info blocks.

    Parameters
    ----------
    df : pandas.DataFrame
    chart_type : str
        One of the CHART_TYPES values.
    subject : str
    selected_countries : list of str
    selected_year : int or None
        Year filter for single-year views. None means all years.
    available_years : list of int
    primary_country : str
    ref_year : int, optional
        Reference year for "Compare two years" mode.
    comp_year : int, optional
        Comparison year for "Compare two years" mode.
    group_key : str, optional
        Pre-selected group breakdown key (used in side-by-side mode to
        avoid a second sidebar selectbox collision).
    """
    if chart_type == "Score distribution":
        missing_cnts = check_missing_countries(
            df, required_cols=[f"PV1{subject}"], 
            countries=selected_countries, year=selected_year
        )
        valid_countries = [c for c in selected_countries if c not in missing_cnts]
        
        if missing_cnts:
            st.warning(f"⚠️ **Data Unavailable:** Excluded **{', '.join(missing_cnts)}** due to missing {SUBJECTS[subject]} scores.")
            
        if valid_countries:
            render_chart_help(chart_type)
            fig = plot_country_distributions(
                df, subject, valid_countries, year=selected_year
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown(country_distribution_text(
                df, subject, valid_countries, year=selected_year
            ))

    elif chart_type == "Gender gap":
        render_chart_help(chart_type, group_key)
        if len(selected_countries) > 1:
            st.info(
                f"Gender gap shows one country at a time — displaying {primary_country}.")
        fig = plot_gender_percentile_line(
            df, subject, primary_country, year=selected_year)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
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
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown(ses_gap_text(df, subject, primary_country, year=selected_year))
        st.info("Students are split into four equal groups by socioeconomic status "
                "(ESCS index). Q1 = lowest SES, Q4 = highest.")


    # Group comparison
    elif chart_type == "Group comparison":
        if group_key is None:
            group_key = st.sidebar.selectbox(
                "Break down by",
                list(GROUP_OPTIONS.keys())
            )

        group_col, group_vals = GROUP_OPTIONS[group_key]

        if len(selected_countries) > 1:
            st.info(
                f"Group comparison shows one country at a time — displaying {primary_country}."
            )

        warns = check_group_sizes(
            df,
            group_col,
            group_vals,
            primary_country,
            year=selected_year
        )

        for w in warns:
            st.warning(w)

        if group_key == "Gender":
            fig = plot_gender_diff_percentile(
                df=df,
                subject=subject,
                cnt=primary_country,
                year=selected_year,
            )

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.info(
                "This chart shows the gender score difference across the distribution. "
                "Y-axis shows Male − Female score difference. Values above zero mean "
                "males score higher; values below zero mean females score higher."
            )

        elif group_key == "Immigration status":
            fig = plot_immigration_score_distribution(
                df=df,
                subject=subject,
                cnt=primary_country,
                year=selected_year,
            )

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.info(
                "Each curve shows the weighted score distribution for one immigration "
                "status group, using score intervals averaged across all 10 plausible values."
            )

        elif group_key == "School location":
            render_chart_help(chart_type, group_key)
            fig = plot_school_location_boxplot(
                df=df, subject=subject, cnt=primary_country, year=selected_year,
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        elif group_key == "School type":
            fig = plot_school_type_distribution(
                df=df,
                subject=subject,
                cnt=primary_country,
                year=selected_year,
            )

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.info(
                "Each curve shows the weighted score distribution for one school type, "
                "using score intervals averaged across all 10 plausible values."
            )

        else:
            fig = plot_group_comparison(
                df=df,
                subject=subject,
                group_col=group_col,
                group_vals=group_vals,
                cnt=primary_country,
                year=selected_year,
                title=f"{SUBJECTS[subject]} by {group_key} | {primary_country}",
            )

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.info(
                f"Score distribution broken down by {group_key} for {primary_country}."
            )

    elif chart_type == "Change over time":
        if len(available_years) < 2:
            st.warning(
                "Only one year of data loaded. Run `make data` to add more years.")
            return

        if len(selected_countries) > 1:
            st.info(
                f"Time comparison shows one country at a time — displaying {primary_country}.")

        # If two specific years were selected, use those; otherwise default
        # to all available years with the latest as reference.
        if ref_year is not None and comp_year is not None:
            reference_year    = ref_year
            comparison_years  = [comp_year]
            fig = plot_year_diff_percentile(
                df=df,
                subject=subject,
                cnt=primary_country,
                reference_year=reference_year,
                comparison_year=comp_year,
            )
        else:
            reference_year    = max(available_years)
            comparison_years  = [y for y in available_years if y != max(available_years)]

            fig = plot_naep_time_comparison(
                df=df,
                subject=subject,
                cnt=primary_country,
                reference_year=reference_year,
                comparison_years=comparison_years,
            )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.info(
            f"X-axis shows {reference_year} scores as the reference. "
            "Points above the diagonal indicate improvement relative to the reference year. "
            "Points below indicate decline."
        )
        
    elif chart_type == "Interval distribution":
        missing_cnts = check_missing_countries(
            df, required_cols=[f"PV1{subject}"], 
            countries=selected_countries, year=selected_year
        )
        valid_countries = [c for c in selected_countries if c not in missing_cnts]

        if missing_cnts:
            st.warning(f"⚠️ **Data Unavailable:** Excluded **{', '.join(missing_cnts)}** due to missing {SUBJECTS[subject]} scores.")
            
        if valid_countries:
            fig = plot_weighted_interval_distribution(
                df, subject, valid_countries, year=selected_year
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.info("Weighted proportion of students per 20-point score interval, averaged across all 10 plausible values.")


    elif chart_type == "Belonging by Immigration":
        missing_cnts = check_missing_countries(
            df, required_cols=["BELONG", "IMMIG", "REPEAT", "ESCS"], 
            countries=selected_countries, year=selected_year
        )
        
        valid_countries = [c for c in selected_countries if c not in missing_cnts]
        
        if missing_cnts:
            st.warning(
                f"⚠️ **Data Unavailable:** Excluded **{', '.join(missing_cnts)}** "
                f"due to missing student context data."
            )

        if valid_countries:
            fig = plot_belonging_by_immigration(
                df=df, countries=valid_countries, year=selected_year
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.info("Distribution of school belonging index by immigration status.")

    elif chart_type == "Country Scatterplot":
        resource_options = {
            "Socioeconomic Status (ESCS)": "ESCS",
            "School Belonging Index": "BELONG",
        }
        if subject == "MATH":
            resource_options["Math Motivation Index"] = "MATHMOT"
            
        selected_resource_label = st.selectbox(
            "Select X-Axis Variable:", 
            list(resource_options.keys()), 
            key=f"scatter_select_{group_key}" 
        )
        selected_col = resource_options[selected_resource_label]
        
        # Check if they are missing the specific resource they just selected
        missing_cnts = check_missing_countries(
            df, required_cols=[f"PV1{subject}", selected_col], 
            countries=selected_countries, year=selected_year
        )
        valid_countries = [c for c in selected_countries if c not in missing_cnts]

        if missing_cnts:
            st.warning(f"⚠️ **Data Unavailable:** Highlighting disabled for **{', '.join(missing_cnts)}** (missing {selected_col} data).")
        
        render_chart_help(chart_type)
        fig = plot_resource_scatter(
            df=df,
            subject=subject,
            resource_col=selected_col,
            resource_label=selected_resource_label,
            year=selected_year,
            highlight_countries=valid_countries
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown(scatter_correlation_text(
            df=df, subject=subject, resource_col=selected_col, 
            resource_label=selected_resource_label, year=selected_year,
            highlight_countries=valid_countries
        ))

# Load data and derive country lists
df = get_data()

available_years = sorted(df["YEAR"].unique().tolist())
all_countries = sorted(df["CNT"].unique().tolist())
oecd_countries = sorted(df[df["OECD"] == 1]["CNT"].unique().tolist())
partner_countries = sorted(df[df["OECD"] == 0]["CNT"].unique().tolist())

# Sidebar controls
st.sidebar.header("Filters")

# 1. Ask for Chart Type FIRST
side_by_side = st.sidebar.toggle("Compare two views side by side", value=False, key="sbs_toggle")

if side_by_side:
    st.sidebar.markdown("**Left panel**")
    chart_type_left = st.sidebar.selectbox("Left chart", CHART_TYPES, key="chart_left")
    st.sidebar.markdown("**Right panel**")
    chart_type_right = st.sidebar.selectbox("Right chart", CHART_TYPES, key="chart_right", index=1)
else:
    chart_type = st.sidebar.radio("View", CHART_TYPES)

st.sidebar.markdown("---")

# 2. Country Group Filter
country_group = st.sidebar.radio(
    "Country group", ["All", "OECD members", "Partner countries"]
)
if country_group == "OECD members":
    country_pool = oecd_countries
elif country_group == "Partner countries":
    country_pool = partner_countries
else:
    country_pool = all_countries

# 3. Dynamically Render Country Selector
SINGLE_COUNTRY_CHARTS = ["Change over time", "Gender gap", "SES gap", "Group comparison"]

if not side_by_side and chart_type in SINGLE_COUNTRY_CHARTS:
    # Show a single selectbox for strict 1-country charts
    selected_country = st.sidebar.selectbox("Country", country_pool, index=0)
    selected_countries = [selected_country]  # Wrap in list so downstream code doesn't break
else:
    # Show the standard multiselect for global charts or Side-by-Side mode
    label = "Countries (Pool)" if side_by_side else "Countries"
    selected_countries = st.sidebar.multiselect(
        label, country_pool, default=country_pool[:2]
    )

if not selected_countries:
    st.warning("Please select at least one country.")
    st.stop()

# 4. Handle Side-by-Side Specific Overrides
if side_by_side:
    country_left = st.sidebar.selectbox(
        "Left country", selected_countries, key="cnt_left"
    )
    right_idx = 1 if len(selected_countries) > 1 else 0
    country_right = st.sidebar.selectbox(
        "Right country", selected_countries, index=right_idx, key="cnt_right"
    )
    
    group_key_left = None
    group_key_right = None
    if chart_type_left == "Group comparison":
        group_key_left = st.sidebar.selectbox(
            "Left: Break down by", list(GROUP_OPTIONS.keys()), key="group_left"
        )
    if chart_type_right == "Group comparison":
        group_key_right = st.sidebar.selectbox(
            "Right: Break down by", list(GROUP_OPTIONS.keys()), key="group_right"
        )

st.sidebar.markdown("---")

subject = st.sidebar.selectbox(
    "Subject", list(SUBJECTS.keys()),
    format_func=lambda x: SUBJECTS[x]
)

st.sidebar.markdown("---")
ref_year = None
comp_year = None

if len(available_years) > 1:
    year_mode = st.sidebar.radio(
        "Year",
        ["Latest (2022)", "All years", "Compare two years"]
    )
    if year_mode == "Latest (2022)":
        selected_year = 2022
    elif year_mode == "All years":
        selected_year = None
    else:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            ref_year = st.selectbox(
                "Reference year",
                available_years[:-1],          
                index=0,
                key="ref_year"
            )
        with col2:
            later_years = [y for y in available_years if y > ref_year]
            comp_year = st.selectbox(
                "Compare to",
                later_years,
                index=len(later_years) - 1,
                key="comp_year"
            )
        selected_year = None   
        st.sidebar.caption(
            f"Reference: {ref_year} (x-axis) → Compare: {comp_year}"
        )
else:
    selected_year = available_years[0]

primary_country = selected_countries[0]

st.title("PISA Score Distribution Dashboard")
st.caption(f"Data: PISA {', '.join(str(y) for y in available_years)}  |  "
           f"{len(all_countries)} countries  |  "
           f"{len(df):,} students")

if side_by_side:
    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader(chart_type_left)
        render_chart(
            df, chart_type_left, subject, selected_countries,
            selected_year, available_years, 
            primary_country=country_left,
            ref_year=ref_year, comp_year=comp_year,
            group_key=group_key_left,
        )
    with right_col:
        st.subheader(chart_type_right)
        render_chart(
            df, chart_type_right, subject, selected_countries,
            selected_year, available_years, 
            primary_country=country_right,
            ref_year=ref_year, comp_year=comp_year,
            group_key=group_key_right,
        )
else:
    # If not side-by-side, it just uses the default global primary_country
    render_chart(
        df, chart_type, subject, selected_countries,
        selected_year, available_years, primary_country,
        ref_year=ref_year, comp_year=comp_year,
    )