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

from src.data_loader import query_pisa
from src.pisa_stats import weighted_percentiles_pv, weighted_mean_pv
from src.config import SUBJECTS, GROUP_OPTIONS, IMMIG_MAP
from src.plotting_plotly import (plot_country_distributions,
                          plot_group_comparison,
                          plot_escs_gap,
                          plot_naep_time_comparison,
                          plot_year_diff_percentile,
                          plot_weighted_interval_distribution,
                          plot_gender_diff_percentile,
                          plot_belonging_by_immigration,
                          plot_immigration_score_distribution,
                          plot_school_location_boxplot,
                          plot_school_type_distribution,
                          plot_resource_scatter,
                          _cnt_label)
from src.text_generator import (country_distribution_text,
                                ses_gap_text,
                                immigration_gap_text,
                                scatter_correlation_text)

st.set_page_config(page_title="PISA Dashboard", layout="wide")

S3_BASE = "https://pisa-dashboard-data.s3.ca-central-1.amazonaws.com"

CHART_TYPES = [
    "Percentile score profile",
    "Score distribution",
    "Score change over time",
    "Belonging by Immigration",
    "Group comparison",
    "Country Scatterplot"
]

# Column sets - each chart only pulls what it needs
BASE_COLS  = ["CNT", "YEAR", "OECD", "W_FSTUWT"]
PV_MATH    = [f"PV{i}MATH" for i in range(1, 11)]
PV_READ    = [f"PV{i}READ" for i in range(1, 11)]
PV_SCIE    = [f"PV{i}SCIE" for i in range(1, 11)]
PV_ALL     = PV_MATH + PV_READ + PV_SCIE
EQUITY_COLS = ["ESCS", "HISEI", "PAREDINT", "HOMEPOS",
               "BELONG", "MATHMOT", "REPEAT", "IMMIG",
               "ST004D01T", "LANGN", "GRADE", "AGE",
               "SC001Q01TA", "SCHLTYPE", "STRATUM", "CNTSCHID", "CNTSTUID"]

PV_BY_SUBJ = {"MATH": PV_MATH, "READ": PV_READ, "SCIE": PV_SCIE}

# Helper function to load data with caching
@st.cache_data(ttl=3600)
def get_meta() -> pd.DataFrame:
    """
    Load the pre-aggregated country-year statistics file for sidebar populaton.

    Reads from a local parquet file when available, falling back to the 
    public S3 copy on deployed environments. This file contains one row 
    per country-year combination with weighted mean scores and equity
    indicators - it is used exclusively to populate sidebar filters 
    (country lists, OECD membership, available years) and never passed to
    plotting functions.

    Returns
    -------
    pd.DataFrame
        One row per country-year. Columns include ``CNT``, ``YEAR``,
        ``OECD``, ``score_math``, ``score_read``, ``score_scie``, and
        weighted means for equity indicators such as ``ESCS`` and ``BELONG``.
    """
    local = Path("data/processed/pisa_country_stats.parquet")
    if local.exists():
        return pd.read_parquet(local)
    return pd.read_parquet(f"{S3_BASE}/pisa_country_stats.parquet")
    

@st.cache_data(ttl=3600, show_spinner="Fetching data...")
def fetch(countries: tuple[str, ...], 
          year: int | None, 
          cols: tuple[str, ...]) -> pd.DataFrame:
    """
    Cached wrapper around ``query_pisa`` for Streamlit chart rendering.

    Calls ``query_pisa`` with the given filters and caches the result for
    one hour. On a cache hit (same countries, year, and columns as a 
    previous call), the DataFrame is returned instantly from memory without
    re-querying the parquet file.

    Arguments are typed as tuples rather than lists because
    ``st.cache_data`` requires all arguments to be hashable. Convert to 
    lists before passing to ``query_pisa``, which accepts lists.

    Parameters
    ----------
    countries : tuple of str
        PISA country codes to include, e.g. ("CAN", "USA").
    year : int or None
        PISA cycle year to filter by (2015, 2018 or 2022).
        Pass ``None`` to return all available years, which is required 
        for the "Score change over time" chart type.
    cols : tuple of str
        Columns to select from the parquet file. Should always include
        "CNT", "YEAR", "W_FSTUWT", plus whichever plausible value and
        equity columns the calling chart needs. Selecting only 
        necessary columns keeps each query small.
    
    Returns
    -------
    pd.DataFrame
        Filtered student-level PISA data. 

    Examples
    --------
    Fetch mathematics plausible values and weights for Canada in 2022:

    >>> pv_cols = tuple(f"PV{i}MATH" for i in range(1, 11))
    >>> df = fetch(("CAN",), 2022, ("CNT", "YEAR", "W_FSTUWT") + pv_cols)
    """
    return query_pisa(list(countries), year=year, cols=list(cols))


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
        
        if 0 < n < 30:
            warnings.append(f"⚠️ Limited data available for **{label}**. Results suppressed to ensure statistical reliability.")
            
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
* **The Slope:** A steeper, wider curve means there is a greater difference in scores between the lowest and highest achievers (higher inequality).
    """,
    "Box Plot": """
**How to read this box plot:**
* **The Box (Middle 50%):** The colored rectangle represents the core of the student population. The bottom edge is the 25th percentile and the top edge is the 75th percentile.
* **The Center Line (Median):** Half the students scored above this thick line, and half scored below.
* **The Whiskers (The Tails):** The lines extending from the box show the 10th and 90th percentiles.
* **The Dots (Jitter):** A representative sample of up to 1,000 students, showing exactly how individual scores are clustered.
    """,
    "Score by gender": """
**How to read this difference chart:**
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
    elif chart_type == "Score by gender":
        help_key = "Score by gender"
    elif chart_type == "Country Scatterplot":
        help_key = "Country Scatterplot"
    elif chart_type == "Group comparison" and group_key == "School location":
        help_key = "Box Plot"

    # Render it if it exists
    if help_key and help_key in CHART_HELP_TEXT:
        with st.expander(f"How to read the {chart_type.lower()} chart"):
            st.markdown(CHART_HELP_TEXT[help_key])
            

def apply_compact_plotly_layout(fig, hide_legend=False):
    """
    Make Plotly charts fit better inside narrow columns.
    Used for side-by-side mode and story tab two-column sections.
    """
    fig.update_layout(
        title=dict(
            font=dict(size=14),
            x=0,
            xanchor="left",
            y=0.96,
        ),
        margin=dict(t=90, b=50, l=60, r=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5,
            font=dict(size=9),
        ),
    )

    fig.update_xaxes(
        title_font=dict(size=11),
        tickfont=dict(size=10),
    )

    fig.update_yaxes(
        title_font=dict(size=11),
        tickfont=dict(size=10),
    )

    if hide_legend:
        fig.update_layout(showlegend=False)

    return fig


def render_chart(chart_type, subject, selected_countries,
                 selected_year, available_years,
                 primary_country, ref_year=None, comp_year=None,
                 compact=False, widget_key="main"):
    """
    Render a single chart panel and its accompanying text/info blocks.

    Parameters
    ----------
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
    pv_cols = PV_BY_SUBJ[subject]

    # 1. Percentile Profile
    if chart_type == "Percentile score profile":
        fetch_cnts = tuple(set(selected_countries) | set(oecd_countries))
        df = fetch(fetch_cnts, selected_year, tuple(BASE_COLS + pv_cols))
        
        missing_cnts = check_missing_countries(
            df, required_cols=[f"PV1{subject}"], 
            countries=selected_countries, year=selected_year
        )
        valid_countries = [c for c in selected_countries if c not in missing_cnts]
        
        if missing_cnts:
            st.warning(f"⚠️ **Data Unavailable:** Excluded **{', '.join(_cnt_label(c) for c in missing_cnts)}** due to missing {SUBJECTS[subject]} scores.")
            
        if valid_countries:
            render_chart_help(chart_type)
            fig = plot_country_distributions(
                df, subject, valid_countries, year=selected_year
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown(country_distribution_text(
                df, subject, valid_countries, year=selected_year
            ))

    # 2. Group Comparison
    elif chart_type == "Group comparison":
        group_key = st.selectbox(
            "Break down by:", 
            list(GROUP_OPTIONS.keys()), 
            key=f"group_select_{widget_key}"
        )

        group_col, group_vals = GROUP_OPTIONS[group_key]

        if len(selected_countries) > 1:
            st.info(
                f"Group comparison shows one country at a time — displaying {_cnt_label(primary_country)}."
            )

        df = fetch((primary_country,), selected_year,
                   tuple(BASE_COLS + pv_cols + [group_col]))
        
        if group_key != "Socioeconomic status":
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
            df = fetch(
                (primary_country,),
                selected_year,
                tuple(BASE_COLS + pv_cols + ["ST004D01T"])
            )
            fig = plot_gender_diff_percentile(
                df=df,
                subject=subject,
                cnt=primary_country,
                year=selected_year,
                active_countries=selected_countries
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False}
            )
            st.info(
                "Y-axis shows Male - Female score difference. "
                "Values above zero mean males score higher; values below zero mean females score higher."
            )
            
        elif group_key == "Socioeconomic status":
            df = fetch(
                (primary_country,),
                selected_year,
                tuple(BASE_COLS + pv_cols + ["ESCS"])
            )

            fig = plot_escs_gap(
                df=df,
                subject=subject,
                cnt=primary_country,
                year=selected_year,
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False}
            )

            st.markdown(
                ses_gap_text(
                    df,
                    subject,
                    primary_country,
                    year=selected_year
                )
            )

            st.info(
                "Students are split into four equal groups by socioeconomic status "
                "(ESCS index). Q1 = lowest SES, Q4 = highest."
            )

        elif group_key == "Immigration status":
            df = fetch((primary_country,), selected_year,
                       tuple(BASE_COLS + pv_cols + ["IMMIG"]))
            fig = plot_immigration_score_distribution(
                df=df,
                subject=subject,
                cnt=primary_country,
                year=selected_year
            )

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.info(
                "Each curve shows the score distribution for one immigration "
                "status group."
            )

        elif group_key == "School location":
            render_chart_help(chart_type, group_key)
            df = fetch((primary_country,), selected_year, tuple(BASE_COLS + pv_cols + ["SC001Q01TA"]))
            
            fig = plot_school_location_boxplot(
                df=df, subject=subject, cnt=primary_country, year=selected_year,
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        elif group_key == "School type":
            df = fetch((primary_country,), selected_year,
                       tuple(BASE_COLS + pv_cols + ["SCHLTYPE"]))
            fig = plot_school_type_distribution(
                df=df,
                subject=subject,
                cnt=primary_country,
                year=selected_year,
            )

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.info(
                "Each curve shows the score distribution for one school type."
            )

        else:
            fig = plot_group_comparison(
                df=df,
                subject=subject,
                group_col=group_col,
                group_vals=group_vals,
                cnt=primary_country,
                year=selected_year,
                title=f"{SUBJECTS[subject]} by {group_key} | {_cnt_label(primary_country)}"
            )

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.info(
                f"Score distribution broken down by {group_key} for {_cnt_label(primary_country)}."
            )

    # 3. Score Change Over Time
    elif chart_type == "Score change over time":
        df = fetch((primary_country,), None, tuple(BASE_COLS + pv_cols))
        if len(available_years) < 2:
            st.warning(
                "Only one year of data loaded. Run `make data` to add more years."
            )
            return

        if len(selected_countries) > 1:
            st.info(
                f"Time comparison shows one country at a time — displaying {_cnt_label(primary_country)}."
            )

        reference_year = min(available_years)
        comparison_years = [y for y in available_years if y != reference_year]
        
        fig = plot_year_diff_percentile(
            df=df,
            subject=subject,
            cnt=primary_country,
            reference_year=reference_year,
            comparison_years=comparison_years
        )

        st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
        st.info(
            f"This chart shows score changes over time. "
            f"Horizontal axis shows {reference_year} scores. "
            f"Each coloured line shows change relative to {reference_year}."
        )
        
    # 4. Score Distribution
    elif chart_type == "Score distribution":
        fetch_cnts = tuple(set(selected_countries) | set(oecd_countries))
        df = fetch(fetch_cnts, selected_year, tuple(BASE_COLS + pv_cols))
        
        missing_cnts = check_missing_countries(
            df, required_cols=[f"PV1{subject}"],
            countries=selected_countries, year=selected_year
        )
        valid_countries = [c for c in selected_countries if c not in missing_cnts]

        if missing_cnts:
            st.warning(f"⚠️ **Data Unavailable:** Excluded **{', '.join(_cnt_label(c) for c in missing_cnts)}** due to missing {SUBJECTS[subject]} scores.")
            
        if valid_countries:
            fig = plot_weighted_interval_distribution(
                df, subject, valid_countries, year=selected_year
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.info("Distribution showing percentage of students per 20-point score interval.")

    # 5. Belonging by Immigration
    elif chart_type == "Belonging by Immigration":
        extra = ["BELONG", "IMMIG", "ESCS", "REPEAT"]
        df = fetch(tuple(selected_countries), selected_year, tuple(BASE_COLS + extra))
        
        missing_cnts = check_missing_countries(
            df, required_cols=["BELONG", "IMMIG", "REPEAT", "ESCS"], 
            countries=selected_countries, year=selected_year
        )
        valid_countries = [c for c in selected_countries if c not in missing_cnts]
        
        if missing_cnts:
            st.warning(
                f"⚠️ **Data Unavailable:** Excluded **{', '.join(_cnt_label(c) for c in missing_cnts)}** "
                f"due to missing student context data."
            )

        if valid_countries:
            fig = plot_belonging_by_immigration(
                df=df, countries=valid_countries, year=selected_year
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.info("Left: Grade repetition rate in different SES quartiles. Right: Distribution of school belonging index by immigration status.")

    # 6. Country Scatterplot
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
            key=f"scatter_select_{widget_key}" 
        )
        selected_col = resource_options[selected_resource_label]
        
        df = fetch(tuple(all_countries), selected_year, tuple(BASE_COLS + pv_cols + [selected_col]))
        
        missing_cnts = check_missing_countries(df, required_cols=[f"PV1{subject}", selected_col], countries=selected_countries, year=selected_year)
        valid_countries = [c for c in selected_countries if c not in missing_cnts]

        if missing_cnts:
            st.warning(f"⚠️ **Data Unavailable:** Highlighting disabled for **{', '.join(_cnt_label(c) for c in missing_cnts)}** (missing {selected_col} data).")
        
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

# ---------------------------------------------------------------------------
# Data Story helpers — Tab 1 only
# ---------------------------------------------------------------------------

def _story_section_header(number, title, subtitle):
    """Render a numbered section header for the story tab."""
    st.markdown(
        f"""
        <div style="
            border-left: 4px solid #0072B2;
            padding: 8px 16px;
            margin: 24px 0 8px 0;
            background: #f7f9fc;
            border-radius: 0 6px 6px 0;
        ">
            <span style="color:#0072B2; font-size:0.8rem; font-weight:700;
                         letter-spacing:0.08em; text-transform:uppercase;">
                Section {number}
            </span><br>
            <span style="font-size:1.25rem; font-weight:700; color:#1a1a2e;">
                {title}
            </span><br>
            <span style="font-size:0.9rem; color:#555;">{subtitle}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _insight_box(text):
    """Render a highlighted key-finding callout above a chart."""
    st.markdown(
        f"""
        <div style="
            background: #e8f4fd;
            border: 1px solid #b3d9f5;
            border-radius: 6px;
            padding: 12px 16px;
            margin-bottom: 10px;
            font-size: 0.95rem;
            color: #1a3a52;
        ">
            💡 <strong>Key finding:</strong> {text}
        </div>
        """,
        unsafe_allow_html=True,
    )

def _pullquote_box(text: str):
    """
    Render an amber callout for a notably important or surprising finding.
    Conditional — only call this when the data warrants it, not for every chapter.
    """
    st.markdown(
        f"""
        <div style="
            background: #fff8e6;
            border-left: 3px solid #BA7517;
            padding: 10px 14px;
            margin: 8px 0 10px 0;
            font-size: 0.95rem;
            font-weight: 500;
            color: #633806;
            line-height: 1.55;
        ">
            ⚠ {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _policy_box(text: str):
    """
    Render a green policy implication callout.
    Should appear once at the bottom of every chapter section.
    """
    st.markdown(
        f"""
        <div style="
            background: #EAF3DE;
            border-left: 3px solid #3B6D11;
            padding: 10px 14px;
            margin: 10px 0 6px 0;
            font-size: 0.88rem;
            color: #27500A;
            line-height: 1.55;
        ">
            <strong>For policymakers:</strong> {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _chart_expander(label: str, fig, how_to_read: str):
    """
    Render a chart inside a collapsible expander with a how-to-read note.
    Replaces the current pattern of st.plotly_chart() + separate st.expander().
    The chart and its explanation are always together — never separated.
    """
    with st.expander(f"📊 {label}", expanded=False):
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown(
            f"""
            <div style="
                font-size: 0.82rem;
                color: #555;
                line-height: 1.6;
                padding: 6px 2px 2px 2px;
                border-top: 0.5px solid #e0e0e0;
                margin-top: 6px;
            ">
                <strong>How to read:</strong> {how_to_read}
            </div>
            """,
            unsafe_allow_html=True,
        )

def render_story_tab(available_years, story_country, story_subject, comparison_countries):
    """
    Render the full Data Story tab content.

    Structure
    ---------
    Intro → Section 1: Global standing → Section 2: Score change over time
          → Section 3: Equity gaps     → Section 4: School context
    """
    story_year        = 2022 if 2022 in available_years else max(available_years)
    subject_label     = SUBJECTS[story_subject]
    display_countries = [story_country] + comparison_countries
    pv_cols           = PV_BY_SUBJ[story_subject]

    # Page header
    st.markdown(f"## PISA {story_year}: {subject_label} Performance — {_cnt_label(story_country)}")
    st.markdown(
        "This story walks you through what PISA data reveals about student performance "
        "— from how countries compare to the OECD average, to whether results have changed over time, "
        "to which groups of students face the largest opportunity differences."
    )

    with st.expander("ℹ️ What is PISA?"):
        st.markdown("""
            **PISA** (Programme for International Student Assessment) is a global study run by 
            the OECD every three years. It measures 15-year-olds' ability to apply reading, 
            mathematics, and science knowledge to real-world problems — not just recall facts.

            - **Why it matters:** International data from PISA allows policymakers to compare 
              education systems on a common scale and track country-level learning outcomes over time.
            - **Who takes it:** ~700,000 students across 80+ countries and economies
            - **Cycles:** 2000, 2003, 2006, 2009, 2012, 2015, 2018, 2022
            - **Terminology:** The "OECD Average" serves as a benchmark representing the 38 member 
              countries. Non-member participants are referred to as "Partner" economies.
        """)

    with st.expander("🔍 Language note: interpreting score differences"):
        st.markdown("""
            When reviewing this data, it is important to use specific language that avoids deficit framing. 
            Score differences between demographic groups reflect **structural inequalities** in access to 
            resources, language support, school quality etc. — not inherent differences in students' ability.
            For more information, see: [Avoiding Deficit Narratives in Education Research](https://files.eric.ed.gov/fulltext/EJ1348584.pdf).
        """)

    st.divider()

    # ── Section 1: Global standing ─────────────────────────────────────────
    _story_section_header(1, "How does this compare to the OECD average?",
        f"Comparing the {subject_label} performance in {_cnt_label(story_country)} to the OECD baseline")

    fetch_cnts = tuple(set(display_countries) | set(oecd_countries))
    df_s1 = fetch(fetch_cnts, story_year, tuple(BASE_COLS + pv_cols))

    missing_cnts = check_missing_countries(
        df_s1, required_cols=[f"PV1{story_subject}"], 
        countries=display_countries, year=story_year
    )
    valid_countries = [c for c in display_countries if c not in missing_cnts]

    if missing_cnts:
        st.warning(f"⚠️ **Data Unavailable:** Excluded **{', '.join(_cnt_label(c) for c in missing_cnts)}** due to missing {subject_label} scores.")

    if valid_countries:
        mean_text = country_distribution_text(df_s1, story_subject, valid_countries, year=story_year)
        if mean_text:
            _insight_box(mean_text)

        with st.expander("📖 How to read this chart"):
            st.markdown("""
                - The **x-axis** is the percentile (P10 = bottom 10%, P90 = top 10%).
                - The **y-axis** is the PISA score for students at that position.
                - A **steeper curve** means more spread in scores — greater inequality within the country.
                - The **dashed black line** is the OECD average across all member countries.
                - If a country's curve sits **above** the OECD line, its students score higher at 
                  every point in the distribution.
            """)

        fig1 = plot_country_distributions(df_s1, story_subject, valid_countries, year=story_year, primary_country=story_country)
        st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})

    ctx_col1, ctx_col2 = st.columns(2)
    
    with ctx_col1:
        with st.expander("🌍 Which countries are OECD members?"):
            st.markdown(
                "The OECD is an international organization of 38 member countries. "
                "PISA reports often use the OECD average as a global benchmark. "
                "Non-member participating nations are referred to as Partner economies."
            )
            
    with ctx_col2:
        with st.expander("🔍 What does a 20-point difference actually mean?"):
            st.markdown("""
                According to the OECD, **20 PISA score points** represents the average 
                annual pace of learning for 15-year-olds. So if two countries differ by 
                40 points at the median, that's approximately two school years of difference 
                in learning outcomes.
                
                *(Source: [PISA 2022 Results (Volume I), OECD](https://www.oecd.org/content/dam/oecd/en/publications/reports/2023/12/pisa-2022-results-volume-i_76772a36/53f23881-en.pdf))*
            """)

    st.divider()

    # ── Section 2: Score change over time ────────────────────────────────────────
    _story_section_header(2, "Has performance changed over time?",
        f"Tracking {_cnt_label(story_country)}'s {subject_label} scores across PISA cycles")

    if len(available_years) < 2:
        st.info("Only one year of data is currently loaded. Load multiple years (2015, 2018, 2022) to unlock this section.")
    else:
        df_s2 = fetch((story_country,), None, tuple(BASE_COLS + pv_cols))
        
        country_years = df_s2["YEAR"].dropna().unique() if "YEAR" in df_s2.columns else []
        
        if len(country_years) < 2:
            st.warning(f"⚠️ **Data Unavailable:** {_cnt_label(story_country)} does not have enough historical data to compare changes over time (only {len(country_years)} year on record).")
        else:
            # Dynamically use the country's actual earliest/latest years
            reference_year   = min(country_years) 
            comparison_years = [y for y in country_years if y != reference_year]
            latest_year      = max(country_years)

            ref_subset  = df_s2[df_s2["YEAR"] == reference_year]
            last_subset = df_s2[df_s2["YEAR"] == latest_year]
            ref_median  = weighted_percentiles_pv(ref_subset,  story_subject, [50])
            last_median = weighted_percentiles_pv(last_subset, story_subject, [50])

            if not (np.isnan(ref_median).all() or np.isnan(last_median).all()):
                delta     = last_median[0] - ref_median[0]
                direction = "increased" if delta > 0 else "decreased"
                _insight_box(
                    f"At the median, {_cnt_label(story_country)}'s {subject_label} score "
                    f"{direction} by {abs(delta):.0f} points between "
                    f"{reference_year} and {latest_year}."
                )

            with st.expander("📖 How to read this chart"):
                st.markdown(f"""
                    This chart tracks score changes across the performance spectrum over time.

                    - The **horizontal axis (x-axis)** shows the {reference_year} baseline score at each percentile (from the lowest to the highest achievers).
                    - The **vertical axis (y-axis)** shows the score for that exact same percentile in a comparison year.
                    - The **solid diagonal line** represents the {reference_year} baseline (exactly zero change).
                    - The **dashed lines** represent the comparison years.
                    - Points **above** the solid diagonal line indicate an **improvement** at that performance level.
                    - Points **below** the solid diagonal line indicate a **decline** at that performance level.
                """)

            fig2 = plot_naep_time_comparison(df=df_s2, subject=story_subject, cnt=story_country,
                                             reference_year=reference_year,
                                             comparison_years=comparison_years)
            st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

            with st.expander("🔍 Why might scores change between cycles?"):
                st.markdown("""
                    - **Changes in school composition** (e.g. due to immigration, urbanisation)
                    - **Disruptions** such as economic crises or the COVID-19 pandemic (relevant to 2022)
                    - **Cohort effects** — the specific group of 15-year-olds tested that year
                    - **Curriculum or policy reforms** implemented before the cycle
                """)

    st.divider()

    # ── Section 3: Equity gaps ─────────────────────────────────────────────
    _story_section_header(3, "Who scores highest — and who is left behind?",
        f"This section examines two key dimensions of equity: score differences by socioeconomic status and immigration status in {_cnt_label(story_country)}")

    st.markdown("High average scores can mask large differences between student groups.")

    df_ses  = fetch((story_country,), story_year, tuple(BASE_COLS + pv_cols + ["ESCS"]))
    df_immig = fetch((story_country,), story_year, tuple(BASE_COLS + pv_cols + ["IMMIG"]))

    eq_col1, eq_col2 = st.columns(2, gap="large")
    with eq_col1:
        st.markdown("#### Scores by Socioeconomic Status")
        
        # Check if data exists before doing any math
        if check_missing_countries(df_ses, ["ESCS"], [story_country], story_year):
            st.warning(f"⚠️ **Data Unavailable:** Insufficient SES data for {_cnt_label(story_country)}.")
        else:
            # Only runs if data is safe
            ses_text = ses_gap_text(df_ses, story_subject, story_country, year=story_year)
            if "Insufficient" not in ses_text:
                _insight_box(ses_text)
            
            with st.expander("📖 How to read the chart showing scores by socioeconomic status"):
                st.markdown("""
                    Students are divided into **four equal-sized groups** (quartiles) based on 
                    the PISA ESCS index — a composite of parental education, occupational status, 
                    and home possessions. **Q1** = bottom 25%, **Q4** = top 25%.
                """)

            fig3a = plot_escs_gap(df_ses, story_subject, story_country, year=story_year)
            st.plotly_chart(fig3a, use_container_width=True, config={'displayModeBar': False})

    with eq_col2:
        st.markdown("#### Scores by Immigration Status")
        if check_missing_countries(df_immig, ["IMMIG"], [story_country], story_year):
            st.warning(f"⚠️ **Data Unavailable:** Insufficient Immigration data for {_cnt_label(story_country)}.")
        else:
            immig_warnings = check_group_sizes(df_immig, "IMMIG", IMMIG_MAP, story_country, year=story_year)
            for w in immig_warnings:
                st.warning(w)

            immig_text = immigration_gap_text(df_immig, story_subject, story_country, year=story_year)
            if "Insufficient" not in immig_text:
                _insight_box(immig_text)

            with st.expander("📖 How to read this chart"):
                st.markdown("""
                    - **Native:** born in the country, both parents born in the country
                    - **Second-generation:** born in the country, at least one parent born abroad
                    - **First-generation:** born abroad, came to the country before or during school age
                    - The **dotted vertical lines** show the exact median (middle) score for each group.
                """)                
            fig3b = plot_immigration_score_distribution(
                df=df_immig, subject=story_subject, cnt=story_country, year=story_year)
            st.plotly_chart(fig3b, use_container_width=True, config={'displayModeBar': False})

    st.divider()

    # ── Section 4: School context ──────────────────────────────────────────
    _story_section_header(4, "Does school context matter?",
        f"How school location and whether school is public or private relate to {subject_label} scores in {_cnt_label(story_country)}")

    df_loc  = fetch((story_country,), story_year, tuple(BASE_COLS + pv_cols + ["SC001Q01TA"]))
    df_type = fetch((story_country,), story_year, tuple(BASE_COLS + pv_cols + ["SCHLTYPE"]))

    ctx_col1, ctx_col2 = st.columns(2, gap="large")
    with ctx_col1:
        st.markdown("#### Scores by School Location")
        if check_missing_countries(df_loc, ["SC001Q01TA"], [story_country], story_year):
            st.warning(f"⚠️ **Data Unavailable:** Insufficient School Location data for {_cnt_label(story_country)}.")
        else:
            with st.expander("📖 How to read the school location chart"):
                st.markdown("""
                    - The **box** spans the middle 50% of students (P25–P75)
                    - The **line** inside the box is the median score (P50)
                    - The **whiskers** extend to P10 and P90
                """)
            fig4a = plot_school_location_boxplot(
                df=df_loc, subject=story_subject, cnt=story_country, year=story_year)
            st.plotly_chart(fig4a, use_container_width=True, config={'displayModeBar': False})

    with ctx_col2:
        st.markdown("#### Scores by School Type")
        if check_missing_countries(df_type, ["SCHLTYPE"], [story_country], story_year):
            st.warning(f"⚠️ **Data Unavailable:** Insufficient School Type data for {_cnt_label(story_country)}.")
        else:
            with st.expander("📖 How to read the school type chart"):
                st.markdown("""
                    - **Public:** government-operated and funded
                    - **Government-dependent private:** privately managed, significant government funding
                    - **Independent private:** privately managed and primarily privately funded
                """)
            fig4b = plot_school_type_distribution(
                df=df_type, subject=story_subject, cnt=story_country, year=story_year)
            st.plotly_chart(fig4b, use_container_width=True, config={'displayModeBar': False})

    st.divider()

    st.markdown("""
        <div style="background:#f0f4f8; border-radius:8px; padding:20px 24px;
                    margin-top:16px; text-align:center;">
            <p style="font-size:1.05rem; color:#333; margin-bottom:8px;">
                <strong>Want to dig deeper?</strong>
            </p>
            <p style="font-size:0.9rem; color:#555;">
                Switch to the <strong>🔍 Explore</strong> tab to choose any country, 
                subject, year, and chart type — and compare two views side by side.
            </p>
        </div>
    """, unsafe_allow_html=True)

# Load data and derive country lists
meta = get_meta()

available_years = sorted(meta["YEAR"].unique().tolist())
all_countries = sorted(meta["CNT"].unique().tolist(), key=_cnt_label)
oecd_countries = sorted(meta[meta["OECD"] == 1]["CNT"].unique().tolist(), key=_cnt_label)
partner_countries = sorted(meta[meta["OECD"] == 0]["CNT"].unique().tolist(), key=_cnt_label)

# ==========================================
# SIDEBAR NAVIGATION & ROUTING
# ==========================================

# 1. The Main App Router
app_mode = st.sidebar.radio(
    "Navigation", 
    ["📖 Data Story", "🔍 Explore"],
    label_visibility="collapsed" # Hides the word "Navigation" for a cleaner look
)

st.sidebar.info(
    "**Interactive Dashboard:** Hover over charts for exact values, "
    "click legend items to hide/show groups, and drag to zoom."
)

st.sidebar.markdown("---")

# ==========================================
# MODE 1: DATA STORY
# ==========================================
if app_mode == "📖 Data Story":
    st.sidebar.header("📖 Story Controls")

    story_country_group = st.sidebar.radio(
        "Country group",
        ["All", "OECD members", "Partner countries"],
        key="story_country_group"
    )
    if story_country_group == "OECD members":
        story_pool = oecd_countries
    elif story_country_group == "Partner countries":
        story_pool = partner_countries
    else:
        story_pool = all_countries

    default_idx = story_pool.index("CAN") if "CAN" in story_pool else 0

    story_country = st.sidebar.selectbox(
        "Focus country", 
        story_pool, 
        index=default_idx, 
        format_func=_cnt_label,
        key="story_country"
    )
    story_subject = st.sidebar.selectbox(
        "Subject", list(SUBJECTS.keys()),
        format_func=lambda x: SUBJECTS[x],
        key="story_subject"
    )
    
    # Draw the main area
    render_story_tab(available_years, story_country, story_subject, [])


# ==========================================
# MODE 2: EXPLORE
# ==========================================
elif app_mode == "🔍 Explore":
    st.sidebar.header("🔍 Explore Filters")

    side_by_side = st.sidebar.toggle("Compare two views side by side", value=False, key="sbs_toggle")

    if side_by_side:
        st.sidebar.markdown("**Left panel**")
        chart_type_left = st.sidebar.selectbox("Left chart", CHART_TYPES, key="chart_left")
        st.sidebar.markdown("**Right panel**")
        chart_type_right = st.sidebar.selectbox("Right chart", CHART_TYPES, key="chart_right", index=1)
    else:
        chart_type = st.sidebar.radio("View", CHART_TYPES)

    st.sidebar.markdown("---")

    country_group = st.sidebar.radio(
        "Country group", ["All", "OECD members", "Partner countries"]
    )
    if country_group == "OECD members":
        country_pool = oecd_countries
    elif country_group == "Partner countries":
        country_pool = partner_countries
    else:
        country_pool = all_countries

    DEFAULT_COUNTRIES = ["CAN", "USA"]
    SINGLE_COUNTRY_CHARTS = ["Score change over time", "Group comparison"]

    if "memory_countries" not in st.session_state:
        st.session_state.memory_countries = [
            c for c in DEFAULT_COUNTRIES if c in country_pool
        ]

    valid_defaults = [
        c for c in st.session_state.memory_countries if c in country_pool]
    if not valid_defaults:
        valid_defaults = [country_pool[0]]

    # Determine selected countries based on chart type requirements
    if not side_by_side and chart_type in SINGLE_COUNTRY_CHARTS:
        # Use the first country from memory as the default, but DO NOT overwrite the memory list
        current_index = country_pool.index(valid_defaults[0])
        selected_country = st.sidebar.selectbox(
            "Country", 
            country_pool, 
            index=current_index,
            format_func=_cnt_label
        )
        selected_countries = [selected_country]
    else:
        # User is in a multi-country view. Update memory ONLY here.
        label = "Countries (Pool)" if side_by_side else "Countries"
        selected_countries = st.sidebar.multiselect(
            label, 
            country_pool, 
            default=valid_defaults,
            format_func=_cnt_label
        )
        if selected_countries:
            st.session_state.memory_countries = selected_countries

    if not selected_countries:
        st.warning("Please select at least one country.")
        st.stop()
        
    if side_by_side:
        country_left = st.sidebar.selectbox(
            "Left country", 
            selected_countries, 
            key="cnt_left",
            format_func=_cnt_label
        )
        right_idx = 1 if len(selected_countries) > 1 else 0
        country_right = st.sidebar.selectbox(
            "Right country",
            selected_countries,
            index=right_idx,
            key="cnt_right",
            format_func=_cnt_label
        )

    st.sidebar.markdown("---")

    subject = st.sidebar.selectbox(
        "Subject", list(SUBJECTS.keys()),
        format_func=lambda x: SUBJECTS[x]
    )

    st.sidebar.markdown("---")
    
    is_time_chart = (
        (not side_by_side and chart_type == "Score change over time") or
        (side_by_side and (
            chart_type_left == "Score change over time" or
            chart_type_right == "Score change over time"
            ))
    )

    if is_time_chart:
        st.sidebar.markdown("### Time comparison")
        ref_year = st.sidebar.selectbox(
            "Baseline year", available_years[:-1], index=0, key="ref_year"
            )
        later_years = [y for y in available_years if y > ref_year]
        comp_year = st.sidebar.selectbox(
            "Compare to", later_years, index=len(later_years) - 1, key="comp_year"
            )
        selected_year = None
    else:
        selected_year = st.sidebar.selectbox(
            "Year", available_years, 
            index=available_years.index(2022) if 2022 in available_years else len(available_years) - 1
        )
        ref_year = None
        comp_year = None

    primary_country = selected_countries[0]

    st.title("PISA Score Distribution Dashboard")
    st.caption(f"Data: PISA {', '.join(str(y) for y in available_years)}  |  "
               f"{len(all_countries)} countries")

    # Chart-Specific Controls injected directly into the main view
    group_key = None
    group_key_left = None
    group_key_right = None

    # ── Draw the main area for Explore ───────────────────────────────────────
    if side_by_side:
        left_col, right_col = st.columns(2)
        with left_col:
            st.subheader(chart_type_left)
            render_chart(
                chart_type=chart_type_left, 
                subject=subject, 
                selected_countries=selected_countries,
                selected_year=selected_year, 
                available_years=available_years,
                primary_country=country_left,
                ref_year=ref_year, 
                comp_year=comp_year,
                compact=True,
                widget_key="left"
            )
        with right_col:
            st.subheader(chart_type_right)
            render_chart(
                chart_type=chart_type_right, 
                subject=subject, 
                selected_countries=selected_countries,
                selected_year=selected_year, 
                available_years=available_years,
                primary_country=country_right,
                ref_year=ref_year, 
                comp_year=comp_year,
                compact=True,
                widget_key="right"
            )
    else:
        render_chart(
            chart_type=chart_type, 
            subject=subject, 
            selected_countries=selected_countries,
            selected_year=selected_year, 
            available_years=available_years, 
            primary_country=primary_country,
            ref_year=ref_year, 
            comp_year=comp_year,
            compact=False,
            widget_key="main"
        )