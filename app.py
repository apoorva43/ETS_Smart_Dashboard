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
import numpy as np
from pathlib import Path

from src.data_loader import query_pisa, load_precomputed
from src.pisa_stats import (
    weighted_percentiles_pv,
    compute_escs_quartile_percentiles,
    compute_group_percentiles
)
from src.config import SUBJECTS, GROUP_OPTIONS, IMMIG_MAP
from src.plotting_plotly import (
                          plot_country_shaded_density,
                          plot_percentile_change_from_baseline,
                          plot_intersectional_heatmap,
                          plot_resource_scatter,
                          plot_group_shaded_density,
                          plot_group_shaded_density_precomputed,
                          plot_country_shaded_density_precomputed,
                          plot_percentile_change_from_baseline_precomputed,
                          _cnt_label)
from src.text_generator import (ses_difference_text,
                                scatter_correlation_text)

st.set_page_config(page_title="PISA Dashboard", layout="wide")

S3_BASE_URL = "https://pisa-dashboard-data.s3.ca-central-1.amazonaws.com"

CHART_TYPES = [
    "Percentile score profile",
    "Score change over time",
    "Intersectional Heatmap",
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
REP_COLS = [f"W_FSTURWT{r}" for r in range(1, 81)]
SE_THRESHOLD = 3.5 
CI_Z = 1.96 # 95% confidence interval

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
    return pd.read_parquet(f"{S3_BASE_URL}/pisa_country_stats.parquet")

@st.cache_data(ttl=3600)
def get_precomputed() -> pd.DataFrame:
    """Load precomputed stats once and cache for the session."""
    from src.data_loader import load_precomputed
    return load_precomputed()


@st.cache_data(ttl=3600)
def get_se_lookup() -> pd.DataFrame:
    """
    Load precomputed standard errors and 95% CIs for every
    (CNT, YEAR, SUBJECT) combination. Indexed for O(1) lookup -
    replaces the runtime 91-column fetch + Fay BRR computation.
    """
    local = Path("data/processed/pisa_se_stats.parquet")
    df = pd.read_parquet(local) if local.exists() else pd.read_parquet(f"{S3_BASE_URL}/pisa_se_stats.parquet")
    return df.set_index(["CNT", "YEAR", "SUBJECT"])
    

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

GROUP_HOW_TO_READ = {
    "Score distribution": (
        "This plot shows the full spread of student scores. "
        "Darker, wider sections indicate where most students are concentrated (the middle 50%), "
        "while the thinner tails show the students at the lower and higher ends of the score distribution. "
        "The vertical line marks the median score. The asterisk shows the average score for the group, which may differ from the median if the distribution is skewed."
    ),
    "Change over time": (
        "The horizontal axis shows the baseline score at each percentile. "
        "The vertical axis shows the change in score in subsequent cycles. "
        "Points on the zero line mean no change. Points above indicate improvement; points below indicate decline."
    ),
    "Socioeconomic status": (
        "Each shape shows the score distribution for one socioeconomic group. "
        "Students are divided into four equal groups based on family background "
        "(parental education, occupation, and home resources):<br>"
        "• Q4: Highest 25%<br>"
        "• Q1: Lowest 25%<br>"
        "The vertical line marks the group's median score."
        "Percentages may not add exactly to 100 percent because some students do not have enough socioeconomic information to be grouped."
    ),
    "Immigration status": (
        "Each shape shows the score distribution for one immigration background:<br>"
        "• Native: Student and both parents born in-country.<br>"
        "• First-generation: Student born abroad.<br>"
        "• Second-generation: Student born in-country, at least one parent born abroad.<br>"
        "The vertical line marks the group's median score. "
        "Percentages may not add exactly to 100 percent because some students do not have enough immigration information to be grouped."
    ),
    "Gender": (
        "Each shape shows the score distribution for male and female students. "
        "The vertical line marks the group's median score."
    ),
    "School type": (
        "Each shape shows the score distribution for one school type:<br>"
        "• Public: Government-operated.<br>"
        "• Government-dependent private: Privately managed but significantly government-funded.<br>"
        "• Independent private: Primarily privately funded.<br>"
        "Percentages indicate the share of students in each school type."
    ),
    "School location": (
        "Each shape shows the score distribution based on the community size where the school is located. "
        "The vertical line marks the group's median score. "
        "Percentages indicate the share of students in each location."
    )
}           

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


def render_plotly_chart_with_note(fig, note, key=None, use_container_width=True):
    """
    Render a Plotly chart with a short interpretation note and interaction tips.
    """
    st.plotly_chart(
        fig,
        use_container_width=use_container_width,
        key=key,
        config={
            "displayModeBar": True,
            "scrollZoom": False,
        },
    )

    st.info(note)

    st.caption(
        "Tip: Click legend items to temporarily hide or show groups. \n"
        "Hover over the chart to see exact values. "
        "Drag across the chart to zoom in, or use the toolbar to pan, reset axes, and download the chart."
    )


def get_plotly_y_range(figs, pad=3):
    """
    Get shared y-axis range from multiple Plotly figures.
    Used to make side-by-side comparisons visually comparable.
    """
    ys = []

    for fig in figs:
        for trace in fig.data:
            if hasattr(trace, "y") and trace.y is not None:
                vals = pd.to_numeric(pd.Series(trace.y),
                                     errors="coerce").dropna()
                ys.extend(vals.tolist())

    if not ys:
        return None

    y_min = min(ys)
    y_max = max(ys)

    # Always include 0 for change-over-time charts
    y_min = min(y_min, 0)
    y_max = max(y_max, 0)

    if y_min == y_max:
        y_min -= 1
        y_max += 1

    return [y_min - pad, y_max + pad]
    

def get_chart_note(chart_type, group_key=None, reference_year=None):
    """
    Short chart-specific explanation shown below each chart.
    """
    if chart_type == "Percentile score profile":
        return (
            "This chart shows the full score distribution for the selected country or countries. "
            "Darker shading shows where most students are concentrated, while the markers show key percentiles. "
            "Use it to compare both the typical score and the spread of student outcomes."
        )

    if chart_type == "Score change over time":
        baseline_text = f" relative to {reference_year}" if reference_year else ""
        return (
            f"This chart shows how scores changed across PISA cycles{baseline_text}. "
            "Each line of coloured markers represents a different percentile group, helping show whether changes were larger "
            "among lower, middle, or higher performing students."
        )

    if chart_type == "Intersectional Heatmap":
        return (
            "This heatmap shows average scores for students who fall into two background categories at the same time. "
            "Darker cells indicate higher average scores. Use it to identify where overlapping student characteristics "
            "are associated with stronger or weaker outcomes."
        )

    if chart_type == "Country Scatterplot":
        return (
            "This scatterplot compares countries by a selected context variable and mean score. "
            "Each point is a country. Use it to identify broad cross-country patterns, but **do not interpret the relationship as causal**."
        )

    if chart_type == "Group comparison":
        if group_key == "Socioeconomic status":
            return (
                "This chart compares score distributions across socioeconomic quartiles. "
                "Students are ranked by the index of economic, social, and cultural status (ESCS) and split into four equal-sized groups: Q1 is the lowest quartile and Q4 is the highest. "
                "Use it to see whether socioeconomic gaps are consistent across the distribution or larger for particular student groups."
            )

        if group_key == "Immigration status":
            return (
                "This chart compares score distributions by immigration background. "
                "Use it to compare group medians while also checking how much the distributions overlap."
            )

        if group_key == "Gender":
            return (
                "This chart compares score distributions by gender. "
                "Use it to see whether differences are concentrated in lower, middle, or higher performing students."
            )

        if group_key == "School location":
            return (
                "This chart compares score distributions by school location. "
                "School location is grouped by the size or type of the community where the school is situated. "
                "Use it to see whether location differences appear mainly in the median or across the full distribution."
            )

        if group_key == "School type":
            return (
                "This chart compares score distributions by school type. "
                "School types typically include public and private institutions, with variations in management and funding depending on the country. "
                "Use it to compare typical performance while also considering the spread and overlap within each school type."
            )

        return (
            f"This chart compares score distributions by {group_key}. "
            "Use it to compare both group medians and within-group variation."
        )

    return (
        "This chart summarizes PISA performance for the selected filters. "
        "Hover over the chart to inspect exact values."
    )


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
            fig = plot_country_shaded_density(
                df, subject, valid_countries, year=selected_year, compact=compact
            )
            render_plotly_chart_with_note(
                fig,
                note=get_chart_note("Percentile score profile"),
                key=f"percentile_profile_{widget_key}_{subject}_{selected_year}",
            )

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

        # Fetch data ONCE for whatever group was selected
        df = fetch((primary_country,), selected_year,
                   tuple(BASE_COLS + pv_cols + [group_col]))
        
        # Size warnings for all groups except SES
        if group_key != "Socioeconomic status":
            warns = check_group_sizes(
                df, group_col, group_vals, primary_country, year=selected_year
            )
            for w in warns:
                st.warning(w)

        # Sorting (SES uses fixed Q1-Q4, others sort by median)
        sort_by_med = False if group_key == "Socioeconomic status" else True

        # Generate the plot ONCE
        fig = plot_group_shaded_density(
            df=df,
            subject=subject,
            cnt=primary_country,
            group_col=group_col, 
            group_labels=group_vals, 
            group_title=group_key,
            year=selected_year,
            compact=compact,
            sort_by_median=sort_by_med
        )


        # Build chart note
        chart_note = get_chart_note("Group comparison", group_key=group_key)

        # Add SES-specific finding into the blue note box instead of showing it above the chart
        if group_key == "Socioeconomic status":
            # Safely check data volume before calling the text generator
            if len(df.dropna(subset=["ESCS", "W_FSTUWT"])) >= 30:
                ses_text = ses_difference_text(
                    df,
                    subject,
                    primary_country,
                    year=selected_year
                )
                chart_note = f"{chart_note}\n\n{ses_text}"
            else:
                chart_note = f"{chart_note}\n\n⚠️ Insufficient socioeconomic data to calculate differences."

        # Render the chart ONCE, formatting the key dynamically
        safe_key = group_key.lower().replace(" ", "_")
        render_plotly_chart_with_note(
            fig,
            note=chart_note,
            key=f"group_{safe_key}_{widget_key}_{subject}_{selected_year}_{primary_country}",
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
        
        fig = plot_percentile_change_from_baseline(
            df=df,
            subject=subject,
            cnt=primary_country,
            reference_year=reference_year
        )

        render_plotly_chart_with_note(
            fig,
            note=get_chart_note(
                "Score change over time",
                reference_year=reference_year
            ),
            key=f"time_change_{widget_key}_{subject}_{reference_year}_{primary_country}",
        )

    # 4. Intersectional Heatmap
    elif chart_type == "Intersectional Heatmap":
        extra = ["BELONG", "IMMIG", "ESCS", "REPEAT"]
        df = fetch(tuple(selected_countries), selected_year, tuple(BASE_COLS + pv_cols + extra))
        
        missing_cnts = check_missing_countries(
            df, required_cols=["BELONG", "IMMIG", "REPEAT", "ESCS"], 
            countries=selected_countries, year=selected_year
        )
        valid_countries = [c for c in selected_countries if c not in missing_cnts]

        if len(selected_countries) > 1:
            st.info(
                f"Intersectional heatmap shows one country at a time — displaying {_cnt_label(primary_country)}."
            )
        
        if missing_cnts:
            st.warning(
                f"⚠️ **Data Unavailable:** Excluded **{', '.join(_cnt_label(c) for c in missing_cnts)}** "
                f"due to missing student context data."
            )

        if valid_countries:
            crossing_options = {
                "School Belonging": "BELONG",
                "Immigration Status": "IMMIG",
                "Gender": "ST004D01T",
            }
            cross_label = st.selectbox(
                "Cross SES with:",
                list(crossing_options.keys()),
                key=f"cross_{widget_key}"
            )
            cross_col = crossing_options[cross_label]

            # Only fetch cross_col if not already in df
            if cross_col not in df.columns:
                df_cross = fetch(
                    tuple(valid_countries),
                    selected_year,
                    tuple(BASE_COLS + pv_cols + ["ESCS", cross_col])
                )
            else:
                df_cross = df  # BELONG, IMMIG, ESCS already fetched above

            fig_hm = plot_intersectional_heatmap(
                df_cross, subject, primary_country,
                row_var="ESCS", col_var=cross_col,
                row_label="SES Quartile", col_label=cross_label,
                year=selected_year
            )
            render_plotly_chart_with_note(
                fig_hm,
                note=get_chart_note("Intersectional Heatmap"),
                key=f"heatmap_{widget_key}_{subject}_{selected_year}_{primary_country}_{cross_col}",
            )

    # 5. Country Scatterplot
    elif chart_type == "Country Scatterplot":
        resource_options = {
            "Socioeconomic Status (ESCS)": "ESCS",
            "School Belonging Index": "BELONG",
        }
            
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
            st.warning(f"⚠️ **Data Unavailable:** Highlighting disabled for **{', '.join(_cnt_label(c) for c in missing_cnts)}** (missing data).")
        
        fig = plot_resource_scatter(
            df=df,
            subject=subject,
            resource_col=selected_col,
            resource_label=selected_resource_label,
            year=selected_year,
            highlight_countries=valid_countries
        )
        scatter_intro = get_chart_note("Country Scatterplot")


        scatter_finding = scatter_correlation_text(
            df=df,
            subject=subject,
            resource_col=selected_col,
            resource_label=selected_resource_label,
            year=selected_year,
            highlight_countries=valid_countries
        )

        render_plotly_chart_with_note(
            fig,
            note=f"{scatter_intro}\n\n{scatter_finding}",
            key=f"scatter_{widget_key}_{subject}_{selected_year}_{selected_col}",
        )

        

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


def _insight_box(findings):
    """Render a highlighted key-finding callout above a chart.
    Accepts either a single string or a list of strings for bullet points."""
    
    if isinstance(findings, list) and len(findings) > 1:
        title = "<span style='font-size: 0.9rem;'>💡 <strong>Key findings:</strong></span>"
        items = "".join([
            f"<li style='margin-bottom: 4px; line-height: 1.4;'>"
            f"<span style='font-size: 0.85rem;'>{f}</span></li>" 
            for f in findings
        ])
        content = f"<ul style='margin-top: 6px; margin-bottom: 0; padding-left: 20px;'>{items}</ul>"
    elif isinstance(findings, list) and len(findings) == 1:
        title = "<span style='font-size: 0.95rem;'>💡 <strong>Key finding:</strong></span>"
        content = f" <span style='font-size: 0.9rem;'>{findings[0]}</span>"
    else:
        title = "<span style='font-size: 0.95rem;'>💡 <strong>Key finding:</strong></span>"
        content = f" <span style='font-size: 0.9rem;'>{findings}</span>"

    st.markdown(
        f"""
        <div style="
            background: #e8f4fd;
            border: 1px solid #b3d9f5;
            border-radius: 6px;
            padding: 12px 16px;
            margin-bottom: 10px;
            color: #1a3a52;
        ">
            {title}{content}
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


def _chart_expander(label: str, fig, how_to_read: str, expanded: bool = True):
    """
    Render a chart inside a collapsible expander with a how-to-read note.
    Replaces the current pattern of st.plotly_chart() + separate st.expander().
    The chart and its explanation are always together — never separated.
    """
    with st.expander(f"📊 {label}", expanded=expanded):
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

def _metric_card(col, label, value, delta=None, help_text=None):
            delta_html = ""
            if delta is not None:
                color = "#27500A" if delta >= 0 else "#A32D2D"
                sign = "▲" if delta >= 0 else "▼"
                delta_html = f"""
                    <div style="font-size:12px;color:{color};margin-top:3px">
                        {sign} {abs(delta):.0f} pts vs OECD
                    </div>
                """
            help_html = ""
            if help_text:
                help_html = f"""
                    <div style="font-size:10px;color:#888;margin-top:3px">
                        {help_text}
                    </div>
                """
            col.markdown(
                f"""
                <div style="
                    background:#f0f4f8;
                    border-radius:8px;
                    padding:14px 16px;
                    border:0.5px solid #dde3ea;
                ">
                    <div style="font-size:11px;color:#666;margin-bottom:4px">{label}</div>
                    <div style="font-size:22px;font-weight:500;color:#1F4E79">{value}</div>
                    {delta_html}
                    {help_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

def render_story_tab(available_years, story_country, story_subject, df_pre=None):
    """
    Render the full Data Story tab content.

    Structure
    ---------
    Intro → Ch 1: Standing → Ch 2: Trend → Ch 3: Equity → Ch 4: School context
    
    Callout hierarchy:
    - Blue _add_story_finding     : key finding, always present, leads each chapter
    - Amber _pullquote_box  : conditional, only when data is notably uneven
    - Green _policy_box     : always present, closes each chapter
    - _chart_expander       : chart + how-to-read, collapsed by default
    """
    story_year    = 2022 if 2022 in available_years else max(available_years)
    subject_label = SUBJECTS[story_subject]
    pv_cols       = PV_BY_SUBJ[story_subject]

    if df_pre is not None:
        country_years = sorted(df_pre[
            (df_pre["CNT"] == story_country) & 
            (df_pre["SUBJECT"] == story_subject)
        ]["YEAR"].dropna().unique().tolist())
    else:
        # If no precomputed data is available, skip the hint to avoid a heavy raw data query
        country_years = []

    if len(country_years) > 0 and story_year not in country_years:
        years_str = ", ".join([str(int(y)) for y in country_years])
        data_hint = f" **Tip:** Data is available for {years_str}. Switch to Explore Mode to view it."
    else:
        data_hint = ""

    st.markdown("""
    <style>
    [data-testid="stMetric"] {
        border: 0.5px solid #dde3ea;
        border-radius: 8px;
        padding: 12px 16px;
        min-height: 120px;
    }
    </style>
""", unsafe_allow_html=True)
    
    story_findings = []

    def _clean_finding_text(text):
        """Remove simple HTML tags before storing finding text for summary."""
        import re
        return re.sub(r"<.*?>", "", str(text)).strip()

    def _add_story_finding(finding):
        """
        Show key finding box and store finding for the final summary section.
        Accepts either a single string or a list of strings.
        """
        _insight_box(finding)

        if isinstance(finding, list):
            for f in finding:
                if f:
                    story_findings.append(_clean_finding_text(f))
        else:
            story_findings.append(_clean_finding_text(finding))

    # ── Page header ────────────────────────────────────────────────────────
    st.markdown(
        f"## PISA {story_year}: {subject_label} Performance "
        f"— {_cnt_label(story_country)}"
    )
    st.markdown(
        "This report walks you through what PISA data reveals about student "
        "performance — from how this country compares to the OECD average, "
        "to whether results have changed over time, to which groups of "
        "students face the largest opportunity differences."
    )

    with st.expander("ℹ️ What is PISA?"):
        st.markdown("""
            **PISA** (Programme for International Student Assessment) is a
            global study run by the OECD every three years. It measures
            15-year-olds' ability to apply reading, mathematics, and science
            knowledge to real-world problems — not just recall facts.

            - **Why it matters:** International data from PISA allows
              policymakers to compare education systems on a common scale
              and track country-level learning outcomes over time.
            - **Who takes it:** ~700,000 students across 80+ countries
            - **Cycles:** 2000, 2003, 2006, 2009, 2012, 2015, 2018, 2022
            - **OECD average:** The benchmark representing the 38 OECD member
              countries. Averages reflect mean scores across students, reported 
              equally across member countries per PISA's official methodology 
              (see [OECD PISA 2022 Results, Volume I](https://www.oecd.org/en/publications/pisa-2022-results-volume-i_53f23881-en.html)). 
              Non-member participants are referred to as Partner economies.
        """)

    with st.expander("🔍 Language note: interpreting score differences"):
        st.markdown("""
            Score differences between demographic groups reflect
            **structural inequalities** in access to resources, language
            support, and school quality — not inherent differences in
            students' ability or potential. For more information, see:
            [Avoiding Deficit Narratives in Education Research](https://files.eric.ed.gov/fulltext/EJ1348584.pdf).
        """)

    st.divider()

    # ── Chapter 1: Standing ────────────────────────────────────────────────
    _story_section_header(
        1,
        "Where does this country stand?",
        f"Comparing {_cnt_label(story_country)} to the {story_year} OECD average in {subject_label}"
    )

    if df_pre is None:
        fetch_cnts = tuple(set([story_country]) | set(oecd_countries))
        df_s1 = fetch(fetch_cnts, story_year, tuple(BASE_COLS + pv_cols))
    else:
        df_s1 = None

    if df_pre is not None:
    # Check against precomputed file instead
        cnt_pre_check = df_pre[
            (df_pre["CNT"]        == story_country) &
            (df_pre["YEAR"]       == story_year)    &
            (df_pre["SUBJECT"]    == story_subject) &
            (df_pre["GROUP_TYPE"] == "country")
        ]
        missing_s1 = [] if not cnt_pre_check.empty else [story_country]
    else:
        missing_s1 = check_missing_countries(
            df_s1, [f"PV1{story_subject}"], [story_country], story_year
        )

    if missing_s1:
        st.warning(
            f"⚠️ **Data unavailable:** No overall {subject_label} scores for {_cnt_label(story_country)} in {story_year}.\n\n{data_hint}"
        )
    else:
            
        if df_pre is not None:
            cnt_pre = df_pre[
                (df_pre["CNT"]        == story_country) &
                (df_pre["YEAR"]       == story_year)    &
                (df_pre["SUBJECT"]    == story_subject) &
                (df_pre["GROUP_TYPE"] == "country")
            ]
            oecd_pre = df_pre[
                (df_pre["CNT"]        == "OECD")        &
                (df_pre["YEAR"]       == story_year)    &
                (df_pre["SUBJECT"]    == story_subject) &
                (df_pre["GROUP_TYPE"] == "oecd")
            ]
            if not cnt_pre.empty and not oecd_pre.empty:
                cnt_mean_score  = float(cnt_pre.iloc[0]["MEAN"]) if not np.isnan(cnt_pre.iloc[0]["MEAN"]) else float(cnt_pre.iloc[0]["P50"])
                oecd_mean_score = float(oecd_pre.iloc[0]["P50"])
                p10_p90_spread  = float(cnt_pre.iloc[0]["P90"]) - float(cnt_pre.iloc[0]["P10"])
                cnt_se          = float(cnt_pre.iloc[0]["SE"]) if not np.isnan(cnt_pre.iloc[0]["SE"]) else np.nan
                delta_val       = cnt_mean_score - oecd_mean_score
            else:
                cnt_mean_score  = np.nan
                oecd_mean_score = np.nan
                p10_p90_spread  = None
                cnt_se          = np.nan
                delta_val       = None
        else:
            # Fall back to live computation from raw data
            s1_pv_cols = [
                f"PV{i}{story_subject}" for i in range(1, 11)
                if f"PV{i}{story_subject}" in df_s1.columns
            ]
            cnt_subset = df_s1[df_s1["CNT"] == story_country]
            if story_year:
                cnt_subset = cnt_subset[cnt_subset["YEAR"] == story_year]
            cnt_mean_score = np.mean([
                np.average(cnt_subset[pv].values, weights=cnt_subset["W_FSTUWT"].values)
                for pv in s1_pv_cols
            ])
            oecd_country_means = []
            for oecd_cnt in df_s1[df_s1["OECD"] == 1]["CNT"].unique():
                c = df_s1[
                    (df_s1["CNT"] == oecd_cnt) & (df_s1["YEAR"] == story_year)
                ].dropna(subset=["W_FSTUWT"] + s1_pv_cols)
                if len(c) < 30:
                    continue
                oecd_country_means.append(np.mean([
                    np.average(c[pv].values, weights=c["W_FSTUWT"].values)
                    for pv in s1_pv_cols
                ]))
            oecd_mean_score = np.mean(oecd_country_means) if oecd_country_means else np.nan
            delta_val = (
                cnt_mean_score - oecd_mean_score
                if not np.isnan(cnt_mean_score) else None
            )
            cnt_p10_p90 = weighted_percentiles_pv(cnt_subset, story_subject, [10, 90])
            p10_p90_spread = (
                cnt_p10_p90[1] - cnt_p10_p90[0]
                if not np.isnan(cnt_p10_p90).all() else None
            )

        # Standard error and 95% CI lookup

        se_lookup = get_se_lookup()

        try: 
            se_row = se_lookup.loc[(story_country, story_year, story_subject)]
            cnt_se = se_row["se"]
            ci_lower = se_row["ci_lower"]
            ci_upper = se_row["ci_upper"]
        except KeyError:
            cnt_se = ci_lower = ci_upper = np.nan

        if delta_val is not None and not np.isnan(delta_val):
            # Round the means FIRST to prevent visual math mismatches
            cnt_mean_r = round(cnt_mean_score)
            oecd_mean_r = round(oecd_mean_score)
            visual_delta = cnt_mean_r - oecd_mean_r
            
            if visual_delta > 0:
                direction = "higher than"
            elif visual_delta < 0:
                direction = "lower than"
            else:
                direction = "equal to"
            
            # Format the text dynamically based on whether there is a gap or a tie
            if visual_delta == 0:
                gap_text = "is equal to"
            else:
                gap_text = f"is {abs(visual_delta)} points {direction}"

            _add_story_finding(
                f"In {story_year}, {_cnt_label(story_country)}'s average {subject_label} score "
                f"{gap_text} the OECD average "
                f"({cnt_mean_r} vs {oecd_mean_r})."
            )
        cols = st.columns(3)
        

        # Card 1: country average score + standard error
        with cols[0]:
            se_suffix = f" ± {cnt_se:.1f}" if not np.isnan(cnt_se) else ""
            ci_text = (
                f"95% confidence interval: [{ci_lower:.0f}, {ci_upper:.0f}]. "
                "This range reflects sampling and measurement uncertainty in the estimate - "
                "the true population mean is likely within this range."
                if not np.isnan(ci_lower) else
                "Standard error unavailable for this selection."
            )
            st.metric(
                f"{_cnt_label(story_country)} average score",
                f"{cnt_mean_score:.0f}{se_suffix}" if not np.isnan(cnt_mean_score) else "N/A",
                delta=f"{delta_val:+.0f} pts vs OECD" if delta_val is not None else None,
                help=ci_text
            )

        # Card 2: OECD average
        with cols[1]:
            st.metric(
                "OECD average",
                f"{oecd_mean_score:.0f}" if not np.isnan(oecd_mean_score) else "N/A",
                help="Mean score averaged equally across all OECD member countries"
            )

        # Card 3: spread
        with cols[2]:
            st.metric(
                "P10 → P90 Spread",
                f"{p10_p90_spread:.0f} pts" if p10_p90_spread is not None else "N/A",
                help="Score range between students at the lower end (10%) and upper end (90%) of the distribution"
            )


        # Chart + how to read
        fig1 = plot_country_shaded_density_precomputed(
            df_pre, story_subject, [
                story_country], year=story_year, show_mean_marker=True,
        ) if df_pre is not None else plot_country_shaded_density(
            df_s1, story_subject, [
                story_country], year=story_year, show_mean_marker=True,
        )
        _chart_expander(
                "Collapse chart", fig1,
                GROUP_HOW_TO_READ.get("Score distribution", "Each bar shows the score distribution for one group.")
            )

        _policy_box(
            f"{_cnt_label(story_country)}'s position relative to the OECD average "
            f"is one signal, but the spread within the country often tells a more "
            f"important story for domestic policy. For example, a large internal spread would suggest "
            f"that raising the floor — not just the average — is the priority."
        )

    st.divider()

    # ── Chapter 2: Trend ───────────────────────────────────────────────────
    _story_section_header(
        2,
        "Has performance changed over time?",
        f"Tracking {_cnt_label(story_country)}'s {subject_label} scores across PISA cycles"
    )

    if len(available_years) < 2:
        st.info(
            "Only one year of data is currently loaded. "
            "Load 2015, 2018, and 2022 to unlock this section."
        )
    else:
        if df_pre is None:
            df_s2 = fetch(
                (story_country,), None, tuple(BASE_COLS + pv_cols)
            )
        else:
            df_s2 = None

        if df_pre is not None:
            trend_rows = df_pre[
                (df_pre["CNT"]        == story_country) &
                (df_pre["SUBJECT"]    == story_subject) &
                (df_pre["GROUP_TYPE"] == "trend")
            ]
            country_years = sorted(trend_rows["YEAR"].unique().tolist())
        else:
            country_years = sorted(
                df_s2["YEAR"].dropna().unique().tolist()
            ) if "YEAR" in df_s2.columns else []

        if len(country_years) < 2:
            st.warning(
                f"⚠️ **Data unavailable:** {_cnt_label(story_country)} "
                f"requires at least two years of data to show historical trends."
            )
        else:
            reference_year   = min(country_years)
            latest_year      = max(country_years)

            # Explicitly extract exactly the 3 percentiles needed
            if df_pre is not None:
                trend_rows = df_pre[
                    (df_pre["CNT"]        == story_country) &
                    (df_pre["SUBJECT"]    == story_subject) &
                    (df_pre["GROUP_TYPE"] == "trend")
                ]
                ref_row  = trend_rows[trend_rows["YEAR"] == reference_year].iloc[0]
                last_row = trend_rows[trend_rows["YEAR"] == latest_year].iloc[0]
                
                ref_p10, ref_p50, ref_p90 = float(ref_row["P10"]), float(ref_row["P50"]), float(ref_row["P90"])
                last_p10, last_p50, last_p90 = float(last_row["P10"]), float(last_row["P50"]), float(last_row["P90"])
            else:
                ref_subset  = df_s2[df_s2["YEAR"] == reference_year]
                last_subset = df_s2[df_s2["YEAR"] == latest_year]
                
                ref_percs  = weighted_percentiles_pv(ref_subset,  story_subject, [10, 50, 90])
                last_percs = weighted_percentiles_pv(last_subset, story_subject, [10, 50, 90])
                
                ref_p10, ref_p50, ref_p90 = ref_percs[0], ref_percs[1], ref_percs[2]
                last_p10, last_p50, last_p90 = last_percs[0], last_percs[1], last_percs[2]

            if not (np.isnan(ref_p50) or np.isnan(last_p50)):
                # Round everything FIRST to sync the math with the visual display
                ref_p10_r, ref_p50_r, ref_p90_r = round(ref_p10), round(ref_p50), round(ref_p90)
                last_p10_r, last_p50_r, last_p90_r = round(last_p10), round(last_p50), round(last_p90)

                # Calculate deltas using the rounded integers
                delta_p10 = last_p10_r - ref_p10_r
                delta_p50 = last_p50_r - ref_p50_r
                delta_p90 = last_p90_r - ref_p90_r
                
                if delta_p50 > 0:
                    direction = "increased"
                elif delta_p50 < 0:
                    direction = "decreased"
                else:
                    direction = "remained unchanged"

                # Initialize the findings list
                findings = []

                # Use the pre-rounded variables (ref_p50_r, last_p50_r) in the text string!
                findings.append(
                    f"At the median, {_cnt_label(story_country)}'s "
                    f"{subject_label} score {direction} by "
                    f"{abs(delta_p50)} points between "
                    f"{reference_year} and {latest_year} "
                    f"({ref_p50_r} → {last_p50_r})."
                )

                # Conditional finding - uneven decline/gain across distribution
                spread = abs(delta_p10 - delta_p90)
                if spread > 10:
                    # Round the deltas first to avoid -0.1 triggering a "decline" but displaying as "-0"
                    d10_r = round(delta_p10)
                    d90_r = round(delta_p90)
                    
                    # Figure out which end performed "worse" relative to the other
                    if d10_r < d90_r:
                        lagging, lag_val = "lower end (P10)", d10_r
                        leading, lead_val = "upper end (P90)", d90_r
                    else:
                        lagging, lag_val = "upper end (P90)", d90_r
                        leading, lead_val = "lower end (P10)", d10_r

                    # Dynamic phrasing based on whether rounded values are positive, negative, or flat
                    if lag_val < 0 and lead_val > 0:
                        comparison = f"scores at the {lagging} declined ({lag_val:+.0f} points) while the {leading} saw gains ({lead_val:+.0f} points)"
                    elif lag_val < 0 and lead_val == 0:
                        comparison = f"scores at the {lagging} declined ({lag_val:+.0f} points) while the {leading} remained flat (0 points)"
                    elif lag_val == 0 and lead_val > 0:
                        comparison = f"scores at the {lagging} remained flat (0 points) while the {leading} saw gains ({lead_val:+.0f} points)"
                    elif lag_val < 0 and lead_val < 0:
                        comparison = f"scores at the {lagging} dropped more ({lag_val:+.0f} points) than at the {leading} ({lead_val:+.0f} points)"
                    else:
                        # Both are strictly > 0
                        comparison = f"scores at the {leading} grew more ({lead_val:+.0f} points) than at the {lagging} ({lag_val:+.0f} points)"

                    findings.append(
                        f"The change is not uniform across the distribution — {comparison}."
                    )
                
                # Render the combined insight box
                _add_story_finding(findings)

            fig2 = plot_percentile_change_from_baseline_precomputed(
                df_pre=df_pre, subject=story_subject,
                cnt=story_country, reference_year=reference_year
            ) if df_pre is not None else plot_percentile_change_from_baseline(
                df=df_s2, subject=story_subject,
                cnt=story_country, reference_year=reference_year
            )
            _chart_expander(
                "Collapse chart", fig2,
                GROUP_HOW_TO_READ.get("Change over time", "Each bar shows the score distribution for one group.")
            )

            _policy_box(
                "A decline that is steeper at the lower end of the distribution "
                "than at the upper end suggests that students who most need "
                "support have been disproportionately impacted. Recovery efforts "
                "should be targeted, not uniform."
            )

    st.divider()

    # ── Chapter 3: Equity ──────────────────────────────────────────────────
    _story_section_header(
        3,
        "How do resources and systemic support shape outcomes?",
        f"Understanding structural differences by student background in {_cnt_label(story_country)}"
    )

    st.markdown(
        "High average scores can mask large differences between student groups. "
        "The figures below show the difference in distribution and median scores between groups — "
        "they reflect structural inequalities in access to resources and support, "
        "not differences in student ability."
    )

    # Fetch all three equity datasets
    # Fetch equity datasets — skip if precomputed available
    if df_pre is None:
        df_ses   = fetch(
            (story_country,), story_year,
            tuple(BASE_COLS + pv_cols + ["ESCS"])
        )
        df_immig = fetch(
            (story_country,), story_year,
            tuple(BASE_COLS + pv_cols + ["IMMIG"])
        )
        df_gender = fetch(
            (story_country,), story_year,
            tuple(BASE_COLS + pv_cols + ["ST004D01T"])
        )
    else:
        df_ses = df_immig = df_gender = None

    # ── Compute all three gaps ─────────────────────────────────────────────
    equity_gaps = []

    if df_pre is not None:
        # Read gaps directly from precomputed file
        ses_rows   = df_pre[(df_pre["CNT"] == story_country) & (df_pre["YEAR"] == story_year) & (df_pre["SUBJECT"] == story_subject) & (df_pre["GROUP_TYPE"] == "ses")]
        immig_rows = df_pre[(df_pre["CNT"] == story_country) & (df_pre["YEAR"] == story_year) & (df_pre["SUBJECT"] == story_subject) & (df_pre["GROUP_TYPE"] == "immigration")]
        gender_rows = df_pre[(df_pre["CNT"] == story_country) & (df_pre["YEAR"] == story_year) & (df_pre["SUBJECT"] == story_subject) & (df_pre["GROUP_TYPE"] == "gender")]

        # SES gap
        q1_row = ses_rows[ses_rows["GROUP_LABEL"] == "Q1 (lowest)"]
        q4_row = ses_rows[ses_rows["GROUP_LABEL"] == "Q4 (highest)"]
        if not q1_row.empty and not q4_row.empty:
            q1_med = float(q1_row.iloc[0]["P50"])
            q4_med = float(q4_row.iloc[0]["P50"])
            equity_gaps.append({
                "label": "Socioeconomic background",
                "value": q4_med - q1_med,
                "sub":   "Highest vs lowest SES quartile",
                "type":  "ses"
            })

        # Immigration gap
        immig_medians = {}
        for lbl in immig_rows["GROUP_LABEL"].unique():
            row = immig_rows[immig_rows["GROUP_LABEL"] == lbl]
            if not row.empty and not np.isnan(float(row.iloc[0]["P50"])):
                immig_medians[lbl] = float(row.iloc[0]["P50"])

        if len(immig_medians) >= 2:
            highest_lbl = max(immig_medians, key=immig_medians.get)
            lowest_lbl  = min(immig_medians, key=immig_medians.get)
            
            clean_labels = {
                "Native": "Native",
                "1st-gen immigrant": "first-generation",
                "2nd-gen immigrant": "second-generation",
                "First-generation": "first-generation",
                "Second-generation": "second-generation"
            }
            
            high_clean = clean_labels.get(highest_lbl, highest_lbl.lower())
            low_clean  = clean_labels.get(lowest_lbl, lowest_lbl.lower())
            
            equity_gaps.append({
                "label": "Immigration status",
                # Round the individual medians BEFORE subtracting
                "value": round(immig_medians[highest_lbl]) - round(immig_medians[lowest_lbl]),
                "sub":   f"{high_clean.capitalize()} vs {low_clean} students",
                "type":  "immig"
            })

        # Gender gap
        male_row   = gender_rows[gender_rows["GROUP_LABEL"] == "Male"]
        female_row = gender_rows[gender_rows["GROUP_LABEL"] == "Female"]
        if not male_row.empty and not female_row.empty:
            male_med   = float(male_row.iloc[0]["P50"])
            female_med = float(female_row.iloc[0]["P50"])
            equity_gaps.append({
                "label": "Gender",
                "value": abs(male_med - female_med),
                "sub":   "Boys vs girls at the median",
                "type":  "gender"
            })

    else:
        # SES gap from raw data
        if not check_missing_countries(df_ses, ["ESCS"], [story_country], story_year):
            curves = compute_escs_quartile_percentiles(
                df_ses, story_subject, [50], cnt=story_country, year=story_year
            )
            q1_med = curves.get("Q1 (low SES)",  [np.nan])[0]
            q4_med = curves.get("Q4 (high SES)", [np.nan])[0]
            if not (np.isnan(q1_med) or np.isnan(q4_med)):
                equity_gaps.append({
                    "label": "Socioeconomic background",
                    "value": q4_med - q1_med,
                    "sub":   "Highest vs lowest SES quartile",
                    "type":  "ses"
                })

        # Immigration gap from raw data
        if not check_missing_countries(df_immig, ["IMMIG"], [story_country], story_year):
            immig_medians = {}
            # Check all three available IMMIG codes
            for code, clean_name in [(1.0, "Native"), (2.0, "second-generation"), (3.0, "first-generation")]:
                subset = df_immig[
                    (df_immig["CNT"] == story_country) & (df_immig["IMMIG"] == code)
                ]
                med = weighted_percentiles_pv(subset, story_subject, [50])
                if not np.isnan(med).all():
                    immig_medians[clean_name] = med[0]

            if len(immig_medians) >= 2:
                highest_group = max(immig_medians, key=immig_medians.get)
                lowest_group  = min(immig_medians, key=immig_medians.get)
                
                equity_gaps.append({
                    "label": "Immigration status",
                    # Round the individual medians BEFORE subtracting
                    "value": round(immig_medians[highest_group]) - round(immig_medians[lowest_group]),
                    "sub":   f"{highest_group.capitalize()} vs {lowest_group} students",
                    "type":  "immig"
                })

        # Gender gap from raw data
        if not check_missing_countries(df_gender, ["ST004D01T"], [story_country], story_year):
            male_subset   = df_gender[
                (df_gender["CNT"] == story_country) & (df_gender["ST004D01T"] == 2.0)
            ]
            female_subset = df_gender[
                (df_gender["CNT"] == story_country) & (df_gender["ST004D01T"] == 1.0)
            ]
            male_med   = weighted_percentiles_pv(male_subset,   story_subject, [50])
            female_med = weighted_percentiles_pv(female_subset, story_subject, [50])
            if not (np.isnan(male_med).all() or np.isnan(female_med).all()):
                equity_gaps.append({
                    "label": "Gender",
                    "value": abs(male_med[0] - female_med[0]),
                    "sub":   "Boys vs girls at the median",
                    "type":  "gender"
                })

    # Sort by size — largest first
    equity_gaps.sort(key=lambda x: x["value"], reverse=True)

    # ── Summary cards ──────────────────────────────────────────────────────
    if equity_gaps:
        findings = []
        
        # Leading insight — biggest gap
        biggest = equity_gaps[0]
        findings.append(
            f"The largest difference in {_cnt_label(story_country)} is by "
            f"<strong>{biggest['label'].lower()}</strong> — "
            f"a {biggest['value']:.0f} point difference at the median "
            f"in {subject_label}. {biggest['sub']}."
        )

        # Conditional finding — SES diff > country vs OECD diff
        ses_gap_entry = next((g for g in equity_gaps if g["type"] == "ses"), None)
        if ses_gap_entry:
            country_vs_oecd = abs(cnt_mean_score - oecd_mean_score) if not (np.isnan(cnt_mean_score) or np.isnan(oecd_mean_score)) else np.nan
            if not np.isnan(country_vs_oecd) and ses_gap_entry["value"] > country_vs_oecd and country_vs_oecd > 5:
                findings.append(
                    f"The socioeconomic difference within "
                    f"{_cnt_label(story_country)} "
                    f"({ses_gap_entry['value']:.0f} points) is larger than "
                    f"the difference between this country and the OECD "
                    f"average ({country_vs_oecd:.0f} points). "
                    f"Domestic equity is a bigger lever than "
                    f"international benchmarking."
                )
        
        # Print combined findings
        _add_story_finding(findings)

        # Print metric cards side-by-side
        card_cols = st.columns(len(equity_gaps))
        for i, (col, gap) in enumerate(zip(card_cols, equity_gaps)):
            with col:
                label = gap["label"] + (" — largest" if i == 0 else "")
                st.metric(
                    label,
                    f"{gap['value']:.0f} pts",
                    help=gap["sub"]
                )

        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

        # ── Combined chart expander ────────────────────────────────────────
        if df_pre is not None:
            ses_ok    = not ses_rows.empty
            immig_ok  = not immig_rows.empty
            gender_ok = not gender_rows.empty
        else:
            ses_ok    = not check_missing_countries(df_ses,    ["ESCS"],       [story_country], story_year)
            immig_ok  = not check_missing_countries(df_immig,  ["IMMIG"],      [story_country], story_year)
            gender_ok = not check_missing_countries(df_gender, ["ST004D01T"],  [story_country], story_year)

        # SES chart
        if ses_ok:
            st.markdown("**By socioeconomic background**")
            group_col_ses, group_labels_ses = GROUP_OPTIONS["Socioeconomic status"]
            fig3a = plot_group_shaded_density_precomputed(
                df_pre=df_pre, subject=story_subject, cnt=story_country,
                group_type="ses", group_title="Socioeconomic Status", year=story_year
            ) if df_pre is not None else plot_group_shaded_density(
                df_ses, story_subject, story_country,
                group_col=group_col_ses, group_labels=group_labels_ses,
                group_title="Socioeconomic Status", year=story_year,
                sort_by_median=False
            )
            _chart_expander(
                "Collapse chart", fig3a,
                GROUP_HOW_TO_READ.get("Socioeconomic status", "Each bar shows the score distribution for one group.")
            )
        else:
            st.warning(f"⚠️ Insufficient socioeconomic data for {_cnt_label(story_country)}.")

        # Immigration chart
        if immig_ok:
            st.markdown("**By immigration status**")
            if df_pre is None:
                immig_warnings = check_group_sizes(
                    df_immig, "IMMIG", IMMIG_MAP, story_country, year=story_year
                )
                for w in immig_warnings:
                    st.warning(w)
            group_col_immig, group_labels_immig = GROUP_OPTIONS["Immigration status"]
            fig3b = plot_group_shaded_density_precomputed(
                df_pre=df_pre, subject=story_subject, cnt=story_country,
                group_type="immigration", group_title="Immigration Status", year=story_year
            ) if df_pre is not None else plot_group_shaded_density(
                df_immig, story_subject, story_country,
                group_col=group_col_immig, group_labels=group_labels_immig,
                group_title="Immigration Status", year=story_year,
                sort_by_median=True
            )
            _chart_expander(
                "Collapse chart", fig3b,
                GROUP_HOW_TO_READ.get("Immigration status", "Each bar shows the score distribution for one group.")
            )
        else:
            st.warning(f"⚠️ Insufficient immigration data for {_cnt_label(story_country)}.")

        # Gender chart
        if gender_ok:
            st.markdown("**By gender**")
            group_col_gen, group_labels_gen = GROUP_OPTIONS["Gender"]
            fig3c = plot_group_shaded_density_precomputed(
                df_pre=df_pre, subject=story_subject, cnt=story_country,
                group_type="gender", group_title="Gender", year=story_year
            ) if df_pre is not None else plot_group_shaded_density(
                df_gender, story_subject, story_country,
                group_col=group_col_gen, group_labels=group_labels_gen,
                group_title="Gender", year=story_year,
                sort_by_median=True
            )
            _chart_expander(
                "Collapse chart", fig3c,
                GROUP_HOW_TO_READ.get("Gender", "Each bar shows the score distribution for one group.")
            )
        else:
            st.warning(f"⚠️ Insufficient gender data for {_cnt_label(story_country)}.")

        _policy_box(
            "Score differences by student background point to structural "
            "inequalities in how resources, language support, and school "
            "quality are distributed. The largest difference for this "
            "country is the most actionable starting point for policy."
        )

    else:
        st.warning(
            f"⚠️ **Data unavailable:** Insufficient data to compute equity differences for {_cnt_label(story_country)} in {story_year}.\n\n{data_hint}"
        )

    st.divider()

    # ── Chapter 4: School context ──────────────────────────────────────────
    _story_section_header(
        4,
        "Does school context matter?",
        f"How school location and type relate to {subject_label} scores "
        f"in {_cnt_label(story_country)}"
    )

    if df_pre is None:
        df_loc  = fetch(
            (story_country,), story_year,
            tuple(BASE_COLS + pv_cols + ["SC001Q01TA"])
        )
        df_type = fetch(
            (story_country,), story_year,
            tuple(BASE_COLS + pv_cols + ["SCHLTYPE"])
        )
    else:
        df_loc = df_type = None

    # ── Public vs private ─────────────────────────────────────────────────
    st.markdown("#### Public vs private")

    if df_pre is None and check_missing_countries(
        df_type, ["SCHLTYPE"], [story_country], story_year
    ):
        st.warning(
            f"⚠️ Insufficient school type data for "
            f"{_cnt_label(story_country)}."
        )
    else:
        # School type key finding:
        # Compare the highest-median school type group with the lowest-median group.
        if df_pre is not None:
            type_rows = df_pre[
                (df_pre["CNT"] == story_country)
                & (df_pre["YEAR"] == story_year)
                & (df_pre["SUBJECT"] == story_subject)
                & (df_pre["GROUP_TYPE"] == "school_type")
            ].copy()

            type_rows = type_rows.dropna(subset=["P50"])

            type_medians = {
                row["GROUP_LABEL"]: float(row["P50"])
                for _, row in type_rows.iterrows()
            }

        else:
            group_col_type, group_labels_type = GROUP_OPTIONS["School type"]

            type_curves = compute_group_percentiles(
                df_type,
                story_subject,
                group_col_type,
                group_labels_type,
                [50],
                cnt=story_country,
                year=story_year,
            )

            type_medians = {
                label: vals[0]
                for label, vals in type_curves.items()
                if vals is not None and len(vals) > 0 and not np.isnan(vals[0])
            }

        if len(type_medians) >= 2:
            highest_group = max(type_medians, key=type_medians.get)
            lowest_group = min(type_medians, key=type_medians.get)

            highest_med = type_medians[highest_group]
            lowest_med = type_medians[lowest_group]
            highest_med_display = int(round(highest_med))
            lowest_med_display = int(round(lowest_med))
            diff_display = highest_med_display - lowest_med_display

            _add_story_finding(
                f"The largest difference in school-type scores in {_cnt_label(story_country)} is between "
                f"{highest_group.lower()} and {lowest_group.lower()} schools: "
                f"students in {highest_group.lower()} schools score "
                f"{diff_display:.0f} points higher at the median in {subject_label} "
                f"in {_cnt_label(story_country)}."
            )
        

        group_col, group_labels = GROUP_OPTIONS["School type"]
        fig4a = plot_group_shaded_density_precomputed(
            df_pre=df_pre, subject=story_subject, cnt=story_country,
            group_type="school_type", group_title="School Type", year=story_year
        ) if df_pre is not None else plot_group_shaded_density(
            df=df_type, subject=story_subject, cnt=story_country,
            group_col=group_col, group_labels=group_labels,
            group_title="School Type", year=story_year,
            sort_by_median=True
        )
        if not fig4a.data:
            st.warning(f"⚠️ **Data unavailable:** School location comparisons are unavailable for {_cnt_label(story_country)} in {story_year}.\n\n{data_hint}")
        else:
            _chart_expander(
                "Collapse chart",
                fig4a,
                GROUP_HOW_TO_READ.get("School type", "Each bar shows the score distribution for one group.")
            )

        _policy_box(
            "Differences between public and private school scores "
            "often reflect the socioeconomic composition of each "
            "school type rather than school quality itself. "
            "Interpreting these figures alongside the SES data "
            "in Chapter 3 is important."
        )

    # ── Urban vs rural ────────────────────────────────────────────────────
    st.markdown("#### Urban vs rural")

    if df_pre is None and check_missing_countries(
        df_loc, ["SC001Q01TA"], [story_country], story_year
    ):
        st.warning(
            f"⚠️ Insufficient school location data for "
            f"{_cnt_label(story_country)}."
        )
    else:


        # School location key finding:
        # Compare the highest-median location group with the lowest-median location group.
        if df_pre is not None:
            loc_rows = df_pre[
                (df_pre["CNT"] == story_country)
                & (df_pre["YEAR"] == story_year)
                & (df_pre["SUBJECT"] == story_subject)
                & (df_pre["GROUP_TYPE"] == "school_loc")
            ].copy()

            loc_rows = loc_rows.dropna(subset=["P50"])

            loc_medians = {
                row["GROUP_LABEL"]: float(row["P50"])
                for _, row in loc_rows.iterrows()
            }

        else:
            group_col, group_labels = GROUP_OPTIONS["School location"]

            loc_curves = compute_group_percentiles(
                df_loc,
                story_subject,
                group_col,
                group_labels,
                [50],
                cnt=story_country,
                year=story_year,
            )

            loc_medians = {
                label: vals[0]
                for label, vals in loc_curves.items()
                if vals is not None and len(vals) > 0 and not np.isnan(vals[0])
            }

        if len(loc_medians) >= 2:
            highest_group = max(loc_medians, key=loc_medians.get)
            lowest_group = min(loc_medians, key=loc_medians.get)

            highest_med = loc_medians[highest_group]
            lowest_med = loc_medians[lowest_group]
            highest_med_display = int(round(highest_med))
            lowest_med_display = int(round(lowest_med))
            diff_display = highest_med_display - lowest_med_display

            _add_story_finding(
                f"The largest difference in school-location scores in {_cnt_label(story_country)} is between "
                f"{highest_group.lower()} and {lowest_group.lower()} schools: "
                f"students in {highest_group.lower()} schools score "
                f"{diff_display:.0f} points higher at the median in {subject_label}. "
            )

        group_col, group_labels = GROUP_OPTIONS["School location"]
        fig4b = plot_group_shaded_density_precomputed(
            df_pre=df_pre, subject=story_subject, cnt=story_country,
            group_type="school_loc", group_title="School Location", year=story_year
        ) if df_pre is not None else plot_group_shaded_density(
            df=df_loc, subject=story_subject, cnt=story_country,
            group_col=group_col, group_labels=group_labels,
            group_title="School Location", year=story_year,
            sort_by_median=True
        )

        if not fig4b.data:
            st.warning(f"⚠️ **Data unavailable:** School type comparisons are unavailable for {_cnt_label(story_country)} in {story_year}.\n\n{data_hint}")
        else:
            _chart_expander(
                "Collapse chart",
                fig4b,
                GROUP_HOW_TO_READ.get("School location", "Each bar shows the score distribution for one group.")
            )

        _policy_box(
            "Urban-rural score differences in PISA typically reflect "
            "resource allocation across school systems — teacher "
            "quality, infrastructure, and support services. "
            "These are addressable through targeted policy."
        )

    st.divider()
    
    # ── Summary of key findings ────────────────────────────────────────────

    st.markdown("## Summary of key findings")
    st.markdown(
        f"Main takeaways for {_cnt_label(story_country)} in {subject_label}."
    )

    if story_findings:
        summary_items = "".join(
            f"<li>{finding}</li>"
            for finding in story_findings
        )

        st.markdown(
            f"""
            <div style="
                background: #e8f4fd;
                border: 1px solid #b3d9f5;
                border-radius: 8px;
                padding: 16px 20px;
                margin: 10px 0 18px 0;
                color: #1a3a52;
                line-height: 1.55;
            ">
                <strong>Summary of key findings:</strong>
                <ul style="margin-top: 8px; margin-bottom: 0; padding-left: 22px;">
                    {summary_items}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning(f"⚠️ **Data unavailable:** No key findings can be generated for {_cnt_label(story_country)} in {story_year}.")

    # ── Footer ─────────────────────────────────────────────────────────────
    st.markdown("""
        <div style="background:#f0f4f8; border-radius:8px; padding:20px 24px;
                    margin-top:16px; text-align:center;">
            <p style="font-size:1.05rem; color:#333; margin-bottom:8px;">
                <strong>Want to dig deeper?</strong>
            </p>
            <p style="font-size:0.9rem; color:#555;">
                Switch to the <strong>🔍 Explore</strong> tab to choose any
                country, subject, year, and chart type.
            </p>
        </div>
    """, unsafe_allow_html=True)

# Load data and derive country lists
meta   = get_meta()
df_pre = get_precomputed()

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

# st.sidebar.info(
#     "**Interactive Dashboard:** Hover over charts for exact values, "
#     "click legend items to hide/show groups, and drag to zoom."
# )

st.sidebar.markdown("---")

# ==========================================
# MODE 1: DATA STORY
# ==========================================
if app_mode == "📖 Data Story":
    st.sidebar.header("📖 Story Controls")

    default_idx = all_countries.index("CAN") if "CAN" in all_countries else 0

    story_country = st.sidebar.selectbox(
        "Focus country", 
        all_countries, 
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
    render_story_tab(available_years, story_country, story_subject, df_pre=df_pre)


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

    country_pool = all_countries

    DEFAULT_COUNTRIES = ["CAN", "USA"]
    SINGLE_COUNTRY_CHARTS = ["Score change over time", "Group comparison", "Intersectional Heatmap"]

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
        selected_year = None
    else:
        selected_year = st.sidebar.selectbox(
            "Year", available_years, 
            index=available_years.index(2022) if 2022 in available_years else len(available_years) - 1
        )

    ref_year = None
    comp_year = None

    primary_country = selected_countries[0]


    # Chart-Specific Controls injected directly into the main view
    group_key = None
    group_key_left = None
    group_key_right = None

    # ── Draw the main area for Explore ───────────────────────────────────────
    if side_by_side:

        # Special case:
        # When comparing two score-change-over-time charts, use the same y-axis
        # so the two countries are visually comparable.
        if (
            chart_type_left == "Score change over time"
            and chart_type_right == "Score change over time"
        ):
            reference_year = min(available_years)

            # Build both figures first
            if df_pre is not None:
                fig_left = plot_percentile_change_from_baseline_precomputed(
                    df_pre=df_pre,
                    subject=subject,
                    cnt=country_left,
                    reference_year=reference_year,
                )

                fig_right = plot_percentile_change_from_baseline_precomputed(
                    df_pre=df_pre,
                    subject=subject,
                    cnt=country_right,
                    reference_year=reference_year,
                )

            else:
                df_time = fetch(
                    tuple(set([country_left, country_right])),
                    None,
                    tuple(BASE_COLS + PV_BY_SUBJ[subject])
                )

                fig_left = plot_percentile_change_from_baseline(
                    df=df_time,
                    subject=subject,
                    cnt=country_left,
                    reference_year=reference_year,
                )

                fig_right = plot_percentile_change_from_baseline(
                    df=df_time,
                    subject=subject,
                    cnt=country_right,
                    reference_year=reference_year,
                )

            # Compute shared y-axis range from both figures
            y_values = []

            for fig in [fig_left, fig_right]:
                for trace in fig.data:
                    if hasattr(trace, "y") and trace.y is not None:
                        vals = pd.to_numeric(
                            pd.Series(trace.y),
                            errors="coerce"
                        ).dropna()
                        y_values.extend(vals.tolist())

            if y_values:
                y_min = min(min(y_values), 0)
                y_max = max(max(y_values), 0)

                # Add padding so the lines do not touch the plot boundary
                pad = max(3, 0.08 * (y_max - y_min))

                shared_y_range = [y_min - pad, y_max + pad]

                fig_left.update_yaxes(range=shared_y_range)
                fig_right.update_yaxes(range=shared_y_range)

            # Apply compact layout after setting shared axis
            fig_left = apply_compact_plotly_layout(fig_left)
            fig_right = apply_compact_plotly_layout(fig_right)

            left_col, right_col = st.columns(2)

            with left_col:
                st.subheader(f"{chart_type_left}: {_cnt_label(country_left)}")
                render_plotly_chart_with_note(
                    fig_left,
                    note=get_chart_note(
                        "Score change over time",
                        reference_year=reference_year,
                    ),
                    key=f"time_change_left_{subject}_{reference_year}_{country_left}",
                )

            with right_col:
                st.subheader(f"{chart_type_right}: {_cnt_label(country_right)}")
                render_plotly_chart_with_note(
                    fig_right,
                    note=get_chart_note(
                        "Score change over time",
                        reference_year=reference_year,
                    ),
                    key=f"time_change_right_{subject}_{reference_year}_{country_right}",
                )

        # All other side-by-side combinations keep the existing logic
        else:
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
