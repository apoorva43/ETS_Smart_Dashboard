"""
Plotting utilities for the PISA dashboard using Plotly.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.config import (
    PERCENTILES_COARSE,
    PERCENTILES_FINE,
    COUNTRY_COLORS,
    PALETTE,
    YEAR_COLORS,
    LOC_MAP,
    IMMIG_MAP,
    SCHLTYPE_MAP,
    SYMBOLS_COARSE,
    OKABE_ITO,
    SUBJECTS,
    COUNTRY_NAMES
)
from src.pisa_stats import (
    weighted_percentiles_pv,
    compute_escs_quartile_percentiles,
    compute_group_percentiles,
    get_oecd_percentiles,
    weighted_mean_pv
)

def _cnt_label(code: str) -> str:
    """
    Return full country name for a CNT code, falling back to the code itself.
    """
    return COUNTRY_NAMES.get(str(code), str(code))

def _base_layout(title: str = "", height: int = 480) -> dict:
    """Shared Plotly layout settings applied to all dashboard charts."""
    return dict(
        title=title,
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", size=12),
        xaxis=dict(
            gridcolor="rgba(128, 128, 128, 0.2)",
            linecolor="rgba(128, 128, 128, 0.4)",
            zeroline=False
        ),
        yaxis=dict(
            gridcolor="rgba(128, 128, 128, 0.2)",
            linecolor="rgba(128, 128, 128, 0.4)",
            zeroline=False
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5
        ),
        hovermode="x unified",
        margin=dict(t=85, b=85, l=50, r=20),
    )


def _check_sufficient_data(df, target_cols, cnt, min_n=100, msg="Insufficient data"):
    """
    Helper function to validate data sufficiency.
    Returns an empty Plotly figure with a warning message if data is insufficient.
    """
    cols_to_check = list(set(target_cols + ["W_FSTUWT"]))
    valid_data = df.dropna(subset=cols_to_check)
    
    if len(valid_data) < min_n:
        fig = go.Figure()
        fig.add_annotation(
            text=f"⚠️ Limited data available for this breakdown.<br>Results suppressed to ensure statistical reliability.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig.update_layout(**_base_layout(height=400))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return None, fig
        
    return valid_data, None

# keep color consistent
def _country_color(cnt: str, active_countries: list) -> str:
    """
    Assign a color based on the country's position in the dashboard's active selection.
    If a user picks ["CAN", "USA"], CAN is always PALETTE[0] and USA is always PALETTE[1]
    across EVERY chart.
    """
    if not active_countries:
        return OKABE_ITO.get("blue", "#0072B2")

    # If the country is in the selected list, use its index to pick the color
    if cnt in active_countries:
        index = active_countries.index(cnt)
        return PALETTE[index % len(PALETTE)]
    
    # If a chart plots a baseline/reference country not in the list, default it
    return OKABE_ITO.get("blue", "#0072B2")

# percentile score profile
def plot_country_distributions(df, subject: str,
                               countries: list,
                               year: int = None,
                               show_oecd: bool = True,
                               primary_country: str = None) -> go.Figure:
    fig = go.Figure()

    for i, cnt in enumerate(countries):
        subset = df[df["CNT"] == cnt]
        if year is not None and "YEAR" in df.columns:
            subset = subset[subset["YEAR"] == year]
            
        percs = weighted_percentiles_pv(subset, subject, PERCENTILES_COARSE)
        if np.isnan(percs).all():
            continue
            
        color = _country_color(cnt, countries)
        
        line_width = 3 if cnt == primary_country else 2
        marker_size = 10 if cnt == primary_country else 8
        
        fig.add_trace(go.Scatter(
            x=PERCENTILES_COARSE, y=percs,
            mode="lines+markers", 
            name=_cnt_label(cnt),
            line=dict(color=color, width=line_width),
            marker=dict(symbol=SYMBOLS_COARSE, size=marker_size, line=dict(color="white", width=1)),
            hovertemplate=f"<b>{_cnt_label(cnt)}</b><br>{SUBJECTS[subject]}: %{{y:.0f}}<extra></extra>"
        ))

    if show_oecd:
        oecd = get_oecd_percentiles(df, subject, PERCENTILES_COARSE, year)
        if not np.isnan(oecd).all():
            fig.add_trace(go.Scatter(
                x=PERCENTILES_COARSE, y=oecd,
                mode="lines+markers", name="OECD avg",
                line=dict(color="#777777", width=2, dash="dash"),
                marker=dict(symbol=SYMBOLS_COARSE, size=9),
                hovertemplate=f"<b>OECD avg</b><br>{SUBJECTS[subject]}: %{{y:.0f}}<extra></extra>"
            ))

    symbol_note = "Percentile symbols: ▼ 10th   ■ 25th   ◆ 50th   ● 75th   ▲ 90th"
    fig.update_layout(**_base_layout(title=f"Percentile Score Profile | {SUBJECTS[subject]}<br><sup>{symbol_note}</sup>"))
    fig.update_xaxes(title="Percentile", tickvals=PERCENTILES_COARSE)
    fig.update_yaxes(title=f"{SUBJECTS[subject]} score")
    return fig

# group - ses
def plot_escs_gap(df, subject, cnt, year=None):
    subset = df[df["CNT"] == cnt].copy()
    if year is not None:
        subset = subset[subset["YEAR"] == year]
        
    valid_data, error_fig = _check_sufficient_data(
        subset, ["ESCS"], cnt, msg=f"Insufficient ESCS data for {cnt}"
    )
    if error_fig is not None:
        return error_fig

    curves = compute_escs_quartile_percentiles(valid_data, subject, PERCENTILES_COARSE, cnt=cnt, year=year)
    fig = go.Figure()

    for color, (label, percs) in zip(PALETTE, curves.items()):
        if np.isnan(percs).all():
            continue
        fig.add_trace(go.Scatter(
            x=PERCENTILES_COARSE, y=percs,
            mode="lines+markers", name=label,
            line=dict(color=color, width=2.5),
            marker=dict(symbol=SYMBOLS_COARSE, size=9, line=dict(color="white", width=1)),
            hovertemplate=f"<b>{label}</b><br>Percentile: %{{x}}<br>{SUBJECTS[subject]}: %{{y:.0f}}<extra></extra>"
        ))

    symbol_note = "Percentile symbols: ▼ 10th   ■ 25th   ◆ 50th   ● 75th   ▲ 90th"
    fig.update_layout(**_base_layout(title=f"Score by Socioeconomic Status | {SUBJECTS[subject]} | {_cnt_label(cnt)}<br><sup>{symbol_note}</sup>"))
    fig.update_xaxes(title="Percentile", tickvals=PERCENTILES_COARSE)
    fig.update_yaxes(title=f"{SUBJECTS[subject]} score")
    return fig


def plot_group_comparison(df, subject: str, group_col: str,
                           group_vals: dict, cnt: str,
                           year: int = None, title: str = "") -> go.Figure:
    subset = df.copy()
    if cnt is not None:
        subset = subset[subset["CNT"] == cnt]
    if year is not None:
        subset = subset[subset["YEAR"] == year]

    valid_data, error_fig = _check_sufficient_data(
        subset, [group_col], cnt, msg=f"Insufficient data to group by {group_col} for {cnt}"
    )
    if error_fig is not None:
        return error_fig

    curves = compute_group_percentiles(valid_data, subject, group_col, group_vals, PERCENTILES_COARSE, cnt=cnt, year=year)
    fig = go.Figure()

    for color, (label, percs) in zip(PALETTE, curves.items()):
        if np.isnan(percs).all():
            continue
        fig.add_trace(go.Scatter(
            x=PERCENTILES_COARSE, y=percs,
            mode="lines+markers", name=label,
            line=dict(color=color, width=2.5),
            marker=dict(symbol=SYMBOLS_COARSE, size=9, line=dict(color="white", width=1))
        ))

    symbol_note = "Percentile symbols: ▼ 10th   ■ 25th   ◆ 50th   ● 75th   ▲ 90th"
    clean_title = title or f"{SUBJECTS[subject]} by Group | {cnt}"
    
    fig.update_layout(**_base_layout(title=f"{clean_title}<br><sup>{symbol_note}</sup>"))
    fig.update_xaxes(title="Percentile", tickvals=PERCENTILES_COARSE)
    fig.update_yaxes(title=f"{SUBJECTS[subject]} score")
    return fig


def plot_naep_time_comparison(df, subject: str, cnt: str,
                               reference_year: int, comparison_years: list,
                               group_col: str = None, group_val: float = None,
                               group_label: str = "All students") -> go.Figure:
    fig = go.Figure()

    def get_subset(year):
        s = df[(df["CNT"] == cnt) & (df["YEAR"] == year)]
        if group_col and group_val is not None:
            s = s[s[group_col] == group_val]
        return s

    ref_percs = weighted_percentiles_pv(get_subset(reference_year), subject, PERCENTILES_COARSE)
    if np.isnan(ref_percs).all():
        return fig

    fig.add_trace(go.Scatter(
        x=ref_percs, y=ref_percs,
        mode="lines+markers", 
        name=f"{reference_year} baseline (no change)",
        line=dict(color="#777777", width=2.5, dash="solid"),
        marker=dict(symbol=SYMBOLS_COARSE, size=9),
        customdata=PERCENTILES_COARSE,
        hovertemplate=f"<b>{reference_year} Baseline</b><br>Percentile: %{{customdata}}<br>Score: %{{y:.0f}}<extra></extra>"
    ))

    for year in comparison_years:
        yr_percs = weighted_percentiles_pv(get_subset(year), subject, PERCENTILES_COARSE)
        if np.isnan(yr_percs).all():
            continue
        
        diff = yr_percs - ref_percs
        
        customdata = np.column_stack([PERCENTILES_COARSE, [f"{d:+.0f}" for d in diff]])
        
        fig.add_trace(go.Scatter(
            x=ref_percs, y=yr_percs,
            mode="lines+markers", 
            name=str(year),
            line=dict(
                color=YEAR_COLORS.get(year, "#185FA5"), 
                width=2.5,
                dash="dash"
            ),
            marker=dict(symbol=SYMBOLS_COARSE, size=9), 
            customdata=customdata,
            hovertemplate=f"<b>{year}</b><br>Percentile: %{{customdata[0]}}<br>Score: %{{y:.0f}}<br>Change from baseline: %{{customdata[1]}}<extra></extra>"
        ))

    fig.update_layout(**_base_layout(
        title=f"Score Change Over Time | {SUBJECTS[subject]} | {_cnt_label(cnt)} | {group_label}<br><sup>(above diagonal = improvement)</sup>"
    ))
    
    fig.update_xaxes(title=f"{reference_year} {SUBJECTS[subject]} score (reference)", hoverformat=".0f")
    fig.update_yaxes(title="Score")
    
    return fig


# change over time
def plot_year_diff_percentile(df, subject: str, cnt: str,
                              reference_year: int,
                              comparison_years) -> go.Figure:
    """
    Plot score change across percentiles relative to a baseline year.

    X-axis = baseline-year percentile score.
    Y-axis = comparison-year score minus baseline-year score.

    If multiple comparison years are provided, one line is drawn per
    comparison year. Percentile symbols are fixed:
    triangle-down 10th, square 25th, diamond 50th, circle 75th, triangle-up 90th.

    Parameters
    ----------
    df : pandas.DataFrame
        PISA dataset containing CNT, YEAR, weights, and plausible value columns.
    subject : str
        Subject code: "MATH", "READ", or "SCIE".
    cnt : str
        Country code.
    reference_year : int
        Baseline year shown on the x-axis.
    comparison_years : int or list[int]
        Year(s) to compare against the baseline year.

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure.
    """
    if isinstance(comparison_years, int):
        comparison_years = [comparison_years]

    comparison_years = [y for y in comparison_years if y != reference_year]

    fig = go.Figure()

    subset = df[df["CNT"] == cnt].copy()

    ref_subset = subset[subset["YEAR"] == reference_year]
    ref_percs = weighted_percentiles_pv(
        ref_subset, subject, PERCENTILES_COARSE)

    if np.isnan(ref_percs).all():
        return _check_sufficient_data(
            pd.DataFrame(), [], cnt,
            msg=f"Insufficient data for baseline year {reference_year}"
        )[1]

    marker_map = {
        10: "triangle-down",
        25: "square",
        50: "diamond",
        75: "circle",
        90: "triangle-up",
    }

    symbol_note = "Percentile symbols: ▼ 10th   ■ 25th   ◆ 50th   ● 75th   ▲ 90th"

    # zero reference line
    fig.add_hline(
        y=0,
        line_dash="solid",
        line_color=OKABE_ITO["vermillion"],
        line_width=1.4
    )

    for i, comp_year in enumerate(comparison_years):
        comp_subset = subset[subset["YEAR"] == comp_year]
        comp_percs = weighted_percentiles_pv(
            comp_subset, subject, PERCENTILES_COARSE)

        if np.isnan(comp_percs).all():
            continue

        delta = comp_percs - ref_percs
        color = YEAR_COLORS.get(comp_year, PALETTE[i % len(PALETTE)])

        # line trace
        fig.add_trace(go.Scatter(
            x=ref_percs,
            y=delta,
            mode="lines",
            name=str(comp_year),
            legendgroup=str(comp_year),
            line=dict(color=color, width=2.5),
            hoverinfo="skip",
        ))

        # one marker per percentile so symbols can differ
        for p, x0, comp0, d0 in zip(PERCENTILES_COARSE, ref_percs, comp_percs, delta):
            fig.add_trace(go.Scatter(
                x=[x0],
                y=[d0],
                mode="markers",
                name=str(comp_year),
                legendgroup=str(comp_year),
                showlegend=False,
                marker=dict(
                    symbol=marker_map[p],
                    size=11,
                    color=color,
                    line=dict(color=OKABE_ITO["black"], width=0.8),
                ),
                customdata=[[
                    p,
                    f"{x0:.0f}",
                    f"{comp0:.0f}",
                    f"{d0:+.0f}"
                ]],
                hovertemplate=(
                    f"Year: {comp_year}<br>"
                    "Percentile: %{customdata[0]}<br>"
                    f"{comp_year} score: %{{customdata[2]:.0f}}<br>"
                    "Change: %{customdata[3]:+.0f}<extra></extra>"
                ), 
            ))
    
    fig.update_layout(**_base_layout(
        title=(
            f"{SUBJECTS[subject]} score change by percentile | {_cnt_label(cnt)}<br>"
            f"<sup>{symbol_note}</sup>"
        )
    ))

    fig.update_xaxes(
        title=f"{reference_year} score (baseline)",
        hoverformat=".0f",
        zeroline=False
    )

    fig.update_yaxes(
        title=f"Change in {SUBJECTS[subject]} score (comparison − {reference_year})",
        zeroline=False
    )

    return fig

# score distribution
def plot_weighted_interval_distribution(df, subject: str, countries: list,
                                        year: int = None, interval_width: int = 20,
                                        score_range: tuple = (0, 1000),
                                        show_oecd: bool = True) -> go.Figure:
    pv_cols = [f"PV{i}{subject}" for i in range(1, 11) if f"PV{i}{subject}" in df.columns]
    bins = np.arange(score_range[0], score_range[1] + interval_width, interval_width)
    midpoints = (bins[:-1] + bins[1:]) / 2

    fig = go.Figure()

    def _country_proportions(subset):
        w = subset["W_FSTUWT"].values
        total_w = w.sum()
        if total_w == 0: return np.full(len(midpoints), np.nan)
        
        pv_props = []
        for pv in pv_cols:
            scores = subset[pv].values
            props = np.array([w[(scores >= bins[i]) & (scores < bins[i + 1])].sum() / total_w for i in range(len(bins) - 1)])
            pv_props.append(props)
        return np.mean(pv_props, axis=0)

    for i, cnt in enumerate(countries):
        subset = df[df["CNT"] == cnt].dropna(subset=pv_cols + ["W_FSTUWT"])
        if year is not None and "YEAR" in df.columns:
            subset = subset[subset["YEAR"] == year]
        
        props = _country_proportions(subset)
        color = _country_color(cnt, countries)

        fig.add_trace(go.Scatter(
            x=midpoints, y=props,
            mode="lines+markers", 
            name=_cnt_label(cnt),
            line=dict(color=color, width=2.5),
            hovertemplate=f"<b>{_cnt_label(cnt)}</b><br>Score: %{{x}}<br>Percentage: %{{y:.0%}}<extra></extra>"
        ))

    if show_oecd:
        all_subset = df.dropna(subset=pv_cols + ["W_FSTUWT"])
        if year is not None and "YEAR" in df.columns:
            all_subset = all_subset[all_subset["YEAR"] == year]
        oecd_props = _country_proportions(all_subset)
        if not np.isnan(oecd_props).all():
            fig.add_trace(go.Scatter(
                x=midpoints, y=oecd_props,
                mode="lines+markers", name="OECD avg",
                line=dict(color="#777777", width=2, dash="dash"),
                marker=dict(size=6),
                hovertemplate=f"OECD avg: %{{y:.1%}}<extra></extra>"
            ))

    fig.update_layout(**_base_layout(title=f"Score Distribution | {SUBJECTS[subject]}"))
    fig.update_xaxes(title="Score")
    fig.update_yaxes(
        title="Percentage of students",
        tickformat=".0%"
    )
    return fig


def plot_gender_diff_percentile(df, subject: str, cnt: str, year: int = None, active_countries: list = None) -> go.Figure:
    """
    Plot the gender score difference across the distribution using Plotly.

    The x-axis shows the female reference score at each percentile. The y-axis
    shows the score difference between male and female students at the matched
    percentile. Positive values indicate higher male scores, while negative
    values indicate higher female scores.

    This version uses neutral, accessible colors rather than gender-coded
    colors. The main line is black, with positive and negative differences
    shaded using Okabe-Ito blue and orange.

    Parameters
    ----------
    df : pandas.DataFrame
        PISA dataset containing gender identifiers, weights, plausible value
        score columns, and optionally a year column.
    subject : str
        Subject code used to select plausible value columns. Expected values
        include ``"MATH"``, ``"READ"``, and ``"SCIE"``.
    cnt : str
        Country code to filter the data, such as ``"CAN"`` or ``"USA"``.
    year : int, optional
        PISA cycle year to filter by. If ``None``, all available years are used.

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure showing gender score difference across percentiles.
    """
    subset = df[df["CNT"] == cnt].copy()

    if year is not None and "YEAR" in subset.columns:
        subset = subset[subset["YEAR"] == year]

    female = subset[subset["ST004D01T"] == 1.0]
    male = subset[subset["ST004D01T"] == 2.0]

    female_percs = weighted_percentiles_pv(
        female, subject, PERCENTILES_FINE
    )
    male_percs = weighted_percentiles_pv(
        male, subject, PERCENTILES_FINE
    )

    fig = go.Figure()

    if np.isnan(female_percs).all() or np.isnan(male_percs).all():
        return _check_sufficient_data(
            pd.DataFrame(), [], cnt, msg="Insufficient data"
        )[1]

    diff = male_percs - female_percs

    if np.isnan(diff).all():
        return _check_sufficient_data(pd.DataFrame(), [], cnt, msg="Insufficient data")[1]

    x_new, diff_new = [], []
    for i in range(len(female_percs)):
        # If the line crosses zero between the previous point and this point
        if i > 0 and diff[i-1] * diff[i] < 0:
            slope = (diff[i] - diff[i-1]) / (female_percs[i] - female_percs[i-1])
            x_zero = female_percs[i-1] - (diff[i-1] / slope)
            x_new.append(x_zero)
            diff_new.append(0.0) # Inject the exact zero point
        
        x_new.append(female_percs[i])
        diff_new.append(diff[i])
    
    x_interp = np.array(x_new)
    diff_interp = np.array(diff_new)

    positive_diff = np.where(diff_interp >= 0, diff_interp, 0)
    negative_diff = np.where(diff_interp < 0, diff_interp, 0)

    # Accessible, non-gender-coded colors
    positive_fill = "rgba(0, 114, 178, 0.22)"   # Okabe-Ito blue
    negative_fill = "rgba(230, 159, 0, 0.22)"   # Okabe-Ito orange
    main_line = _country_color(cnt, active_countries) if active_countries else OKABE_ITO.get("vermillion", "#D85A30")

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#999999",
        line_width=1.5,
        annotation_text="No difference",
        annotation_position="top left",
    )

    fig.add_trace(go.Scatter(
        x=x_interp,
        y=positive_diff,
        mode="lines",
        line=dict(width=0),
        fill="tozeroy",
        fillcolor=positive_fill,
        name="Positive difference",
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=x_interp,
        y=negative_diff,
        mode="lines",
        line=dict(width=0),
        fill="tozeroy",
        fillcolor=negative_fill,
        name="Negative difference",
        hoverinfo="skip",
    ))

    customdata = np.column_stack([
        PERCENTILES_FINE,
        np.round(female_percs).astype(int),
        np.round(male_percs).astype(int),
        [f"{d:+.0f}" for d in diff],
    ])


    fig.add_trace(go.Scatter(
        x=female_percs,
        y=diff,
        mode="lines+markers",
        name="Male - Female",
        line=dict(
            color=main_line,
            width=2.5,
        ),
        marker=dict(
            size=5,
            color=main_line,
        ),
        customdata=customdata,
        hovertemplate=(
            "Percentile: %{customdata[0]}<br>"
            "Female score: %{customdata[1]:.0f}<br>"
            "Male score: %{customdata[2]:.0f}<br>"
            "Male - Female difference: %{customdata[3]:+.0f}<extra></extra>"
        ),
    ))

    fig.update_layout(**_base_layout(
        title=(
            f"Scores by Gender | {SUBJECTS[subject]} | {_cnt_label(cnt)}<br>"
        )
    ))

    fig.update_xaxes(
        title=f"Female {SUBJECTS[subject]} score (reference)",
        hoverformat=".0f"
    )

    fig.update_yaxes(
        title="Score difference (Male - Female)",
        zeroline=False,
    )

    return fig

def plot_intersectional_heatmap(df, subject, cnt, row_var="ESCS", col_var="BELONG",
                                 row_label="SES Quartile", col_label="Belonging Quartile",
                                 year=None, n_bins=4, min_cell_n=30):
    pv_cols = [f"PV{i}{subject}" for i in range(1, 11) if f"PV{i}{subject}" in df.columns]
    subset = df[df["CNT"] == cnt].copy()
    if year is not None and "YEAR" in subset.columns:
        subset = subset[subset["YEAR"] == year]

    subset = subset.dropna(subset=[row_var, col_var, "W_FSTUWT"] + pv_cols)

    if len(subset) < min_cell_n * n_bins:
        fig = go.Figure()
        fig.add_annotation(text="⚠️ Insufficient data for intersectional breakdown.",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14, color="gray"))
        fig.update_layout(**_base_layout(height=400))
        return fig

    subset["row_bin"] = pd.qcut(subset[row_var].rank(method="first"), q=n_bins,
                                 labels=[f"{row_label} Q{i+1}" for i in range(n_bins)])
    subset["col_bin"] = pd.qcut(subset[col_var].rank(method="first"), q=n_bins,
                                 labels=[f"{col_label} Q{i+1}" for i in range(n_bins)])

    row_cats = [f"{row_label} Q{i+1}" for i in range(n_bins)]
    col_cats = [f"{col_label} Q{i+1}" for i in range(n_bins)]

    z = np.full((n_bins, n_bins), np.nan)
    text = [[""] * n_bins for _ in range(n_bins)]

    for r_idx, r_cat in enumerate(row_cats):
        for c_idx, c_cat in enumerate(col_cats):
            cell = subset[(subset["row_bin"] == r_cat) & (subset["col_bin"] == c_cat)]
            if len(cell) < min_cell_n:
                text[r_idx][c_idx] = "n/a"
                continue
            pv_means = [np.average(cell[pv].values, weights=cell["W_FSTUWT"].values)
                        for pv in pv_cols if pv in cell.columns]
            if pv_means:
                z[r_idx][c_idx] = np.mean(pv_means)
                text[r_idx][c_idx] = f"{z[r_idx][c_idx]:.0f}"

    fig = go.Figure(go.Heatmap(
        z=z,
        x=col_cats,
        y=row_cats,
        text=text,
        texttemplate="%{text}",
        colorscale="Blues",
        colorbar=dict(title=f"Mean {SUBJECTS[subject]} score"),
        hovertemplate=(
            f"{row_label}: %{{y}}<br>"
            f"{col_label}: %{{x}}<br>"
            f"Mean score: %{{z:.0f}}<extra></extra>"
        )
    ))

    fig.update_layout(**_base_layout(
        title=f"Mean {SUBJECTS[subject]} Score | {row_label} × {col_label} | {_cnt_label(cnt)}",
        height=480
    ))
    return fig

def plot_belonging_by_immigration(df, countries: list, year: int = None,
                                min_group_n: int = 30) -> go.Figure:
                                
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Grade repetition by SES quartile", 
        "School belonging by immigration status"
    ))

    if not countries:
        return _check_sufficient_data(pd.DataFrame(), [], "", msg="Select a country")[1]

    subset = df[df["CNT"].isin(countries)].copy()
    if year is not None and "YEAR" in subset.columns:
        subset = subset[subset["YEAR"] == year]

    def _weighted_mean(values, weights):
        valid = np.isfinite(values) & np.isfinite(weights)
        return np.average(values[valid], weights=weights[valid]) if valid.sum() > 0 else np.nan

    color_map = {}
    for i, cnt in enumerate(countries):
        color_map[cnt] = _country_color(cnt, countries)

    # Panel 1: Repetition
    if all(c in subset.columns for c in ["REPEAT", "ESCS", "CNT", "W_FSTUWT"]):
        df_rep = subset.dropna(subset=["REPEAT", "ESCS", "W_FSTUWT"]).copy()
        
        if len(df_rep) >= 4:
            df_rep["ESCS_Q"] = pd.qcut(df_rep["ESCS"].rank(method="first"), q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
            
            for cnt in countries:
                rates = []
                for q in ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]:
                    sub = df_rep[(df_rep["ESCS_Q"] == q) & (df_rep["CNT"] == cnt)]
                    if len(sub) < min_group_n:
                        rates.append(np.nan)
                    else:
                        rates.append(_weighted_mean((sub["REPEAT"] == 1).to_numpy(), sub["W_FSTUWT"].to_numpy()) * 100)
                
                fig.add_trace(go.Bar(
                    x=["Q1", "Q2", "Q3", "Q4"], y=rates, name=_cnt_label(cnt), 
                    legendgroup=_cnt_label(cnt),
                    marker_color=color_map[cnt],
                    texttemplate="%{y:.0f}%", textposition="outside",
                    hoverinfo="skip"
                ), row=1, col=1)

    # Panel 2: Belonging
    if all(c in subset.columns for c in ["BELONG", "IMMIG", "CNT", "W_FSTUWT"]):
        df_bel = subset.dropna(subset=["BELONG", "IMMIG", "W_FSTUWT"]).copy()
        
        for cnt in countries:
            means = []
            for code in IMMIG_MAP:
                sub = df_bel[(df_bel["IMMIG"] == code) & (df_bel["CNT"] == cnt)]
                means.append(_weighted_mean(sub["BELONG"].to_numpy(), sub["W_FSTUWT"].to_numpy()) if len(sub) >= min_group_n else np.nan)
                
            fig.add_trace(go.Bar(
                x=list(IMMIG_MAP.values()), y=means, name=_cnt_label(cnt), 
                legendgroup=_cnt_label(cnt),
                marker_color=color_map[cnt],
                showlegend=False, 
                texttemplate="%{y:.2f}", textposition="outside",
                hoverinfo="skip"
            ), row=1, col=2)

    fig.update_layout(**_base_layout(title="Student Context"))
    fig.update_layout(barmode='group')
    fig.update_yaxes(
        title_text="Grade repetition rate (%)",
        row=1,
        col=1
    )
    fig.update_yaxes(
        title_text="Mean Sense of Belonging index",
        row=1,
        col=2
    )
    fig.update_xaxes(
        title_text="SES quartile",
        row=1,
        col=1
    )
    fig.update_xaxes(
        title_text="Immigration status",
        row=1,
        col=2
    )
    return fig

def plot_country_shaded_density(df, subject, countries, year, min_group_n=30):
    from scipy.stats import gaussian_kde

    pv_cols = [f"PV{i}{subject}" for i in range(1, 11) if f"PV{i}{subject}" in df.columns]
    subset = df[df["CNT"].isin(countries)].copy()
    if year is not None and "YEAR" in subset.columns:
        subset = subset[subset["YEAR"] == year]

    subset = subset.dropna(subset=["W_FSTUWT"] + pv_cols)
    if len(subset) < min_group_n:
        fig = go.Figure()
        fig.add_annotation(text="⚠️ Insufficient data.", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="gray"))
        return fig

    quartiles = countries
    x_grid = np.linspace(100, 900, 500)

    # Percentile bands to shade: (low_p, high_p, opacity)
    BANDS = [
        (0,  10,  0.10),
        (10, 25,  0.20),
        (25, 75,  0.45),   # IQR — darkest
        (75, 90,  0.20),
        (90, 100, 0.10),
    ]

    fig = go.Figure()

    for row_idx, cnt_code in enumerate(countries):
        group = subset[subset["CNT"] == cnt_code]
        q_label = _cnt_label(cnt_code)
        color = PALETTE[row_idx % len(PALETTE)]
        if len(group) < min_group_n:
            continue

        if color.startswith("#") and len(color) == 7:
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        elif color.startswith("rgb"):
            # Strip out letters and brackets to just get the numbers
            clean_rgb = color.replace("rgba", "").replace("rgb", "").replace("(", "").replace(")", "")
            parts = clean_rgb.split(",")
            r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
        else:
            r, g, b = 128, 128, 128 # Safe fallback gray

        # Average KDE across all 10 plausible values
        kde_vals = []
        for pv in pv_cols:
            scores = group[pv].dropna().values
            weights = group.loc[group[pv].notna(), "W_FSTUWT"].values
            if len(scores) < 10:
                continue
            try:
                kde = gaussian_kde(scores, weights=weights, bw_method="scott")
                kde_vals.append(kde(x_grid))
            except Exception:
                continue

        if not kde_vals:
            continue

        density = np.mean(kde_vals, axis=0)
        density /= density.max()   # normalise to 1 so all rows same height

        # Compute weighted percentile scores for band boundaries
        all_scores = np.concatenate([group[pv].values for pv in pv_cols])
        all_weights = np.tile(group["W_FSTUWT"].values, len(pv_cols))
        sort_idx = np.argsort(all_scores)
        sorted_scores = all_scores[sort_idx]
        sorted_weights = all_weights[sort_idx]
        cumw = np.cumsum(sorted_weights) / sorted_weights.sum()

        def score_at_p(p):
            idx = np.searchsorted(cumw, p / 100)
            return sorted_scores[min(idx, len(sorted_scores) - 1)]

        # Row position: each quartile is a horizontal band on y
        y_center = len(quartiles) - row_idx   # Q4 on top, Q1 on bottom
        bar_height = 0.7

        # Draw shaded bands from darkest (IQR) to lightest (tails)
        for (lo_p, hi_p, alpha) in BANDS:
            x_lo = score_at_p(lo_p)
            x_hi = score_at_p(hi_p)

            # Clip density shape to this band's score range
            mask = (x_grid >= x_lo) & (x_grid <= x_hi)
            if mask.sum() < 2:
                continue

            band_x = np.concatenate([[x_lo], x_grid[mask], [x_hi]])
            band_density = np.concatenate([[0], density[mask], [0]])

            # Scale density shape to bar_height, centred on y_center
            scaled_y_top = y_center + (band_density / 2) * bar_height
            scaled_y_bot = y_center - (band_density / 2) * bar_height

            # Build closed polygon path for the filled shape
            poly_x = np.concatenate([band_x, band_x[::-1]])
            poly_y = np.concatenate([scaled_y_top, scaled_y_bot[::-1]])

            fill_color = f"rgba({r},{g},{b},{alpha})"
            line_color = f"rgba({r},{g},{b},0)"  # no border between bands

            fig.add_trace(go.Scatter(
                x=poly_x, y=poly_y,
                fill="toself",
                fillcolor=fill_color,
                line=dict(color=line_color, width=0),
                mode="lines",
                showlegend=False,
                hoverinfo="skip"
            ))

        # Median line
        med = score_at_p(50)
        fig.add_trace(go.Scatter(
            x=[med, med],
            y=[y_center - bar_height / 2, y_center + bar_height / 2],
            mode="lines",
            line=dict(color=f"rgb({r},{g},{b})", width=2.5),
            showlegend=False,
            hovertemplate=f"<b>{q_label}</b><br>Median: {med:.0f}<extra></extra>"
        ))

        # Dense invisible hover points across the full row
        hover_x = np.linspace(score_at_p(2), score_at_p(98), 200)
        hover_y = np.full(200, y_center)
        fig.add_trace(go.Scatter(
            x=hover_x,
            y=hover_y,
            mode="markers",
            marker=dict(color="rgba(0,0,0,0)", size=8),
            name=q_label,
            showlegend=True,
            hovertemplate=(
                f"<b>{q_label}</b><br>"
                f"P10: {round(score_at_p(10))}<br>"
                f"P25: {round(score_at_p(25))}<br>"
                f"Median: {round(score_at_p(50))}<br>"
                f"P75: {round(score_at_p(75))}<br>"
                f"P90: {round(score_at_p(90))}<extra></extra>"
            )
        ))

        # Visible markers at P10, P25, P50, P75, P90
        marker_ps = [10, 25, 50, 75, 90]
        marker_xs = [round(score_at_p(p)) for p in marker_ps]
        marker_sizes = [6, 6, 10, 6, 6]  # median slightly bigger

        fig.add_trace(go.Scatter(
            x=marker_xs,
            y=[y_center] * len(marker_xs),
            mode="markers",
            marker=dict(
                color=f"rgb({r},{g},{b})",
                size=marker_sizes,
                symbol="line-ns",           # vertical tick mark
                line=dict(color=f"rgb({r},{g},{b})", width=2)
            ),
            showlegend=False,
            hoverinfo="skip"
        ))

    fig.update_layout(**_base_layout(
        title=f"Score Distribution | {SUBJECTS[subject]}"
    ))
    fig.update_xaxes(title=f"{SUBJECTS[subject]} score", range=[100, 900])
    fig.update_yaxes(
        tickvals=list(range(1, len(quartiles) + 1)),
        ticktext=list(reversed(quartiles)),
        showgrid=False,
        zeroline=False,
        range=[0.3, len(quartiles) + 0.7]
    )
    return fig

def plot_escs_shaded_density(df, subject, cnt, year=None, min_group_n=30):
    from scipy.stats import gaussian_kde

    pv_cols = [f"PV{i}{subject}" for i in range(1, 11) if f"PV{i}{subject}" in df.columns]
    subset = df[df["CNT"] == cnt].copy()
    if year is not None and "YEAR" in subset.columns:
        subset = subset[subset["YEAR"] == year]

    subset = subset.dropna(subset=["ESCS", "W_FSTUWT"] + pv_cols)
    if len(subset) < min_group_n:
        fig = go.Figure()
        fig.add_annotation(text="⚠️ Insufficient data.", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="gray"))
        return fig

    subset["ESCS_Q"] = pd.qcut(
        subset["ESCS"].rank(method="first"), q=4,
        labels=["Q1 (lowest SES)", "Q2", "Q3", "Q4 (highest SES)"]
    )
    quartiles = ["Q1 (lowest SES)", "Q2", "Q3", "Q4 (highest SES)"]
    x_grid = np.linspace(100, 900, 500)

    # Percentile bands to shade: (low_p, high_p, opacity)
    BANDS = [
        (0,  10,  0.10),
        (10, 25,  0.20),
        (25, 75,  0.45),   # IQR — darkest
        (75, 90,  0.20),
        (90, 100, 0.10),
    ]

    fig = go.Figure()

    for row_idx, (q_label, color) in enumerate(zip(quartiles, PALETTE)):
        group = subset[subset["ESCS_Q"] == q_label]
        if len(group) < min_group_n:
            continue

        if color.startswith("#") and len(color) == 7:
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        elif color.startswith("rgb"):
            # Strip out letters and brackets to just get the numbers
            clean_rgb = color.replace("rgba", "").replace("rgb", "").replace("(", "").replace(")", "")
            parts = clean_rgb.split(",")
            r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
        else:
            r, g, b = 128, 128, 128 # Safe fallback gray

        # Average KDE across all 10 plausible values
        kde_vals = []
        for pv in pv_cols:
            scores = group[pv].dropna().values
            weights = group.loc[group[pv].notna(), "W_FSTUWT"].values
            if len(scores) < 10:
                continue
            try:
                kde = gaussian_kde(scores, weights=weights, bw_method="scott")
                kde_vals.append(kde(x_grid))
            except Exception:
                continue

        if not kde_vals:
            continue

        density = np.mean(kde_vals, axis=0)
        density /= density.max()   # normalise to 1 so all rows same height

        # Compute weighted percentile scores for band boundaries
        all_scores = np.concatenate([group[pv].values for pv in pv_cols])
        all_weights = np.tile(group["W_FSTUWT"].values, len(pv_cols))
        sort_idx = np.argsort(all_scores)
        sorted_scores = all_scores[sort_idx]
        sorted_weights = all_weights[sort_idx]
        cumw = np.cumsum(sorted_weights) / sorted_weights.sum()

        def score_at_p(p):
            idx = np.searchsorted(cumw, p / 100)
            return sorted_scores[min(idx, len(sorted_scores) - 1)]

        # Row position: each quartile is a horizontal band on y
        y_center = len(quartiles) - row_idx   # Q4 on top, Q1 on bottom
        bar_height = 0.7

        # Draw shaded bands from darkest (IQR) to lightest (tails)
        for (lo_p, hi_p, alpha) in BANDS:
            x_lo = score_at_p(lo_p)
            x_hi = score_at_p(hi_p)

            # Clip density shape to this band's score range
            mask = (x_grid >= x_lo) & (x_grid <= x_hi)
            if mask.sum() < 2:
                continue

            band_x = np.concatenate([[x_lo], x_grid[mask], [x_hi]])
            band_density = np.concatenate([[0], density[mask], [0]])

            # Scale density shape to bar_height, centred on y_center
            scaled_y_top = y_center + (band_density / 2) * bar_height
            scaled_y_bot = y_center - (band_density / 2) * bar_height

            # Build closed polygon path for the filled shape
            poly_x = np.concatenate([band_x, band_x[::-1]])
            poly_y = np.concatenate([scaled_y_top, scaled_y_bot[::-1]])

            fill_color = f"rgba({r},{g},{b},{alpha})"
            line_color = f"rgba({r},{g},{b},0)"  # no border between bands

            fig.add_trace(go.Scatter(
                x=poly_x, y=poly_y,
                fill="toself",
                fillcolor=fill_color,
                line=dict(color=line_color, width=0),
                mode="lines",
                showlegend=False,
                hoverinfo="skip"
            ))

        # Median line
        med = score_at_p(50)
        fig.add_trace(go.Scatter(
            x=[med, med],
            y=[y_center - bar_height / 2, y_center + bar_height / 2],
            mode="lines",
            line=dict(color=f"rgb({r},{g},{b})", width=2.5),
            showlegend=False,
            hovertemplate=f"<b>{q_label}</b><br>Median: {med:.0f}<extra></extra>"
        ))

        # Invisible wide hover bar for the whole row
        fig.add_trace(go.Scatter(
            x=[score_at_p(5), score_at_p(95)],
            y=[y_center, y_center],
            mode="lines",
            line=dict(color="rgba(0,0,0,0)", width=bar_height * 40),
            name=q_label,
            showlegend=True,
            marker=dict(color=f"rgb({r},{g},{b})"),
            hovertemplate=(
                f"<b>{q_label}</b><br>"
                f"P10: {score_at_p(10):.0f} | "
                f"Median: {med:.0f} | "
                f"P90: {score_at_p(90):.0f}"
                "<extra></extra>"
            )
        ))

    fig.update_layout(**_base_layout(
        title=f"Score Distribution by Socioeconomic Status | {SUBJECTS[subject]} | {_cnt_label(cnt)}"
    ))
    fig.update_xaxes(title=f"{SUBJECTS[subject]} score", range=[100, 900])
    fig.update_yaxes(
        tickvals=list(range(1, len(quartiles) + 1)),
        ticktext=list(reversed(quartiles)),
        showgrid=False,
        zeroline=False,
        range=[0.3, len(quartiles) + 0.7]
    )
    return fig

def plot_percentile_change_from_baseline(df, subject, cnt, reference_year=2015):
    subset = df[df["CNT"] == cnt].copy()
    years = sorted(subset["YEAR"].dropna().unique())

    ref_subset = subset[subset["YEAR"] == reference_year]
    ref_percs = weighted_percentiles_pv(ref_subset, subject, PERCENTILES_COARSE)

    if np.isnan(ref_percs).all():
        return _check_sufficient_data(pd.DataFrame(), [], cnt,
            msg=f"No data for baseline year {reference_year}")[1]

    percentile_labels = {
        0: "10th percentile", 1: "25th percentile",
        2: "50th (median)", 3: "75th percentile", 4: "90th percentile"
    }

    fig = go.Figure()
    fig.add_hline(y=0, line_dash="solid", line_color=OKABE_ITO["vermillion"],
                  line_width=1.5, annotation_text=f"{reference_year} baseline",
                  annotation_position="top left")

    for p_idx, (p_val, p_label) in enumerate(zip(PERCENTILES_COARSE, percentile_labels.values())):
        color = PALETTE[p_idx % len(PALETTE)]
        x_years, y_deltas = [], []

        for yr in years:
            yr_subset = subset[subset["YEAR"] == yr]
            yr_percs = weighted_percentiles_pv(yr_subset, subject, PERCENTILES_COARSE)
            if np.isnan(yr_percs).all():
                continue
            x_years.append(yr)
            y_deltas.append(float(yr_percs[p_idx] - ref_percs[p_idx]))

        if not x_years:
            continue

        fig.add_trace(go.Scatter(
            x=x_years, y=y_deltas,
            mode="lines+markers",
            name=p_label,
            line=dict(color=color, width=2.5),
            marker=dict(size=9, color=color, line=dict(color="white", width=1)),
            customdata=[[yr, f"{d:+.0f}"] for yr, d in zip(x_years, y_deltas)],
            hovertemplate=(
                f"<b>{p_label}</b><br>"
                "Year: %{customdata[0]}<br>"
                "Change from baseline: %{customdata[1]}<extra></extra>"
            )
        ))

    fig.update_layout(**_base_layout(
        title=f"{SUBJECTS[subject]} score change by percentile | {_cnt_label(cnt)}<br>"
              f"<sup>Relative to {reference_year} baseline</sup>"
    ))
    fig.update_xaxes(title="Year", tickvals=years, tickformat="d")
    fig.update_yaxes(title=f"Score change from {reference_year}")
    return fig

def plot_immigration_kde(df, subject, cnt, year=None, min_group_n=30):
    from scipy.stats import gaussian_kde

    pv_cols = [f"PV{i}{subject}" for i in range(1, 11) if f"PV{i}{subject}" in df.columns]
    subset = df[df["CNT"] == cnt].copy()
    if year is not None and "YEAR" in subset.columns:
        subset = subset[subset["YEAR"] == year]

    x_grid = np.linspace(100, 900, 400)
    fig = go.Figure()

    for color, (code, label) in zip(PALETTE, IMMIG_MAP.items()):
        group = subset[subset["IMMIG"] == code].dropna(subset=pv_cols + ["W_FSTUWT"])
        if len(group) < min_group_n:
            continue

        kde_vals = []
        for pv in pv_cols:
            scores = group[pv].values
            weights = group["W_FSTUWT"].values
            try:
                kde = gaussian_kde(scores, weights=weights, bw_method="scott") # change scott to 0.4 if not smooth
                kde_vals.append(kde(x_grid))
            except Exception:
                continue

        if not kde_vals:
            continue

        density = np.mean(kde_vals, axis=0)

        med_val = weighted_percentiles_pv(group, subject, [50])
        med_label = f" (Median: {med_val[0]:.0f})" if not np.isnan(med_val).all() else ""

        if not np.isnan(med_val).all():
            fig.add_vline(x=med_val[0], line_width=1.5, line_dash="dot",
                          line_color=color, opacity=0.7)

        fig.add_trace(go.Scatter(
            x=x_grid, y=density,
            mode="lines", name=f"{label}{med_label}",
            line=dict(color=color, width=2.5),
            hovertemplate=f"<b>{label}</b><br>Score: %{{x:.0f}}<extra></extra>"
        ))

    fig.update_layout(**_base_layout(
        title=f"Score Distribution by Immigration Status | {SUBJECTS[subject]} | {_cnt_label(cnt)}"
    ), showlegend=True)
    fig.update_xaxes(title="Score")
    fig.update_yaxes(title="Density", showticklabels=False, showgrid=False)
    return fig


def plot_immigration_score_distribution(df, subject: str, cnt: str, year: int = None,
                                        interval_width: int = 20, score_range: tuple = (0, 1000),
                                        min_group_n: int = 30) -> go.Figure:
    pv_cols = [f"PV{i}{subject}" for i in range(1, 11) if f"PV{i}{subject}" in df.columns]
    bins = np.arange(score_range[0], score_range[1] + interval_width, interval_width)
    midpoints = (bins[:-1] + bins[1:]) / 2

    subset = df[df["CNT"] == cnt].copy()
    if year is not None and "YEAR" in subset.columns:
        subset = subset[subset["YEAR"] == year]

    fig = go.Figure()

    for color, (code, label) in zip(PALETTE, IMMIG_MAP.items()):
        group_df = subset[subset["IMMIG"] == code].dropna(subset=pv_cols + ["W_FSTUWT"])
        if len(group_df) < min_group_n: 
            continue

        w = group_df["W_FSTUWT"].values
        total_w = w.sum()
        if total_w == 0: continue

        # Calculate the curve proportions
        pv_props = []
        for pv in pv_cols:
            scores = group_df[pv].values
            pv_props.append(np.array([w[(scores >= bins[i]) & (scores < bins[i + 1])].sum() / total_w for i in range(len(bins) - 1)]))

        props = np.mean(pv_props, axis=0)

        # Calculate the exact median and update the legend label
        med_val = weighted_percentiles_pv(group_df, subject, [50])
        med_label = f" (Median: {med_val[0]:.0f})" if not np.isnan(med_val).all() else ""
        full_label = f"{label}{med_label}"

        # Dotted vertical line exactly at the median
        if not np.isnan(med_val).all():
            fig.add_vline(x=med_val[0], line_width=1.5, line_dash="dot", line_color=color, opacity=0.7)

        customdata = np.column_stack([
            [full_label] * len(midpoints),
            [f"{p:.0%}" for p in props]
        ])

        fig.add_trace(go.Scatter(
            x=midpoints, y=props,
            mode="lines", name=full_label,
            line=dict(color=color, width=2.5),
            customdata=customdata,
            hovertemplate=(
                "Status: %{customdata[0]}<br>"
                "Percentage: %{customdata[1]}<extra></extra>"
            ),
        ))

    fig.update_layout(**_base_layout(title=f"Score by Immigration Status | {SUBJECTS[subject]} | {_cnt_label(cnt)}"),
                      showlegend=True)
    fig.update_xaxes(title="Score")
    fig.update_yaxes(title="Percentage of Students", tickformat=".0%")
    return fig

def plot_school_location_boxplot(df, subject: str, cnt: str, year: int = None,
                                 location_col: str = "SC001Q01TA", min_group_n: int = 30) -> go.Figure:
    """
    Creates a 'Jitter Quantile Plot' using a weighted sample of students.
    Uses an invisible bar overlay to bypass Plotly's default box tooltips
    and deliver exact mathematical percentiles and demographic breakdowns.
    """
    subset = df[df["CNT"] == cnt].copy()
    if year is not None and "YEAR" in subset.columns:
        subset = subset[subset["YEAR"] == year]

    fig = go.Figure()
    pv_col = f"PV1{subject}"
    
    # Calculate total valid country weight FIRST to ensure accurate percentages
    valid_subset = subset.dropna(subset=[pv_col, "W_FSTUWT", location_col])
    total_w = valid_subset["W_FSTUWT"].sum()
    if total_w == 0:
        return fig

    for i, (code, label) in enumerate(LOC_MAP.items()):
        group = valid_subset[valid_subset[location_col] == code]
        if len(group) < min_group_n:
            continue

        # Calculate population percentage breakdown
        group_w = group["W_FSTUWT"].sum()
        pct = (group_w / total_w) * 100
        label_with_pct = f"{label}<br>({pct:.0f}%)" # Appends % directly to the X-axis label

        # Calculate true weighted percentiles mathematically
        percs = weighted_percentiles_pv(group, subject, [10, 25, 50, 75, 90])
        if np.isnan(percs).all():
            continue
        p10, p25, p50, p75, p90 = percs

        # Create clean tooltip string
        hover_text = (
            f"<b>{label}</b> ({pct:.0f}% of students)<br><br>"
            f"90th Percentile: {p90:.0f}<br>"
            f"75th Percentile: {p75:.0f}<br>"
            f"Median (50th): {p50:.0f}<br>"
            f"25th Percentile: {p25:.0f}<br>"
            f"10th Percentile: {p10:.0f}"
            f"<extra></extra>"
        )

        min_val = group[pv_col].min()
        max_val = group[pv_col].max()
        
        fig.add_trace(go.Bar(
            x=[label_with_pct],
            y=[max_val - min_val],
            base=[min_val],
            width=0.7,
            marker_color="rgba(0,0,0,0)",
            hovertext=[hover_text],
            hovertemplate="%{hovertext}",
            textposition="none",
            hoverlabel=dict(align="left"),
            showlegend=False
        ))

        sample_size = min(1000, len(group))
        sampled = group.sample(n=sample_size, weights="W_FSTUWT", random_state=42)

        fig.add_trace(go.Box(
            y=sampled[pv_col],
            x=[label_with_pct] * len(sampled),
            name=label,
            marker_color=PALETTE[i % len(PALETTE)],
            boxpoints='all',
            jitter=0.5,
            pointpos=0,
            fillcolor='rgba(0,0,0,0)',
            opacity=0.8,
            marker=dict(size=4, opacity=0.4, line=dict(width=0)),
            line=dict(width=2),
            hoverinfo="skip",
            showlegend=False
        ))

    fig.update_layout(
        **_base_layout(title=f"Score by School Location | {SUBJECTS[subject]} | {_cnt_label(cnt)}")
    )
    
    fig.update_layout(
        hovermode="closest", 
        barmode="overlay"
    )
    
    fig.update_yaxes(title=f"{SUBJECTS[subject]} score", hoverformat=".0f")

    return fig


def plot_school_type_distribution(df, subject: str, cnt: str, year: int = None,
                                  school_type_col: str = "SCHLTYPE", interval_width: int = 20,
                                  score_range: tuple = (0, 1000), min_group_n: int = 30) -> go.Figure:

    pv_cols = [f"PV{i}{subject}" for i in range(1, 11) if f"PV{i}{subject}" in df.columns]
    bins = np.arange(score_range[0], score_range[1] + interval_width, interval_width)
    midpoints = (bins[:-1] + bins[1:]) / 2

    subset = df[df["CNT"] == cnt].copy()
    if year is not None and "YEAR" in subset.columns:
        subset = subset[subset["YEAR"] == year]

    fig = go.Figure()

    for color, (code, label) in zip(PALETTE, SCHLTYPE_MAP.items()):
        group_df = subset[subset[school_type_col] == code].dropna(subset=pv_cols + ["W_FSTUWT"])
        if len(group_df) < min_group_n: continue

        w = group_df["W_FSTUWT"].values
        total_w = w.sum()
        if total_w == 0: continue

        pv_props = []
        for pv in pv_cols:
            scores = group_df[pv].values
            pv_props.append(np.array([w[(scores >= bins[i]) & (scores < bins[i + 1])].sum() / total_w for i in range(len(bins) - 1)]))

        props = np.mean(pv_props, axis=0)
        
        fig.add_trace(go.Scatter(
            x=midpoints, y=props,
            mode="lines+markers", name=label,
            line=dict(color=color, width=2.5),
            marker=dict(size=4),
            hovertemplate=(
                f"School type: {label}<br>"
                "Score: %{x}<br>"
                "Percentage: %{y:.0%}<extra></extra>"
            ),
        ))

    fig.update_layout(**_base_layout(title=f"Score by School Type | {SUBJECTS[subject]} | {_cnt_label(cnt)}"))
    fig.update_xaxes(title="Score")
    fig.update_yaxes(
        title="Percentage of students",
        tickformat=".0%"
    )
    return fig

def plot_resource_scatter(df, subject: str, resource_col: str,
                           resource_label: str, year: int = None,
                           highlight_countries: list = None) -> go.Figure:
    """
    Plotly Scatterplot: country-level resource variable vs mean score.
    """
    subset = df.copy()
    if year and "YEAR" in df.columns:
        subset = subset[subset["YEAR"] == year]

    rows = []
    for cnt in subset["CNT"].unique():
        cnt_data = subset[subset["CNT"] == cnt]
        mean_score    = weighted_mean_pv(cnt_data, subject)
        mean_resource = cnt_data[resource_col].mean()  
        oecd_flag     = cnt_data["OECD"].iloc[0] if "OECD" in cnt_data.columns else 0
        
        if not np.isnan(mean_score) and not np.isnan(mean_resource):
            rows.append({"CNT": cnt, 
                         "CNT_LABEL": _cnt_label(cnt),
                         "score": mean_score,
                         "resource": mean_resource, 
                         "OECD": oecd_flag})

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
         return _check_sufficient_data(pd.DataFrame(), [], "", msg="Insufficient data for scatter")[1]

    fig = go.Figure()

    # Separate data into groups to control legend and colors
    highlights = highlight_countries or []
    oecd = plot_df[~plot_df["CNT"].isin(highlights) & (plot_df["OECD"] == 1)]
    partner = plot_df[~plot_df["CNT"].isin(highlights) & (plot_df["OECD"] == 0)]
    hi = plot_df[plot_df["CNT"].isin(highlights)]

    # Standardize the hover tooltip
    htemp = (
        "<b>%{customdata[0]}</b><br>" + 
        f"{resource_label}: %{{x:.2f}}<br>" + 
        f"Mean {SUBJECTS[subject]}: %{{y:.0f}}<extra></extra>"
    )

    if not partner.empty:
        fig.add_trace(go.Scatter(
            x=partner["resource"], y=partner["score"],
            mode="markers", name="Partner countries",
            marker=dict(color="#cccccc", size=10, opacity=0.7),
            customdata=partner[["CNT_LABEL"]], hovertemplate=htemp
        ))

    if not oecd.empty:
        fig.add_trace(go.Scatter(
            x=oecd["resource"], y=oecd["score"],
            mode="markers", name="OECD members",
            marker=dict(color="#185FA5", size=10, opacity=0.8),
            customdata=oecd[["CNT_LABEL"]], hovertemplate=htemp
        ))

    if not hi.empty:
        # Highlighting adds a border and text label directly to the plot
        fig.add_trace(go.Scatter(
            x=hi["resource"], y=hi["score"],
            mode="markers+text", name="Selected",
            marker=dict(
                color=OKABE_ITO.get("vermillion", "#D85A30"), 
                size=14, 
                line=dict(width=2, color="white")
            ),
            text=hi["CNT_LABEL"], 
            textposition="top center", 
            textfont=dict(size=11, color=OKABE_ITO.get("vermillion", "#D85A30")),
            customdata=hi[["CNT_LABEL"]], hovertemplate=htemp
        ))

    # Quadrant label annotations — positioned at 10th/90th percentile corners of data
    x_lo = float(np.nanpercentile(plot_df["resource"], 12))
    x_hi = float(np.nanpercentile(plot_df["resource"], 88))
    y_lo = float(np.nanpercentile(plot_df["score"], 12))
    y_hi = float(np.nanpercentile(plot_df["score"], 88))

    quadrant_labels = [
        (x_hi, y_hi, "High Performance /<br>High " + resource_label.split(" ")[0]),
        (x_lo, y_hi, "High Performance /<br>Low " + resource_label.split(" ")[0]),
        (x_hi, y_lo, "Low Performance /<br>High " + resource_label.split(" ")[0]),
        (x_lo, y_lo, "Low Performance /<br>Low " + resource_label.split(" ")[0]),
    ]
    for qx, qy, qtext in quadrant_labels:
        fig.add_annotation(x=qx, y=qy, text=qtext, showarrow=False,
                           font=dict(size=10, color="#999999"),
                           align="center", xanchor="center", yanchor="middle")

    fig.update_layout(**_base_layout(
        title=f"{resource_label} vs {SUBJECTS[subject]} performance<br><sup>(each point = one country)</sup>"
    ))
    fig.update_xaxes(title=resource_label)
    fig.update_yaxes(title=f"Mean {SUBJECTS[subject]} score")
    
    fig.update_layout(hovermode="closest")

    return fig

def plot_immigration_shaded_density(df, subject, cnt, year=None, min_group_n=30):
    from scipy.stats import gaussian_kde

    pv_cols = [f"PV{i}{subject}" for i in range(1, 11) if f"PV{i}{subject}" in df.columns]
    subset = df[df["CNT"] == cnt].copy()
    if year is not None and "YEAR" in subset.columns:
        subset = subset[subset["YEAR"] == year]

    subset = subset.dropna(subset=["IMMIG", "W_FSTUWT"] + pv_cols)
    if len(subset) < min_group_n:
        fig = go.Figure()
        fig.add_annotation(text="⚠️ Insufficient data.", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="gray"))
        return fig

    quartiles = list(IMMIG_MAP.values())   # ["Native", "Second-generation", "First-generation"]
    x_grid = np.linspace(100, 900, 500)

    # Percentile bands to shade: (low_p, high_p, opacity)
    BANDS = [
        (0,  10,  0.10),
        (10, 25,  0.20),
        (25, 75,  0.45),   # IQR — darkest
        (75, 90,  0.20),
        (90, 100, 0.10),
    ]

    fig = go.Figure()

    for row_idx, (q_label, color) in enumerate(zip(quartiles, PALETTE)):
        code = list(IMMIG_MAP.keys())[row_idx]
        group = subset[subset["IMMIG"] == code]
        if len(group) < min_group_n:
            continue

        if color.startswith("#") and len(color) == 7:
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        elif color.startswith("rgb"):
            # Strip out letters and brackets to just get the numbers
            clean_rgb = color.replace("rgba", "").replace("rgb", "").replace("(", "").replace(")", "")
            parts = clean_rgb.split(",")
            r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
        else:
            r, g, b = 128, 128, 128 # Safe fallback gray

        # Average KDE across all 10 plausible values
        kde_vals = []
        for pv in pv_cols:
            scores = group[pv].dropna().values
            weights = group.loc[group[pv].notna(), "W_FSTUWT"].values
            if len(scores) < 10:
                continue
            try:
                kde = gaussian_kde(scores, weights=weights, bw_method="scott")
                kde_vals.append(kde(x_grid))
            except Exception:
                continue

        if not kde_vals:
            continue

        density = np.mean(kde_vals, axis=0)
        density /= density.max()   # normalise to 1 so all rows same height

        # Compute weighted percentile scores for band boundaries
        all_scores = np.concatenate([group[pv].values for pv in pv_cols])
        all_weights = np.tile(group["W_FSTUWT"].values, len(pv_cols))
        sort_idx = np.argsort(all_scores)
        sorted_scores = all_scores[sort_idx]
        sorted_weights = all_weights[sort_idx]
        cumw = np.cumsum(sorted_weights) / sorted_weights.sum()

        def score_at_p(p):
            idx = np.searchsorted(cumw, p / 100)
            return sorted_scores[min(idx, len(sorted_scores) - 1)]

        # Row position: each quartile is a horizontal band on y
        y_center = len(quartiles) - row_idx   # Q4 on top, Q1 on bottom
        bar_height = 0.7

        # Draw shaded bands from darkest (IQR) to lightest (tails)
        for (lo_p, hi_p, alpha) in BANDS:
            x_lo = score_at_p(lo_p)
            x_hi = score_at_p(hi_p)

            # Clip density shape to this band's score range
            mask = (x_grid >= x_lo) & (x_grid <= x_hi)
            if mask.sum() < 2:
                continue

            band_x = np.concatenate([[x_lo], x_grid[mask], [x_hi]])
            band_density = np.concatenate([[0], density[mask], [0]])

            # Scale density shape to bar_height, centred on y_center
            scaled_y_top = y_center + (band_density / 2) * bar_height
            scaled_y_bot = y_center - (band_density / 2) * bar_height

            # Build closed polygon path for the filled shape
            poly_x = np.concatenate([band_x, band_x[::-1]])
            poly_y = np.concatenate([scaled_y_top, scaled_y_bot[::-1]])

            fill_color = f"rgba({r},{g},{b},{alpha})"
            line_color = f"rgba({r},{g},{b},0)"  # no border between bands

            fig.add_trace(go.Scatter(
                x=poly_x, y=poly_y,
                fill="toself",
                fillcolor=fill_color,
                line=dict(color=line_color, width=0),
                mode="lines",
                showlegend=False,
                hoverinfo="skip"
            ))

        # Median line
        med = score_at_p(50)
        fig.add_trace(go.Scatter(
            x=[med, med],
            y=[y_center - bar_height / 2, y_center + bar_height / 2],
            mode="lines",
            line=dict(color=f"rgb({r},{g},{b})", width=2.5),
            showlegend=False,
            hovertemplate=f"<b>{q_label}</b><br>Median: {med:.0f}<extra></extra>"
        ))

        # Invisible wide hover bar for the whole row
        fig.add_trace(go.Scatter(
            x=[score_at_p(5), score_at_p(95)],
            y=[y_center, y_center],
            mode="lines",
            line=dict(color="rgba(0,0,0,0)", width=bar_height * 40),
            name=q_label,
            showlegend=True,
            marker=dict(color=f"rgb({r},{g},{b})"),
            hovertemplate=(
                f"<b>{q_label}</b><br>"
                f"P10: {score_at_p(10):.0f} | "
                f"Median: {med:.0f} | "
                f"P90: {score_at_p(90):.0f}"
                "<extra></extra>"
            )
        ))

    fig.update_layout(**_base_layout(
        title=f"Score Distribution by Immigration Status | {SUBJECTS[subject]} | {_cnt_label(cnt)}"
    ))
    fig.update_xaxes(title=f"{SUBJECTS[subject]} score", range=[100, 900])
    fig.update_yaxes(
        tickvals=list(range(1, len(quartiles) + 1)),
        ticktext=list(reversed(quartiles)),
        showgrid=False,
        zeroline=False,
        range=[0.3, len(quartiles) + 0.7]
    )
    return fig

def plot_jitter_boxplot(df, subject, cnt, group_col, group_labels,
                         group_title, year=None, min_group_n=30):
    pv_cols = [f"PV{i}{subject}" for i in range(1, 11) if f"PV{i}{subject}" in df.columns]
    subset = df[df["CNT"] == cnt].copy()
    if year is not None and "YEAR" in subset.columns:
        subset = subset[subset["YEAR"] == year]

    # Handle continuous variables (e.g. ESCS) by binning
    if group_labels is None:
        subset = subset.dropna(subset=[group_col])
        subset["_group_bin"] = pd.qcut(
            subset[group_col].rank(method="first"), q=4,
            labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
        )
        group_col = "_group_bin"
        group_labels = {
            "Q1 (lowest)": "Q1 (lowest)",
            "Q2": "Q2",
            "Q3": "Q3",
            "Q4 (highest)": "Q4 (highest)"
        }

    subset["mean_score"] = subset[pv_cols].mean(axis=1)
    subset = subset.dropna(subset=[group_col, "mean_score", "W_FSTUWT"])

    # Proficiency thresholds — PISA standard levels
    def get_tier(score):
        if score < 413:   return "Below Basic (<413)"
        elif score < 545: return "Basic/Proficient (413–544)"
        else:             return "Advanced (545+)"

    subset["Tier"] = subset["mean_score"].apply(get_tier)

    labels = list(group_labels.values())
    codes  = list(group_labels.keys())

    # Tier percentages per group for right panel
    tier_order = ["Below Basic (<413)", "Basic/Proficient (413–544)", "Advanced (545+)"]
    tier_colors = {
        "Below Basic (<413)":        "#D85A30",
        "Basic/Proficient (413–544)":"#cccccc",
        "Advanced (545+)":           "#0072B2"
    }

    tier_pct = {}
    for code, label in zip(codes, labels):
        grp = subset[subset[group_col] == code]
        total = len(grp)
        tier_pct[label] = {
            t: round(len(grp[grp["Tier"] == t]) / total * 100, 1) if total > 0 else 0
            for t in tier_order
        }

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Score Distribution", "Proficiency Tiers"),
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.15
    )

    for i, (code, label) in enumerate(zip(codes, labels)):
        color = PALETTE[i % len(PALETTE)]
        if color.startswith("#") and len(color) == 7:
            r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
        else:
            r, g, b = 128, 128, 128

        grp = subset[subset[group_col] == code]
        if len(grp) < min_group_n:
            continue

        scores = grp["mean_score"].values
        percs = weighted_percentiles_pv(grp, subject, [10, 25, 50, 75, 90])
        p10, p25, p50, p75, p90 = [round(p) for p in percs]

        q1, med, q3 = p25, p50, p75
        iqr = q3 - q1
        lower_fence = max(scores.min(), q1 - 1.5 * iqr)

        # Left: box plot, no hover
        fig.add_trace(go.Box(
            y=scores, name=label,
            marker_color=color,
            line=dict(color=color, width=2),
            fillcolor=f"rgba({r},{g},{b},0.15)",
            boxpoints=False,
            hoverinfo="skip",
            showlegend=False,
            q1=[p25], median=[p50], q3=[p75],
            lowerfence=[p10], upperfence=[p90]
        ), row=1, col=1)

        # Static annotation at lower fence
        fig.add_annotation(
            x=label, y=lower_fence,
            text=f"Q3: {q3}<br>Med: {med}<br>Q1: {q1}",
            showarrow=False,
            xanchor="left", yanchor="bottom", xshift=8,
            align="left",
            font=dict(color=color, size=10),
            row=1, col=1
        )

    # Right: horizontal stacked bars
    for tier in tier_order:
        fig.add_trace(go.Bar(
            y=labels,
            x=[tier_pct[label][tier] for label in labels],
            name=tier,
            orientation="h",
            marker=dict(color=tier_colors[tier]),
            hovertemplate=f"<b>{tier}</b><br>%{{x:.1f}}%<extra></extra>"
        ), row=1, col=2)

    fig.update_layout(
        **_base_layout(title=f"{SUBJECTS[subject]} by {group_title} | {_cnt_label(cnt)}"),
        barmode="stack",
        # height=500,
        # margin=dict(t=80, b=100)
    )
    fig.update_layout(height=500,
        margin=dict(t=80, b=100)
    )
    fig.update_yaxes(title_text=f"{SUBJECTS[subject]} score", row=1, col=1)
    fig.update_xaxes(title_text="% of students", row=1, col=2)

    return fig