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
    MIN_GROUP_N,
    OKABE_ITO,
    SUBJECTS,
)
from src.pisa_stats import (
    weighted_percentiles_pv,
    compute_escs_quartile_percentiles,
    compute_group_percentiles,
    get_oecd_percentiles,
    weighted_mean_pv
)

def _base_layout(title: str = "", height: int = 480) -> dict:
    """Shared Plotly layout settings applied to all dashboard charts."""
    return dict(
        title=title,
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="sans-serif", size=12),
        xaxis=dict(
            gridcolor="#eeeeee", 
            linecolor="#cccccc",
            zeroline=False
        ),
        yaxis=dict(
            gridcolor="#eeeeee", 
            linecolor="#cccccc",
            zeroline=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified",
        margin=dict(t=60, b=40, l=50, r=20),
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
            text=msg,
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


def plot_country_distributions(df, subject: str,
                               countries: list,
                               year: int = None,
                               show_oecd: bool = True) -> go.Figure:
    fig = go.Figure()
    
    color_list = list(OKABE_ITO.values())

    for i, cnt in enumerate(countries):
        subset = df[df["CNT"] == cnt]
        if year is not None and "YEAR" in df.columns:
            subset = subset[subset["YEAR"] == year]
            
        percs = weighted_percentiles_pv(subset, subject, PERCENTILES_COARSE)
        if np.isnan(percs).all():
            continue
            
        color = COUNTRY_COLORS.get(cnt, color_list[i % len(color_list)])
        
        fig.add_trace(go.Scatter(
            x=PERCENTILES_COARSE, y=percs,
            mode="lines+markers", name=cnt,
            line=dict(color=color, width=2.5),
            marker=dict(size=6),
            hovertemplate=f"<b>{cnt}</b><br>Percentile: %{{x}}<br>{SUBJECTS[subject]}: %{{y:.0f}}<extra></extra>"
        ))

    if show_oecd:
        oecd = get_oecd_percentiles(df, subject, PERCENTILES_COARSE, year)
        if not np.isnan(oecd).all():
            fig.add_trace(go.Scatter(
                x=PERCENTILES_COARSE, y=oecd,
                mode="lines", name="OECD avg",
                line=dict(color="black", width=2, dash="dash"),
                hovertemplate=f"<b>OECD avg</b><br>Percentile: %{{x}}<br>{SUBJECTS[subject]}: %{{y:.0f}}<extra></extra>"
            ))

    fig.update_layout(**_base_layout(title=f"Score Distribution | {SUBJECTS[subject]}"))
    fig.update_xaxes(title="Percentile", tickvals=PERCENTILES_COARSE)
    fig.update_yaxes(title=f"{SUBJECTS[subject]} score")
    return fig


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
            marker=dict(size=6)
        ))

    fig.update_layout(**_base_layout(title=f"SES Gap | {SUBJECTS[subject]} | {cnt}"))
    fig.update_xaxes(title="Percentile", tickvals=PERCENTILES_COARSE)
    fig.update_yaxes(title=f"{SUBJECTS[subject]} score")
    return fig


def plot_gender_percentile_line(df, subject: str, cnt: str, year: int = None) -> go.Figure:
    subset = df[df["CNT"] == cnt].copy()
    if year is not None and "YEAR" in df.columns:
        subset = subset[subset["YEAR"] == year]

    valid_data, error_fig = _check_sufficient_data(
        subset, ["ST004D01T"], cnt, msg=f"Insufficient gender data for {cnt}"
    )
    if error_fig is not None:
        return error_fig

    female = valid_data[valid_data["ST004D01T"] == 1.0]
    male   = valid_data[valid_data["ST004D01T"] == 2.0]

    female_percs = weighted_percentiles_pv(female, subject, PERCENTILES_FINE)
    male_percs   = weighted_percentiles_pv(male,   subject, PERCENTILES_FINE)

    fig = go.Figure()
    if np.isnan(female_percs).all():
        return fig

    # Reference diagonal
    fig.add_trace(go.Scatter(
        x=female_percs, y=female_percs,
        mode="lines", name="Female (reference)",
        line=dict(color="#cccccc", width=1.5, dash="dot"),
        hoverinfo="skip"
    ))

    # Male line
    fig.add_trace(go.Scatter(
        x=female_percs, y=male_percs,
        mode="lines", name="Male",
        line=dict(color=COUNTRY_COLORS.get(cnt, "#185FA5"), width=2.5),
        customdata=PERCENTILES_FINE,
        hovertemplate="Percentile: %{customdata}<br>Female Score: %{x:.0f}<br>Male Score: %{y:.0f}<extra></extra>"
    ))

    fig.update_layout(**_base_layout(
        title=f"Gender Gap | {SUBJECTS[subject]} | {cnt}<br><sup>(above diagonal = males score higher)</sup>"
    ))
    fig.update_xaxes(title=f"Female {SUBJECTS[subject]} score (reference)")
    fig.update_yaxes(title="Male Score")
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
            line=dict(color=color, width=2.5)
        ))

    fig.update_layout(**_base_layout(title=title or f"{SUBJECTS[subject]} by Group | {cnt}"))
    fig.update_xaxes(title="Percentile", tickvals=PERCENTILES_COARSE)
    fig.update_yaxes(title=f"{SUBJECTS[subject]} score")
    return fig


def plot_naep_time_comparison(df, subject: str, cnt: str,
                               reference_year: int, comparison_years: list,
                               group_col: str = None, group_val: float = None,
                               group_label: str = "All students") -> go.Figure:
    fig = go.Figure()
    all_years = [reference_year] + comparison_years

    def get_subset(year):
        s = df[(df["CNT"] == cnt) & (df["YEAR"] == year)]
        if group_col and group_val is not None:
            s = s[s[group_col] == group_val]
        return s

    ref_percs = weighted_percentiles_pv(get_subset(reference_year), subject, PERCENTILES_FINE)
    if np.isnan(ref_percs).all():
        return fig

    # Neutral diagonal
    fig.add_trace(go.Scatter(
        x=ref_percs, y=ref_percs,
        mode="lines", name="No change",
        line=dict(color="#dddddd", width=1, dash="dot"),
        hoverinfo="skip"
    ))

    for year in all_years:
        yr_percs = weighted_percentiles_pv(get_subset(year), subject, PERCENTILES_FINE)
        if np.isnan(yr_percs).all():
            continue
        
        is_ref = (year == reference_year)
        fig.add_trace(go.Scatter(
            x=ref_percs, y=yr_percs,
            mode="lines", name=f"{year} (ref)" if is_ref else str(year),
            line=dict(
                color=YEAR_COLORS.get(year, "#333333"), 
                width=3.0 if is_ref else 2.0,
                dash="solid" if is_ref else "dash"
            ),
            customdata=PERCENTILES_FINE,
            hovertemplate=f"<b>{year}</b><br>Percentile: %{{customdata}}<br>{reference_year} score: %{{x:.0f}}<br>{year} score: %{{y:.0f}}<extra></extra>"
        ))

    fig.update_layout(**_base_layout(
        title=f"Score Change Over Time | {SUBJECTS[subject]} | {cnt} | {group_label}<br><sup>(above diagonal = improvement)</sup>"
    ))
    fig.update_xaxes(title=f"{reference_year} {SUBJECTS[subject]} score (reference)")
    fig.update_yaxes(title="Score")
    return fig


def plot_year_diff_percentile(df, subject: str, cnt: str,
                               reference_year: int, comparison_year: int,
                               group_col: str = None, group_val: float = None,
                               group_label: str = "All students") -> go.Figure:
    
    def get_subset(year):
        subset = df[(df["CNT"] == cnt) & (df["YEAR"] == year)]
        if group_col is not None and group_val is not None:
            subset = subset[subset[group_col] == group_val]
        return subset

    ref_percs = weighted_percentiles_pv(get_subset(reference_year), subject, PERCENTILES_FINE)
    comp_percs = weighted_percentiles_pv(get_subset(comparison_year), subject, PERCENTILES_FINE)

    fig = go.Figure()
    if np.isnan(ref_percs).all() or np.isnan(comp_percs).all():
        return _check_sufficient_data(pd.DataFrame(), [], cnt, msg="Insufficient data to compute difference")[1]

    delta = comp_percs - ref_percs

    fig.add_hline(y=0, line_dash="dash", line_color="#777777", line_width=1.5)

    fig.add_trace(go.Scatter(
        x=PERCENTILES_FINE, y=delta,
        mode="lines", name=f"{comparison_year} - {reference_year}",
        line=dict(color=COUNTRY_COLORS.get(cnt, "#185FA5"), width=2.5),
        hovertemplate="Percentile: %{x}<br>Score Diff: %{y:.1f}<extra></extra>"
    ))

    fig.update_layout(**_base_layout(
        title=f"Score Change Across Distribution | {SUBJECTS[subject]} | {cnt}<br><sup>({comparison_year} vs {reference_year})</sup>"
    ))
    fig.update_xaxes(title="Percentile")
    fig.update_yaxes(title=f"Score Difference ({comparison_year} - {reference_year})")
    return fig


def plot_weighted_interval_distribution(df, subject: str, countries: list,
                                        year: int = None, interval_width: int = 20,
                                        score_range: tuple = (0, 1000),
                                        show_oecd: bool = True) -> go.Figure:
    pv_cols = [f"PV{i}{subject}" for i in range(1, 11) if f"PV{i}{subject}" in df.columns]
    bins = np.arange(score_range[0], score_range[1] + interval_width, interval_width)
    midpoints = (bins[:-1] + bins[1:]) / 2

    fig = go.Figure()
    color_cycle = list(COUNTRY_COLORS.values()) + ["#1D9E75", "#BA7517", "#888780"]

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
        color = COUNTRY_COLORS.get(cnt, color_cycle[i % len(color_cycle)])
        
        fig.add_trace(go.Scatter(
            x=midpoints, y=props,
            mode="lines+markers", name=cnt,
            line=dict(color=color, width=2.5),
            hovertemplate=f"<b>{cnt}</b><br>Score: %{{x}}<br>Proportion: %{{y:.3f}}<extra></extra>"
        ))

    if show_oecd:
        all_subset = df.dropna(subset=pv_cols + ["W_FSTUWT"])
        if year is not None and "YEAR" in df.columns:
            all_subset = all_subset[all_subset["YEAR"] == year]
        oecd_props = _country_proportions(all_subset)
        if not np.isnan(oecd_props).all():
            fig.add_trace(go.Scatter(
                x=midpoints, y=oecd_props,
                mode="lines", name="OECD avg",
                line=dict(color="black", width=2, dash="dash")
            ))

    fig.update_layout(**_base_layout(title=f"Score Distribution | {SUBJECTS[subject]}<br><sup>(averaged across 10 PVs)</sup>"))
    fig.update_xaxes(title="Score")
    fig.update_yaxes(title="Weighted proportion of students")
    return fig


def plot_gender_diff_percentile(df, subject: str, cnt: str, year: int = None) -> go.Figure:
    subset = df[df["CNT"] == cnt]
    if year is not None and "YEAR" in df.columns:
        subset = subset[subset["YEAR"] == year]

    female = subset[subset["ST004D01T"] == 1.0]
    male = subset[subset["ST004D01T"] == 2.0]

    female_percs = weighted_percentiles_pv(female, subject, PERCENTILES_FINE)
    male_percs = weighted_percentiles_pv(male, subject, PERCENTILES_FINE)

    fig = go.Figure()
    if np.isnan(female_percs).all() or np.isnan(male_percs).all():
        return _check_sufficient_data(pd.DataFrame(), [], cnt, msg="Insufficient data")[1]

    diff = male_percs - female_percs

    fig.add_hline(y=0, line_dash="dash", line_color="#cccccc", line_width=1.5, annotation_text="No gap")

    line_color = COUNTRY_COLORS.get(cnt, "#185FA5") if diff.mean() >= 0 else OKABE_ITO.get("pink", "#CC79A7")

    fig.add_trace(go.Scatter(
        x=female_percs, y=diff,
        mode="lines", name="Difference",
        line=dict(color=line_color, width=2.5),
        customdata=PERCENTILES_FINE,
        hovertemplate="Percentile: %{customdata}<br>Score Diff: %{y:.1f}<extra></extra>"
    ))

    fig.update_layout(**_base_layout(
        title=f"Gender Gap | {SUBJECTS[subject]} | {cnt}<br><sup>(above zero = males score higher)</sup>"
    ))
    fig.update_xaxes(title=f"Female {SUBJECTS[subject]} score (reference)")
    fig.update_yaxes(title="Score Difference (Male - Female)")
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
        color_map[cnt] = COUNTRY_COLORS.get(cnt, PALETTE[i % len(PALETTE)])

    # Panel 1: Repetition
    if all(c in subset.columns for c in ["REPEAT", "ESCS", "CNT", "W_FSTUWT"]):
        df_rep = subset.dropna(subset=["REPEAT", "ESCS", "W_FSTUWT"]).copy()
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
                x=["Q1", "Q2", "Q3", "Q4"], y=rates, name=cnt, 
                marker_color=color_map[cnt],
                texttemplate="%{y:.1f}%", textposition="outside"
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
                x=list(IMMIG_MAP.values()), y=means, name=cnt, 
                marker_color=color_map[cnt],
                showlegend=False, 
                texttemplate="%{y:.2f}", textposition="outside"
            ), row=1, col=2)

    fig.update_layout(**_base_layout(title="Student Context"))
    fig.update_layout(barmode='group')
    return fig


def plot_immigration_score_distribution(df, subject: str, cnt: str, year: int = None,
                                        interval_width: int = 20, score_range: tuple = (0, 1000)) -> go.Figure:
    pv_cols = [f"PV{i}{subject}" for i in range(1, 11) if f"PV{i}{subject}" in df.columns]
    bins = np.arange(score_range[0], score_range[1] + interval_width, interval_width)
    midpoints = (bins[:-1] + bins[1:]) / 2

    subset = df[df["CNT"] == cnt].copy()
    if year is not None and "YEAR" in subset.columns:
        subset = subset[subset["YEAR"] == year]

    fig = go.Figure()

    for color, (code, label) in zip(PALETTE, IMMIG_MAP.items()):
        group_df = subset[subset["IMMIG"] == code].dropna(subset=pv_cols + ["W_FSTUWT"])
        if group_df.empty: continue

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
            mode="lines", name=label,
            line=dict(color=color, width=2.5)
        ))

    fig.update_layout(**_base_layout(title=f"Score by Immigration Status | {SUBJECTS[subject]} | {cnt}"))
    fig.update_xaxes(title="Score")
    fig.update_yaxes(title="Weighted proportion")
    return fig


def plot_school_location_boxplot(df, subject: str, cnt: str, year: int = None,
                                 location_col: str = "SC001Q01TA", min_group_n: int = 30) -> go.Figure:
    subset = df[df["CNT"] == cnt].copy()
    if year is not None and "YEAR" in subset.columns:
        subset = subset[subset["YEAR"] == year]

    fig = go.Figure()

    for i, (code, label) in enumerate(LOC_MAP.items()):
        group = subset[subset[location_col] == code]
        if len(group.dropna(subset=["W_FSTUWT"])) < min_group_n:
            continue

        percs = weighted_percentiles_pv(group, subject, [10, 25, 50, 75, 90])
        if np.isnan(percs).all():
            continue

        fig.add_trace(go.Box(
            name=label,
            lowerfence=[percs[0]],
            q1=[percs[1]],
            median=[percs[2]],
            q3=[percs[3]],
            upperfence=[percs[4]],
            marker_color=PALETTE[i % len(PALETTE)],
            boxpoints=False
        ))

    fig.update_layout(**_base_layout(title=f"Score Distribution by School Location | {SUBJECTS[subject]} | {cnt}"))
    fig.update_yaxes(title=f"{SUBJECTS[subject]} score")
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
            marker=dict(size=4)
        ))

    fig.update_layout(**_base_layout(title=f"Score by School Type | {SUBJECTS[subject]} | {cnt}"))
    fig.update_xaxes(title="Score")
    fig.update_yaxes(title="Weighted proportion")
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
            rows.append({"CNT": cnt, "score": mean_score,
                          "resource": mean_resource, "OECD": oecd_flag})

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
            customdata=partner[["CNT"]], hovertemplate=htemp
        ))

    if not oecd.empty:
        fig.add_trace(go.Scatter(
            x=oecd["resource"], y=oecd["score"],
            mode="markers", name="OECD members",
            marker=dict(color="#185FA5", size=10, opacity=0.8),
            customdata=oecd[["CNT"]], hovertemplate=htemp
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
            text=hi["CNT"], 
            textposition="top center", 
            textfont=dict(size=11, color=OKABE_ITO.get("vermillion", "#D85A30")),
            customdata=hi[["CNT"]], hovertemplate=htemp
        ))

    fig.update_layout(**_base_layout(
        title=f"{resource_label} vs {SUBJECTS[subject]} performance<br><sup>(each point = one country)</sup>"
    ))
    fig.update_xaxes(title=resource_label)
    fig.update_yaxes(title=f"Mean {SUBJECTS[subject]} score")

    return fig