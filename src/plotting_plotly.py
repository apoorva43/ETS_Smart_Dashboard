"""
Plotting utilities for the PISA dashboard using Plotly.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.config import (
    PERCENTILES_COARSE,
    PALETTE,
    OKABE_ITO,
    SUBJECTS,
    COUNTRY_NAMES
)
from src.pisa_stats import (
    weighted_percentiles_pv,
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

def plot_intersectional_heatmap(df, subject, cnt, row_var="ESCS", col_var="BELONG",
                                 row_label="SES Quartile", col_label="Belonging",
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

    # Handle categorical vs continuous variables dynamically
    def assign_bins(df_col, var_name, label_prefix):
        if var_name == "IMMIG":
            mapping = {1.0: "Native", 2.0: "Second-generation", 3.0: "First-generation"}
            cats = ["Native", "Second-generation", "First-generation"]
            return df_col.map(mapping), cats
        elif var_name in ["ST004D01T", "GENDER"]: # 1=Female, 2=Male
            mapping = {1.0: "Female", 2.0: "Male"}
            cats = ["Female", "Male"]
            return df_col.map(mapping), cats
        else:
            # Strip out "Quartile" if it's already in the label to prevent duplication with Q1/Q2
            clean_prefix = label_prefix.replace(" Quartile", "").replace(" quartile", "")
            cats = [f"{clean_prefix} Q{i+1}" for i in range(n_bins)]
            binned = pd.qcut(df_col.rank(method="first"), q=n_bins, labels=cats)
            return binned, cats

    subset["row_bin"], row_cats = assign_bins(subset[row_var], row_var, row_label)
    subset["col_bin"], col_cats = assign_bins(subset[col_var], col_var, col_label)

    # Dynamic dimensions based on the actual categories returned
    z = np.full((len(row_cats), len(col_cats)), np.nan)
    text = [[""] * len(col_cats) for _ in range(len(row_cats))]
    customdata = [[0.0] * len(col_cats) for _ in range(len(row_cats))]
    
    total_weight = subset["W_FSTUWT"].sum()

    for r_idx, r_cat in enumerate(row_cats):
        for c_idx, c_cat in enumerate(col_cats):
            cell = subset[(subset["row_bin"] == r_cat) & (subset["col_bin"] == c_cat)]
            
            if len(cell) < min_cell_n:
                text[r_idx][c_idx] = "n/a"
                customdata[r_idx][c_idx] = 0.0
                continue
                
            pv_means = [np.average(cell[pv].values, weights=cell["W_FSTUWT"].values)
                        for pv in pv_cols if pv in cell.columns]
            
            if pv_means:
                # Calculate % of population for this cell using weights
                cell_weight = cell["W_FSTUWT"].sum()
                pop_share = (cell_weight / total_weight) * 100
                
                z[r_idx][c_idx] = np.mean(pv_means)
                text[r_idx][c_idx] = f"{z[r_idx][c_idx]:.0f}"
                customdata[r_idx][c_idx] = pop_share

    fig = go.Figure(go.Heatmap(
        z=z,
        x=col_cats,
        y=row_cats,
        text=text,
        texttemplate="%{text}",
        customdata=customdata,
        colorscale="Blues",
        colorbar=dict(title=f"Mean {SUBJECTS[subject]} score"),
        hovertemplate=(
            f"{row_label}: %{{y}}<br>"
            f"{col_label}: %{{x}}<br>"
            f"Mean score: %{{z:.0f}}<br>"
            f"Population share: %{{customdata:.1f}}%<extra></extra>"
        )
    ))

    # Standardize title replacing "Quartile" if a categorical variable is used
    clean_col_label = col_label.replace(' Quartile', '') if col_var in ['IMMIG', 'ST004D01T'] else col_label
    clean_row_label = row_label.replace(' Quartile', '') if row_var in ['IMMIG', 'ST004D01T'] else row_label

    fig.update_layout(**_base_layout(
        title=f"Mean {SUBJECTS[subject]} Score | {clean_row_label} x {clean_col_label} | {_cnt_label(cnt)}",
        height=480
    ))
    return fig

def _parse_color(color):
    if color.startswith("#") and len(color) == 7:
        return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    elif color.startswith("rgb"):
        clean = color.replace("rgba","").replace("rgb","").replace("(","").replace(")","")
        parts = clean.split(",")
        return int(parts[0]), int(parts[1]), int(parts[2])
    return 128, 128, 128

def _render_shaded_density_rows(fig, rows, x_grid, BANDS, bar_height, show_bottom_percentile_labels=False, show_oecd_labels=False, show_percentile_text=True, short_percentile_labels=False):
    """
    rows: list of dicts with keys:
        - group: DataFrame (already filtered)
        - label: str (y-axis label)
        - color_rgb: tuple (r, g, b)
        - y_center: float
        - pv_cols: list
        - legendgroup: str
    """
    from scipy.stats import gaussian_kde

    bottom_y_center = min([row["y_center"] for row in rows]) if rows else None
    for row in rows:
        group = row["group"]
        q_label = row["label"]
        r, g, b = row["color_rgb"]
        y_center = row["y_center"]
        pv_cols = row["pv_cols"]
        legendgroup = row["legendgroup"]

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
        density /= density.max()

        all_scores = np.concatenate([group[pv].values for pv in pv_cols])
        all_weights = np.tile(group["W_FSTUWT"].values, len(pv_cols))
        valid = np.isfinite(all_scores) & np.isfinite(all_weights)
        all_scores, all_weights = all_scores[valid], all_weights[valid]
        sort_idx = np.argsort(all_scores)
        sorted_scores = all_scores[sort_idx]
        sorted_weights = all_weights[sort_idx]
        cumw = np.cumsum(sorted_weights) / sorted_weights.sum()

        def score_at_p(p):
            idx = np.searchsorted(cumw, p / 100)
            return float(sorted_scores[min(idx, len(sorted_scores) - 1)])

        for (lo_p, hi_p, alpha) in BANDS:
            x_lo = score_at_p(lo_p)
            x_hi = score_at_p(hi_p)
            mask = (x_grid >= x_lo) & (x_grid <= x_hi)
            if mask.sum() < 2:
                continue
            band_x = np.concatenate([[x_lo], x_grid[mask], [x_hi]])
            band_density = np.concatenate([[0], density[mask], [0]])
            scaled_y_top = y_center + (band_density / 2) * bar_height
            scaled_y_bot = y_center - (band_density / 2) * bar_height
            poly_x = np.concatenate([band_x, band_x[::-1]])
            poly_y = np.concatenate([scaled_y_top, scaled_y_bot[::-1]])

            fig.add_trace(go.Scatter(
                x=poly_x, y=poly_y,
                fill="toself",
                fillcolor=f"rgba({r},{g},{b},{alpha})",
                line=dict(color=f"rgba({r},{g},{b},0)", width=0),
                mode="lines",
                legendgroup=legendgroup,
                showlegend=False,
                hoverinfo="skip"
            ))

        med = score_at_p(50)
        fig.add_trace(go.Scatter(
            x=[med, med],
            y=[y_center - bar_height / 2, y_center + bar_height / 2],
            mode="lines",
            line=dict(color=f"rgb({r},{g},{b})", width=3),
            legendgroup=legendgroup,
            showlegend=False,
            hoverinfo="skip"
        ))

        marker_ps = [10, 25, 50, 75, 90]
        fig.add_trace(go.Scatter(
            x=[round(score_at_p(p)) for p in marker_ps],
            y=[y_center] * len(marker_ps),
            mode="markers",
            marker=dict(
                color=f"rgb({r},{g},{b})",
                size=[6, 6, 10, 6, 6],
                # symbol="line-ns",
                line=dict(color=f"rgb({r},{g},{b})", width=2)
            ),
            legendgroup=legendgroup,
            showlegend=False,
            hoverinfo="skip"
        ))

        if show_percentile_text:
            fig.add_trace(go.Scatter(
                x=[round(score_at_p(50)) + 8],
                y=[y_center + 0.12],
                mode="text",
                text=[f"<b>{round(score_at_p(50))}</b>"],
                textposition="middle right",
                textfont=dict(size=12, color=f"rgba({r},{g},{b},0.9)"),
                legendgroup=legendgroup,
                showlegend=False,
                hoverinfo="skip"
            ))
        
        # Add percentile labels only under the bottom row.
        # This is useful for group-comparison shaded density plots:
        # label the five vertical percentile markers once instead of repeating
        # the same labels on every row.
        if show_bottom_percentile_labels and y_center == bottom_y_center:
            if short_percentile_labels:
                p_annotate = {
                    10: "P10",
                    25: "P25",
                    50: "Median",
                    75: "P75",
                    90: "P90",
                }
            else:
                p_annotate = {
                    10: "10th Percentile",
                    25: "25th Percentile",
                    50: "Median",
                    75: "75th Percentile",
                    90: "90th Percentile",
                }

            fig.add_trace(go.Scatter(
                x=[round(score_at_p(p)) for p in p_annotate],
                y=[y_center - 0.55] * len(p_annotate),
                mode="text",
                text=list(p_annotate.values()),
                textposition="top center",
                textfont=dict(size=9, color="rgba(85,85,85,0.9)"),
                legendgroup=legendgroup,
                showlegend=False,
                hoverinfo="skip"
            ))
            
        if show_oecd_labels:
            if short_percentile_labels:
                p_annotate = {
                    10: "P10",
                    25: "P25",
                    50: "Median",
                    75: "P75",
                    90: "P90",
                }
            else:
                p_annotate = {
                    10: "10th Percentile",
                    25: "25th Percentile",
                    50: "Median",
                    75: "75th Percentile",
                    90: "90th Percentile",
                }

            fig.add_trace(go.Scatter(
                x=[round(score_at_p(p)) for p in p_annotate],
                y=[y_center - 0.55] * len(p_annotate),
                mode="text",
                text=list(p_annotate.values()),
                textposition="top center",
                textfont=dict(size=9, color="rgba(85,85,85,0.9)"),
                legendgroup=legendgroup,
                showlegend=False,
                hoverinfo="skip"
            ))

            # Add OECD median value to match Canada / selected countries
            fig.add_trace(go.Scatter(
                x=[round(score_at_p(50)) + 8],
                y=[y_center + 0.12],
                mode="text",
                text=[f"<b>{round(score_at_p(50))}</b>"],
                textposition="middle right",
                textfont=dict(size=12, color="rgba(85,85,85,0.95)"),
                legendgroup=legendgroup,
                showlegend=False,
                hoverinfo="skip"
            ))

        if not show_oecd_labels:
            fig.add_trace(go.Scatter(
                x=[round(score_at_p(p)) for p in [10, 25, 50, 75, 90]],
                y=[y_center] * 5,
                mode="markers",
                marker=dict(
                    color=f"rgb({r},{g},{b})",
                    size=[6, 6, 10, 6, 6],
                    symbol="line-ns",
                    line=dict(color=f"rgb({r},{g},{b})", width=2)
                ),
                legendgroup=legendgroup,
                showlegend=False,
                hoverinfo="skip"
            ))

        p_vals = np.arange(2, 98.2, 0.2)
        score_vals_raw = np.array([score_at_p(p) for p in p_vals])
        unique_mask = np.concatenate([[True], np.diff(score_vals_raw.round(0)) != 0])
        score_vals = score_vals_raw[unique_mask]
        p_vals_clean = p_vals[unique_mask]

        fig.add_trace(go.Scatter(
            x=score_vals,
            y=[y_center] * len(score_vals),
            mode="markers",
            marker=dict(color="rgba(0,0,0,0)", size=8),
            name=q_label,
            legendgroup=legendgroup,
            showlegend=True,
            customdata=np.stack([p_vals_clean], axis=1),
            hovertemplate=(
                f"<b>{q_label}</b><br>"
                "Score: %{x:.0f}<br>"
                "Percentile: %{customdata[0]:.0f}"
                "<extra></extra>"
            )
        ))

# Percentile score profile
def plot_country_shaded_density(df, subject, countries, year, min_group_n=30, compact=False):
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

    x_grid = np.linspace(100, 900, 500)
    BANDS = [(0,10,0.10),(10,25,0.20),(25,75,0.45),(75,90,0.20),(90,100,0.10)]
    bar_height = 0.7
    fig = go.Figure()

    rows = []
    for row_idx, cnt_code in enumerate(countries):
        group = subset[subset["CNT"] == cnt_code]
        if len(group) < min_group_n:
            continue
        q_label = _cnt_label(cnt_code)
        color = PALETTE[row_idx % len(PALETTE)]
        r, g, b = _parse_color(color)
        rows.append(dict(
            group=group, label=q_label, color_rgb=(r, g, b),
            y_center=len(countries) - row_idx,
            pv_cols=pv_cols, legendgroup=q_label
        ))

    _render_shaded_density_rows(fig, rows, x_grid, BANDS, bar_height,
                                 show_percentile_text=True, show_oecd_labels=False,
                                short_percentile_labels=compact)

    # OECD row — same helper, y=0
    oecd_df = df[df["OECD"] == 1].copy()
    if year is not None and "YEAR" in oecd_df.columns:
        oecd_df = oecd_df[oecd_df["YEAR"] == year]
    oecd_df = oecd_df.dropna(subset=["W_FSTUWT"] + pv_cols)

    if len(oecd_df) > 30:
        _render_shaded_density_rows(fig, [dict(
            group=oecd_df, label="OECD Average", color_rgb=(85, 85, 85),
            y_center=0, pv_cols=pv_cols, legendgroup="OECD Average"
        )], x_grid, BANDS, bar_height, show_percentile_text=False,
            show_oecd_labels=True, short_percentile_labels=compact,)

    all_tickvals = list(range(len(countries), 0, -1)) + [0]
    all_ticktext = [_cnt_label(c) for c in countries] + ["OECD Average"]

    fig.update_layout(**_base_layout(title=f"Score Distribution | {SUBJECTS[subject]}"))
    fig.update_layout(hovermode="closest", hoverlabel=dict(namelength=-1))
    fig.update_xaxes(title=f"{SUBJECTS[subject]} score", range=[100, 900],
                     showspikes=True, spikemode="across", spikesnap="data",
                     tickformat="d", hoverformat="d")
    fig.update_yaxes(tickvals=all_tickvals, ticktext=all_ticktext,
                     showgrid=False, zeroline=False,
                     range=[-0.7, len(countries) + 0.7])
    return fig

# Group comparison
def plot_group_shaded_density(df, subject, cnt, group_col, group_labels,
                               group_title, year=None, min_group_n=30, sort_by_median=False, compact=False):
    pv_cols = [f"PV{i}{subject}" for i in range(1, 11) if f"PV{i}{subject}" in df.columns]
    subset = df[df["CNT"] == cnt].copy()
    if year is not None and "YEAR" in subset.columns:
        subset = subset[subset["YEAR"] == year]

    # Handle continuous variables (ESCS) by binning into quartiles
    if group_labels is None:
        subset = subset.dropna(subset=[group_col])
        subset["_group_bin"] = pd.qcut(
            subset[group_col].rank(method="first"), q=4,
            labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
        )
        group_col = "_group_bin"
        group_labels = {
            "Q1 (lowest)": "Q1 (lowest)", "Q2": "Q2",
            "Q3": "Q3", "Q4 (highest)": "Q4 (highest)"
        }

    subset = subset.dropna(subset=["W_FSTUWT"] + pv_cols)

    if len(subset) < min_group_n:
        fig = go.Figure()
        fig.add_annotation(text="⚠️ Insufficient data.", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="gray"))
        return fig

    # Compute group sizes and percentages for y-axis labels
    total = len(subset)
    group_sizes = {
        code: len(subset[subset[group_col] == code])
        for code in group_labels
    }

    codes = list(group_labels.keys())

    # Helper to enforce OECD standard sorting via semantic label matching
    def _get_oecd_rank(code):
        lbl = str(group_labels.get(code, "")).lower()
        
        # a. Immigration status
        if "native" in lbl: return 1
        if "1st" in lbl or "first" in lbl: return 2
        if "2nd" in lbl or "second" in lbl: return 3
        
        # b. School location (Megacity to village)
        # Note: Must check specific multi-word labels before broad ones 
        # so 'megacity'/'large city' aren't caught by 'city', etc.
        if "megacity" in lbl: return 10
        if "large city" in lbl or "1 000 000" in lbl: return 11
        if "small town" in lbl or "3 000" in lbl: return 14
        if "town" in lbl or "15 000" in lbl: return 13
        if "city" in lbl: return 12
        if "village" in lbl or "rural" in lbl: return 15
        
        # c. School type (Public -> Govt-dep. private -> Independent private)
        if "public" in lbl: return 20
        if "govt" in lbl: return 21
        if "independent" in lbl: return 22
        
        # d. Socioeconomic status (Q4 to Q1)
        if "q4" in lbl: return 30
        if "q3" in lbl: return 31
        if "q2" in lbl: return 32
        if "q1" in lbl: return 33
        
        # e. Gender (Male -> Female)
        # Note: Must check 'female' first so it doesn't accidentally match 'male'
        if "female" in lbl: return 41
        if "male" in lbl: return 40
        
        return 999 # Fallback for unrecognized variables

    # Apply custom sort if the variable matches any OECD patterns
    ranks = {c: _get_oecd_rank(c) for c in codes}
    has_custom_order = any(r != 999 for r in ranks.values())

    if has_custom_order:
        codes.sort(key=lambda c: ranks[c])
    else:
        # Fallback to median or size
        if sort_by_median:
            def _get_median(code):
                grp = subset[subset[group_col] == code]
                if len(grp) < min_group_n:
                    return -1
                m = weighted_percentiles_pv(grp, subject, [50])
                return float(m[0]) if not np.isnan(m[0]) else -1
            codes.sort(key=_get_median, reverse=True)
        else:
            codes.sort(key=lambda c: group_sizes.get(c, 0), reverse=True)

    x_grid = np.linspace(100, 900, 500)
    BANDS = [(0,10,0.10),(10,25,0.20),(25,75,0.45),(75,90,0.20),(90,100,0.10)]
    bar_height = 0.7
    fig = go.Figure()

    # filter out the suppressed groups entirely
    valid_groups = []
    for code in codes:
        group = subset[subset[group_col] == code]
        if len(group) >= min_group_n:
            valid_groups.append((code, group))

    rows = []
    ticktext = []
    tickvals = []
    num_valid = len(valid_groups)

    # Iterate only over the valid groups
    for row_idx, (code, group) in enumerate(valid_groups):
        pct = group_sizes.get(code, 0) / total * 100
        base_label = group_labels[code]
        y_label = f"{base_label} ({pct:.0f}%)"
        
        color = PALETTE[row_idx % len(PALETTE)]
        r, g, b = _parse_color(color)

        y_center = num_valid - row_idx
        rows.append(dict(
            group=group, label=y_label, color_rgb=(r, g, b),
            y_center=y_center,
            pv_cols=pv_cols, legendgroup=y_label
        ))
        ticktext.append(y_label)
        tickvals.append(y_center)

    _render_shaded_density_rows(
        fig,
        rows,
        x_grid,
        BANDS,
        bar_height,
        show_bottom_percentile_labels=True,
        short_percentile_labels=compact
    )

    fig.update_layout(**_base_layout(
        title=f"Score by {group_title} | {SUBJECTS[subject]} | {_cnt_label(cnt)}"
    ))
    fig.update_layout(hovermode="closest", hoverlabel=dict(namelength=-1))
    fig.update_xaxes(title=f"{SUBJECTS[subject]} score", range=[100, 900],
                     showspikes=True, spikemode="across", spikesnap="data",
                     tickformat="d", hoverformat="d")
    
    fig.update_yaxes(tickvals=tickvals, ticktext=ticktext,
                     showgrid=False, zeroline=False,
                     range=[0.3, num_valid + 0.7])
                     
    return fig

# Score change over time
def plot_percentile_change_from_baseline(df, subject, cnt, reference_year=2015):
    subset = df[df["CNT"] == cnt].copy()
    years = sorted(subset["YEAR"].dropna().unique())

    ref_subset = subset[subset["YEAR"] == reference_year]
    ref_percs = weighted_percentiles_pv(ref_subset, subject, PERCENTILES_COARSE)

    if np.isnan(ref_percs).all():
        return _check_sufficient_data(pd.DataFrame(), [], cnt,
            msg=f"No data for baseline year {reference_year}")[1]

    # Map each index to its label and a distinct Plotly symbol
    percentile_config = {
        0: {"label": "10th percentile", "symbol": "triangle-down"},
        1: {"label": "25th percentile", "symbol": "square"},
        2: {"label": "50th (median)", "symbol": "diamond"},
        3: {"label": "75th percentile", "symbol": "circle"},
        4: {"label": "90th percentile", "symbol": "triangle-up"}
    }

    fig = go.Figure()
    
    # Baseline uses a neutral dark grey
    fig.add_hline(y=0, line_dash="solid", line_color="#666666",
                  line_width=1.5, annotation_text=f"{reference_year} baseline",
                  annotation_position="top left")

    # 90th percentile is added first to put
    # it at the top of the hover tooltip and the legend
    for p_idx, p_val in reversed(list(enumerate(PERCENTILES_COARSE))):
        config = percentile_config[p_idx]
        p_label = config["label"]
        p_symbol = config["symbol"]
        
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

        # Base sizes
        base_size = 13 if "triangle" in p_symbol else 10
        
        # Dynamically set size and border width to 0 for all but the 50th percentile (p_idx == 2) at the baseline year
        marker_sizes = [base_size if (yr != reference_year or p_idx == 2) else 0 for yr in x_years]
        border_widths = [1 if (yr != reference_year or p_idx == 2) else 0 for yr in x_years]

        fig.add_trace(go.Scatter(
            x=x_years, y=y_deltas,
            mode="lines+markers",
            name=p_label,
            line=dict(color="#B0B0B0", width=2),
            marker=dict(symbol=p_symbol, size=marker_sizes, color=color, line=dict(color="white", width=border_widths)),
            customdata=[[yr, f"{d:+.0f}"] for yr, d in zip(x_years, y_deltas)],
            hovertemplate=(
                f"<b>{p_label}</b><br>"
                "Year: %{customdata[0]}<br>"
                "Change from baseline: %{customdata[1]}<extra></extra>"
            )
        ))

    fig.update_layout(**_base_layout(
        title=f"{SUBJECTS[subject]} Score Change by Percentile | {_cnt_label(cnt)}<br>"
              f"<sup>Relative to {reference_year} baseline</sup>"
    ))
    fig.update_xaxes(title="Year", tickvals=years, tickformat="d")
    fig.update_yaxes(title=f"Score change from {reference_year}")
    return fig

# Scatter plot
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

    # Quadrant label annotations — positioned at the extreme edges of the plot using 'paper' references
    short_label = resource_label.split(" ")[0]
    quadrant_labels = [
        (0.98, 0.98, f"High Performance /<br>High {short_label}", "right", "top"),
        (0.02, 0.98, f"High Performance /<br>Low {short_label}", "left", "top"),
        (0.98, 0.02, f"Low Performance /<br>High {short_label}", "right", "bottom"),
        (0.02, 0.02, f"Low Performance /<br>Low {short_label}", "left", "bottom"),
    ]
    
    for qx, qy, qtext, xanch, yanch in quadrant_labels:
        fig.add_annotation(
            x=qx, y=qy, xref="paper", yref="paper", 
            text=qtext, showarrow=False,
            font=dict(size=11, color="#999999"),
            align=xanch, xanchor=xanch, yanchor=yanch
        )

    layout_args = _base_layout(
        title=f"{resource_label} vs {SUBJECTS[subject]} performance<br><sup>(each point = one country)</sup>"
    )
    
    # Force the chart to be taller to spread the points out vertically
    layout_args["height"] = 700 
    
    fig.update_layout(**layout_args)
    fig.update_xaxes(title=resource_label)
    fig.update_yaxes(title=f"Mean {SUBJECTS[subject]} score")
    
    fig.update_layout(hovermode="closest")

    return fig
