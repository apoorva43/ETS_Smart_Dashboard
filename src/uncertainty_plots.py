# src/uncertainty_plots.py
"""
Uncertainty visualisation prototypes for the PISA dashboard, based on Kay et al. (2016) "When(ish) is My Bus?".
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from src.pisa_stats import weighted_mean_pv, weighted_percentiles_pv
from src.config import (
    PERCENTILES_FINE,
    COUNTRY_COLORS, OKABE_ITO, PALETTE
)

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#eeeeee",
    "grid.linewidth":    0.7,
    "font.family":       "sans-serif",
    "font.size":         11,
})



def _brr_mean_distribution(data: pd.DataFrame, subject: str) -> np.ndarray:
    """
    Return the point estimate + 80 BRR replicate mean estimates.

    This gives us 81 plausible values for the true mean, which we use
    as the raw distribution for the quantile dotplot.

    Parameters
    ----------
    data : pd.DataFrame
        Filtered PISA subset (single country / group / year).
    subject : str
        Subject code, e.g. ``"MATH"``.

    Returns
    -------
    np.ndarray of shape (81,)
        Index 0 is the main estimate; indices 1–80 are BRR replicates.
        Returns array of NaN if data is insufficient.
    """
    point = weighted_mean_pv(data, subject)
    if np.isnan(point):
        return np.full(81, np.nan)

    replicates = [point]
    for i in range(1, 81):
        rep_col = f"W_FSTURWT{i}"
        if rep_col not in data.columns:
            replicates.append(np.nan)
            continue
        replicates.append(weighted_mean_pv(data, subject, weight_col=rep_col))

    return np.array(replicates)



def plot_quantile_dotplot(df: pd.DataFrame, subject: str,
                          countries: list,
                          year: int = None,
                          n_dots: int = 20,
                          oecd_mean: float = None) -> plt.Figure:
    """
    Quantile dotplot of the BRR distribution of mean scores per country.

    Each dot represents one equally-likely outcome drawn from the 80 BRR
    replicate estimates. Users can count dots above/below a threshold
    without needing to interpret a confidence interval.

    Based on Kay et al. (2016): frequency-based displays like this 
    outperform CI ribbons for non-expert decision makers because counting 
    is more intuitive than interpreting CIs.

    Parameters
    ----------
    df : pd.DataFrame
        Full PISA dataset.
    subject : str
        Subject code, e.g. ``"MATH"``.
    countries : list of str
        Country codes to compare, e.g. ``["CAN", "USA"]``.
    year : int, optional
        Filter to a single PISA cycle. ``None`` uses all available years.
    n_dots : int
        Number of dots in the grid. Each dot = 1/n_dots probability.
        20 is recommended (each dot = 5%). Defaults to 20.
    oecd_mean : float, optional
        OECD average score to draw as a reference line. If ``None``,
        computed from all OECD==1 countries in the filtered data.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_countries = len(countries)
    fig, axes = plt.subplots(
        1, n_countries,
        figsize=(5 * n_countries, 4.5),
        sharey=False
    )
    if n_countries == 1:
        axes = [axes]

    color_cycle = list(COUNTRY_COLORS.values()) + \
        [OKABE_ITO["green"], OKABE_ITO["orange"], OKABE_ITO["pink"]]

    # Compute OECD mean from data if not provided
    oecd_subset = df[df["OECD"] == 1]
    if year is not None and "YEAR" in df.columns:
        oecd_subset = oecd_subset[oecd_subset["YEAR"] == year]
    if oecd_mean is None:
        oecd_mean = weighted_mean_pv(oecd_subset, subject)
    oecd_label = f"OECD avg: {oecd_mean:.0f}"

    for ax, cnt, color in zip(axes, countries,
                               [color_cycle[i % len(color_cycle)]
                                for i in range(n_countries)]):
        subset = df[df["CNT"] == cnt]
        if year is not None and "YEAR" in df.columns:
            subset = subset[subset["YEAR"] == year]

        dist = _brr_mean_distribution(subset, subject)
        valid = dist[~np.isnan(dist)]

        if len(valid) < 5:
            ax.text(0.5, 0.5, f"Insufficient data\nfor {cnt}",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        # Quantile-sample n_dots equally-spaced quantiles from the
        # BRR distribution - this is the core of the dotplot encoding.
        quantile_probs = np.linspace(1 / (2 * n_dots), 1 - 1 / (2 * n_dots),
                                      n_dots)
        dot_values = np.quantile(valid, quantile_probs)

        point_est = dist[0]   # main weight estimate

        # Layout: stack dots in columns of fixed width
        # Each unique score value gets a column; dots stack vertically
        dot_size   = 18
        cols       = 5        # dots per row before wrapping
        rows_needed = int(np.ceil(n_dots / cols))

        xs, ys = [], []
        for k, val in enumerate(sorted(dot_values)):
            col_idx = k % cols
            row_idx = k // cols
            xs.append(val)
            ys.append(row_idx + 1)

        ax.scatter(xs, ys, s=dot_size ** 1.5,
                   color=color, alpha=0.85, zorder=3,
                   edgecolors="white", linewidths=0.5)

        # Point estimate line
        ax.axvline(point_est, color=color, lw=2, ls="-",
                   label=f"Mean: {point_est:.0f}", zorder=4)

        # OECD average reference
        ax.axvline(oecd_mean, color="#888888", lw=1.5, ls="--",
                   label=oecd_label, zorder=2)

        ax.set_xlabel(f"{subject} mean score", fontsize=10)
        ax.set_yticks([])
        ax.set_title(cnt, fontsize=12, fontweight="600", color=color)
        ax.legend(fontsize=8.5, loc="upper left")
        ax.spines["left"].set_visible(False)

    year_str = str(year) if year else "all years"
    fig.suptitle(
        f"How certain are we about mean {subject} scores? ({year_str})\n"
        "Quantile dotplot - count dots to reason about uncertainty  "
        "[Kay et al. 2016]",
        fontsize=11, fontweight="500", y=1.02
    )
    plt.tight_layout()
    return fig


def plot_quantile_dotplot_shared_axis(df: pd.DataFrame, subject: str,
                                      countries: list,
                                      year: int = None,
                                      n_dots: int = 20) -> plt.Figure:
    """
    Quantile dotplot of the BRR distribution of mean scores per country,
    following Kay et al. (2016), with a horizontal boxplot overlay on
    the x-axis in the style of a raincloud plot.

    Dots are binned into discrete score columns and stacked vertically,
    forming a histogram-like mountain shape. A compact horizontal boxplot
    sits on the x-axis baseline to show median and IQR at a glance. 
    Both country panels share the same x-axis range so distributions 
    are directly comparable in position and spread.

    Parameters
    ----------
    df : pd.DataFrame
        Full PISA dataset containing BRR replicate weight columns
        ``W_FSTURWT1`` through ``W_FSTURWT80``.
    subject : str
        Subject code, e.g. ``"MATH"``, ``"READ"``, or ``"SCIE"``.
    countries : list of str
        Country codes to compare side by side, e.g. ``["CAN", "USA"]``.
    year : int, optional
        Filter to a single PISA cycle (2015, 2018, or 2022).
        ``None`` uses all available years.
    n_dots : int, optional
        Number of dots. Each dot represents ``100 / n_dots`` percent
        probability. 20 (5% per dot) is recommended. Defaults to 20.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with one panel per country showing a vertically stacked
        quantile dotplot above a horizontal boxplot baseline, with a
        shared x-axis range across all panels.
    """
    n_countries = len(countries)
    fig, axes = plt.subplots(
        1, n_countries,
        figsize=(4.5 * n_countries, 4.5),
        sharey=False
    )
    if n_countries == 1:
        axes = [axes]

    color_cycle = list(COUNTRY_COLORS.values()) + \
        [OKABE_ITO["green"], OKABE_ITO["orange"], OKABE_ITO["pink"]]

    DOT_PT = 7
    DOT_S  = DOT_PT ** 2 * 3.14

    # Pre-compute all BRR distributions and shared x-limits
    country_dists = {}
    all_vals = []
    for cnt in countries:
        subset = df[df["CNT"] == cnt]
        if year is not None and "YEAR" in df.columns:
            subset = subset[subset["YEAR"] == year]
        dist  = _brr_mean_distribution(subset, subject)
        valid = dist[~np.isnan(dist)]
        country_dists[cnt] = (dist, valid)
        if len(valid) >= 5:
            all_vals.extend(valid.tolist())

    if all_vals:
        global_pad = (np.max(all_vals) - np.min(all_vals)) * 0.12
        x_lo = np.min(all_vals) - global_pad
        x_hi = np.max(all_vals) + global_pad
    else:
        x_lo, x_hi = None, None

    # Draw each panel 
    for ax, cnt, color in zip(
        axes,
        countries,
        [color_cycle[i % len(color_cycle)] for i in range(n_countries)]
    ):
        dist, valid = country_dists[cnt]

        if len(valid) < 5:
            ax.text(0.5, 0.5, f"Insufficient data\nfor {cnt}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="gray")
            ax.axis("off")
            continue

        # Step 1: sample n_dots equally-spaced quantiles 
        quantile_probs = np.linspace(
            1 / (2 * n_dots),
            1 - 1 / (2 * n_dots),
            n_dots
        )
        dot_values = np.quantile(valid, quantile_probs)

        # Step 2: bin into ~sqrt(n_dots) wide columns 
        n_bins = max(int(np.round(np.sqrt(n_dots))), 3)
        bin_edges = np.linspace(dot_values.min(), dot_values.max(), n_bins + 1)
        bin_edges[0]  -= 1e-6
        bin_edges[-1] += 1e-6
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Step 3: stack dots vertically within each bin 
        bin_indices = np.clip(
            np.digitize(dot_values, bin_edges) - 1, 0, n_bins - 1
        )
        dot_x, dot_y = [], []
        bin_counts: dict = {}
        for b in bin_indices:
            count = bin_counts.get(b, 0)
            dot_x.append(bin_centers[b])
            dot_y.append(count + 1)
            bin_counts[b] = count + 1

        # Step 4: dot cloud
        ax.scatter(dot_x, dot_y,
                   s=DOT_S, color=color, alpha=0.88,
                   zorder=3, edgecolors="white", linewidths=0.4)

        # Step 5: horizontal boxplot on baseline (y=0)
        q1, med, q3 = np.percentile(valid, [25, 50, 75])
        iqr  = q3 - q1
        lo   = max(valid.min(), q1 - 1.5 * iqr)
        hi   = min(valid.max(), q3 + 1.5 * iqr)
        bh   = 0.35

        ax.plot([lo, q1], [0, 0], color=color, lw=1.2, zorder=4)
        ax.plot([q3, hi], [0, 0], color=color, lw=1.2, zorder=4)

        box = plt.Rectangle(
            (q1, -bh), q3 - q1, 2 * bh,
            facecolor=color, alpha=0.25,
            edgecolor=color, linewidth=1.2, zorder=4
        )
        ax.add_patch(box)

        ax.plot([med, med], [-bh, bh], color=color, lw=2, zorder=5)

        cap_h = bh * 0.6
        for x in [lo, hi]:
            ax.plot([x, x], [-cap_h, cap_h], color=color, lw=1.2, zorder=4)

        # Axis formatting 
        if x_lo is not None:
            ax.set_xlim(x_lo, x_hi)        # shared range

        ax.set_ylim(-1, max(dot_y) + 0.8)

        ax.axhline(0, color="#cccccc", lw=0.8, zorder=1)
        ax.set_xlabel(f"{subject} mean score", fontsize=10)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        # Consistent ticks across both panels 
        if x_lo is not None:
            tick_vals = np.linspace(x_lo, x_hi, 5)
            ax.set_xticks(tick_vals)
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{x:.0f}")
            )

        ax.set_title(cnt, fontsize=13, fontweight="500", color=color, pad=8)

    year_str = str(year) if year else "all years"
    prob_per_dot = round(100 / n_dots, 1)
    fig.suptitle(
        f"How certain are we about mean {subject} scores? ({year_str})\n"
        f"Each dot = {prob_per_dot}% chance of this outcome  "
        "[Kay et al. 2016]",
        fontsize=10, fontweight="500", y=1.04
    )

    plt.tight_layout()

    # Force equal axes widths after layout
    if n_countries > 1:
        positions = [ax.get_position() for ax in axes]
        min_width = min(p.width for p in positions)
        for ax, pos in zip(axes, positions):
            ax.set_position([pos.x0, pos.y0, min_width, pos.height])

    return fig