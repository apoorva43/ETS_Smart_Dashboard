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


