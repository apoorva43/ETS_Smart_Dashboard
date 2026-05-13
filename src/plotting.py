#plotting.py
"""
Plotting utilities for the PISA dashboard.

This module contains reusable functions for visualizing weighted
PISA score distributions. 

Functions include country-level distribution comparisons and socioeconomic
status (ESCS) quartile comparisons.

"""

import numpy as np
import matplotlib.pyplot as plt
from src.config import (PERCENTILES_COARSE,
                        COUNTRY_COLORS, PALETTE)
from src.pisa_stats import (weighted_percentiles_pv,
                            compute_escs_quartile_percentiles,
                            get_oecd_percentiles)

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


def _base_percentile_ax(ax, percentiles=PERCENTILES_COARSE):
    """
    Apply standard axis formatting for percentile-based score plots.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axis object to format.
    percentiles : list of int, optional
        Percentile values to display on the x-axis. Defaults to
        ``PERCENTILES_COARSE``.

    Returns
    -------
    None
        The function modifies the provided axis in place.
    """
    ax.set_xticks(percentiles)
    ax.set_xlabel("Percentile", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)


def plot_country_distributions(df, subject: str,
                               countries: list,
                               year: int = None,
                               show_oecd: bool = True) -> plt.Figure:
    """
    For each selected country, this function computes weighted percentiles
    averaged across all plausible values for the chosen subject. The resulting
    percentile curves show how student performance varies across the score
    distribution.

    Parameters
    ----------
    df : pandas.DataFrame
        PISA dataset containing country identifiers, weights, plausible values,
        and optionally a year column.
    subject : str
        Subject code used to select plausible value columns. Expected values
        include ``"MATH"``, ``"READ"``, and ``"SCIE"``.
    countries : list of str
        Country codes to include in the plot, such as ``["CAN", "USA"]``.
    year : int, optional
        PISA cycle year to filter by. If ``None``, all available years are used.
    show_oecd : bool, optional
        Whether to include an OECD average percentile curve as a reference line.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure containing the percentile line plot.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    color_cycle = list(COUNTRY_COLORS.values()) + \
        ["#1D9E75", "#BA7517", "#888780"]

    for i, cnt in enumerate(countries):
        subset = df[df["CNT"] == cnt]
        if year is not None and "YEAR" in df.columns:
            subset = subset[df["YEAR"] == year]
        percs = weighted_percentiles_pv(subset, subject, PERCENTILES_COARSE)
        if np.isnan(percs).all():
            continue
        color = COUNTRY_COLORS.get(cnt, color_cycle[i % len(color_cycle)])
        ax.plot(PERCENTILES_COARSE, percs,
                color=color, lw=2.5, marker="o", ms=5, label=cnt)

    if show_oecd:
        oecd = get_oecd_percentiles(df, subject, PERCENTILES_COARSE, year)
        if not np.isnan(oecd).all():
            ax.plot(PERCENTILES_COARSE, oecd,
                    color="black", lw=2, ls="--", label="OECD avg", zorder=5)

    _base_percentile_ax(ax)
    ax.set_title(
        f"Score distribution – {subject}", fontsize=12, fontweight="500")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_escs_gap(df, subject: str, cnt: str,
                  year: int = None) -> plt.Figure:
    """
    The function splits students into socioeconomic status quartiles using the
    ESCS index within the selected country and year, then plots weighted score
    percentile curves for each quartile.

    Parameters
    ----------
    df : pandas.DataFrame
        PISA dataset containing ESCS, country identifiers, weights, and
        plausible value score columns.
    subject : str
        Subject code used to select plausible value columns. Expected values
        include ``"MATH"``, ``"READ"``, and ``"SCIE"``.
    cnt : str
        Country code to filter the data, such as ``"CAN"`` or ``"USA"``.
    year : int, optional
        PISA cycle year to filter by. If ``None``, all available years are used.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure containing ESCS quartile percentile curves.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    curves = compute_escs_quartile_percentiles(df, subject,
                                               PERCENTILES_COARSE,
                                               cnt=cnt, year=year)
    for color, (label, percs) in zip(PALETTE, curves.items()):
        if np.isnan(percs).all():
            continue
        ax.plot(PERCENTILES_COARSE, percs,
                color=color, lw=2.5, marker="o", ms=5, label=label)

    _base_percentile_ax(ax)
    ax.set_title(f"SES gap – {subject} – {cnt}", fontsize=12, fontweight="500")
    ax.legend(title="ESCS quartile", fontsize=9)
    plt.tight_layout()
    return fig
