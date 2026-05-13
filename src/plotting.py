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
from src.config import (
    PERCENTILES_COARSE,
    PERCENTILES_FINE,
    COUNTRY_COLORS, 
    PALETTE,
    YEAR_COLORS
    )
from src.pisa_stats import (
    weighted_percentiles_pv,
    compute_escs_quartile_percentiles,
    compute_group_percentiles,
    get_oecd_percentiles
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

def plot_gender_percentile_line(df, subject: str, cnt: str,
                                 year: int = None) -> plt.Figure:
    """
    NAEP-style gender plot: x-axis = Female score at each percentile.
    Female is the reference group; Male line plotted against it.
    """
    from src.config import GENDER_MAP
    fig, ax = plt.subplots(figsize=(9, 5))

    subset = df[df["CNT"] == cnt]
    if year is not None and "YEAR" in df.columns:
        subset = subset[subset["YEAR"] == year]

    female = subset[subset["ST004D01T"] == 1.0]
    male   = subset[subset["ST004D01T"] == 2.0]

    female_percs = weighted_percentiles_pv(female, subject, PERCENTILES_FINE)
    male_percs   = weighted_percentiles_pv(male,   subject, PERCENTILES_FINE)

    if np.isnan(female_percs).all():
        return fig

    # Reference diagonal (Female = Female, i.e. no gap)
    ax.plot(female_percs, female_percs,
            color="#cccccc", lw=1.5, ls=":", label="Female (reference)", zorder=0)

    # Male line against Female x-axis
    ax.plot(female_percs, male_percs,
            color=COUNTRY_COLORS.get(cnt, "#185FA5"),
            lw=2.5, label="Male", zorder=5)

    # Percentile tick labels on top axis
    perc_label_positions = weighted_percentiles_pv(female, subject,
                                                    [10, 25, 50, 75, 90])
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(perc_label_positions)
    ax2.set_xticklabels(["P10", "P25", "P50", "P75", "P90"], fontsize=8)
    ax2.set_xlabel("Female percentile", fontsize=9)

    ax.set_xlabel(f"Female {subject} score (reference)", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(
        f"Gender gap – {subject} – {cnt}\n"
        "(above diagonal = males score higher at that percentile)",
        fontsize=12, fontweight="500"
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig

def plot_group_comparison(df, subject: str, group_col: str,
                           group_vals: dict, cnt: str,
                           year: int = None,
                           title: str = "") -> plt.Figure:
    """
    Percentile line plot broken down by any group variable.
    Works for gender, immigration, school type, etc.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    curves = compute_group_percentiles(df, subject, group_col,
                                        group_vals, PERCENTILES_COARSE,
                                        cnt=cnt, year=year)
    for color, (label, percs) in zip(PALETTE, curves.items()):
        if np.isnan(percs).all():
            continue
        ax.plot(PERCENTILES_COARSE, percs,
                color=color, lw=2.5, marker="o", ms=5, label=label)

    _base_percentile_ax(ax)
    ax.set_title(title or f"{subject} by group – {cnt}", fontsize=12, fontweight="500")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig

def plot_naep_time_comparison(df, subject: str, cnt: str,
                               reference_year: int,
                               comparison_years: list,
                               group_col: str = None,
                               group_val: float = None,
                               group_label: str = "All students") -> plt.Figure:
    """
    NAEP-style time comparison:
    x-axis = reference year scores; each other year plotted against it.
    Optionally filtered to one group (e.g. Female only).
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    all_years = [reference_year] + comparison_years

    def get_subset(year):
        s = df[(df["CNT"] == cnt) & (df["YEAR"] == year)]
        if group_col and group_val is not None:
            s = s[s[group_col] == group_val]
        return s

    ref_percs = weighted_percentiles_pv(get_subset(reference_year),
                                         subject, PERCENTILES_FINE)
    if np.isnan(ref_percs).all():
        return fig

    # Neutral diagonal
    ax.plot(ref_percs, ref_percs, color="#dddddd", lw=1, ls=":", zorder=0)

    for year in all_years:
        yr_percs = weighted_percentiles_pv(get_subset(year), subject, PERCENTILES_FINE)
        if np.isnan(yr_percs).all():
            print(f"  Skipping {year}: insufficient data")
            continue
        lw    = 3.0 if year == reference_year else 2.0
        ls    = "-"  if year == reference_year else "--"
        label = f"{year} (reference)" if year == reference_year else str(year)
        ax.plot(ref_percs, yr_percs,
                color=YEAR_COLORS.get(year, "#333333"),
                lw=lw, ls=ls, label=label, marker="o", ms=3)

    # Percentile annotation on top axis
    perc_ticks = weighted_percentiles_pv(get_subset(reference_year),
                                          subject, [10, 25, 50, 75, 90])
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(perc_ticks)
    ax2.set_xticklabels(["P10", "P25", "P50", "P75", "P90"], fontsize=8)
    ax2.set_xlabel(f"{reference_year} percentile", fontsize=9)

    ax.set_xlabel(f"{reference_year} {subject} score", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(
        f"Score change over time – {subject} – {cnt} – {group_label}\n"
        f"(x-axis = {reference_year} reference; above diagonal = improvement)",
        fontsize=11, fontweight="500"
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig