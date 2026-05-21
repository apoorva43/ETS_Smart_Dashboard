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
import pandas as pd
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


def _check_sufficient_data(df, target_cols, cnt, min_n=100, msg="Insufficient data"):
    """
    Helper function to validate data sufficiency before plotting.
    
    Drops NaNs for the required columns (plus the survey weight column) and 
    checks if the remaining sample size meets the minimum threshold.
    
    Returns a tuple of (valid_data, error_fig).
    If data is sufficient, error_fig is None. 
    If data is insufficient, valid_data is None and error_fig contains a warning plot.
    """
    cols_to_check = list(set(target_cols + ["W_FSTUWT"]))
    valid_data = df.dropna(subset=cols_to_check)
    
    if len(valid_data) < min_n:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.text(0.5, 0.5, msg, 
                ha='center', va='center', fontsize=12, color='gray')
        ax.axis('off')
        return None, fig
        
    return valid_data, None


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
        Defaults to True.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure containing the percentile line plot.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, cnt in enumerate(countries):
        subset = df[df["CNT"] == cnt]
        if year is not None and "YEAR" in df.columns:
            subset = subset[subset["YEAR"] == year]
        percs = weighted_percentiles_pv(subset, subject, PERCENTILES_COARSE)
        if np.isnan(percs).all():
            continue
        color = COUNTRY_COLORS.get(cnt, PALETTE[i % len(PALETTE)])
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


def plot_escs_gap(df, subject, cnt, year=None):
    """
    Create a socioeconomic status (ESCS) gap comparison plot across score percentiles.
    
    Splits the student population of a given country into four equal quartiles
    based on their ESCS index, and plots the score distribution for each.

    Parameters
    ----------
    df : pandas.DataFrame
        PISA dataset containing the ESCS index, weights, plausible value 
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
    matplotlib.figure.Figure
        Matplotlib figure containing the ESCS quartile comparison curves.
    """
    subset = df[df["CNT"] == cnt].copy()
    if year is not None:
        subset = subset[subset["YEAR"] == year]
        
    valid_data, error_fig = _check_sufficient_data(
        subset, ["ESCS"], cnt, 
        msg=f"Insufficient ESCS data for {cnt}"
    )
    if error_fig is not None:
        return error_fig

    fig, ax = plt.subplots(figsize=(9, 5))

    curves = compute_escs_quartile_percentiles(valid_data, subject,
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
    Create a NAEP-style gender comparison plot across score percentiles.

    Female student scores are used as the reference distribution on the 
    x-axis, while male student scores are plotted on the y-axis. The 
    diagonal reference line represents equal performance between genders.

    Curves above the diagonal indicate higher male performance at a given
    percentile, while curves below indicate higher female performance.

    Parameters
    ----------
    df : pandas.DataFrame
        PISA dataset containing gender identifiers, weights, plausible
        value score columns, and optionally a year column.
    subject : str
        Subject code used to select plausible value columns. Expected values
        include ``"MATH"``, ``"READ"``, and ``"SCIE"``.
    cnt : str
        Country code to filter the data, such as ``"CAN"`` or ``"USA"``.
    year : int, optional
        PISA cycle year to filter by. If ``None``, all available years are 
        used.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure containing the gender percentile comparison plot.
    """
    subset = df[df["CNT"] == cnt].copy()
    if year is not None and "YEAR" in df.columns:
        subset = subset[subset["YEAR"] == year]

    valid_data, error_fig = _check_sufficient_data(
        subset, ["ST004D01T"], cnt, 
        msg=f"Insufficient gender data for {cnt}"
    )
    if error_fig is not None:
        return error_fig

    fig, ax = plt.subplots(figsize=(9, 5))

    female = valid_data[valid_data["ST004D01T"] == 1.0]
    male   = valid_data[valid_data["ST004D01T"] == 2.0]

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
    Plot percentile score distributions for arbitrary demographic or
    contextual groups.

    This function generalizes percentile comparison plots across grouping
    variables such as gender, immigration status, school type, or school 
    location.

    Parameters
    ----------
    df : pandas.DataFrame
        PISA dataset containing grouping variables, weights, plausible value
        score columns, and optionally a year column.
    subject : str
        Subject code used to select plausible value columns. Expected values
        include ``"MATH"``, ``"READ"``, and ``"SCIE"``.
    group_col : str
        Column name containing the grouping variable.
    group_vals : dict
        Mapping from raw coded values to readable group labels.
    cnt : str
        Country code to filter the data, such as ``"CAN"`` or ``"USA"``.
    year : int, optional
        PISA cycle year to filter by. If ``None``, all available years are
        used.
    title : str, optional
        Custom plot title. If empty, a default title is generated.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure containing percentile comparison curves by group.
    """
    subset = df.copy()
    if cnt is not None:
        subset = subset[subset["CNT"] == cnt]
    if year is not None:
        subset = subset[subset["YEAR"] == year]

    valid_data, error_fig = _check_sufficient_data(
        subset, [group_col], cnt, 
        msg=f"Insufficient data to group by {group_col} for {cnt}"
    )
    if error_fig is not None:
        return error_fig

    fig, ax = plt.subplots(figsize=(9, 5))

    curves = compute_group_percentiles(valid_data, subject, group_col,
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
    Create a NAEP-style time comparison plot across score percentiles.

    Percentile scores from a reference year are placed on the x-axis, while
    percentile scores from comparison years are plotted on the y-axis. The
    diagonal reference line represents no change relative to the reference
    year.

    Curves above the diagonal indicate score improvements relative to the
    reference year, while curves below indicate declines.

    Parameters
    ----------
    df : pandas.DataFrame
        PISA dataset containing country identifiers, weights, plausible
        value score columns, and a year column.
    subject : str
        Subject code used to select plausible value columns. Expected values
        include ``"MATH"``, ``"READ"``, and ``"SCIE"``.
    cnt : str
        Country code to filter the data, such as ``"CAN"`` or ``"USA"``.
    reference_year : int
        Baseline PISA cycle year used as the x-axis reference distribution.
    comparison_years : list of int
        Additional PISA cycle years to compare against the reference year.
    group_col : str, optional
        Column name for subgroup filtering, such as gender or immigration
        status.
    group_val : float, optional
        Specific subgroup value used for filtering.
    group_label : str, optional
        Human-readable subgroup label used in the plot title.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure containing percentile-based time comparison curves.    
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


def plot_weighted_interval_distribution(df, subject: str,
                                        countries: list,
                                        year: int = None,
                                        interval_width: int = 20,
                                        score_range: tuple = (0, 1000),
                                        show_oecd: bool = True) -> plt.Figure:
    """
    Plot weighted score interval proportions averaged across all 10 plausible
    values, following the PISA-recommended approach for distribution plots.

    For each country and each plausible value, students are binned into equal
    score intervals of ``interval_width`` points. The weighted proportion of
    students falling in each interval is computed, then averaged across all 10
    PVs. Proportions are plotted against the midpoint of each interval.

    Parameters
    ----------
    df : pandas.DataFrame
        PISA dataset containing country identifiers, weights, and plausible
        value score columns.
    subject : str
        Subject code used to select plausible value columns. Expected values
        include ``"MATH"``, ``"READ"``, and ``"SCIE"``.
    countries : list of str
        Country codes to include in the plot, such as ``["CAN", "USA"]``.
    year : int, optional
        PISA cycle year to filter by. If ``None``, all available years are used.
    interval_width : int, optional
        Width of each score bin in points. Defaults to 20.
    score_range : tuple of (int, int), optional
        (min, max) of the score scale. Defaults to (0, 1000).
    show_oecd : bool, optional
        Whether to include an OECD average curve as a reference line.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure containing the weighted interval distribution plot.
    """
    pv_cols = [f"PV{i}{subject}" for i in range(1, 11)]
    pv_cols = [c for c in pv_cols if c in df.columns]

    bins = np.arange(score_range[0], score_range[1] +
                     interval_width, interval_width)
    midpoints = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots(figsize=(10, 5))
    color_cycle = list(COUNTRY_COLORS.values()) + \
        ["#1D9E75", "#BA7517", "#888780"]

    def _country_proportions(subset):
        """Average weighted proportions across all PVs for one subset."""
        pv_props = []
        w = subset["W_FSTUWT"].values
        total_w = w.sum()
        if total_w == 0:
            return np.full(len(midpoints), np.nan)
        for pv in pv_cols:
            scores = subset[pv].values
            props = np.array([
                w[(scores >= bins[i]) & (scores < bins[i + 1])].sum() / total_w
                for i in range(len(bins) - 1)
            ])
            pv_props.append(props)
        return np.mean(pv_props, axis=0)

    for i, cnt in enumerate(countries):
        subset = df[df["CNT"] == cnt].dropna(subset=pv_cols + ["W_FSTUWT"])
        if year is not None and "YEAR" in df.columns:
            subset = subset[subset["YEAR"] == year]
        props = _country_proportions(subset)
        color = COUNTRY_COLORS.get(cnt, color_cycle[i % len(color_cycle)])
        ax.plot(midpoints, props, color=color,
                lw=2.5, marker="o", ms=4, label=cnt)

    if show_oecd:
        all_subset = df[df["CNT"].isin(df["CNT"].unique())].dropna(
            subset=pv_cols + ["W_FSTUWT"])
        if year is not None and "YEAR" in df.columns:
            all_subset = all_subset[all_subset["YEAR"] == year]
        oecd_props = _country_proportions(all_subset)
        if not np.isnan(oecd_props).all():
            ax.plot(midpoints, oecd_props,
                    color="black", lw=2, ls="--", label="OECD avg", zorder=5)

    ax.set_xlabel("Score", fontsize=10)
    ax.set_ylabel("Weighted proportion of students", fontsize=10)
    ax.set_title(f"Score distribution – {subject}\n"
                 f"(weighted intervals, averaged across 10 PVs)",
                 fontsize=12, fontweight="500")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_gender_diff_percentile(df, subject: str, cnt: str,
                                year: int = None) -> plt.Figure:
    """
    Plot the gender score gap (Male minus Female) on the y-axis against the
    female reference percentile distribution on the x-axis.

    This is a modified version of the NAEP-style gender comparison: instead of
    plotting male scores against the female x-axis, the y-axis shows the raw
    score *difference* (Male − Female) at each matched percentile. The
    horizontal zero line represents no gap. Values above zero indicate males
    score higher at that percentile; values below indicate females score higher.

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
    matplotlib.figure.Figure
        Matplotlib figure showing gender score difference across percentiles.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    subset = df[df["CNT"] == cnt]
    if year is not None and "YEAR" in df.columns:
        subset = subset[subset["YEAR"] == year]

    female = subset[subset["ST004D01T"] == 1.0]
    male = subset[subset["ST004D01T"] == 2.0]

    female_percs = weighted_percentiles_pv(female, subject, PERCENTILES_FINE)
    male_percs = weighted_percentiles_pv(male,   subject, PERCENTILES_FINE)

    if np.isnan(female_percs).all() or np.isnan(male_percs).all():
        return fig

    diff = male_percs - female_percs

    # Zero reference line
    ax.axhline(0, color="#cccccc", lw=1.5, ls=":", zorder=0, label="No gap")

    # Fill above/below zero for easy reading
    ax.fill_between(female_percs, diff, 0,
                    where=(diff >= 0), alpha=0.15,
                    color=COUNTRY_COLORS.get(cnt, "#185FA5"),
                    label="Male advantage")
    ax.fill_between(female_percs, diff, 0,
                    where=(diff < 0), alpha=0.15,
                    color=OKABE_ITO["pink"],
                    label="Female advantage")

    ax.plot(female_percs, diff,
            color=COUNTRY_COLORS.get(cnt, "#185FA5"),
            lw=2.5, zorder=5)

    # Percentile tick labels on top axis
    perc_label_positions = weighted_percentiles_pv(female, subject,
                                                   [10, 25, 50, 75, 90])
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(perc_label_positions)
    ax2.set_xticklabels(["P10", "P25", "P50", "P75", "P90"], fontsize=8)
    ax2.set_xlabel("Female percentile", fontsize=9)

    ax.set_xlabel(f"Female {subject} score (reference)", fontsize=10)
    ax.set_ylabel("Score difference (Male − Female)", fontsize=10)
    ax.set_title(
        f"Gender gap – {subject} – {cnt}\n"
        "(above zero = males score higher at that percentile)",
        fontsize=12, fontweight="500"
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_belonging_by_immigration(df,
                                countries: list,
                                year: int = None,
                                min_group_n: int = 30,
                                repeat_col: str = "REPEAT",
                                escs_col: str = "ESCS",
                                belonging_col: str = "BELONG",
                                immig_col: str = "IMMIG",
                                weight_col: str = "W_FSTUWT") -> plt.Figure:
    """
    Plot contextual comparisons for selected countries.

    This function creates a two-panel comparison plot:
    1. Grade repetition rate by SES quartile.
    2. Weighted mean school belonging by immigration status.

    Countries are shown as side-by-side bars within each category. The function
    supports any number of selected countries, although the plot is easiest to
    read with two to four countries.

    Parameters
    ----------
    df : pandas.DataFrame
        PISA dataset containing country identifiers, ESCS, repeat status,
        school belonging, immigration status, and sampling weights.
    countries : list of str
        Country codes to compare, such as ``["CAN", "USA"]``.
    year : int, optional
        PISA cycle year to filter by. If ``None``, all available years are used.
    min_group_n : int, optional
        Minimum number of valid observations required to show a bar.
    repeat_col : str, optional
        Column indicating grade repetition. In PISA coding, ``1`` means yes
        and ``2`` means no.
    escs_col : str, optional
        Column containing the socioeconomic status index.
    belonging_col : str, optional
        Column containing the school belonging index.
    immig_col : str, optional
        Column containing immigration status.
    weight_col : str, optional
        Column containing student sampling weights.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure with two side-by-side bar charts.
    """
    countries = list(countries)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    if len(countries) == 0:
        for ax in axes:
            ax.text(
                0.5, 0.5,
                "Please select at least one country.",
                ha="center",
                va="center",
                transform=ax.transAxes
            )
        return fig

    subset = df[df["CNT"].isin(countries)].copy()

    if year is not None and "YEAR" in subset.columns:
        subset = subset[subset["YEAR"] == year]

    colors = {
        cnt: COUNTRY_COLORS.get(cnt, PALETTE[i % len(PALETTE)])
        for i, cnt in enumerate(countries)
    }

    n_countries = len(countries)
    width = min(0.8 / n_countries, 0.35)

    def _weighted_mean(values, weights):
        valid = np.isfinite(values) & np.isfinite(weights)
        if valid.sum() == 0:
            return np.nan
        return np.average(values[valid], weights=weights[valid])

    def _weighted_rate(condition, weights):
        valid = np.isfinite(weights)
        if valid.sum() == 0:
            return np.nan
        return np.average(condition[valid], weights=weights[valid]) * 100

    # ------------------------------------------------------------------
    # Left panel: Grade repetition rate by SES quartile
    # ------------------------------------------------------------------
    ax = axes[0]

    required_rep_cols = [repeat_col, escs_col, "CNT", weight_col]
    if all(c in subset.columns for c in required_rep_cols):
        df_rep = subset.dropna(
            subset=[repeat_col, escs_col, weight_col]).copy()

        df_rep["ESCS_Q"] = pd.qcut(
            df_rep[escs_col],
            q=4,
            labels=["Q1\n(low SES)", "Q2", "Q3", "Q4\n(high SES)"],
            duplicates="drop"
        )

        q_tick_labels = ["Q1\n(low SES)", "Q2", "Q3", "Q4\n(high SES)"]
        x = np.arange(len(q_tick_labels))

        for i, cnt in enumerate(countries):
            rates = []

            for q in q_tick_labels:
                sub = df_rep[
                    (df_rep["ESCS_Q"] == q) &
                    (df_rep["CNT"] == cnt)
                ]

                if len(sub) < min_group_n:
                    rates.append(np.nan)
                else:
                    rate = _weighted_rate(
                        (sub[repeat_col] == 1).to_numpy(),
                        sub[weight_col].to_numpy()
                    )
                    rates.append(rate)

            offset = (i - (n_countries - 1) / 2) * width

            bars = ax.bar(
                x + offset,
                rates,
                width,
                label=cnt,
                color=colors[cnt],
                alpha=0.88
            )

            ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(q_tick_labels, fontsize=9)
        ax.set_ylabel("Grade repetition rate (%)", fontsize=10)
        ax.set_title(
            "Grade repetition by SES quartile\n"
            "(low-SES students repeat at higher rates)",
            fontsize=11,
            fontweight="500"
        )
        ax.legend(fontsize=9)

    else:
        missing = [c for c in required_rep_cols if c not in subset.columns]
        ax.text(
            0.5, 0.5,
            f"Missing required column(s): {', '.join(missing)}",
            ha="center",
            va="center",
            transform=ax.transAxes
        )

    # ------------------------------------------------------------------
    # Right panel: School belonging by immigration status
    # ------------------------------------------------------------------
    ax = axes[1]

    required_bel_cols = [belonging_col, immig_col, "CNT", weight_col]
    if all(c in subset.columns for c in required_bel_cols):
        df_bel = subset.dropna(
            subset=[belonging_col, immig_col, weight_col]
        ).copy()

        x = np.arange(len(IMMIG_MAP))

        for i, cnt in enumerate(countries):
            means = []

            for code in IMMIG_MAP:
                sub = df_bel[
                    (df_bel[immig_col] == code) &
                    (df_bel["CNT"] == cnt)
                ]

                if len(sub) < min_group_n:
                    means.append(np.nan)
                else:
                    mean = _weighted_mean(
                        sub[belonging_col].to_numpy(),
                        sub[weight_col].to_numpy()
                    )
                    means.append(mean)

            offset = (i - (n_countries - 1) / 2) * width

            bars = ax.bar(
                x + offset,
                means,
                width,
                label=cnt,
                color=colors[cnt],
                alpha=0.88
            )

            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(list(IMMIG_MAP.values()), fontsize=9)
        ax.set_ylabel("Weighted mean BELONG index", fontsize=10)
        ax.set_title(
            "School belonging by immigration status\n"
            "(higher = greater sense of belonging)",
            fontsize=11,
            fontweight="500"
        )
        ax.legend(fontsize=9)

    else:
        missing = [c for c in required_bel_cols if c not in subset.columns]
        ax.text(
            0.5, 0.5,
            f"Missing required column(s): {', '.join(missing)}",
            ha="center",
            va="center",
            transform=ax.transAxes
        )

    plt.tight_layout()
    return fig


def plot_immigration_score_distribution(df, subject: str,
                                        cnt: str,
                                        year: int = None,
                                        interval_width: int = 20,
                                        score_range: tuple = (0, 1000)) -> plt.Figure:
    """
    Plot weighted score interval distributions by immigration status.

    Within a selected country and subject, students are split by immigration
    status. For each immigration group and each plausible value, students are
    binned into equal score intervals. The weighted proportion of students in
    each interval is computed and then averaged across all ten plausible
    values.

    Parameters
    ----------
    df : pandas.DataFrame
        PISA dataset containing country identifiers, immigration status,
        sampling weights, and plausible value score columns.
    subject : str
        Subject code used to select plausible value columns. Expected values
        include ``"MATH"``, ``"READ"``, and ``"SCIE"``.
    cnt : str
        Country code to filter the data, such as ``"CAN"`` or ``"USA"``.
    year : int, optional
        PISA cycle year to filter by. If ``None``, all available years are used.
    interval_width : int, optional
        Width of each score bin in points. Defaults to 20.
    score_range : tuple of (int, int), optional
        Minimum and maximum score values used to define score intervals.
        Defaults to ``(0, 1000)``.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure containing weighted score distribution curves by
        immigration status.
    """
    pv_cols = [f"PV{i}{subject}" for i in range(1, 11)]
    pv_cols = [c for c in pv_cols if c in df.columns]

    bins = np.arange(score_range[0], score_range[1] + interval_width,
                     interval_width)
    midpoints = (bins[:-1] + bins[1:]) / 2


    fig, ax = plt.subplots(figsize=(10, 5))

    subset = df[df["CNT"] == cnt].copy()
    if year is not None and "YEAR" in subset.columns:
        subset = subset[subset["YEAR"] == year]

    def _group_proportions(group_df):
        """Average weighted interval proportions across all PVs for one group."""
        group_df = group_df.dropna(subset=pv_cols + ["W_FSTUWT"])

        if group_df.empty:
            return np.full(len(midpoints), np.nan)

        w = group_df["W_FSTUWT"].values
        total_w = w.sum()

        if total_w == 0:
            return np.full(len(midpoints), np.nan)

        pv_props = []

        for pv in pv_cols:
            scores = group_df[pv].values

            props = np.array([
                w[(scores >= bins[i]) & (scores < bins[i + 1])].sum() / total_w
                for i in range(len(bins) - 1)
            ])

            pv_props.append(props)

        return np.mean(pv_props, axis=0)

    for color, (code, label) in zip(PALETTE, IMMIG_MAP.items()):
        group_df = subset[subset["IMMIG"] == code]
        props = _group_proportions(group_df)

        if np.isnan(props).all():
            continue

        ax.plot(midpoints, props,
                color=color,
                lw=2.5,
                marker="o",
                ms=4,
                label=label)

    ax.set_xlabel("Score", fontsize=10)
    ax.set_ylabel("Weighted proportion of students", fontsize=10)
    ax.set_title(
        f"Score distribution by immigration status – {subject} – {cnt}\n"
        "(weighted intervals, averaged across 10 PVs)",
        fontsize=12,
        fontweight="500"
    )
    ax.legend(title="Immigration status", fontsize=9)
    plt.tight_layout()
    return fig


def plot_school_location_boxplot(df,
                                 subject: str,
                                 cnt: str,
                                 year: int = None,
                                 location_col: str = "SC001Q01TA",
                                 min_group_n: int = 30) -> plt.Figure:
    """
    Plot weighted score distribution by school location/community type.

    This function creates a weighted boxplot-style chart using weighted
    percentiles averaged across all plausible values. For each school location
    group, the box shows P25 to P75, the center line shows the median, and the
    whiskers show P10 to P90.

    Parameters
    ----------
    df : pandas.DataFrame
        PISA dataset containing country identifiers, school location,
        sampling weights, and plausible value score columns.
    subject : str
        Subject code used to select plausible value columns. Expected values
        include ``"MATH"``, ``"READ"``, and ``"SCIE"``.
    cnt : str
        Country code to filter the data, such as ``"CAN"`` or ``"USA"``.
    year : int, optional
        PISA cycle year to filter by. If ``None``, all available years are used.
    location_col : str, optional
        Column describing school community/location type. Defaults to
        ``"SC001Q01TA"``.
    min_group_n : int, optional
        Minimum number of observations required to show a group.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure containing weighted boxplots by school location.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    required_cols = [location_col, "CNT", "W_FSTUWT"]
    pv_cols = [f"PV{i}{subject}" for i in range(1, 11)]
    pv_cols = [c for c in pv_cols if c in df.columns]

    missing = [c for c in required_cols if c not in df.columns]
    if missing or not pv_cols:
        ax.text(
            0.5, 0.5,
            f"Missing required column(s): {', '.join(missing)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    subset = df[df["CNT"] == cnt].copy()

    if year is not None and "YEAR" in subset.columns:
        subset = subset[subset["YEAR"] == year]

    stats = []
    colors = []

    for i, (code, label) in enumerate(LOC_MAP.items()):
        group = subset[subset[location_col] == code]

        if len(group.dropna(subset=["W_FSTUWT"])) < min_group_n:
            continue

        percs = weighted_percentiles_pv(
            group,
            subject,
            [10, 25, 50, 75, 90]
        )

        if np.isnan(percs).all():
            continue

        stats.append({
            "label": label,
            "whislo": percs[0],
            "q1": percs[1],
            "med": percs[2],
            "q3": percs[3],
            "whishi": percs[4],
            "fliers": [],
        })

        colors.append(PALETTE[i % len(PALETTE)])

    if not stats:
        ax.text(
            0.5, 0.5,
            "Insufficient data to plot school location groups.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    box = ax.bxp(
        stats,
        showfliers=False,
        patch_artist=True,
        widths=0.55
    )

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(1.8)

    ax.set_ylabel(f"{subject} score", fontsize=10)
    ax.set_xlabel("School community type", fontsize=10)
    ax.set_title(
        f"Score distribution by school location – {subject} – {cnt}\n"
        "(weighted P10–P90, averaged across 10 plausible values)",
        fontsize=12,
        fontweight="500"
    )

    ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout()
    return fig


def plot_school_type_distribution(df,
                                  subject: str,
                                  cnt: str,
                                  year: int = None,
                                  school_type_col: str = "SCHLTYPE",
                                  interval_width: int = 20,
                                  score_range: tuple = (0, 1000),
                                  min_group_n: int = 30) -> plt.Figure:
    """
    Plot weighted score distributions by school type.

    Students are grouped by school type. For each group and each plausible
    value, students are binned into equal score intervals. The weighted
    proportion of students in each interval is computed and then averaged
    across all ten plausible values.

    Parameters
    ----------
    df : pandas.DataFrame
        PISA dataset containing country identifiers, school type, sampling
        weights, and plausible value score columns.
    subject : str
        Subject code used to select plausible value columns. Expected values
        include ``"MATH"``, ``"READ"``, and ``"SCIE"``.
    cnt : str
        Country code to filter the data, such as ``"CAN"`` or ``"USA"``.
    year : int, optional
        PISA cycle year to filter by. If ``None``, all available years are used.
    school_type_col : str, optional
        Column describing school type. Defaults to ``"SCHLTYPE"``.
    interval_width : int, optional
        Width of each score interval.
    score_range : tuple of (int, int), optional
        Minimum and maximum score values used to define score intervals.
    min_group_n : int, optional
        Minimum number of observations required to show a group.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure containing weighted score distribution curves by
        school type.
    """

    pv_cols = [f"PV{i}{subject}" for i in range(1, 11)]
    pv_cols = [c for c in pv_cols if c in df.columns]

    bins = np.arange(
        score_range[0],
        score_range[1] + interval_width,
        interval_width
    )
    midpoints = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots(figsize=(10, 5))

    required_cols = [school_type_col, "CNT", "W_FSTUWT"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing or not pv_cols:
        ax.text(
            0.5, 0.5,
            f"Missing required column(s): {', '.join(missing)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    subset = df[df["CNT"] == cnt].copy()

    if year is not None and "YEAR" in subset.columns:
        subset = subset[subset["YEAR"] == year]

    def _group_proportions(group_df):
        group_df = group_df.dropna(subset=pv_cols + ["W_FSTUWT"])

        if len(group_df) < min_group_n:
            return np.full(len(midpoints), np.nan)

        weights = group_df["W_FSTUWT"].to_numpy()
        total_weight = weights.sum()

        if total_weight == 0:
            return np.full(len(midpoints), np.nan)

        pv_props = []

        for pv in pv_cols:
            scores = group_df[pv].to_numpy()

            props = np.array([
                weights[(scores >= bins[i]) & (scores < bins[i + 1])].sum()
                / total_weight
                for i in range(len(bins) - 1)
            ])

            pv_props.append(props)

        return np.mean(pv_props, axis=0)

    for color, (code, label) in zip(PALETTE, SCHLTYPE_MAP.items()):
        group = subset[subset[school_type_col] == code]
        props = _group_proportions(group)

        if np.isnan(props).all():
            continue

        ax.plot(
            midpoints,
            props,
            color=color,
            lw=2.5,
            marker="o",
            ms=4,
            label=label,
        )

    ax.set_xlabel(f"{subject} score", fontsize=10)
    ax.set_ylabel("Weighted proportion of students", fontsize=10)
    ax.set_title(
        f"Score distribution by school type – {subject} – {cnt}\n"
        "(weighted intervals, averaged across 10 plausible values)",
        fontsize=12,
        fontweight="500"
    )

    ax.legend(title="School type", fontsize=9)
    plt.tight_layout()
    return fig
