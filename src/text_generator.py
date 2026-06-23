#text_generator.py
"""
This module contains helper functions that generate short explanatory text
based on computed PISA statistics. These summaries are intended to accompany
dashboard visualizations and help users interpret results without relying only
on charts.

The generated text is designed to be concise, non-technical, and suitable for
policy-oriented dashboard users.
"""

import numpy as np
import pandas as pd
from src.pisa_stats import (
    weighted_mean_pv,
    weighted_percentiles_pv,
    compute_escs_quartile_percentiles,
    get_oecd_percentiles
    )
from src.config import (
    SUBJECTS,
    PERCENTILES_COARSE,
    COUNTRY_NAMES
    )

def _cnt_label(code: str) -> str:
    """
    Return full country name for a CNT code, falling back to the code itself.
    """
    return COUNTRY_NAMES.get(str(code), str(code))

def _oecd_mean_score(df, subject, year=None):
    pv_cols = [f"PV{i}{subject}" for i in range(1, 11) if f"PV{i}{subject}" in df.columns]
    oecd_df = df[df["OECD"] == 1].copy()
    if year is not None and "YEAR" in oecd_df.columns:
        oecd_df = oecd_df[oecd_df["YEAR"] == year]
    country_means = []
    for cnt in oecd_df["CNT"].unique():
        c = oecd_df[oecd_df["CNT"] == cnt].dropna(subset=["W_FSTUWT"] + pv_cols)
        if len(c) < 30:
            continue
        country_means.append(np.mean([
            np.average(c[pv].values, weights=c["W_FSTUWT"].values)
            for pv in pv_cols
        ]))
    return np.mean(country_means) if country_means else np.nan

def country_distribution_text(df, subject: str, countries: list, year: int = None) -> str:
    """
    Generates insight text comparing the primary country's score distribution
    to the OECD median. Makes a claim across the spectrum, not just the median.
    Avoids 'weighted', 'gap', and years-of-schooling framing in main text.
    """
    if not countries:
        return ""

    cnt = countries[0]
    subset = df[df["CNT"] == cnt]
    if year is not None and "YEAR" in subset.columns:
        subset = subset[subset["YEAR"] == year]

    cnt_percs = weighted_percentiles_pv(subset, subject, [10, 50, 90])
    pv_cols = [f"PV{i}{subject}" for i in range(1, 11) if f"PV{i}{subject}" in subset.columns]
    cnt_mean = np.mean([
        np.average(subset[pv].values, weights=subset["W_FSTUWT"].values)
        for pv in pv_cols if len(subset[pv].dropna()) > 0
    ])
    oecd_mean = _oecd_mean_score(df, subject, year=year)
    oecd_percs = get_oecd_percentiles(df, subject, [10, 50, 90], year=year)

    if np.isnan(cnt_percs).all() or np.isnan(oecd_mean) or np.isnan(cnt_mean):
        return "Insufficient data to compare scores."

    cnt_p10, cnt_p50, cnt_p90 = cnt_percs
    oecd_p10, oecd_p50, oecd_p90 = oecd_percs
    diff_mean = cnt_mean - oecd_mean
    diff_mean_abs = abs(diff_mean)
    diff_p10 = cnt_p10 - oecd_p10
    diff_p90 = cnt_p90 - oecd_p90
    subject_label = SUBJECTS.get(subject, subject)

    # Main sentence — country median vs OECD mean
    if diff_mean_abs < 3:
        main = (
            f"On average, students in {_cnt_label(cnt)} score in line with "
            f"the OECD average in {subject_label}."
        )
        direction_general = "in line with"
    else:
        direction = "above" if diff_mean > 0 else "below"
        main = (
            f"On average, students in {_cnt_label(cnt)} score "
            f"{diff_mean_abs:.0f} points {direction} the OECD average "
            f"in {subject_label}."
        )
        direction_general = direction

    # Spectrum sentence
    diff_spread = abs(diff_p90 - diff_p10)
    if not np.isnan(oecd_percs).all() and diff_spread > 15:
        if abs(diff_p10) < abs(diff_p90):
            spectrum = (
                f" The difference is larger at the upper end of the distribution "
                f"({diff_p90:+.0f} pts at P90) than at the lower end "
                f"({diff_p10:+.0f} pts at P10)."
            )
        else:
            spectrum = (
                f" The difference is larger at the lower end of the distribution "
                f"({diff_p10:+.0f} pts at P10) than at the upper end "
                f"({diff_p90:+.0f} pts at P90)."
            )
    else:
        if direction_general == "in line with":
            spectrum = (
                " This pattern is consistent across the distribution — "
                "from the lower end to the upper end, students score similarly to the OECD average."
            )
        else:
            spectrum = (
                f" This pattern is consistent across the distribution — "
                f"from the lower end to the upper end, students score similarly "
                f"{direction_general} the OECD average."
            )

    return main + spectrum


def gender_gap_text(df, subject, cnt, year=None):
    """
    Generate interpretive summary text for the gender percentile comparisons.

    The generated text summarizes median gender differences and whether the
    gap changes across the score distribution.

    Parameters
    ----------
    df : pandas.DataFrame
        PISA dataset containing gender identifiers, weights, and plausible
        value score columns.
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
    str
        Concise interpretation of the gender achievement gap across the
        score distribution.
    """
    subset = df[df["CNT"] == cnt]
    if year and "YEAR" in df.columns:
        subset = subset[subset["YEAR"] == year]

    valid_data = subset.dropna(subset=["ST004D01T", "W_FSTUWT"])
    if len(valid_data) < 30:
        return f"Insufficient data to compute gender difference for {_cnt_label(cnt)}."

    female = valid_data[valid_data["ST004D01T"] == 1.0]
    male   = valid_data[valid_data["ST004D01T"] == 2.0]

    f_percs = weighted_percentiles_pv(female, subject, PERCENTILES_COARSE)
    m_percs = weighted_percentiles_pv(male,   subject, PERCENTILES_COARSE)

    if np.isnan(f_percs).all() or np.isnan(m_percs).all():
        return f"Insufficient data to compute gender difference for {_cnt_label(cnt)}."

    diffs = m_percs - f_percs
    p10_diff, median_diff, p90_diff = diffs[0], diffs[2], diffs[-1]
    subject_label = SUBJECTS[subject]

    if abs(median_diff) < 5:
        overall = f"At the median, male and female students in {_cnt_label(cnt)} score similarly in {subject_label} ({median_diff:+.0f} points)."
    elif median_diff > 0:
        overall = f"Male students in {_cnt_label(cnt)} score {median_diff:.0f} points higher than female students at the median in {subject_label}."
    else:
        overall = f"Female students in {_cnt_label(cnt)} score {abs(median_diff):.0f} points higher than male students at the median in {subject_label}."

    if abs(p90_diff - p10_diff) > 10:
        spread = (
            f"The difference widens toward the upper end of the distribution: "
            f"{p10_diff:+.0f} pts at P10 vs {p90_diff:+.0f} pts at P90. "
            f"This pattern is invisible in average-only reporting."
        )
    else:
        spread = f"The difference is relatively consistent across the distribution ({p10_diff:+.0f} pts at P10, {p90_diff:+.0f} pts at P90)."

    return f"{overall} {spread}"


def ses_difference_text(df, subject: str, cnt: str, year: int = None) -> str:
    """
    Generates insight text for the SES chart.
    Makes a claim across the spectrum, not just the median.
    Avoids 'gap', 'weighted', and ESCS jargon in user-facing text.
    """
    curves = compute_escs_quartile_percentiles(
        df, subject, [10, 50, 90], cnt=cnt, year=year
    )

    q1_p10, q1_p50, q1_p90 = curves.get("Q1 (low SES)",  [np.nan, np.nan, np.nan])
    q4_p10, q4_p50, q4_p90 = curves.get("Q4 (high SES)", [np.nan, np.nan, np.nan])

    if np.isnan(q1_p50) or np.isnan(q4_p50):
        return "Insufficient data to compute socioeconomic differences."

    diff_p50 = q4_p50 - q1_p50
    diff_p10 = q4_p10 - q1_p10
    diff_p90 = q4_p90 - q1_p90
    
    subject_label = SUBJECTS[subject]
    
    # Use absolute values
    diff_p50_abs = abs(diff_p50)
    direction_p50 = "higher" if diff_p50 >= 0 else "lower"

    # Main sentence
    main = (
        f"In {_cnt_label(cnt)}, students from the highest socioeconomic backgrounds "
        f"score {diff_p50_abs:.0f} points {direction_p50} than those from the lowest backgrounds "
        f"at the median in {subject_label}."
    )

    # Spectrum — does the difference widen or narrow across the distribution?
    if not (np.isnan(diff_p10) or np.isnan(diff_p90)):
        # Compare the absolute magnitudes just in case of negative differences
        spread = abs(abs(diff_p90) - abs(diff_p10))
        
        if spread > 15:
            if abs(diff_p90) > abs(diff_p10):
                spectrum = (
                    f" This difference widens at the upper end of the distribution "
                    f"({abs(diff_p90):.0f} pts at P90 vs {abs(diff_p10):.0f} pts at P10), "
                    f"suggesting that high-SES students particularly pull ahead "
                    f"among the highest achievers."
                )
            else:
                spectrum = (
                    f" This difference is largest among lower-performing students "
                    f"({abs(diff_p10):.0f} pts at P10 vs {abs(diff_p90):.0f} pts at P90)."
                )
        else:
            spectrum = (
                f" This difference is fairly consistent across the distribution "
                f"({abs(diff_p10):.0f} pts at P10, {abs(diff_p90):.0f} pts at P90)."
            )
    else:
        spectrum = ""

    return main + spectrum

def scatter_correlation_text(df, subject, resource_col, resource_label, year=None, highlight_countries=None):
    """
    Computes the global country-level correlation, and dynamically 
    reports the specific values for highlighted countries.
    """
    if df.empty or "CNT" not in df.columns:
        return "Insufficient data to compute macro-correlation."
    subset = df.copy()
    if year and "YEAR" in df.columns:
        subset = subset[subset["YEAR"] == year]

    rows = []
    for cnt in subset["CNT"].unique():
        cnt_data = subset[subset["CNT"] == cnt]
        mean_score    = weighted_mean_pv(cnt_data, subject)
        mean_resource = cnt_data[resource_col].mean()
        if not np.isnan(mean_score) and not np.isnan(mean_resource):
            rows.append({"CNT": cnt, "score": mean_score, "resource": mean_resource})

    plot_df = pd.DataFrame(rows)
    
    if len(plot_df) < 5: 
        return "Insufficient data to compute macro-correlation."

    corr = plot_df["resource"].corr(plot_df["score"])
    subject_label = SUBJECTS.get(subject, subject)

    strength = "strong" if abs(corr) > 0.6 else "moderate" if abs(corr) > 0.3 else "weak"
    direction = "positive" if corr > 0 else "negative"
    
    # Emphasize that this is a GLOBAL metric so the user understands why it is static
    if abs(corr) < 0.1:
        base_text = f"Across **all global PISA countries**, there is virtually no correlation (r = {corr:.2f}) between the {resource_label} and mean {subject_label} scores."
    else:
        base_text = (f"Across **all global PISA countries**, there is a **{strength} {direction} correlation** "
                     f"(r = {corr:.2f}) between the {resource_label} and mean {subject_label} scores.")
                     
    # Make the UI feel responsive by reporting their specific selections
    if highlight_countries:
        hi_df = plot_df[plot_df["CNT"].isin(highlight_countries)]
        if not hi_df.empty:
            details = []
            for _, row in hi_df.iterrows():
                details.append(f"**{_cnt_label(row['CNT'])}** (Index: {row['resource']:.2f}, Score: {row['score']:.0f})")
            
            base_text += f"\n\n**Highlighted Countries:** " + ", ".join(details) + "."

    return base_text

def immigration_gap_text(df, subject: str, cnt: str, year: int = None) -> str:
    """
    Calculates the median score gaps between Native students and both 1st- and 2nd-generation 
    immigrant students, generating an automated insight string.
    """
    subset = df[df["CNT"] == cnt].copy()
    if year is not None and "YEAR" in subset.columns:
        subset = subset[subset["YEAR"] == year]

    if "IMMIG" not in subset.columns or "W_FSTUWT" not in subset.columns:
        return "Insufficient data to calculate the immigration difference."

    # Calculate medians for all three groups
    nat_med = weighted_percentiles_pv(subset[subset["IMMIG"] == 1.0], subject, [50])
    gen2_med = weighted_percentiles_pv(subset[subset["IMMIG"] == 2.0], subject, [50])
    gen1_med = weighted_percentiles_pv(subset[subset["IMMIG"] == 3.0], subject, [50])

    # Safety check: We need the Native score to act as our baseline
    if np.isnan(nat_med).all():
        return "Insufficient Native student data to establish a baseline for comparison."

    base_score = nat_med[0]
    comparisons = []

    # Calculate gap for 2nd-generation (if data exists)
    if not np.isnan(gen2_med).all():
        diff2 = gen2_med[0] - base_score
        dir2 = "higher" if diff2 > 0 else "lower"
        # If the gap is exactly 0, handle the language gracefully
        if round(diff2) == 0:
            comparisons.append("exactly the same for 2nd-generation immigrants")
        else:
            comparisons.append(f"{abs(diff2):.0f} points {dir2} for 2nd-generation immigrants")

    # Calculate gap for 1st-generation (if data exists)
    if not np.isnan(gen1_med).all():
        diff1 = gen1_med[0] - base_score
        dir1 = "higher" if diff1 > 0 else "lower"
        if round(diff1) == 0:
            comparisons.append("exactly the same for 1st-generation immigrants")
        else:
            comparisons.append(f"{abs(diff1):.0f} points {dir1} for 1st-generation immigrants")

    if not comparisons:
        return "Insufficient data to compare immigration groups."

    # Join them together into a professional, readable sentence
    joined_comparisons = " and ".join(comparisons)
    
    return (
        f"Compared to the median Native student in {_cnt_label(cnt)}, "
        f"the median {SUBJECTS[subject]} score is {joined_comparisons}."
    )