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
    compute_escs_quartile_percentiles
    )
from src.config import (
    SUBJECTS,
    PERCENTILES_COARSE
    )


def country_distribution_text(df, subject, countries, year=None):
    """
    Generate a short summary of weighted mean scores by country.

    Parameters
    ----------
    df : pandas.DataFrame
        PISA dataset containing country identifiers, weights, and plausible
        value score columns.
    subject : str
        Subject code used to select plausible value columns. Expected values
        include ``"MATH"``, ``"READ"``, and ``"SCIE"``.
    countries : list of str
        Country codes to summarize, such as ``["CAN", "USA"]``.
    year : int, optional
        PISA cycle year to filter by. If ``None``, all available years are used.

    Returns
    -------
    str
        One-sentence summary of weighted mean scores. Returns an empty string
        if no valid mean scores can be computed.
    """
    lines = []
    for cnt in countries:
        subset = df[df["CNT"] == cnt]
        if year and "YEAR" in df.columns:
            subset = subset[subset["YEAR"] == year]
        mean = weighted_mean_pv(subset, subject)
        if not np.isnan(mean):
            lines.append(f"{cnt}: {mean:.0f}")
    if not lines:
        return ""
    subject_label = SUBJECTS[subject]
    return f"Weighted mean {subject_label} scores | " + ",  ".join(lines) + "."


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
    if len(valid_data) < 100:
        return f"Insufficient data to compute gender gap for {cnt}."

    female = valid_data[valid_data["ST004D01T"] == 1.0]
    male   = valid_data[valid_data["ST004D01T"] == 2.0]

    f_percs = weighted_percentiles_pv(female, subject, PERCENTILES_COARSE)
    m_percs = weighted_percentiles_pv(male,   subject, PERCENTILES_COARSE)

    if np.isnan(f_percs).all() or np.isnan(m_percs).all():
        return f"Insufficient data to compute gender gap for {cnt}."

    gaps = m_percs - f_percs
    p10_gap, median_gap, p90_gap = gaps[0], gaps[2], gaps[-1]
    subject_label = SUBJECTS[subject]

    if abs(median_gap) < 5:
        overall = f"At the median, male and female students in {cnt} score similarly in {subject_label} ({median_gap:+.0f} points)."
    elif median_gap > 0:
        overall = f"Male students in {cnt} score {median_gap:.0f} points higher than female students at the median in {subject_label}."
    else:
        overall = f"Female students in {cnt} score {abs(median_gap):.0f} points higher than male students at the median in {subject_label}."

    if abs(p90_gap - p10_gap) > 10:
        spread = (f"The gap widens toward the top of the distribution: "
                  f"{p10_gap:+.0f} pts at P10 vs {p90_gap:+.0f} pts at P90. "
                  f"This pattern is invisible in average-only reporting.")
    else:
        spread = f"The gap is relatively consistent across the distribution ({p10_gap:+.0f} pts at P10, {p90_gap:+.0f} pts at P90)."

    return f"{overall} {spread}"


def ses_gap_text(df, subject, cnt, year=None):
    """
    Generate interpretive summary text for socioeconomic achievement gaps.

    The generated text compares median scores between students in the 
    highest and lowest ESCS quartiles.

    Parameters
    ----------
    df : pandas.DataFrame
        PISA dataset containing ESCS values, weights, and plausible value
        score columns.
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
        Concise interpretation of the socioeconomic achievement gap.
    """
    curves = compute_escs_quartile_percentiles(df, subject,
                                               [50], cnt=cnt, year=year)
    q1_med = curves.get("Q1 (low SES)", [np.nan])[0]
    q4_med = curves.get("Q4 (high SES)", [np.nan])[0]

    if np.isnan(q1_med) or np.isnan(q4_med):
        return "Insufficient data to compute SES gap."

    gap = q4_med - q1_med
    subject_label = SUBJECTS[subject]
    return (
        f"In {cnt}, students in the highest SES quartile score "
        f"{gap:.0f} points higher than those in the lowest quartile "
        f"at the median in {subject_label}. "
        f"This gap represents the combined effect of home resources, "
        f"parental education, and occupational status."
    )

def scatter_correlation_text(df, subject, resource_col, resource_label, year=None, highlight_countries=None):
    """
    Computes the global country-level correlation, and dynamically 
    reports the specific values for highlighted countries.
    """
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
                details.append(f"**{row['CNT']}** (Index: {row['resource']:.2f}, Score: {row['score']:.0f})")
            
            base_text += f"\n\n**Highlighted Countries:** " + ", ".join(details) + "."

    return base_text