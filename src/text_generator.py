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
from src.pisa_stats import weighted_mean_pv
from src.config import SUBJECTS


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
    return f"Weighted mean {subject_label} scores — " + ",  ".join(lines) + "."
