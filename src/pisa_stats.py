import numpy as np
from statsmodels.stats.weightstats import DescrStatsW
import pandas as pd
from src.config import MIN_GROUP_N


def weighted_mean_pv(data: pd.DataFrame, subject: str,
                     weight_col: str = "W_FSTUWT") -> float:
    """
    Correct PISA weighted mean: average of 10 per-PV weighted means.
    Returns np.nan if fewer than MIN_GROUP_N valid rows or if columns are missing.
    """
    # Expected columns
    expected_pvs = [f"PV{i}{subject}" for i in range(1, 11)]
    
    # Filter to only the columns that actually exist in the dataframe
    available_pvs = [c for c in expected_pvs if c in data.columns]
    
    if not available_pvs or weight_col not in data.columns:
        return np.nan

    valid = data.dropna(subset=[weight_col] + available_pvs)
    
    if len(valid) < MIN_GROUP_N:
        return np.nan
        
    w = valid[weight_col].values
    
    return float(np.mean([np.average(valid[pv].values, weights=w) for pv in available_pvs]))


def weighted_percentiles_pv(data: pd.DataFrame, subject: str,
                             percentiles: list,
                             weight_col: str = "W_FSTUWT") -> np.ndarray:
    """
    Weighted percentiles averaged across all 10 PVs.
    Returns array of NaN if fewer than MIN_GROUP_N valid rows.
    """
    pv_cols = [f"PV{i}{subject}" for i in range(1, 11)]
    valid   = data.dropna(subset=[weight_col] + pv_cols)
    if len(valid) < MIN_GROUP_N:
        return np.full(len(percentiles), np.nan)
    w = valid[weight_col].values
    pv_percs = []
    for pv in pv_cols:
        ds = DescrStatsW(valid[pv].values, weights=w)
        pv_percs.append(ds.quantile(np.array(percentiles) / 100,
                                    return_pandas=False))
    return np.mean(pv_percs, axis=0)

def compute_group_percentiles(df: pd.DataFrame, subject: str,
                               group_col: str, group_vals: dict,
                               percentiles: list,
                               cnt: str = None,
                               year: int = None) -> dict:
    """
    Compute percentile curves for each category in group_vals.
    Optionally filter by country and/or year first.
    Returns dict: {label: np.ndarray of length len(percentiles)}
    """
    subset = df.copy()
    if cnt  is not None: subset = subset[subset["CNT"]  == cnt]
    if year is not None and "YEAR" in subset.columns:
        subset = subset[subset["YEAR"] == year]

    return {
        label: weighted_percentiles_pv(
            subset[subset[group_col] == code], subject, percentiles
        )
        for code, label in group_vals.items()
    }


def compute_escs_quartile_percentiles(df: pd.DataFrame, subject: str,
                                       percentiles: list,
                                       cnt: str = None,
                                       year: int = None) -> dict:
    """
    Split data into ESCS quartiles within the filtered subset,
    then compute percentile curves for each quartile.
    Uses within-group quartile cuts (not global), which is PISA convention.
    """
    subset = df.dropna(subset=["ESCS"]).copy()
    if cnt  is not None: subset = subset[subset["CNT"]  == cnt]
    if year is not None and "YEAR" in subset.columns:
        subset = subset[subset["YEAR"] == year]

    labels = ["Q1 (low SES)", "Q2", "Q3", "Q4 (high SES)"]

    if len(subset) < 4:
        return {q: np.full(len(percentiles), np.nan) for q in labels}

    subset["ESCS_Q"] = pd.qcut(
        subset["ESCS"].rank(method="first"), 
        q=4,
        labels=labels
    )
    
    return {
        q: weighted_percentiles_pv(subset[subset["ESCS_Q"] == q], subject, percentiles)
        for q in labels
    }


def get_oecd_percentiles(df: pd.DataFrame, subject: str,
                          percentiles: list,
                          year: int = None) -> np.ndarray:
    """
    OECD average percentile curve.
    Computed across all OECD==1 countries, then averaged per percentile.
    Returns np.ndarray of length len(percentiles).
    """
    subset = df[df["OECD"] == 1]
    if year is not None and "YEAR" in subset.columns:
        subset = subset[subset["YEAR"] == year]

    country_curves = []
    for cnt in subset["CNT"].unique():
        curve = weighted_percentiles_pv(subset[subset["CNT"] == cnt],
                                         subject, percentiles)
        if not np.isnan(curve).all():
            country_curves.append(curve)

    return np.nanmean(country_curves, axis=0) if country_curves else np.full(len(percentiles), np.nan)


def compute_weighted_se_pv(
    df: pd.DataFrame,
    subject: str,
    weight_col: str = "W_FSTUWT"
) -> float:
    """
    Standard error of the weighted mean score based on the PISA Data
    Analysis Manual with 10 PVs and 80 Fay BRR replicates.

    Formula
    -------
    stderr² = sampling_var + (1 + 1/M) * imputation_var

    Parameters
    ----------
    df : pd.DataFrame
        Student-level rows for a SINGLE country and year, already filtered.
        Must contain weight_col, W_FSTURWT1-W_FSTURWT80, and PV1-PV10{subject}.
    subject : str
        One of "MATH", "READ", "SCIE".
    weight_col : str
        Final student weight column. Default "W_FSTUWT".

    Returns
    -------
    float
        Estimated standard error in score points.
        Returns np.nan if fewer than MIN_GROUP_N rows or replicate weights
        are absent from df.
    """
    pv_cols  = [f"PV{i}{subject}" for i in range(1, 11)]
    rep_cols = [f"W_FSTURWT{r}" for r in range(1, 81)]

    # Drop rows missing the main weight or any PV
    valid = df.dropna(subset=[weight_col] + pv_cols)
    if len(valid) < MIN_GROUP_N:
        return np.nan

    # Replicate weights must be present 
    missing_reps = [c for c in rep_cols if c not in valid.columns]
    if missing_reps:
        return np.nan

    w_full = valid[weight_col].values
    rep_matrix = valid[rep_cols].values   # (n_students, 80)
    M = 10

    # Imputation variance: variance across 10 PV weighted means 
    pv_means = np.array([
        np.average(valid[col].values, weights=w_full)
        for col in pv_cols
    ])
    imputation_var = float(np.var(pv_means, ddof=1))   

    # Sampling variance: Fay BRR averaged across 10 PVs
    pv_sampling_vars = []
    for col in pv_cols:
        scores    = valid[col].values
        mean_full = np.average(scores, weights=w_full)
        rep_means = np.array([
            np.average(scores, weights=rep_matrix[:, r])
            for r in range(80)
        ])
        pv_sampling_vars.append(
            np.sum((1.0 / 20.0) * (rep_means - mean_full) ** 2)
        )
    sampling_var = float(np.mean(pv_sampling_vars))

    # Combine
    stderr2 = sampling_var + (1.0 + 1.0 / M) * imputation_var
    return float(np.sqrt(stderr2))