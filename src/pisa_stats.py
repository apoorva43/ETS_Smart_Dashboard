import numpy as np
from statsmodels.stats.weightstats import DescrStatsW
import pandas as pd
from src.config import MIN_GROUP_N


def weighted_mean_pv(data: pd.DataFrame, subject: str,
                     weight_col: str = "W_FSTUWT") -> float:
    """
    Correct PISA weighted mean: average of 10 per-PV weighted means.
    Returns np.nan if fewer than MIN_GROUP_N valid rows.
    """
    pv_cols = [f"PV{i}{subject}" for i in range(1, 11)]
    valid   = data.dropna(subset=[weight_col] + pv_cols)
    if len(valid) < MIN_GROUP_N:
        return np.nan
    w = valid[weight_col].values
    return np.mean([np.average(valid[pv].values, weights=w) for pv in pv_cols])


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