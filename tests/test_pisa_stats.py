"""
Unit tests for src/pisa_stats.py
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from src.pisa_stats import (
    weighted_mean_pv,
    weighted_percentiles_pv,
    compute_escs_quartile_percentiles,
    get_oecd_percentiles,
    compute_weighted_se_pv,
)


# weighted_mean_pv
class TestWeightedMeanPv:
    def test_returns_float_for_valid_data(self, base_df):
        # Returns a finite float when given sufficient data
        result = weighted_mean_pv(base_df[base_df["CNT"] == "CAN"], "MATH")
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_score_in_plausible_range(self, base_df):
        # Mean score must fall within the PISA score range
        result = weighted_mean_pv(base_df[base_df["CNT"] == "CAN"], "MATH")
        assert 100 <= result <= 900

    def test_returns_nan_for_insufficient_data(self, tiny_df):
        # Returns NaN rather than raising when n < MIN_GROUP_N
        result = weighted_mean_pv(tiny_df, "MATH")
        assert np.isnan(result)

    def test_weights_influence_mean(self, base_df):
        # Doubling weights on high scorers should push the mean upward
        sub = base_df[base_df["CNT"] == "CAN"].copy()
        high = sub["PV1MATH"] > sub["PV1MATH"].median()
        sub_boosted = sub.copy()
        sub_boosted.loc[high, "W_FSTUWT"] *= 10
        mean_orig = weighted_mean_pv(sub, "MATH")
        mean_boost = weighted_mean_pv(sub_boosted, "MATH")
        assert mean_boost > mean_orig

    def test_all_subjects_return_values(self, base_df):
        # All three PISA subjects produce a valid mean
        sub = base_df[base_df["CNT"] == "CAN"]
        for subj in ["MATH", "READ", "SCIE"]:
            assert np.isfinite(weighted_mean_pv(sub, subj))

    def test_missing_pv_columns_returns_nan(self, base_df):
        # Drops gracefully when PV columns are absent
        sub = base_df[base_df["CNT"] == "CAN"].drop(
            columns=[f"PV{i}MATH" for i in range(1, 11)]
        )
        result = weighted_mean_pv(sub, "MATH")
        assert np.isnan(result)


# weighted_percentiles_pv
class TestWeightedPercentilesPv:
    def test_returns_array_of_correct_length(self, base_df):
        # Output length matches the number of requested percentiles
        sub = base_df[base_df["CNT"] == "CAN"]
        result = weighted_percentiles_pv(sub, "MATH", [10, 50, 90])
        assert len(result) == 3

    def test_percentiles_are_monotone(self, base_df):
        # P10 <= P25 <= P50 <= P75 <= P90
        sub = base_df[base_df["CNT"] == "CAN"]
        result = weighted_percentiles_pv(sub, "MATH", [10, 25, 50, 75, 90])
        assert all(result[i] <= result[i + 1] for i in range(len(result) - 1))

    def test_returns_nan_array_for_insufficient_data(self, tiny_df):
        # All NaN when n < MIN_GROUP_N
        result = weighted_percentiles_pv(tiny_df, "MATH", [10, 50, 90])
        assert np.isnan(result).all()

    def test_median_near_mean_for_normal_distribution(self, base_df):
        # For symmetric data, median and mean should be close
        sub = base_df[base_df["CNT"] == "CAN"]
        median = weighted_percentiles_pv(sub, "MATH", [50])[0]
        mean = weighted_mean_pv(sub, "MATH")
        assert abs(median - mean) < 20   # within 20 points for normal data

    def test_single_percentile_returns_scalar_array(self, base_df):
        # Single-element list returns a length-1 array, not a bare float
        sub = base_df[base_df["CNT"] == "CAN"]
        result = weighted_percentiles_pv(sub, "MATH", [50])
        assert result.shape == (1,)


# compute_escs_quartile_percentiles
class TestEscsQuartilePercentiles:
    def test_returns_four_quartile_keys(self, base_df):
        # Result dict always has exactly four SES quartile keys
        result = compute_escs_quartile_percentiles(base_df, "MATH", [50], cnt="CAN", year=2022)
        assert len(result) == 4

    def test_q4_median_above_q1_median(self, base_df):
        # High-SES quartile should score above low-SES quartile at the median
        result = compute_escs_quartile_percentiles(base_df, "MATH", [50], cnt="CAN", year=2022)
        q1 = result.get("Q1 (low SES)", [np.nan])[0]
        q4 = result.get("Q4 (high SES)", [np.nan])[0]
        assert q4 > q1

    def test_filters_by_country_and_year(self, base_df):
        # Different countries produce different results
        can = compute_escs_quartile_percentiles(base_df, "MATH", [50], cnt="CAN", year=2022)
        bra = compute_escs_quartile_percentiles(base_df, "MATH", [50], cnt="BRA", year=2022)
        assert can["Q1 (low SES)"][0] != bra["Q1 (low SES)"][0]

    def test_handles_missing_escs_gracefully(self, base_df):
        # NaN ESCS values are dropped without error
        sub = base_df.copy()
        sub.loc[sub["CNT"] == "CAN", "ESCS"] = np.nan
        result = compute_escs_quartile_percentiles(sub, "MATH", [50], cnt="CAN")
        assert all(np.isnan(v).all() for v in result.values())


# get_oecd_percentiles
class TestGetOecdPercentiles:
    def test_returns_array_of_correct_length(self, base_df):
        # Output matches requested percentile count
        result = get_oecd_percentiles(base_df, "MATH", [10, 50, 90], year=2022)
        assert len(result) == 3

    def test_oecd_average_in_plausible_range(self, base_df):
        # OECD average must be a valid PISA score
        result = get_oecd_percentiles(base_df, "MATH", [50], year=2022)
        assert 100 <= result[0] <= 900

    def test_only_uses_oecd_countries(self, base_df):
        # Removing OECD flag from all non-OECD countries should not change result
        oecd_only = base_df[base_df["OECD"] == 1]
        result_full = get_oecd_percentiles(base_df, "MATH", [50], year=2022)
        result_oecd = get_oecd_percentiles(oecd_only, "MATH", [50], year=2022)
        np.testing.assert_allclose(result_full, result_oecd, rtol=1e-5)

    def test_returns_nan_when_no_oecd_data(self, base_df):
        # All NaN when OECD column is all zero
        no_oecd = base_df.copy()
        no_oecd["OECD"] = 0
        result = get_oecd_percentiles(no_oecd, "MATH", [50], year=2022)
        assert np.isnan(result).all()


# compute_weighted_se_pv
class TestComputeWeightedSePv:
    def test_returns_positive_float(self, base_df):
        # SE must be a positive finite number for valid data
        sub = base_df[base_df["CNT"] == "CAN"]
        result = compute_weighted_se_pv(sub, "MATH")
        assert np.isfinite(result) and result > 0

    def test_returns_nan_for_insufficient_data(self, tiny_df):
        # Returns NaN rather than raising for n < MIN_GROUP_N
        result = compute_weighted_se_pv(tiny_df, "MATH")
        assert np.isnan(result)

    def test_returns_nan_when_replicate_weights_missing(self, base_df):
        # Returns NaN gracefully when BRR replicate columns are absent
        sub = base_df[base_df["CNT"] == "CAN"].drop(
            columns=[f"W_FSTURWT{r}" for r in range(1, 81)]
        )
        result = compute_weighted_se_pv(sub, "MATH")
        assert np.isnan(result)

    def test_se_smaller_for_larger_sample(self, base_df):
        # Larger samples should produce smaller standard errors
        sub = base_df[base_df["CNT"] == "CAN"]
        half = sub.sample(frac=0.3, random_state=42)
        se_full = compute_weighted_se_pv(sub, "MATH")
        se_half = compute_weighted_se_pv(half, "MATH")
        assert se_full < se_half
