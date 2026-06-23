"""
Unit tests for src/precompute.py

Tests focus on the helper functions (_compute_kde, _compute_se,
_compute_ses_groups) and the schema/shape of build_precomputed output
using synthetic data so no real Parquet files are required.
"""
import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.precompute import (
    _compute_kde,
    _compute_se,
    _compute_ses_groups,
    build_precomputed,
    PERCENTILES,
    X_GRID,
    GROUP_CONFIGS,
)


PV_COLS = [f"PV{i}MATH" for i in range(1, 11)]
REP_COLS = [f"W_FSTURWT{r}" for r in range(1, 81)]


# ── _compute_ses_groups ───────────────────────────────────────────────────────

class TestComputeSesGroups:
    def test_returns_four_groups(self, base_df):
        # SES binning produces exactly four quartile groups
        sub = base_df[base_df["CNT"] == "CAN"]
        result = _compute_ses_groups(sub)
        assert len(result) == 4

    def test_group_labels_are_strings(self, base_df):
        # All group keys are strings
        sub = base_df[base_df["CNT"] == "CAN"]
        result = _compute_ses_groups(sub)
        assert all(isinstance(k, str) for k in result)

    def test_returns_empty_for_small_data(self, tiny_df):
        # Returns empty dict when fewer than 120 valid rows
        result = _compute_ses_groups(tiny_df)
        assert result == {}

    def test_groups_are_roughly_equal_size(self, base_df):
        # Each quartile should have approximately N/4 students
        sub = base_df[base_df["CNT"] == "CAN"]
        result = _compute_ses_groups(sub)
        sizes = [len(v) for v in result.values()]
        assert max(sizes) - min(sizes) < len(sub) * 0.1   # within 10%


# ── _compute_kde ──────────────────────────────────────────────────────────────

class TestComputeKde:
    def test_returns_json_strings(self, base_df):
        # Returns two JSON strings for valid data
        sub = base_df[base_df["CNT"] == "CAN"]
        dx, dy = _compute_kde(sub, PV_COLS)
        assert isinstance(dx, str) and isinstance(dy, str)

    def test_json_is_valid(self, base_df):
        # Both returned strings parse as valid JSON lists
        sub = base_df[base_df["CNT"] == "CAN"]
        dx, dy = _compute_kde(sub, PV_COLS)
        x_arr = json.loads(dx)
        y_arr = json.loads(dy)
        assert isinstance(x_arr, list) and isinstance(y_arr, list)

    def test_density_normalised_to_one(self, base_df):
        # Density is normalised so max value is 1.0
        sub = base_df[base_df["CNT"] == "CAN"]
        _, dy = _compute_kde(sub, PV_COLS)
        density = np.array(json.loads(dy))
        assert abs(density.max() - 1.0) < 1e-3

    def test_x_grid_matches_expected(self, base_df):
        # Returned x values match the module-level X_GRID
        sub = base_df[base_df["CNT"] == "CAN"]
        dx, _ = _compute_kde(sub, PV_COLS)
        x_arr = np.array(json.loads(dx))
        assert len(x_arr) == len(X_GRID)

    def test_returns_none_for_empty_data(self):
        # Returns (None, None) for an empty DataFrame
        empty = pd.DataFrame(columns=["W_FSTUWT"] + PV_COLS)
        dx, dy = _compute_kde(empty, PV_COLS)
        assert dx is None and dy is None

    def test_returns_none_when_pv_cols_missing(self, base_df):
        # Returns (None, None) gracefully when PV columns not in DataFrame
        sub = base_df[base_df["CNT"] == "CAN"].drop(columns=PV_COLS)
        dx, dy = _compute_kde(sub, PV_COLS)
        assert dx is None and dy is None


# ── _compute_se ───────────────────────────────────────────────────────────────

class TestComputeSe:
    def test_returns_positive_float(self, base_df):
        # SE is a finite positive number for valid data
        sub = base_df[base_df["CNT"] == "CAN"]
        result = _compute_se(sub, "MATH")
        assert np.isfinite(result) and result > 0

    def test_returns_nan_for_small_data(self, tiny_df):
        # Returns NaN when fewer than 30 valid rows
        result = _compute_se(tiny_df, "MATH")
        assert np.isnan(result)

    def test_returns_nan_when_rep_weights_missing(self, base_df):
        # Returns NaN gracefully when BRR columns are absent
        sub = base_df[base_df["CNT"] == "CAN"].drop(columns=REP_COLS)
        result = _compute_se(sub, "MATH")
        assert np.isnan(result)


# ── build_precomputed (integration) ───────────────────────────────────────────

class TestBuildPrecomputed:
    def test_output_parquet_created(self, base_df, tmp_path):
        # build_precomputed writes a Parquet file to the specified path
        out = tmp_path / "test_precomputed.parquet"
        with patch("src.precompute.pd.read_parquet", return_value=base_df):
            build_precomputed(
                processed_dir=str(tmp_path),
                out_path=str(out)
            )
        assert out.exists()

    def test_output_has_required_columns(self, base_df, tmp_path):
        # Output Parquet contains all expected schema columns
        out = tmp_path / "test_precomputed.parquet"
        with patch("src.precompute.pd.read_parquet", return_value=base_df):
            build_precomputed(processed_dir=str(tmp_path), out_path=str(out))
        df_out = pd.read_parquet(out)
        required = {"CNT", "YEAR", "SUBJECT", "GROUP_TYPE",
                    "GROUP_LABEL", "P10", "P25", "P50", "P75", "P90"}
        assert required.issubset(set(df_out.columns))

    def test_output_contains_country_and_oecd_records(self, base_df, tmp_path):
        # Output includes both country-level and OECD average records
        out = tmp_path / "test_precomputed.parquet"
        with patch("src.precompute.pd.read_parquet", return_value=base_df):
            build_precomputed(processed_dir=str(tmp_path), out_path=str(out))
        df_out = pd.read_parquet(out)
        assert "country" in df_out["GROUP_TYPE"].values
        assert "oecd" in df_out["GROUP_TYPE"].values

    def test_output_contains_all_group_types(self, base_df, tmp_path):
        # Output includes records for all six group types
        out = tmp_path / "test_precomputed.parquet"
        with patch("src.precompute.pd.read_parquet", return_value=base_df):
            build_precomputed(processed_dir=str(tmp_path), out_path=str(out))
        df_out = pd.read_parquet(out)
        expected_types = {"country", "oecd", "trend", "gender",
                          "immigration", "ses", "school_loc", "school_type"}
        found = set(df_out["GROUP_TYPE"].unique())
        assert expected_types.issubset(found)

    def test_percentiles_are_monotone_in_output(self, base_df, tmp_path):
        # P10 <= P25 <= P50 <= P75 <= P90 for every output row
        out = tmp_path / "test_precomputed.parquet"
        with patch("src.precompute.pd.read_parquet", return_value=base_df):
            build_precomputed(processed_dir=str(tmp_path), out_path=str(out))
        df_out = pd.read_parquet(out).dropna(subset=["P10", "P25", "P50", "P75", "P90"])
        assert (df_out["P10"] <= df_out["P25"]).all()
        assert (df_out["P25"] <= df_out["P50"]).all()
        assert (df_out["P50"] <= df_out["P75"]).all()
        assert (df_out["P75"] <= df_out["P90"]).all()

    def test_pct_sums_to_100_per_group_type(self, base_df, tmp_path):
        # PCT values for non-country group types sum to ~100 per country/year/subject
        out = tmp_path / "test_precomputed.parquet"
        with patch("src.precompute.pd.read_parquet", return_value=base_df):
            build_precomputed(processed_dir=str(tmp_path), out_path=str(out))
        df_out = pd.read_parquet(out)
        for gtype in ["gender", "immigration", "ses"]:
            sub = df_out[df_out["GROUP_TYPE"] == gtype]
            if sub.empty:
                continue
            totals = sub.groupby(["CNT", "YEAR", "SUBJECT"])["PCT"].sum()
            assert (totals > 95).all() and (totals <= 101).all()
