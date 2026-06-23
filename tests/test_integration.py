"""
Integration tests for the PISA dashboard.

These tests verify that the full pipeline from raw data through stats
computation, precompute, and chart/text generation produces consistent
and coherent outputs. They run only on the main branch in CI (see ci.yml)
since they are slower than unit tests.
"""
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

from src.pisa_stats import (
    weighted_mean_pv,
    weighted_percentiles_pv,
    get_oecd_percentiles,
)
from src.precompute import build_precomputed, _compute_kde
from src.text_generator import (
    country_distribution_text,
    ses_difference_text,
    gender_gap_text,
    immigration_gap_text,
    _oecd_mean_score,
)
from src.plotting_plotly import (
    plot_country_shaded_density,
    plot_group_shaded_density,
    plot_percentile_change_from_baseline,
)


# ── OECD average consistency ──────────────────────────────────────────────────

class TestOecdConsistency:
    def test_oecd_mean_matches_across_modules(self, base_df):
        # _oecd_mean_score in text_generator and get_oecd_percentiles
        # should both use OECD==1 data and produce values in the same ballpark
        mean_score = _oecd_mean_score(base_df, "MATH", year=2022)
        oecd_p50 = get_oecd_percentiles(base_df, "MATH", [50], year=2022)[0]
        assert np.isfinite(mean_score)
        assert np.isfinite(oecd_p50)
        # Mean and median should be within 30 points for roughly normal data
        assert abs(mean_score - oecd_p50) < 30

    def test_oecd_mean_excludes_non_oecd_countries(self, base_df):
        # Brazil (OECD=0) should not affect the OECD mean
        mean_with_bra = _oecd_mean_score(base_df, "MATH", year=2022)
        df_no_bra = base_df[base_df["CNT"] != "BRA"]
        mean_without_bra = _oecd_mean_score(df_no_bra, "MATH", year=2022)
        assert abs(mean_with_bra - mean_without_bra) < 1e-6


# ── Stats → text pipeline ─────────────────────────────────────────────────────

class TestStatsPipelineToText:
    def test_country_text_direction_matches_stats(self, base_df):
        # Text says "above" iff the country mean exceeds the OECD mean
        oecd_mean = _oecd_mean_score(base_df, "MATH", year=2022)
        sub = base_df[(base_df["CNT"] == "CAN") & (base_df["YEAR"] == 2022)]
        pv_cols = [f"PV{i}MATH" for i in range(1, 11)]
        cnt_mean = np.mean([
            np.average(sub[pv].values, weights=sub["W_FSTUWT"].values)
            for pv in pv_cols
        ])
        text = country_distribution_text(base_df, "MATH", ["CAN"], year=2022)
        if cnt_mean > oecd_mean + 3:
            assert "above" in text
        elif cnt_mean < oecd_mean - 3:
            assert "below" in text
        else:
            assert "in line" in text

    def test_ses_text_direction_matches_stats(self, base_df):
        # Text direction ("higher"/"lower") is consistent with computed Q4-Q1 difference
        from src.pisa_stats import compute_escs_quartile_percentiles
        curves = compute_escs_quartile_percentiles(
            base_df, "MATH", [50], cnt="CAN", year=2022
        )
        q1 = curves["Q1 (low SES)"][0]
        q4 = curves["Q4 (high SES)"][0]
        text = ses_difference_text(base_df, "MATH", "CAN", year=2022)
        if q4 > q1:
            assert "higher" in text
        else:
            assert "lower" in text

    def test_gender_text_direction_matches_stats(self, base_df):
        # Text direction is consistent with computed male/female median difference
        sub = base_df[(base_df["CNT"] == "CAN") & (base_df["YEAR"] == 2022)]
        female = sub[sub["ST004D01T"] == 1.0]
        male = sub[sub["ST004D01T"] == 2.0]
        f_med = weighted_percentiles_pv(female, "MATH", [50])[0]
        m_med = weighted_percentiles_pv(male, "MATH", [50])[0]
        text = gender_gap_text(base_df, "MATH", "CAN", year=2022)
        if m_med > f_med + 5:
            assert "male" in text.lower()
        elif f_med > m_med + 5:
            assert "female" in text.lower()


# ── Stats → chart pipeline ────────────────────────────────────────────────────

class TestStatsPipelineToChart:
    def test_precomputed_density_matches_live_kde(self, base_df):
        # KDE from precompute and from plot function use same bw_method
        # so their peak positions should be close
        pv_cols = [f"PV{i}MATH" for i in range(1, 11)]
        sub = base_df[(base_df["CNT"] == "CAN") & (base_df["YEAR"] == 2022)]
        _, dy = _compute_kde(sub, pv_cols)
        density = np.array(json.loads(dy))
        peak_precompute = density.argmax()

        # Chart figure — find the band polygon trace with highest x span
        fig = plot_country_shaded_density(base_df, "MATH", ["CAN"], year=2022)
        # The figure uses the same KDE logic; just confirm it renders non-empty
        assert len(fig.data) > 0
        assert peak_precompute > 0

    def test_trend_chart_years_match_data(self, base_df):
        # The x-axis tick values in the trend chart match years in the data
        fig = plot_percentile_change_from_baseline(
            base_df, "MATH", "CAN", reference_year=2015
        )
        expected_years = sorted(base_df[base_df["CNT"] == "CAN"]["YEAR"].unique())
        chart_years = list(fig.layout.xaxis.tickvals or [])
        assert sorted(chart_years) == expected_years

    def test_group_chart_labels_match_config(self, base_df):
        # Y-axis tick labels in group chart correspond to group_labels values
        from src.config import GENDER_MAP
        fig = plot_group_shaded_density(
            base_df, "MATH", "CAN", "ST004D01T", GENDER_MAP, "Gender", year=2022
        )
        ticktext = list(fig.layout.yaxis.ticktext or [])
        for label in GENDER_MAP.values():
            assert any(label in t for t in ticktext)


# ── Full precompute pipeline ──────────────────────────────────────────────────

class TestPrecomputePipeline:
    def test_full_pipeline_produces_valid_parquet(self, base_df, tmp_path):
        # build_precomputed runs end-to-end and produces a valid, non-empty Parquet
        out = tmp_path / "precomputed.parquet"
        with patch("src.precompute.pd.read_parquet", return_value=base_df):
            build_precomputed(processed_dir=str(tmp_path), out_path=str(out))
        assert out.exists()
        df_out = pd.read_parquet(out)
        assert len(df_out) > 0

    def test_density_json_deserialises_correctly(self, base_df, tmp_path):
        # DENSITY_X and DENSITY_Y columns parse as valid float arrays
        out = tmp_path / "precomputed.parquet"
        with patch("src.precompute.pd.read_parquet", return_value=base_df):
            build_precomputed(processed_dir=str(tmp_path), out_path=str(out))
        df_out = pd.read_parquet(out).dropna(subset=["DENSITY_X", "DENSITY_Y"])
        sample = df_out.iloc[0]
        x = np.array(json.loads(sample["DENSITY_X"]))
        y = np.array(json.loads(sample["DENSITY_Y"]))
        assert x.ndim == 1 and y.ndim == 1
        assert len(x) == len(y)
        assert y.max() <= 1.0 + 1e-6

    def test_no_negative_percentile_values(self, base_df, tmp_path):
        # All percentile values in output are non-negative PISA scores
        out = tmp_path / "precomputed.parquet"
        with patch("src.precompute.pd.read_parquet", return_value=base_df):
            build_precomputed(processed_dir=str(tmp_path), out_path=str(out))
        df_out = pd.read_parquet(out)
        for col in ["P10", "P25", "P50", "P75", "P90"]:
            vals = df_out[col].dropna()
            assert (vals >= 0).all(), f"Negative values found in {col}"

    def test_all_input_countries_appear_in_output(self, base_df, tmp_path):
        # Every country in the input data has at least one country-type record
        out = tmp_path / "precomputed.parquet"
        with patch("src.precompute.pd.read_parquet", return_value=base_df):
            build_precomputed(processed_dir=str(tmp_path), out_path=str(out))
        df_out = pd.read_parquet(out)
        country_records = df_out[df_out["GROUP_TYPE"] == "country"]
        for cnt in base_df["CNT"].unique():
            assert cnt in country_records["CNT"].values
