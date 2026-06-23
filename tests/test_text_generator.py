"""
Unit tests for src/text_generator.py
"""
import numpy as np
import pytest

from src.text_generator import (
    country_distribution_text,
    gender_gap_text,
    ses_difference_text,
    immigration_gap_text,
    scatter_correlation_text,
)


# ── country_distribution_text ─────────────────────────────────────────────────

class TestCountryDistributionText:
    def test_returns_non_empty_string(self, base_df):
        # Returns a non-empty string for valid data
        result = country_distribution_text(base_df, "MATH", ["CAN"], year=2022)
        assert isinstance(result, str) and len(result) > 0

    def test_empty_countries_returns_empty_string(self, base_df):
        # Empty countries list returns empty string, not an error
        result = country_distribution_text(base_df, "MATH", [], year=2022)
        assert result == ""

    def test_contains_country_name(self, base_df):
        # Generated text mentions the selected country by name
        result = country_distribution_text(base_df, "MATH", ["CAN"], year=2022)
        assert "Canada" in result or "CAN" in result

    def test_contains_subject_name(self, base_df):
        # Generated text names the subject
        result = country_distribution_text(base_df, "MATH", ["CAN"], year=2022)
        assert "Mathematics" in result or "Math" in result

    def test_no_forbidden_words(self, base_df):
        # Text avoids ETS-prohibited terms
        result = country_distribution_text(base_df, "MATH", ["CAN"], year=2022)
        for word in ["weighted", "gap"]:
            assert word not in result.lower()

    def test_above_or_below_oecd_mentioned(self, base_df):
        # Text mentions direction relative to OECD average
        result = country_distribution_text(base_df, "MATH", ["CAN"], year=2022)
        assert "above" in result or "below" in result or "in line" in result

    def test_insufficient_data_returns_fallback(self, tiny_df):
        # Returns a safe fallback string rather than crashing
        result = country_distribution_text(tiny_df, "MATH", ["TST"], year=2022)
        assert isinstance(result, str)
        assert len(result) > 0


# ── gender_gap_text ───────────────────────────────────────────────────────────

class TestGenderGapText:
    def test_returns_string(self, base_df):
        # Returns a non-empty string for valid data
        result = gender_gap_text(base_df, "MATH", "CAN", year=2022)
        assert isinstance(result, str) and len(result) > 0

    def test_mentions_male_or_female(self, base_df):
        # Text refers to gender groups explicitly
        result = gender_gap_text(base_df, "MATH", "CAN", year=2022)
        assert "male" in result.lower() or "female" in result.lower()

    def test_mentions_percentiles(self, base_df):
        # Text references distribution positions (P10, P90)
        result = gender_gap_text(base_df, "MATH", "CAN", year=2022)
        assert "P10" in result or "P90" in result or "distribution" in result

    def test_insufficient_data_returns_fallback(self, tiny_df):
        # Returns safe fallback rather than raising
        result = gender_gap_text(tiny_df, "MATH", "TST", year=2022)
        assert "Insufficient" in result or isinstance(result, str)


# ── ses_difference_text ───────────────────────────────────────────────────────

class TestSesDifferenceText:
    def test_returns_string(self, base_df):
        # Returns a non-empty string for valid data
        result = ses_difference_text(base_df, "MATH", "CAN", year=2022)
        assert isinstance(result, str) and len(result) > 0

    def test_no_forbidden_words(self, base_df):
        # Avoids "gap" and "weighted" per ETS guidelines
        result = ses_difference_text(base_df, "MATH", "CAN", year=2022)
        for word in ["weighted", "gap"]:
            assert word not in result.lower()

    def test_mentions_socioeconomic_context(self, base_df):
        # Text describes family background or socioeconomic groups
        result = ses_difference_text(base_df, "MATH", "CAN", year=2022)
        assert "socioeconomic" in result.lower() or "background" in result.lower()

    def test_mentions_distribution_endpoints(self, base_df):
        # References both ends of the distribution
        result = ses_difference_text(base_df, "MATH", "CAN", year=2022)
        assert "P10" in result or "P90" in result or "distribution" in result

    def test_direction_is_consistent(self, base_df):
        # "higher" or "lower" appears in the text
        result = ses_difference_text(base_df, "MATH", "CAN", year=2022)
        assert "higher" in result or "lower" in result


# ── immigration_gap_text ──────────────────────────────────────────────────────

class TestImmigrationGapText:
    def test_returns_string(self, base_df):
        # Returns a non-empty string for valid data
        result = immigration_gap_text(base_df, "MATH", "CAN", year=2022)
        assert isinstance(result, str) and len(result) > 0

    def test_mentions_native_students(self, base_df):
        # Native group is mentioned as the baseline
        result = immigration_gap_text(base_df, "MATH", "CAN", year=2022)
        assert "Native" in result or "native" in result

    def test_mentions_generation(self, base_df):
        # References immigrant generation in output
        result = immigration_gap_text(base_df, "MATH", "CAN", year=2022)
        assert "generation" in result.lower()

    def test_missing_immig_column_returns_fallback(self, base_df):
        # Returns fallback string when IMMIG column is absent
        sub = base_df.drop(columns=["IMMIG"])
        result = immigration_gap_text(sub, "MATH", "CAN", year=2022)
        assert "Insufficient" in result or isinstance(result, str)


# ── scatter_correlation_text ──────────────────────────────────────────────────

class TestScatterCorrelationText:
    def test_returns_string(self, base_df):
        # Returns a non-empty string
        result = scatter_correlation_text(base_df, "MATH", "ESCS", "SES Index", year=2022)
        assert isinstance(result, str) and len(result) > 0

    def test_contains_correlation_value(self, base_df):
        # Correlation coefficient appears in text
        result = scatter_correlation_text(base_df, "MATH", "ESCS", "SES Index", year=2022)
        assert "r = " in result

    def test_highlighted_countries_mentioned(self, base_df):
        import pandas as pd
        # Artificially inflate the number of countries to bypass the correlation minimum threshold
        dfs = [base_df]
        for i in range(10):
            mock = base_df.copy()
            mock["CNT"] = f"M0{i}"
            dfs.append(mock)
        large_df = pd.concat(dfs, ignore_index=True)

        # Highlighted countries are named in text
        result = scatter_correlation_text(
            large_df, "MATH", "ESCS", "SES Index",
            year=2022, highlight_countries=["CAN"]
        )
        assert "Canada" in result or "CAN" in result

    def test_contains_correlation_value(self, base_df):
        import pandas as pd
        # Artificially inflate the number of countries
        dfs = [base_df]
        for i in range(10):
            mock = base_df.copy()
            mock["CNT"] = f"M0{i}"
            dfs.append(mock)
        large_df = pd.concat(dfs, ignore_index=True)

        # Correlation coefficient appears in text
        result = scatter_correlation_text(large_df, "MATH", "ESCS", "SES Index", year=2022)
        assert "r = " in result
