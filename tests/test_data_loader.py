"""
Unit tests for src/data_loader.py

Network calls and file I/O are mocked so tests run offline and without
touching the filesystem outside of tmp_path.
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from src.data_loader import validate_url, build_country_stats


# validate_url
class TestValidateUrl:
    def test_returns_true_for_200(self):
        # Returns True when the server responds with 200
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-length": "1000000"}
        with patch("src.data_loader.requests.head", return_value=mock_resp):
            assert validate_url("https://example.com/file.zip", 2022) is True

    def test_returns_false_for_404(self):
        # Returns False for a 404 response
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        with patch("src.data_loader.requests.head", return_value=mock_resp):
            assert validate_url("https://example.com/missing.zip", 2022) is False

    def test_returns_false_for_empty_url(self):
        # Returns False immediately for an empty URL string
        assert validate_url("", 2022) is False

    def test_returns_false_on_timeout(self):
        # Returns False gracefully on network timeout
        import requests as req_lib
        with patch("src.data_loader.requests.head",
                   side_effect=req_lib.exceptions.Timeout):
            assert validate_url("https://example.com/file.zip", 2022) is False

    def test_returns_false_on_connection_error(self):
        # Returns False gracefully on connection error
        import requests as req_lib
        with patch("src.data_loader.requests.head",
                   side_effect=req_lib.exceptions.ConnectionError):
            assert validate_url("https://example.com/file.zip", 2022) is False


# build_country_stats
class TestBuildCountryStats:
    def test_output_file_created(self, base_df, tmp_path):
        # Creates a Parquet output file at the expected path
        (tmp_path / "pisa_all.parquet").touch()  # Create dummy file to pass .exists() check
        
        with patch("src.data_loader.pd.read_parquet", return_value=base_df):
            out = build_country_stats(processed_dir=str(tmp_path))
        assert Path(out).exists()

    def test_output_has_cnt_and_year_columns(self, base_df, tmp_path):
        # Output contains CNT and YEAR as identifier columns
        (tmp_path / "pisa_all.parquet").touch()  # Create dummy file
        
        with patch("src.data_loader.pd.read_parquet", return_value=base_df):
            out = build_country_stats(processed_dir=str(tmp_path))
        df_out = pd.read_parquet(out)
        assert "CNT" in df_out.columns
        assert "YEAR" in df_out.columns

    def test_one_row_per_country_year(self, base_df, tmp_path):
        # Output has exactly one row per unique CNT/YEAR combination
        (tmp_path / "pisa_all.parquet").touch()  # Create dummy file
        
        with patch("src.data_loader.pd.read_parquet", return_value=base_df):
            out = build_country_stats(processed_dir=str(tmp_path))
        df_out = pd.read_parquet(out)
        assert df_out.duplicated(subset=["CNT", "YEAR"]).sum() == 0
