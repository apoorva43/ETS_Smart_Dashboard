"""
Shared fixtures for PISA dashboard tests.
All fixtures use synthetic data that mirrors the real PISA schema
so tests run without any actual data files.
"""
import numpy as np
import pandas as pd
import pytest


N = 500  # students per country


def _make_students(cnt: str, oecd: int, year: int, rng: np.random.Generator) -> pd.DataFrame:
    n = N
    scores = rng.normal(500, 80, n).clip(100, 900)
    rows = {
        "CNT": cnt,
        "YEAR": year,
        "OECD": oecd,
        "W_FSTUWT": rng.uniform(0.5, 2.5, n),
        "ESCS": rng.normal(0, 1, n),
        "IMMIG": rng.choice([1.0, 2.0, 3.0], n, p=[0.7, 0.15, 0.15]),
        "ST004D01T": rng.choice([1.0, 2.0], n),   # 1=Female 2=Male
        "SC001Q01TA": rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], n),
        "SCHLTYPE": rng.choice([1.0, 2.0, 3.0], n, p=[0.05, 0.15, 0.80]),
        "BELONG": rng.normal(0, 1, n),
        "REPEAT": rng.choice([0, 1], n, p=[0.85, 0.15]),
    }
    for i in range(1, 11):
        rows[f"PV{i}MATH"] = scores + rng.normal(0, 5, n)
        rows[f"PV{i}READ"] = scores + rng.normal(0, 5, n)
        rows[f"PV{i}SCIE"] = scores + rng.normal(0, 5, n)
    for r in range(1, 81):
        rows[f"W_FSTURWT{r}"] = rng.uniform(0.5, 2.5, n)
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def base_df():
    rng = np.random.default_rng(42)
    frames = []
    for cnt, oecd in [("CAN", 1), ("USA", 1), ("BRA", 0)]:
        for year in [2015, 2018, 2022]:
            frames.append(_make_students(cnt, oecd, year, rng))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def single_country_df(base_df):
    return base_df[base_df["CNT"] == "CAN"].copy()


@pytest.fixture(scope="session")
def tiny_df():
    """DataFrame with fewer rows than MIN_GROUP_N — triggers insufficient-data paths."""
    rng = np.random.default_rng(99)
    return _make_students("TST", 0, 2022, rng).head(5)
