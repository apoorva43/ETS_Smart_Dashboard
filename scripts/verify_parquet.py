import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from src.pisa_stats import weighted_mean_pv, weighted_percentiles_pv
from src.config import PERCENTILES_COARSE

YEARS = [2022, 2018, 2015]


def check_student_id(df: pd.DataFrame, year: int) -> None:
    """
    Verify CNTSTUID exists and is populated.
    Also surfaces any other student ID candidate columns as a sanity check.
    """
    print(f"\n Student ID check ({year}):")
    if "CNTSTUID" not in df.columns:
        print("  ERROR: CNTSTUID missing from columns")
        id_candidates = [c for c in df.columns if "STU" in c.upper() and "ID" in c.upper()]
        print(f"  Candidate columns found: {id_candidates}")
        return
    n_missing = df["CNTSTUID"].isna().sum()
    n_total   = len(df)
    print(f"  CNTSTUID present: {n_total - n_missing:,}/{n_total:,} non-null")
    if n_missing > 0:
        print(f"  WARNING: {n_missing:,} rows missing CNTSTUID")
    id_candidates = [c for c in df.columns if "STU" in c.upper() and "ID" in c.upper()]
    print(f"  Student ID candidates: {id_candidates}")


def check_duplicates(df: pd.DataFrame, year: int) -> None:
    """
    Check for duplicate students within the same country.
    A real duplicate is same CNT + CNTSTUID appearing more than once.
    """
    print(f"\n Duplicate check ({year}):")
    if "CNTSTUID" not in df.columns:
        print("  SKIPPED: CNTSTUID not available")
        return
    unique_students = df["CNTSTUID"].nunique()
    dupes           = len(df) - unique_students
    print(f"  Unique students: {unique_students:,}  (duplicates: {dupes:,})")

    cnt_dupes = df.duplicated(subset=["CNT", "CNTSTUID"], keep=False).sum()
    if cnt_dupes == 0:
        print("  No CNT+CNTSTUID duplicates found")
    else:
        print(f"  WARNING: {cnt_dupes:,} rows share CNT+CNTSTUID")
        print(df[df.duplicated(subset=["CNT", "CNTSTUID"], keep=False)]
              [["CNT", "CNTSTUID"]].value_counts().head(10))


def missing_value_audit(df: pd.DataFrame, year: int) -> None:
    """
    Report missingness for key columns.
    Flags anything above 5% as a WARNING.
    """
    print(f"\n Missing value audit ({year}):")
    audit_cols = {
        "Score (PV1MATH)":   "PV1MATH",
        "Weight (W_FSTUWT)": "W_FSTUWT",
        "Gender":            "ST004D01T",
        "ESCS":              "ESCS",
        "Immigration":       "IMMIG",
    }
    for label, col in audit_cols.items():
        if col not in df.columns:
            print(f"  {label:25s}: NOT IN DATA")
            continue
        n_miss = df[col].isna().sum()
        pct    = round(n_miss / len(df) * 100, 1)
        status = "OK" if pct < 5 else "WARNING"
        print(f"  {label:25s}: {n_miss:>6,} missing ({pct}%)  [{status}]")


for year in YEARS:
    path = Path(f"data/processed/pisa_{year}.parquet")
    if not path.exists():
        print(f"\n{year}: parquet not found - skipping")
        continue

    df = pd.read_parquet(path)

    print(f"\n{'='*50}")
    print(f"PISA {year}")
    print(f"{'='*50}")
    print(f"Rows:          {len(df):,}")
    print(f"Countries:     {df['CNT'].nunique()}")
    print(f"OECD members:  {df[df['OECD']==1]['CNT'].nunique()}")
    print(f"YEAR values:   {sorted(df['YEAR'].unique())}")

    missing_weight = df["W_FSTUWT"].isna().sum()
    print(f"Missing W_FSTUWT: {missing_weight:,} ({missing_weight/len(df)*100:.1f}%)")
    pv1_missing = df["PV1MATH"].isna().sum()
    print(f"Missing PV1MATH:  {pv1_missing:,} ({pv1_missing/len(df)*100:.1f}%)")

    check_student_id(df, year)
    check_duplicates(df, year)
    missing_value_audit(df, year)

    print()
    for cnt in ["CAN", "USA"]:
        subset = df[df["CNT"] == cnt]
        if len(subset) == 0:
            print(f"  {cnt}: not found in {year} data")
            continue
        mean  = weighted_mean_pv(subset, "MATH")
        percs = weighted_percentiles_pv(subset, "MATH", PERCENTILES_COARSE)
        print(f"  {cnt}  n={len(subset):,}  "
              f"math mean={mean:.1f}  "
              f"P10={percs[0]:.0f}  P50={percs[2]:.0f}  P90={percs[4]:.0f}")

    print()
    print("  Top 5 countries by student count:")
    top5 = df["CNT"].value_counts().head(5)
    for cnt, count in top5.items():
        print(f"    {cnt}: {count:,}")

print(f"\n{'='*50}")
print("All years verified")
print(f"{'='*50}")