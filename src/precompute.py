# src/precompute.py
"""
Precomputes group-level percentile statistics and KDE density curves
for the PISA dashboard.

Record types (GROUP_TYPE values):
- "country"     : full country distribution KDE + percentiles + SE + MEAN (Section 1)
- "oecd"        : OECD average KDE + percentiles (Section 1 OECD line)
- "trend"       : percentiles per year for trend chart (Section 2)
- "gender"      : gender group percentiles + KDE (Section 3)
- "immigration" : immigration group percentiles + KDE (Section 3)
- "ses"         : SES quartile percentiles + KDE (Section 3)
- "school_loc"  : school location percentiles + KDE (Section 4)
- "school_type" : school type percentiles + KDE (Section 4)

Output: data/processed/pisa_precomputed.parquet

Run via:
    python src/build_data.py precompute
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import gaussian_kde
from src.pisa_stats import weighted_percentiles_pv, weighted_mean_pv
from src.config import (
    SUBJECTS, GENDER_MAP, IMMIG_MAP, LOC_MAP, SCHLTYPE_MAP
)

S3_BASE_URL  = "https://pisa-dashboard-data.s3.ca-central-1.amazonaws.com"
PERCENTILES  = [10, 25, 50, 75, 90]
X_GRID       = np.linspace(100, 900, 200)

GROUP_CONFIGS = [
    ("gender",       "ST004D01T",  GENDER_MAP),
    ("immigration",  "IMMIG",      IMMIG_MAP),
    ("school_loc",   "SC001Q01TA", LOC_MAP),
    ("school_type",  "SCHLTYPE",   SCHLTYPE_MAP),
    ("ses",          "ESCS",       None),
]

SES_LABELS = {
    "Q1 (lowest)":  "Q1 (lowest)",
    "Q2":           "Q2",
    "Q3":           "Q3",
    "Q4 (highest)": "Q4 (highest)",
}

REP_COLS = [f"W_FSTURWT{i}" for i in range(1, 81)]


def _compute_ses_groups(subset: pd.DataFrame) -> dict:
    valid = subset.dropna(subset=["ESCS"])
    if len(valid) < 120:
        return {}
    valid = valid.copy()
    valid["_ses_q"] = pd.qcut(
        valid["ESCS"].rank(method="first"), q=4,
        labels=list(SES_LABELS.keys())
    )
    return {label: valid[valid["_ses_q"] == key]
            for key, label in SES_LABELS.items()}


def _compute_kde(group: pd.DataFrame, pv_cols: list) -> tuple:
    """Compute averaged KDE density. Returns (x_json, density_json) or (None, None)."""
    available_pvs = [pv for pv in pv_cols if pv in group.columns]
    
    if not available_pvs:
        return None, None
    
    if len(group) > 2000:
        group = group.sample(n=2000, weights="W_FSTUWT", random_state=42)

    kde_vals = []
    for pv in pv_cols:
        scores  = group[pv].dropna().values
        weights = group.loc[group[pv].notna(), "W_FSTUWT"].values
        if len(scores) < 10:
            continue
        try:
            kde = gaussian_kde(scores, weights=weights, bw_method="scott")
            kde_vals.append(kde(X_GRID))
        except Exception:
            continue

    if not kde_vals:
        return None, None

    density = np.mean(kde_vals, axis=0)
    density /= density.max()

    return (
        json.dumps(np.round(X_GRID, 1).tolist()),
        json.dumps(np.round(density, 4).tolist())
    )


def _compute_se(group: pd.DataFrame, subject: str) -> float:
    """
    Compute standard error using Fay BRR method.
    Returns np.nan if replicate weights are missing.
    """
    pv_cols = [f"PV{i}{subject}" for i in range(1, 11)]
    valid   = group.dropna(subset=["W_FSTUWT"] + pv_cols)
    if len(valid) < 30:
        return np.nan

    missing_reps = [c for c in REP_COLS if c not in valid.columns]
    if missing_reps:
        return np.nan

    w_full     = valid["W_FSTUWT"].values
    rep_matrix = valid[REP_COLS].values
    M          = 10

    pv_means = np.array([
        np.average(valid[pv].values, weights=w_full)
        for pv in pv_cols
    ])
    imputation_var = float(np.var(pv_means, ddof=1))

    pv_sampling_vars = []
    for pv in pv_cols:
        scores    = valid[pv].values
        mean_full = np.average(scores, weights=w_full)
        rep_means = np.array([
            np.average(scores, weights=rep_matrix[:, r])
            for r in range(80)
        ])
        pv_sampling_vars.append(
            np.sum((1.0 / 20.0) * (rep_means - mean_full) ** 2)
        )
    sampling_var = float(np.mean(pv_sampling_vars))
    stderr2      = sampling_var + (1.0 + 1.0 / M) * imputation_var
    return float(np.sqrt(stderr2))


def build_precomputed(
    processed_dir: str = "data/processed",
    out_path: str = "data/processed/pisa_precomputed.parquet"
) -> Path:
    processed_dir = Path(processed_dir)
    source        = processed_dir / "pisa_all.parquet"

    if source.exists():
        print(f"Loading {source}...")
        df = pd.read_parquet(source)
    else:
        url = f"{S3_BASE_URL}/pisa_all.parquet"
        print(f"Local file not found, loading from {url}...")
        df = pd.read_parquet(url)

    records   = []
    countries = df["CNT"].unique()
    years     = sorted(df["YEAR"].unique())

    print(f"Processing {len(countries)} countries x {len(years)} years x {len(SUBJECTS)} subjects...")

    # ── OECD average records (one per YEAR/SUBJECT) ───────────────────────
    print("Computing OECD average records...")
    for year in years:
        year_df = df[(df["OECD"] == 1) & (df["YEAR"] == year)]
        for subject in SUBJECTS:
            pv_cols = [f"PV{i}{subject}" for i in range(1, 11)
                       if f"PV{i}{subject}" in year_df.columns]
            if not pv_cols:
                continue

            # Average percentiles and means across OECD countries
            country_percs = []
            country_means = []
            for cnt in year_df["CNT"].unique():
                c = year_df[year_df["CNT"] == cnt]
                p = weighted_percentiles_pv(c, subject, PERCENTILES)
                m = weighted_mean_pv(c, subject)
                if not np.isnan(p).all():
                    country_percs.append(p)
                if not np.isnan(m):
                    country_means.append(m)

            if not country_percs:
                continue

            avg_percs     = np.nanmean(country_percs, axis=0)
            avg_oecd_mean = float(np.mean(country_means)) if country_means else np.nan
            density_x, density_y = _compute_kde(year_df, pv_cols)
            if density_x is None:
                continue

            records.append({
                "CNT":         "OECD",
                "YEAR":        int(year),
                "SUBJECT":     subject,
                "GROUP_TYPE":  "oecd",
                "GROUP_LABEL": "OECD Average",
                "N":           len(year_df),
                "PCT":         100.0,
                "P10":         round(float(avg_percs[0]), 1),
                "P25":         round(float(avg_percs[1]), 1),
                "P50":         round(float(avg_percs[2]), 1),
                "P75":         round(float(avg_percs[3]), 1),
                "P90":         round(float(avg_percs[4]), 1),
                "MEAN":        round(avg_oecd_mean, 1) if not np.isnan(avg_oecd_mean) else np.nan,
                "SE":          np.nan,
                "DENSITY_X":   density_x,
                "DENSITY_Y":   density_y,
            })

    # ── Per-country records ───────────────────────────────────────────────
    done  = 0
    total = len(countries) * len(years) * len(SUBJECTS)

    for cnt in countries:
        cnt_df = df[df["CNT"] == cnt]

        for year in years:
            year_df = cnt_df[cnt_df["YEAR"] == year]
            if len(year_df) < 30:
                continue
            total_n = len(year_df)

            for subject in SUBJECTS:
                done += 1
                if done % 100 == 0:
                    print(f"  {done}/{total} country/year/subject combos...")

                pv_cols = [f"PV{i}{subject}" for i in range(1, 11)
                           if f"PV{i}{subject}" in year_df.columns]
                if not pv_cols:
                    continue

                percs      = weighted_percentiles_pv(year_df, subject, PERCENTILES)
                if np.isnan(percs).all():
                    continue

                density_x, density_y = _compute_kde(year_df, pv_cols)
                if density_x is None:
                    continue

                se         = _compute_se(year_df, subject)
                mean_score = weighted_mean_pv(year_df, subject)

                # Country-level record (Section 1 chart)
                records.append({
                    "CNT":         str(cnt),
                    "YEAR":        int(year),
                    "SUBJECT":     subject,
                    "GROUP_TYPE":  "country",
                    "GROUP_LABEL": str(cnt),
                    "N":           total_n,
                    "PCT":         100.0,
                    "P10":         round(float(percs[0]), 1),
                    "P25":         round(float(percs[1]), 1),
                    "P50":         round(float(percs[2]), 1),
                    "P75":         round(float(percs[3]), 1),
                    "P90":         round(float(percs[4]), 1),
                    "MEAN":        round(float(mean_score), 1) if not np.isnan(mean_score) else np.nan,
                    "SE":          round(float(se), 4) if not np.isnan(se) else np.nan,
                    "DENSITY_X":   density_x,
                    "DENSITY_Y":   density_y,
                })

                # Trend record (Section 2) — same percentiles, no KDE needed
                records.append({
                    "CNT":         str(cnt),
                    "YEAR":        int(year),
                    "SUBJECT":     subject,
                    "GROUP_TYPE":  "trend",
                    "GROUP_LABEL": str(year),
                    "N":           total_n,
                    "PCT":         100.0,
                    "P10":         round(float(percs[0]), 1),
                    "P25":         round(float(percs[1]), 1),
                    "P50":         round(float(percs[2]), 1),
                    "P75":         round(float(percs[3]), 1),
                    "P90":         round(float(percs[4]), 1),
                    "MEAN":        np.nan,
                    "SE":          np.nan,
                    "DENSITY_X":   None,
                    "DENSITY_Y":   None,
                })

                # Group records (Sections 3 and 4)
                for group_type, group_col, group_map in GROUP_CONFIGS:
                    if group_map is None:
                        if group_col not in year_df.columns:
                            continue
                        groups = _compute_ses_groups(year_df)
                    else:
                        if group_col not in year_df.columns:
                            continue
                        groups = {}
                        for code, label in group_map.items():
                            grp = year_df[year_df[group_col] == code]
                            if len(grp) >= 30:
                                groups[label] = grp

                    if not groups:
                        continue

                    for label, grp in groups.items():
                        g_percs = weighted_percentiles_pv(grp, subject, PERCENTILES)
                        if np.isnan(g_percs).all():
                            continue

                        g_dx, g_dy = _compute_kde(grp, pv_cols)
                        if g_dx is None:
                            continue

                        records.append({
                            "CNT":         str(cnt),
                            "YEAR":        int(year),
                            "SUBJECT":     subject,
                            "GROUP_TYPE":  group_type,
                            "GROUP_LABEL": label,
                            "N":           len(grp),
                            "PCT":         round(len(grp) / total_n * 100, 1),
                            "P10":         round(float(g_percs[0]), 1),
                            "P25":         round(float(g_percs[1]), 1),
                            "P50":         round(float(g_percs[2]), 1),
                            "P75":         round(float(g_percs[3]), 1),
                            "P90":         round(float(g_percs[4]), 1),
                            "MEAN":        np.nan,
                            "SE":          np.nan,
                            "DENSITY_X":   g_dx,
                            "DENSITY_Y":   g_dy,
                        })

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(records)
    result.to_parquet(out, compression="zstd", index=False)
    size_mb = out.stat().st_size / 1e6
    print(f"Saved: {out} ({len(result):,} rows, {size_mb:.1f} MB)")
    return out