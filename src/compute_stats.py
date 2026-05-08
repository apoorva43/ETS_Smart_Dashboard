import pandas as pd
import numpy as np
import json
import os

# Load the data
df = pd.read_csv("data/raw/sampledat.csv")

pv_math_cols = [f"PV{i}MATH" for i in range(1, 11)]
weight_col = "W_FSTUWT"

def weighted_pv_mean(sub_df, weight_col, pv_cols):
    means = []
    for pv in pv_cols:
        valid = sub_df[[pv, weight_col]].dropna()
        if len(valid) == 0:
            continue
        means.append(np.average(valid[pv], weights=valid[weight_col]))
    return round(float(np.mean(means)), 1) if means else None

def weighted_percentile(sub_df, weight_col, percentile, pv_cols):
    pctiles = []
    for pv in pv_cols:
        valid = sub_df[[pv, weight_col]].dropna()
        if len(valid) == 0:
            continue
        sorted_idx = valid[pv].argsort()
        sorted_scores = valid[pv].iloc[sorted_idx].values
        sorted_weights = valid[weight_col].iloc[sorted_idx].values
        cumulative = sorted_weights.cumsum() / sorted_weights.sum()
        pctiles.append(np.interp(percentile / 100, cumulative, sorted_scores))
    return round(float(np.mean(pctiles)), 1) if pctiles else None

# Compute statistics
can = df[df["CNT"] == "CAN"]
usa = df[df["CNT"] == "USA"]

can_mean = weighted_pv_mean(can, weight_col, pv_math_cols)
usa_mean = weighted_pv_mean(usa, weight_col, pv_math_cols)
can_p25  = weighted_percentile(can, weight_col, 25, pv_math_cols)
usa_p25  = weighted_percentile(usa, weight_col, 25, pv_math_cols)
can_p75  = weighted_percentile(can, weight_col, 75, pv_math_cols)
usa_p75  = weighted_percentile(usa, weight_col, 75, pv_math_cols)

stats = {
    # Sample sizes
    "n_total":   int(len(df)),
    "n_canada":  int(len(can)),
    "n_usa":     int(len(usa)),
    "n_cols":    int(df.shape[1]),
    "n_cols_selected": 132,

    # Weighted means
    "can_math_mean": can_mean,
    "usa_math_mean": usa_mean,
    "mean_gap":      round(can_mean - usa_mean, 1),

    # Percentile gaps
    "can_p25":  can_p25,
    "usa_p25":  usa_p25,
    "gap_p25":  round(can_p25 - usa_p25, 1),

    "can_p75":  can_p75,
    "usa_p75":  usa_p75,
    "gap_p75":  round(can_p75 - usa_p75, 1),

    # Missingness
    "missing_gender":      int(df["ST004D01T"].isna().sum()),
    "missing_gender_pct":  round(df["ST004D01T"].isna().mean() * 100, 1),
    "missing_school_type": int(df["SC001Q01TA"].isna().sum()),
    "missing_school_pct":  round(df["SC001Q01TA"].isna().mean() * 100, 1),
    "missing_lang":        int(df["ST022Q01TA"].isna().sum()),
    "missing_lang_pct":    round(df["ST022Q01TA"].isna().mean() * 100, 1),
    "missing_escs":        int(df["ESCS"].isna().sum()),
    "missing_escs_pct":    round(df["ESCS"].isna().mean() * 100, 1),
    "missing_immig":       int(df["IMMIG"].isna().sum()),
    "missing_immig_pct":   round(df["IMMIG"].isna().mean() * 100, 1),
}

# Save values
os.makedirs("data/processed", exist_ok=True)
with open("data/processed/stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("Stats saved to data/processed/stats.json")
print(json.dumps(stats, indent=2))