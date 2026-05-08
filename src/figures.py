import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import gaussian_kde
import os

# Load the data
df = pd.read_csv("data/raw/sampledat.csv")
os.makedirs("data/images", exist_ok=True)

pv_math_cols = [f"PV{i}MATH" for i in range(1, 11)]
pv_read_cols = [f"PV{i}READ" for i in range(1, 11)]
pv_scie_cols = [f"PV{i}SCIE" for i in range(1, 11)]
weight_col   = "W_FSTUWT"
np.random.seed(42)

def weighted_pv_mean(sub_df, weight_col, pv_cols):
    means = []
    for pv in pv_cols:
        valid = sub_df[[pv, weight_col]].dropna()
        if len(valid) == 0:
            continue
        means.append(np.average(valid[pv], weights=valid[weight_col]))
    return np.mean(means) if means else np.nan

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
    return np.mean(pctiles) if pctiles else np.nan

def avg_pv_scores(sub_df, pv_cols):
    """
    Row-wise PV average per student - for visualization shape only.
    """
    return sub_df[pv_cols].mean(axis=1).dropna().values

def weighted_kde(scores, weights, x_range, bw_method=0.3):
    """
    Weighted KDE via weighted resampling.
    """
    weights = np.array(weights)
    weights = weights / weights.sum()
    n_resample = min(10000, len(scores) * 3)
    resampled  = np.random.choice(scores, size=n_resample,
                                  replace=True, p=weights)
    return gaussian_kde(resampled, bw_method=bw_method)(x_range)

# Figure 1: Math Score Distributions 
stats_fig1 = {}
for country in ["CAN", "USA"]:
    sub = df[df["CNT"] == country]
    stats_fig1[country] = {
        "mean": weighted_pv_mean(sub, weight_col, pv_math_cols),
        "p25": weighted_percentile(sub, weight_col, 25, pv_math_cols),
        "p75": weighted_percentile(sub, weight_col, 75, pv_math_cols),
        "scores": avg_pv_scores(sub, pv_math_cols)
    }

pct_colors = {"p25": "#7570b3", "p75": "#d4890a"}
mean_color = "#e24b4a"
country_labels = {"CAN": "Canada", "USA": "United States"}

fig, axes = plt.subplots(2, 1, figsize=(11, 6),
                          sharex=True,
                          gridspec_kw={"hspace": 0.12})

for ax, country in zip(axes, ["CAN", "USA"]):
    s = stats_fig1[country]
    scores = s["scores"]
    mean = s["mean"]
    p25 = s["p25"]
    p75 = s["p75"]

    counts, bins, _ = ax.hist(
        scores, bins=50, density=True,
        color="#5b9bd5", alpha=0.75,
        edgecolor="white", linewidth=0.4
    )
    bin_width = bins[1] - bins[0]
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y * bin_width * 100:.1f}%")
    )
    y_top = counts.max()
    label_y = y_top * 1.02

    ax.axvline(p25, color=pct_colors["p25"], linewidth=1.8,
               linestyle="--", zorder=3)
    ax.axvline(p75, color=pct_colors["p75"], linewidth=1.8,
               linestyle="--", zorder=3)
    ax.axvline(mean, color=mean_color, linewidth=2,
               linestyle="-", zorder=4)

    ax.text(p25, label_y, f"P25\n{p25:.0f}",
            fontsize=9, color=pct_colors["p25"],
            fontweight="bold", ha="center", va="bottom")
    ax.text(mean + 8, label_y, f"Mean\n{mean:.0f}",
            fontsize=9, color=mean_color,
            fontweight="bold", ha="center", va="bottom")
    ax.text(p75 + 8, label_y, f"P75\n{p75:.0f}",
            fontsize=9, color=pct_colors["p75"],
            fontweight="bold", ha="center", va="bottom")

    ax.text(0.02, 0.93, country_labels[country],
            transform=ax.transAxes, fontsize=11,
            fontweight="bold", color="#1a2e3b", va="top")

    ax.set_xlim(100, 850)
    ax.set_ylim(0, y_top * 1.22)
    ax.set_ylabel("% of Students", fontsize=9, color="black")
    ax.tick_params(colors="black", labelsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("black")
    ax.grid(axis="y", alpha=0.25, linestyle="--")

axes[1].set_xlabel("Math Score", fontsize=10, color="black")
legend_elements = [
    mlines.Line2D([0], [0], color=mean_color, linewidth=2,
                  linestyle="-", label="Weighted mean"),
    mlines.Line2D([0], [0], color=pct_colors["p25"], linewidth=1.8,
                  linestyle="--", label="25th percentile"),
    mlines.Line2D([0], [0], color=pct_colors["p75"], linewidth=1.8,
                  linestyle="--", label="75th percentile"),
]
axes[0].legend(handles=legend_elements, loc="upper right",
               fontsize=8.5, framealpha=0.9, edgecolor="#dde3e8")
plt.suptitle("PISA 2022 Math: What's Hiding Behind the Average?",
             fontsize=13, fontweight="bold", color="#1a2e3b", y=1.01)
plt.tight_layout()
plt.savefig("data/images/fig1_math_dist.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Figure 1 saved.")

# Figure 2: Gender Gap
percentiles = [10, 25, 50, 75, 90]
subjects = {
    "Mathematics": pv_math_cols,
    "Reading": pv_read_cols,
    "Science": pv_scie_cols
}
country_styles = {
    "CAN": {"color": "#2166ac", "linestyle": "-",
            "marker": "o", "label": "CAN"},
    "USA": {"color": "#d6604d", "linestyle": "--",
            "marker": "o", "label": "USA"}
}

fig, axes = plt.subplots(1, 3, figsize=(14, 5),
                          sharey=False, sharex=True)
fig.suptitle(
    "Gender gap across the score distribution\n"
    "(above zero = males higher; below zero = females higher)",
    fontsize=11, y=1.02
)

for ax, (subject, pv_cols) in zip(axes, subjects.items()):
    for country, style in country_styles.items():
        sub = df[df["CNT"] == country]
        male = sub[sub["ST004D01T"] == 2]
        female = sub[sub["ST004D01T"] == 1]

        gaps = []
        for p in percentiles:
            p_male = weighted_percentile(male,   weight_col, p, pv_cols)
            p_female = weighted_percentile(female, weight_col, p, pv_cols)
            gaps.append(p_male - p_female)

        ax.plot(percentiles, gaps,
                color=style["color"], linestyle=style["linestyle"],
                marker=style["marker"], markersize=6,
                linewidth=2, label=style["label"])

    ax.axhline(0, color="black", linewidth=0.8,
               linestyle=":", alpha=0.6)
    ax.set_title(f"{subject}\ngender gap (Male \u2212 Female)",
                 fontsize=10)
    ax.set_xlabel("Percentile", fontsize=9)
    ax.set_ylabel("Score gap (pts)", fontsize=9)
    ax.set_xticks(percentiles)
    ax.tick_params(labelsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linestyle="--")

axes[2].legend(fontsize=9, loc="upper left",
               framealpha=0.9, edgecolor="#dde3e8")
plt.tight_layout()
plt.savefig("data/images/fig2_gender_gap.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Figure 2 saved.")

# Figure 3: ESCS KDE 
df["ESCS_quartile"] = df.groupby("CNT")["ESCS"].transform(
    lambda x: pd.qcut(
        x, q=4,
        labels=["Q1 (Low ESCS)", "Q2", "Q3", "Q4 (High ESCS)"],
        duplicates="drop"
    )
)

quartile_colors = {
    "Q1 (Low ESCS)":  "#d73027",
    "Q2": "#fc8d59",
    "Q3": "#91bfdb",
    "Q4 (High ESCS)": "#2166ac"
}
x_range = np.linspace(100, 850, 400)

fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                          sharex=True, sharey=False)

for ax, country in zip(axes, ["CAN", "USA"]):
    sub = df[df["CNT"] == country].dropna(
        subset=["ESCS_quartile", "W_FSTUWT"] + pv_math_cols
    )

    for quartile, color in quartile_colors.items():
        group = sub[sub["ESCS_quartile"] == quartile].copy()
        group["avg_score"] = group[pv_math_cols].mean(axis=1)
        group = group.dropna(subset=["avg_score", "W_FSTUWT"])

        scores = group["avg_score"].values
        weights = group["W_FSTUWT"].values
        n = len(scores)

        if n < 10:
            continue

        y = weighted_kde(scores, weights, x_range, bw_method=0.3)

        ax.plot(x_range, y, color=color, linewidth=2.2,
                label=f"{quartile} (n={n})")
        ax.fill_between(x_range, y, color=color, alpha=0.12)

    ax.set_title(f"{country} - Math by ESCS Quartile",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Math Score", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_xlim(100, 850)
    ax.tick_params(labelsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(fontsize=8, framealpha=0.9,
              edgecolor="#dde3e8", loc="upper right")

plt.suptitle("PISA 2022: Math Score Distribution by ESCS (KDE)",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("data/images/fig3_escs_dist.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Figure 3 saved.")