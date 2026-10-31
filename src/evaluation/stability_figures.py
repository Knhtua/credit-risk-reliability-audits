from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
root = PROJECT_ROOT / "run_artifacts" / "rolling_time_backtest"
folds = pd.read_csv(root / "fold_uncertainty_summary.csv")
quarters = pd.read_csv(root / "quarter_stability_metrics.csv")

plt.style.use("seaborn-v0_8-whitegrid")

# Figure 1: fold-level uncertainty
fig, ax = plt.subplots(figsize=(7.2, 4.6))

lower = folds["pr_auc"] - folds["pr_auc_ci_lower"]
upper = folds["pr_auc_ci_upper"] - folds["pr_auc"]

ax.errorbar(
    folds["fold"],
    folds["pr_auc"],
    yerr=[lower, upper],
    fmt="o",
    markersize=8,
    capsize=6,
    linewidth=2,
    color="#1f77b4",
)

ax.set_ylabel("PR-AUC")
ax.set_xlabel("Rolling validation fold")
ax.set_title("Rolling Backtest PR-AUC with 95% Bootstrap Intervals")
ax.set_ylim(0, max(folds["pr_auc_ci_upper"]) * 1.18)

for _, row in folds.iterrows():
    ax.annotate(
        f'{row["pr_auc"]:.3f}',
        (row["fold"], row["pr_auc"]),
        xytext=(8, 7),
        textcoords="offset points",
    )

fig.tight_layout()
fig.savefig(root / "fold_pr_auc_uncertainty.png", dpi=220)
fig.savefig(root / "fold_pr_auc_uncertainty.pdf")
plt.close(fig)

# Figure 2: quarterly stability
quarters["period"] = quarters["period"].astype(str)

fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

axes[0].plot(
    quarters["period"],
    quarters["pr_auc"],
    marker="o",
    linewidth=2,
    color="#1f77b4",
)
axes[0].set_ylabel("PR-AUC")
axes[0].set_title("Quarterly Out-of-Time Performance and Calibration")

axes[1].plot(
    quarters["period"],
    quarters["prevalence"],
    marker="o",
    label="Observed event rate",
    color="#2ca02c",
)
axes[1].plot(
    quarters["period"],
    quarters["prediction_mean"],
    marker="s",
    label="Mean predicted risk",
    color="#ff7f0e",
)
axes[1].set_ylabel("Rate")
axes[1].legend()

colors = [
    "#d62728" if value > 0 else "#2ca02c"
    for value in quarters["calibration_gap"]
]
axes[2].bar(
    quarters["period"],
    quarters["calibration_gap"],
    color=colors,
)
axes[2].axhline(0, color="black", linewidth=1)
axes[2].set_ylabel("Prediction mean − event rate")
axes[2].set_xlabel("Validation quarter")

for ax in axes:
    ax.tick_params(axis="x", rotation=45)

fig.tight_layout()
fig.savefig(root / "quarterly_stability.png", dpi=220)
fig.savefig(root / "quarterly_stability.pdf")
plt.close(fig)

print("Created:")
print(root / "fold_pr_auc_uncertainty.png")
print(root / "fold_pr_auc_uncertainty.pdf")
print(root / "quarterly_stability.png")
print(root / "quarterly_stability.pdf")
