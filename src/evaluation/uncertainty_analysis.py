from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "run_artifacts" / "rolling_time_backtest"
FOLDS = ["fold_1", "fold_2"]

N_BOOTSTRAP = 2000
RANDOM_SEED = 20260812


def stratified_bootstrap_metrics(y_true, y_pred, rng):
    positive_idx = np.flatnonzero(y_true == 1)
    negative_idx = np.flatnonzero(y_true == 0)

    sampled_positive = rng.choice(
        positive_idx,
        size=len(positive_idx),
        replace=True,
    )
    sampled_negative = rng.choice(
        negative_idx,
        size=len(negative_idx),
        replace=True,
    )

    sampled_idx = np.concatenate([
        sampled_positive,
        sampled_negative,
    ])
    rng.shuffle(sampled_idx)

    sampled_y = y_true[sampled_idx]
    sampled_pred = y_pred[sampled_idx]

    return {
        "pr_auc": average_precision_score(
            sampled_y,
            sampled_pred,
        ),
        "brier_score": brier_score_loss(
            sampled_y,
            sampled_pred,
        ),
        "prediction_mean": float(sampled_pred.mean()),
        "prevalence": float(sampled_y.mean()),
    }


fold_summary_rows = []
bootstrap_rows = []
quarter_rows = []

rng = np.random.default_rng(RANDOM_SEED)

for fold in FOLDS:
    path = ROOT / f"{fold}_validation_predictions.csv"
    df = pd.read_csv(path)

    y_true = df["y_true"].to_numpy(dtype=int)
    y_pred = df["y_pred"].to_numpy(dtype=float)

    observed_pr_auc = average_precision_score(
        y_true,
        y_pred,
    )
    observed_brier = brier_score_loss(
        y_true,
        y_pred,
    )
    observed_prevalence = float(y_true.mean())
    observed_prediction_mean = float(y_pred.mean())

    for bootstrap_id in range(N_BOOTSTRAP):
        metrics = stratified_bootstrap_metrics(
            y_true,
            y_pred,
            rng,
        )

        bootstrap_rows.append({
            "fold": fold,
            "bootstrap_id": bootstrap_id,
            **metrics,
        })

    fold_bootstrap = pd.DataFrame([
        row for row in bootstrap_rows
        if row["fold"] == fold
    ])

    pr_auc_lower, pr_auc_upper = np.quantile(
        fold_bootstrap["pr_auc"],
        [0.025, 0.975],
    )
    brier_lower, brier_upper = np.quantile(
        fold_bootstrap["brier_score"],
        [0.025, 0.975],
    )

    fold_summary_rows.append({
        "fold": fold,
        "rows": len(df),
        "positives": int(y_true.sum()),
        "prevalence": observed_prevalence,
        "prediction_mean": observed_prediction_mean,
        "calibration_gap": (
            observed_prediction_mean
            - observed_prevalence
        ),
        "pr_auc": observed_pr_auc,
        "pr_auc_ci_lower": pr_auc_lower,
        "pr_auc_ci_upper": pr_auc_upper,
        "brier_score": observed_brier,
        "brier_ci_lower": brier_lower,
        "brier_ci_upper": brier_upper,
    })

    for (year, quarter), group in df.groupby([
        "SNAPSHOT_YEAR",
        "SNAPSHOT_QUARTER",
    ]):
        quarter_y = group["y_true"].to_numpy(dtype=int)
        quarter_pred = group["y_pred"].to_numpy(dtype=float)

        if quarter_y.sum() > 0:
            quarter_pr_auc = average_precision_score(
                quarter_y,
                quarter_pred,
            )
        else:
            quarter_pr_auc = np.nan

        quarter_rows.append({
            "fold": fold,
            "period": f"{int(year)}Q{int(quarter)}",
            "rows": len(group),
            "positives": int(quarter_y.sum()),
            "prevalence": float(quarter_y.mean()),
            "prediction_mean": float(quarter_pred.mean()),
            "calibration_gap": float(
                quarter_pred.mean() - quarter_y.mean()
            ),
            "pr_auc": quarter_pr_auc,
            "brier_score": brier_score_loss(
                quarter_y,
                quarter_pred,
            ),
        })


fold_summary = pd.DataFrame(fold_summary_rows)
bootstrap_results = pd.DataFrame(bootstrap_rows)
quarter_results = pd.DataFrame(quarter_rows)

fold_summary.to_csv(
    ROOT / "fold_uncertainty_summary.csv",
    index=False,
)
bootstrap_results.to_csv(
    ROOT / "bootstrap_metric_samples.csv",
    index=False,
)
quarter_results.to_csv(
    ROOT / "quarter_stability_metrics.csv",
    index=False,
)

summary = {
    "stage": "STATISTICAL_UNCERTAINTY_ANALYSIS",
    "method": "stratified_row_bootstrap",
    "bootstrap_repetitions": N_BOOTSTRAP,
    "random_seed": RANDOM_SEED,
    "row_id_parsed": False,
    "hidden_test_labels_accessed": False,
    "limitation": (
        "Row bootstrap preserves class counts but does not "
        "fully account for repeated observations within issuer."
    ),
    "folds": fold_summary.to_dict(orient="records"),
}

(
    ROOT / "uncertainty_analysis_summary.json"
).write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

print("=== Fold uncertainty summary ===")
print(fold_summary.to_string(index=False))

print("\n=== Quarterly stability metrics ===")
print(quarter_results.to_string(index=False))

print("\nSaved:")
print(ROOT / "fold_uncertainty_summary.csv")
print(ROOT / "bootstrap_metric_samples.csv")
print(ROOT / "quarter_stability_metrics.csv")
print(ROOT / "uncertainty_analysis_summary.json")
