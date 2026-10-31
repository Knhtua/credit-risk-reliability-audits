from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


TARGET = "DOWNGRADE_6M"
ROW_ID = "ROW_ID"
TOP_FRACTIONS = (0.01, 0.05, 0.10)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_prediction(path: Path, probability_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)

    if list(frame.columns) != [ROW_ID, TARGET]:
        raise ValueError(
            f"{path}: expected exactly [{ROW_ID}, {TARGET}], "
            f"found {list(frame.columns)}"
        )

    if frame[ROW_ID].duplicated().any():
        raise ValueError(f"{path}: duplicated ROW_ID")

    probability = pd.to_numeric(frame[TARGET], errors="raise")

    if probability.isna().any():
        raise ValueError(f"{path}: NaN predictions")

    if not probability.between(0, 1).all():
        raise ValueError(f"{path}: predictions outside [0, 1]")

    return frame.rename(columns={TARGET: probability_name})


def read_labels(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)

    if list(frame.columns) != [ROW_ID, TARGET]:
        raise ValueError(
            f"{path}: expected exactly [{ROW_ID}, {TARGET}], "
            f"found {list(frame.columns)}"
        )

    if frame[ROW_ID].duplicated().any():
        raise ValueError(f"{path}: duplicated ROW_ID")

    label = pd.to_numeric(frame[TARGET], errors="raise")

    if not label.isin([0, 1]).all():
        raise ValueError(f"{path}: labels must be 0/1")

    frame[TARGET] = label.astype(np.int8)
    return frame


def parse_control_columns(row_id: pd.Series) -> pd.DataFrame:
    """
    Parse ROW_ID strictly for external evaluation controls.

    These variables are never returned to, or used by, a predictive model.
    """
    text = row_id.astype(str)
    parts = text.str.split("_", n=1, expand=True)

    if parts.shape[1] != 2:
        raise ValueError("ROW_ID does not follow YYYYMMDD_ISSUER format")

    snapshot_dt = pd.to_datetime(
        parts[0],
        format="%Y%m%d",
        errors="raise",
    )

    issuer_id = parts[1].astype(str)

    return pd.DataFrame(
        {
            "SNAPSHOT_DT_CONTROL": snapshot_dt,
            "QUARTER_CONTROL": snapshot_dt.dt.to_period("Q").astype(str),
            "ISSUER_ID_CONTROL": issuer_id,
        },
        index=row_id.index,
    )


def top_k_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    fraction: float,
) -> dict:
    n = len(y)
    selected_n = max(1, int(np.ceil(n * fraction)))

    # Stable sorting makes tied-score handling reproducible.
    order = np.argsort(-probability, kind="mergesort")
    selected = order[:selected_n]

    events_total = int(y.sum())
    events_caught = int(y[selected].sum())
    prevalence = float(y.mean())

    precision = events_caught / selected_n
    recall = events_caught / events_total if events_total else np.nan

    return {
        "top_fraction": float(fraction),
        "selected_rows": int(selected_n),
        "events_caught": events_caught,
        "precision": float(precision),
        "recall": float(recall),
        "precision_lift_over_event_rate": (
            float(precision / prevalence)
            if prevalence > 0
            else None
        ),
        "random_expected_events": float(selected_n * prevalence),
        "events_over_random_expectation": (
            float(events_caught / (selected_n * prevalence))
            if prevalence > 0
            else None
        ),
    }


def model_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    model_name: str,
) -> tuple[dict, pd.DataFrame]:
    prevalence = float(y.mean())
    ap = float(average_precision_score(y, probability))

    summary = {
        "model": model_name,
        "n": int(len(y)),
        "events": int(y.sum()),
        "event_rate": prevalence,
        "average_precision": ap,
        "ap_over_event_rate": (
            float(ap / prevalence)
            if prevalence > 0
            else None
        ),
        "roc_auc": float(roc_auc_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "log_loss": float(
            log_loss(
                y,
                np.clip(probability, 1e-15, 1 - 1e-15),
                labels=[0, 1],
            )
        ),
        "mean_predicted_risk": float(probability.mean()),
        "prediction_minus_event_rate": float(
            probability.mean() - prevalence
        ),
    }

    top_rows = [
        {
            "model": model_name,
            **top_k_metrics(y, probability, fraction),
        }
        for fraction in TOP_FRACTIONS
    ]

    return summary, pd.DataFrame(top_rows)


def weighted_average_precision_sorted(
    sorted_y: np.ndarray,
    sorted_group_code: np.ndarray,
    group_multiplicity: np.ndarray,
) -> float:
    """
    Average precision after cluster resampling without physically duplicating
    all rows. group_multiplicity contains the number of times each issuer was
    selected in the bootstrap replicate.
    """
    weights = group_multiplicity[sorted_group_code].astype(float)

    weighted_positive = weights * sorted_y
    total_positive = weighted_positive.sum()

    if total_positive <= 0:
        return np.nan

    cumulative_positive = np.cumsum(weighted_positive)
    cumulative_total = np.cumsum(weights)

    valid = cumulative_total > 0

    precision = np.zeros_like(cumulative_positive, dtype=float)
    precision[valid] = (
        cumulative_positive[valid] / cumulative_total[valid]
    )

    return float(
        np.sum(precision * weighted_positive) / total_positive
    )


def paired_cluster_bootstrap(
    frame: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    issuer_codes, issuer_values = pd.factorize(
        frame["ISSUER_ID_CONTROL"],
        sort=True,
    )

    n_issuers = len(issuer_values)
    y = frame[TARGET].to_numpy(dtype=np.int8)
    human = frame["human_probability"].to_numpy(dtype=float)
    baseline = frame["baseline_probability"].to_numpy(dtype=float)

    human_order = np.argsort(-human, kind="mergesort")
    baseline_order = np.argsort(-baseline, kind="mergesort")

    human_y_sorted = y[human_order]
    human_group_sorted = issuer_codes[human_order]

    baseline_y_sorted = y[baseline_order]
    baseline_group_sorted = issuer_codes[baseline_order]

    rng = np.random.default_rng(seed)
    records = []

    for replicate in range(repetitions):
        sampled_issuers = rng.integers(
            low=0,
            high=n_issuers,
            size=n_issuers,
        )

        multiplicity = np.bincount(
            sampled_issuers,
            minlength=n_issuers,
        )

        human_ap = weighted_average_precision_sorted(
            human_y_sorted,
            human_group_sorted,
            multiplicity,
        )

        baseline_ap = weighted_average_precision_sorted(
            baseline_y_sorted,
            baseline_group_sorted,
            multiplicity,
        )

        records.append(
            {
                "replicate": replicate,
                "human_average_precision": human_ap,
                "baseline_average_precision": baseline_ap,
                "human_minus_baseline_ap": human_ap - baseline_ap,
            }
        )

        if (replicate + 1) % 250 == 0:
            print(
                f"Bootstrap progress: {replicate + 1}/{repetitions}",
                flush=True,
            )

    return pd.DataFrame(records)


def interval_record(
    values: pd.Series,
    metric: str,
) -> dict:
    clean = values.dropna()

    return {
        "metric": metric,
        "bootstrap_repetitions_valid": int(len(clean)),
        "bootstrap_mean": float(clean.mean()),
        "bootstrap_median": float(clean.median()),
        "ci_2_5_pct": float(clean.quantile(0.025)),
        "ci_97_5_pct": float(clean.quantile(0.975)),
    }


def quarterly_metrics(
    frame: pd.DataFrame,
    probability_column: str,
    model_name: str,
) -> pd.DataFrame:
    records = []

    for quarter, group in frame.groupby(
        "QUARTER_CONTROL",
        sort=True,
    ):
        y = group[TARGET].to_numpy(dtype=np.int8)
        probability = group[probability_column].to_numpy(dtype=float)

        events = int(y.sum())
        event_rate = float(y.mean())

        records.append(
            {
                "model": model_name,
                "quarter": quarter,
                "n": int(len(group)),
                "events": events,
                "event_rate": event_rate,
                "average_precision": (
                    float(average_precision_score(y, probability))
                    if events > 0
                    else np.nan
                ),
                "ap_over_event_rate": (
                    float(
                        average_precision_score(y, probability)
                        / event_rate
                    )
                    if events > 0 and event_rate > 0
                    else np.nan
                ),
                "brier_score": float(
                    brier_score_loss(y, probability)
                ),
                "mean_predicted_risk": float(probability.mean()),
                "prediction_minus_event_rate": float(
                    probability.mean() - event_rate
                ),
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--labels", required=True)
    parser.add_argument("--human", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260813)

    args = parser.parse_args()

    labels_path = Path(args.labels)
    human_path = Path(args.human)
    baseline_path = Path(args.baseline)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    labels = read_labels(labels_path)
    human = read_prediction(human_path, "human_probability")
    baseline = read_prediction(
        baseline_path,
        "baseline_probability",
    )

    frame = (
        labels
        .merge(
            human,
            on=ROW_ID,
            how="outer",
            validate="one_to_one",
        )
        .merge(
            baseline,
            on=ROW_ID,
            how="outer",
            validate="one_to_one",
        )
    )

    required = [
        TARGET,
        "human_probability",
        "baseline_probability",
    ]

    if frame[required].isna().any().any():
        raise ValueError(
            "Labels, human predictions and baseline predictions "
            "do not contain identical ROW_ID sets"
        )

    control = parse_control_columns(frame[ROW_ID])
    frame = pd.concat([frame, control], axis=1)

    y = frame[TARGET].to_numpy(dtype=np.int8)

    human_summary, human_top = model_metrics(
        y,
        frame["human_probability"].to_numpy(dtype=float),
        "human_protocol_v2_2_lightgbm",
    )

    baseline_summary, baseline_top = model_metrics(
        y,
        frame["baseline_probability"].to_numpy(dtype=float),
        "class_weighted_logistic_baseline",
    )

    summaries = pd.DataFrame(
        [human_summary, baseline_summary]
    )

    top_k = pd.concat(
        [human_top, baseline_top],
        ignore_index=True,
    )

    human_quarterly = quarterly_metrics(
        frame,
        "human_probability",
        "human_protocol_v2_2_lightgbm",
    )

    baseline_quarterly = quarterly_metrics(
        frame,
        "baseline_probability",
        "class_weighted_logistic_baseline",
    )

    quarterly = pd.concat(
        [human_quarterly, baseline_quarterly],
        ignore_index=True,
    )

    bootstrap = paired_cluster_bootstrap(
        frame,
        repetitions=args.bootstrap,
        seed=args.seed,
    )

    intervals = pd.DataFrame(
        [
            interval_record(
                bootstrap["human_average_precision"],
                "human_average_precision",
            ),
            interval_record(
                bootstrap["baseline_average_precision"],
                "baseline_average_precision",
            ),
            interval_record(
                bootstrap["human_minus_baseline_ap"],
                "human_minus_baseline_ap",
            ),
        ]
    )

    summaries.to_csv(
        output_dir / "overall_metrics.csv",
        index=False,
    )

    top_k.to_csv(
        output_dir / "top_k_capacity_metrics.csv",
        index=False,
    )

    quarterly.to_csv(
        output_dir / "quarterly_metrics.csv",
        index=False,
    )

    bootstrap.to_csv(
        output_dir / "issuer_cluster_bootstrap.csv",
        index=False,
    )

    intervals.to_csv(
        output_dir / "issuer_cluster_bootstrap_intervals.csv",
        index=False,
    )

    scope = {
        "analysis_type": (
            "POST_HOC_DIAGNOSTIC_OF_ALREADY_LOCKED_AND_SCORED_MODELS"
        ),
        "model_selection_use_allowed": False,
        "feature_selection_use_allowed": False,
        "hyperparameter_selection_use_allowed": False,
        "new_rdagent_candidate_selection_use_allowed": False,
        "labels_path": str(labels_path),
        "human_prediction_path": str(human_path),
        "baseline_prediction_path": str(baseline_path),
        "labels_sha256": sha256_file(labels_path),
        "human_prediction_sha256": sha256_file(human_path),
        "baseline_prediction_sha256": sha256_file(baseline_path),
        "human_submission_lock_expected": (
            "ec7b22e30544afc6db184d155ae30a7e8"
            "ef97ffbd729d39b02a515967a5c99fa"
        ),
        "human_submission_lock_passed": (
            sha256_file(human_path)
            == (
                "ec7b22e30544afc6db184d155ae30a7e8"
                "ef97ffbd729d39b02a515967a5c99fa"
            )
        ),
        "bootstrap_unit": "issuer",
        "bootstrap_repetitions": args.bootstrap,
        "bootstrap_seed": args.seed,
        "row_id_role": (
            "EXTERNAL_EVALUATION_CONTROL_ONLY; "
            "NOT_A_PREDICTIVE_FEATURE"
        ),
    }

    with (output_dir / "evaluation_scope.json").open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            scope,
            stream,
            ensure_ascii=False,
            indent=2,
        )

    result = {
        "human": human_summary,
        "baseline": baseline_summary,
        "human_minus_baseline_ap": (
            human_summary["average_precision"]
            - baseline_summary["average_precision"]
        ),
        "cluster_bootstrap_intervals": (
            intervals.to_dict(orient="records")
        ),
    }

    with (output_dir / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            result,
            stream,
            ensure_ascii=False,
            indent=2,
        )

    print("OVERALL METRICS")
    print(summaries.to_string(index=False))

    print()
    print("TOP-K CAPACITY METRICS")
    print(top_k.to_string(index=False))

    print()
    print("ISSUER CLUSTER BOOTSTRAP INTERVALS")
    print(intervals.to_string(index=False))

    print()
    print("PHASE1_TEST_DIAGNOSTICS=PASS")


if __name__ == "__main__":
    main()
