from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "DOWNGRADE_6M"
SPLIT = "SPLIT"

EPSILON = 1e-6
NUMERIC_QUANTILE_BINS = 10
LOW_CARDINALITY_THRESHOLD = 10
RARE_CATEGORY_THRESHOLD = 0.001

PSI_LEVELS = {
    "low": 0.10,
    "moderate": 0.25,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def psi_level(value: float) -> str:
    if not np.isfinite(value):
        return "not_computable"

    if value < PSI_LEVELS["low"]:
        return "low"

    if value < PSI_LEVELS["moderate"]:
        return "moderate"

    return "high"


def stable_psi(
    reference_bucket: pd.Series,
    comparison_bucket: pd.Series,
) -> tuple[float, pd.DataFrame]:
    reference_text = (
        reference_bucket
        .astype("string")
        .fillna("__MISSING__")
    )

    comparison_text = (
        comparison_bucket
        .astype("string")
        .fillna("__MISSING__")
    )

    levels = sorted(
        set(reference_text.unique())
        | set(comparison_text.unique())
    )

    reference_distribution = (
        reference_text
        .value_counts(normalize=True)
        .reindex(levels, fill_value=0.0)
    )

    comparison_distribution = (
        comparison_text
        .value_counts(normalize=True)
        .reindex(levels, fill_value=0.0)
    )

    reference_smoothed = (
        reference_distribution + EPSILON
    )

    comparison_smoothed = (
        comparison_distribution + EPSILON
    )

    contribution = (
        comparison_smoothed - reference_smoothed
    ) * np.log(
        comparison_smoothed / reference_smoothed
    )

    detail = pd.DataFrame(
        {
            "bucket": levels,
            "reference_proportion": (
                reference_distribution.to_numpy()
            ),
            "comparison_proportion": (
                comparison_distribution.to_numpy()
            ),
            "psi_contribution": contribution.to_numpy(),
        }
    )

    return float(contribution.sum()), detail


def fit_numeric_bins(
    reference: pd.Series,
) -> np.ndarray | None:
    numeric = pd.to_numeric(
        reference,
        errors="coerce",
    )

    observed = numeric.dropna()

    if observed.nunique() <= 1:
        return None

    quantiles = observed.quantile(
        np.linspace(
            0,
            1,
            NUMERIC_QUANTILE_BINS + 1,
        )
    ).to_numpy(dtype=float)

    edges = np.unique(quantiles)

    if len(edges) < 3:
        return None

    edges[0] = -np.inf
    edges[-1] = np.inf

    return edges


def apply_numeric_bins(
    series: pd.Series,
    edges: np.ndarray,
) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    bucket = pd.cut(
        numeric,
        bins=edges,
        include_lowest=True,
        duplicates="drop",
    ).astype("string")

    return bucket.fillna("__MISSING__")


def fit_category_mapping(
    reference: pd.Series,
) -> dict:
    text = (
        reference
        .astype("string")
        .fillna("__MISSING__")
    )

    frequencies = text.value_counts(
        normalize=True,
        dropna=False,
    )

    retained = set(
        frequencies.loc[
            frequencies >= RARE_CATEGORY_THRESHOLD
        ].index.astype(str)
    )

    retained.add("__MISSING__")

    return {
        "retained": retained,
        "observed": set(text.astype(str).unique()),
    }


def apply_category_mapping(
    series: pd.Series,
    mapping: dict,
    is_reference: bool,
) -> pd.Series:
    text = (
        series
        .astype("string")
        .fillna("__MISSING__")
        .astype(str)
    )

    observed = mapping["observed"]
    retained = mapping["retained"]

    def transform(value: str) -> str:
        if value in retained:
            return value

        if value not in observed and not is_reference:
            return "__UNSEEN__"

        return "__RARE__"

    return text.map(transform)


def comparison_psi(
    feature: str,
    reference: pd.Series,
    comparison: pd.Series,
    field_type: str,
    comparison_name: str,
) -> tuple[dict, pd.DataFrame]:
    reference_missing = float(
        reference.isna().mean()
    )

    comparison_missing = float(
        comparison.isna().mean()
    )

    reference_unique = int(
        reference.nunique(dropna=True)
    )

    if (
        field_type == "numeric"
        and reference_unique
        > LOW_CARDINALITY_THRESHOLD
    ):
        method = "numeric_reference_quantile_bins"
        edges = fit_numeric_bins(reference)

        if edges is None:
            method = "discrete_category_mapping"

            mapping = fit_category_mapping(reference)

            reference_bucket = apply_category_mapping(
                reference,
                mapping,
                is_reference=True,
            )

            comparison_bucket = apply_category_mapping(
                comparison,
                mapping,
                is_reference=False,
            )
        else:
            reference_bucket = apply_numeric_bins(
                reference,
                edges,
            )

            comparison_bucket = apply_numeric_bins(
                comparison,
                edges,
            )
    else:
        method = "discrete_category_mapping"

        mapping = fit_category_mapping(reference)

        reference_bucket = apply_category_mapping(
            reference,
            mapping,
            is_reference=True,
        )

        comparison_bucket = apply_category_mapping(
            comparison,
            mapping,
            is_reference=False,
        )

    value, detail = stable_psi(
        reference_bucket,
        comparison_bucket,
    )

    comparison_text = (
        comparison_bucket.astype(str)
    )

    record = {
        "feature": feature,
        "comparison": comparison_name,
        "method": method,
        "reference_unique": reference_unique,
        "comparison_unique": int(
            comparison.nunique(dropna=True)
        ),
        "reference_missing_rate": reference_missing,
        "comparison_missing_rate": comparison_missing,
        "missing_rate_shift": (
            comparison_missing - reference_missing
        ),
        "absolute_missing_rate_shift": abs(
            comparison_missing - reference_missing
        ),
        "unseen_bucket_rate": float(
            comparison_text.eq("__UNSEEN__").mean()
        ),
        "psi": value,
        "psi_level": psi_level(value),
        "bucket_count": int(len(detail)),
    }

    detail.insert(0, "feature", feature)
    detail.insert(1, "comparison", comparison_name)
    detail.insert(2, "method", method)

    return record, detail


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-valid", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    train_valid_path = Path(args.train_valid)
    test_path = Path(args.test)
    preflight_dir = Path(args.preflight_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    inventory = pd.read_csv(
        preflight_dir / "field_inventory.csv"
    )

    eligible = inventory.loc[
        inventory["eligible_model_feature"]
    ].copy()

    feature_columns = eligible["column"].tolist()

    required_train_columns = (
        feature_columns + [SPLIT]
    )

    print(
        "Reading selected TRAIN/VALID columns...",
        flush=True,
    )

    train_valid = pd.read_csv(
        train_valid_path,
        usecols=required_train_columns,
        low_memory=False,
    )

    print(
        "Reading selected TEST columns...",
        flush=True,
    )

    test = pd.read_csv(
        test_path,
        usecols=required_train_columns,
        low_memory=False,
    )

    train = train_valid.loc[
        train_valid[SPLIT].eq("TRAIN")
    ]

    valid = train_valid.loc[
        train_valid[SPLIT].eq("VALID")
    ]

    if len(train) != 172566:
        raise ValueError(
            f"Unexpected TRAIN rows: {len(train)}"
        )

    if len(valid) != 71194:
        raise ValueError(
            f"Unexpected VALID rows: {len(valid)}"
        )

    if len(test) != 114603:
        raise ValueError(
            f"Unexpected TEST rows: {len(test)}"
        )

    if not test[SPLIT].eq("TEST").all():
        raise ValueError(
            "Unexpected SPLIT values in test.csv"
        )

    records = []
    detail_frames = []

    field_type_map = dict(
        zip(
            eligible["column"],
            eligible["drift_field_type"],
        )
    )

    comparisons = [
        (
            "train_to_valid",
            train,
            valid,
        ),
        (
            "train_to_test",
            train,
            test,
        ),
        (
            "valid_to_test",
            valid,
            test,
        ),
    ]

    total_operations = (
        len(feature_columns) * len(comparisons)
    )

    operation = 0

    for feature in feature_columns:
        field_type = field_type_map[feature]

        for (
            comparison_name,
            reference_frame,
            comparison_frame,
        ) in comparisons:
            operation += 1

            if operation % 25 == 0:
                print(
                    f"Drift progress: "
                    f"{operation}/{total_operations}",
                    flush=True,
                )

            record, detail = comparison_psi(
                feature=feature,
                reference=reference_frame[feature],
                comparison=comparison_frame[feature],
                field_type=field_type,
                comparison_name=comparison_name,
            )

            record["field_type"] = field_type

            records.append(record)
            detail_frames.append(detail)

    drift = pd.DataFrame(records)

    drift = drift[
        [
            "feature",
            "field_type",
            "comparison",
            "method",
            "reference_unique",
            "comparison_unique",
            "reference_missing_rate",
            "comparison_missing_rate",
            "missing_rate_shift",
            "absolute_missing_rate_shift",
            "unseen_bucket_rate",
            "psi",
            "psi_level",
            "bucket_count",
        ]
    ]

    detail = pd.concat(
        detail_frames,
        ignore_index=True,
    )

    drift.to_csv(
        output_dir / "feature_drift_all_comparisons.csv",
        index=False,
    )

    detail.to_csv(
        output_dir / "psi_bucket_contributions.csv",
        index=False,
    )

    for comparison_name in (
        "train_to_valid",
        "train_to_test",
        "valid_to_test",
    ):
        comparison_frame = drift.loc[
            drift["comparison"].eq(
                comparison_name
            )
        ].sort_values(
            [
                "psi",
                "absolute_missing_rate_shift",
            ],
            ascending=False,
        )

        comparison_frame.to_csv(
            output_dir
            / f"{comparison_name}_drift_ranked.csv",
            index=False,
        )

    pivot = drift.pivot(
        index=[
            "feature",
            "field_type",
        ],
        columns="comparison",
        values="psi",
    ).reset_index()

    pivot.columns.name = None

    pivot["test_acceleration_vs_valid"] = (
        pivot["train_to_test"]
        - pivot["train_to_valid"]
    )

    pivot = pivot.sort_values(
        "train_to_test",
        ascending=False,
    )

    pivot.to_csv(
        output_dir / "psi_comparison_matrix.csv",
        index=False,
    )

    train_test = drift.loc[
        drift["comparison"].eq(
            "train_to_test"
        )
    ]

    valid_test = drift.loc[
        drift["comparison"].eq(
            "valid_to_test"
        )
    ]

    summary = {
        "analysis": "PHASE2_FORMAL_FEATURE_DRIFT",
        "hidden_test_labels_accessed": False,
        "model_predictions_accessed": False,
        "model_selection_use_allowed": False,
        "feature_selection_use_allowed": False,
        "train_valid_sha256": sha256(
            train_valid_path
        ),
        "test_sha256": sha256(test_path),
        "preflight_inventory_sha256": sha256(
            preflight_dir / "field_inventory.csv"
        ),
        "eligible_features": len(feature_columns),
        "numeric_features": int(
            eligible["drift_field_type"]
            .eq("numeric")
            .sum()
        ),
        "categorical_features": int(
            eligible["drift_field_type"]
            .eq("categorical")
            .sum()
        ),
        "comparisons": [
            "train_to_valid",
            "train_to_test",
            "valid_to_test",
        ],
        "numeric_bin_rule": (
            "Reference-period quantile bins; "
            "missing is a separate bucket"
        ),
        "low_cardinality_numeric_rule": (
            "Treated as discrete categories when "
            f"reference unique <= "
            f"{LOW_CARDINALITY_THRESHOLD}"
        ),
        "categorical_rule": (
            "Reference categories below 0.1% pooled "
            "as __RARE__; unseen comparison categories "
            "mapped to __UNSEEN__"
        ),
        "psi_epsilon": EPSILON,
        "psi_thresholds_descriptive": {
            "low": "<0.10",
            "moderate": "0.10 to <0.25",
            "high": ">=0.25",
        },
        "train_test_high_psi_count": int(
            train_test["psi"].ge(0.25).sum()
        ),
        "train_test_moderate_psi_count": int(
            (
                train_test["psi"].ge(0.10)
                & train_test["psi"].lt(0.25)
            ).sum()
        ),
        "valid_test_high_psi_count": int(
            valid_test["psi"].ge(0.25).sum()
        ),
        "valid_test_moderate_psi_count": int(
            (
                valid_test["psi"].ge(0.10)
                & valid_test["psi"].lt(0.25)
            ).sum()
        ),
    }

    with (
        output_dir / "drift_summary.json"
    ).open("w", encoding="utf-8") as stream:
        json.dump(
            summary,
            stream,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("DRIFT SUMMARY")

    for key, value in summary.items():
        print(f"{key}={value}")

    print()
    print("TOP 25 TRAIN-TEST PSI")

    columns = [
        "feature",
        "field_type",
        "method",
        "psi",
        "psi_level",
        "absolute_missing_rate_shift",
        "unseen_bucket_rate",
    ]

    print(
        train_test
        .sort_values("psi", ascending=False)
        [columns]
        .head(25)
        .to_string(index=False)
    )

    print()
    print("TOP 20 VALID-TEST PSI")

    print(
        valid_test
        .sort_values("psi", ascending=False)
        [columns]
        .head(20)
        .to_string(index=False)
    )

    print()
    print("TOP 20 TEST ACCELERATION VS VALID")

    print(
        pivot[
            [
                "feature",
                "field_type",
                "train_to_valid",
                "train_to_test",
                "valid_to_test",
                "test_acceleration_vs_valid",
            ]
        ]
        .sort_values(
            "test_acceleration_vs_valid",
            ascending=False,
        )
        .head(20)
        .to_string(index=False)
    )

    print()
    print("PHASE2_FORMAL_DRIFT=PASS")


if __name__ == "__main__":
    main()
