from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "DOWNGRADE_6M"
SPLIT = "SPLIT"
ROW_ID = "ROW_ID"

MODEL_ROLES = {
    "NUMERIC_FEATURE",
    "CATEGORICAL_FEATURE",
}

PROTOCOL_EXCLUDED = {
    ROW_ID,
    SPLIT,
    TARGET,
    "SNAPSHOT_YEAR",
    "SNAPSHOT_QUARTER",
}

ALLOWED_REMOVED_ROLES = {
    "IDENTIFIER_NOT_FEATURE",
    "AUDIT_NOT_FEATURE",
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


def normalize_category(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("__MISSING__")
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-valid", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    train_valid_path = Path(args.train_valid)
    test_path = Path(args.test)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=False)

    print("Reading train.csv...", flush=True)

    train_valid = pd.read_csv(
        train_valid_path,
        low_memory=False,
    )

    print("Reading test.csv...", flush=True)

    test = pd.read_csv(
        test_path,
        low_memory=False,
    )

    manifest = pd.read_csv(manifest_path)

    train = train_valid.loc[
        train_valid[SPLIT].eq("TRAIN")
    ].copy()

    valid = train_valid.loc[
        train_valid[SPLIT].eq("VALID")
    ].copy()

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

    if TARGET in test.columns:
        raise ValueError(
            "TEST feature file unexpectedly contains labels"
        )

    if not test[SPLIT].eq("TEST").all():
        raise ValueError(
            "test.csv contains non-TEST rows"
        )

    manifest_columns = set(manifest["column"])
    export_columns = set(train_valid.columns)

    export_only = sorted(
        export_columns - manifest_columns
    )

    if export_only != [ROW_ID]:
        raise ValueError(
            f"Unexpected export-only fields: {export_only}"
        )

    removed_manifest = manifest.loc[
        ~manifest["column"].isin(train_valid.columns)
    ].copy()

    disallowed_removed = removed_manifest.loc[
        ~removed_manifest["role"].isin(
            ALLOWED_REMOVED_ROLES
        )
    ]

    if not disallowed_removed.empty:
        raise ValueError(
            "A predictive field was removed unexpectedly"
        )

    manifest_lookup = manifest.set_index("column")

    records = []

    common_columns = [
        column
        for column in train.columns
        if column in test.columns
    ]

    for index, column in enumerate(
        common_columns,
        start=1,
    ):
        if index % 20 == 0:
            print(
                f"Field progress: "
                f"{index}/{len(common_columns)}",
                flush=True,
            )

        if column == ROW_ID:
            role = (
                "EXPORT_CONTROL_IDENTIFIER_NOT_FEATURE"
            )
            manifest_dtype = "export-generated"
        elif column in manifest_lookup.index:
            role = str(
                manifest_lookup.loc[column, "role"]
            )
            manifest_dtype = str(
                manifest_lookup.loc[column, "dtype"]
            )
        else:
            raise ValueError(
                f"Undocumented field: {column}"
            )

        train_series = train[column]
        valid_series = valid[column]
        test_series = test[column]

        categorical = (
            role == "CATEGORICAL_FEATURE"
            or not pd.api.types.is_numeric_dtype(
                train_series
            )
        )

        eligible = (
            role in MODEL_ROLES
            and column not in PROTOCOL_EXCLUDED
        )

        train_missing = float(
            train_series.isna().mean()
        )

        valid_missing = float(
            valid_series.isna().mean()
        )

        test_missing = float(
            test_series.isna().mean()
        )

        record = {
            "column": column,
            "role": role,
            "manifest_dtype": manifest_dtype,
            "loaded_dtype": str(train_series.dtype),
            "eligible_model_feature": eligible,
            "drift_field_type": (
                "categorical"
                if categorical
                else "numeric"
            ),
            "train_missing_rate": train_missing,
            "valid_missing_rate": valid_missing,
            "test_missing_rate": test_missing,
            "missing_shift_train_valid": (
                valid_missing - train_missing
            ),
            "missing_shift_valid_test": (
                test_missing - valid_missing
            ),
            "missing_shift_train_test": (
                test_missing - train_missing
            ),
            "train_nunique": int(
                train_series.nunique(dropna=True)
            ),
            "valid_nunique": int(
                valid_series.nunique(dropna=True)
            ),
            "test_nunique": int(
                test_series.nunique(dropna=True)
            ),
            "valid_unknown_count_vs_train": 0,
            "valid_unknown_rate_vs_train": 0.0,
            "test_unknown_count_vs_train": 0,
            "test_unknown_rate_vs_train": 0.0,
        }

        if categorical and eligible:
            train_category = normalize_category(
                train_series
            )

            valid_category = normalize_category(
                valid_series
            )

            test_category = normalize_category(
                test_series
            )

            known = set(train_category.unique())

            valid_unknown = ~valid_category.isin(
                known
            )

            test_unknown = ~test_category.isin(
                known
            )

            record[
                "valid_unknown_count_vs_train"
            ] = int(valid_unknown.sum())

            record[
                "valid_unknown_rate_vs_train"
            ] = float(valid_unknown.mean())

            record[
                "test_unknown_count_vs_train"
            ] = int(test_unknown.sum())

            record[
                "test_unknown_rate_vs_train"
            ] = float(test_unknown.mean())

        records.append(record)

    inventory = pd.DataFrame(records)

    eligible = inventory.loc[
        inventory["eligible_model_feature"]
    ].copy()

    missingness = eligible.copy()

    for comparison in (
        "train_valid",
        "valid_test",
        "train_test",
    ):
        missingness[
            f"abs_missing_shift_{comparison}"
        ] = missingness[
            f"missing_shift_{comparison}"
        ].abs()

    missingness = missingness.sort_values(
        [
            "abs_missing_shift_train_test",
            "abs_missing_shift_valid_test",
        ],
        ascending=False,
    )

    categorical_unknowns = eligible.loc[
        eligible["drift_field_type"].eq(
            "categorical"
        )
    ].copy()

    categorical_unknowns = (
        categorical_unknowns.sort_values(
            [
                "test_unknown_rate_vs_train",
                "valid_unknown_rate_vs_train",
            ],
            ascending=False,
        )
    )

    inventory.to_csv(
        output_dir / "field_inventory.csv",
        index=False,
    )

    missingness.to_csv(
        output_dir / "missingness_shifts.csv",
        index=False,
    )

    categorical_unknowns.to_csv(
        output_dir / "categorical_unknowns.csv",
        index=False,
    )

    removed_manifest[
        ["column", "role", "dtype", "missing_pct"]
    ].to_csv(
        output_dir
        / "structurally_removed_audit_fields.csv",
        index=False,
    )

    train_ids = (
        train_valid[ROW_ID]
        .astype(str)
        .str.split("_", n=1, expand=True)
    )

    test_ids = (
        test[ROW_ID]
        .astype(str)
        .str.split("_", n=1, expand=True)
    )

    development_issuers = set(train_ids[1])
    test_issuers = set(test_ids[1])

    summary = {
        "analysis": "PHASE2_DRIFT_PREFLIGHT",
        "hidden_test_labels_accessed": False,
        "model_predictions_accessed": False,
        "row_id_role": (
            "EXPORT_CONTROL_IDENTIFIER_NOT_FEATURE"
        ),
        "row_id_allowed_for_model": False,
        "train_valid_sha256": sha256(
            train_valid_path
        ),
        "test_sha256": sha256(test_path),
        "manifest_sha256": sha256(
            manifest_path
        ),
        "train_rows": len(train),
        "valid_rows": len(valid),
        "test_rows": len(test),
        "development_unique_issuers": len(
            development_issuers
        ),
        "test_unique_issuers": len(
            test_issuers
        ),
        "overlapping_issuers": len(
            development_issuers & test_issuers
        ),
        "new_test_issuers": len(
            test_issuers - development_issuers
        ),
        "manifest_fields": len(manifest),
        "model_export_fields_including_target": (
            len(train_valid.columns)
        ),
        "structurally_removed_manifest_fields": (
            len(removed_manifest)
        ),
        "export_generated_control_fields": [
            ROW_ID
        ],
        "eligible_model_features": int(
            len(eligible)
        ),
        "eligible_numeric_features": int(
            eligible["drift_field_type"]
            .eq("numeric")
            .sum()
        ),
        "eligible_categorical_features": int(
            eligible["drift_field_type"]
            .eq("categorical")
            .sum()
        ),
        "categorical_fields_with_test_unknowns": int(
            (
                categorical_unknowns[
                    "test_unknown_rate_vs_train"
                ] > 0
            ).sum()
        ),
        "fields_with_abs_train_test_missing_shift_ge_5pct": int(
            (
                missingness[
                    "abs_missing_shift_train_test"
                ] >= 0.05
            ).sum()
        ),
    }

    with (
        output_dir / "dataset_summary.json"
    ).open("w", encoding="utf-8") as stream:
        json.dump(
            summary,
            stream,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("DATASET SUMMARY")

    for key, value in summary.items():
        print(f"{key}={value}")

    print()
    print(
        "TOP 20 ABSOLUTE TRAIN-TEST "
        "MISSINGNESS SHIFTS"
    )

    missing_columns = [
        "column",
        "role",
        "train_missing_rate",
        "valid_missing_rate",
        "test_missing_rate",
        "missing_shift_train_test",
    ]

    print(
        missingness[
            missing_columns
        ].head(20).to_string(index=False)
    )

    print()
    print("CATEGORICAL UNKNOWN RATES")

    category_columns = [
        "column",
        "train_nunique",
        "valid_nunique",
        "test_nunique",
        "valid_unknown_count_vs_train",
        "valid_unknown_rate_vs_train",
        "test_unknown_count_vs_train",
        "test_unknown_rate_vs_train",
    ]

    print(
        categorical_unknowns[
            category_columns
        ].to_string(index=False)
    )

    print()
    print("PHASE2_DRIFT_PREFLIGHT=PASS")


if __name__ == "__main__":
    main()
