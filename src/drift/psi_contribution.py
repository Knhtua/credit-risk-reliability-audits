from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def classify_bucket(bucket: str) -> str:
    text = str(bucket)

    if text == "__MISSING__":
        return "missing"

    if text == "__UNSEEN__":
        return "unseen"

    if text == "__RARE__":
        return "rare"

    return "observed"


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--formal-drift-dir",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    formal_dir = Path(args.formal_drift_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    drift_path = (
        formal_dir
        / "feature_drift_all_comparisons.csv"
    )

    detail_path = (
        formal_dir
        / "psi_bucket_contributions.csv"
    )

    matrix_path = (
        formal_dir
        / "psi_comparison_matrix.csv"
    )

    drift = pd.read_csv(drift_path)
    detail = pd.read_csv(detail_path)
    matrix = pd.read_csv(matrix_path)

    detail["bucket_type"] = (
        detail["bucket"].map(classify_bucket)
    )

    selected_pairs = set()

    train_test = drift.loc[
        drift["comparison"].eq("train_to_test")
    ].sort_values("psi", ascending=False)

    valid_test = drift.loc[
        drift["comparison"].eq("valid_to_test")
    ].sort_values("psi", ascending=False)

    for _, row in train_test.head(25).iterrows():
        selected_pairs.add(
            (row["feature"], row["comparison"])
        )

    for _, row in valid_test.head(20).iterrows():
        selected_pairs.add(
            (row["feature"], row["comparison"])
        )

    summaries = []
    top_bucket_frames = []

    for feature, comparison in sorted(selected_pairs):
        feature_drift = drift.loc[
            drift["feature"].eq(feature)
            & drift["comparison"].eq(comparison)
        ]

        if len(feature_drift) != 1:
            raise ValueError(
                f"Expected one drift row for "
                f"{feature}/{comparison}"
            )

        row = feature_drift.iloc[0]

        contribution = detail.loc[
            detail["feature"].eq(feature)
            & detail["comparison"].eq(comparison)
        ].copy()

        if contribution.empty:
            raise ValueError(
                f"No bucket detail for "
                f"{feature}/{comparison}"
            )

        contribution = contribution.sort_values(
            "psi_contribution",
            ascending=False,
        )

        total = float(
            contribution["psi_contribution"].sum()
        )

        missing_contribution = float(
            contribution.loc[
                contribution["bucket_type"].eq(
                    "missing"
                ),
                "psi_contribution",
            ].sum()
        )

        unseen_contribution = float(
            contribution.loc[
                contribution["bucket_type"].eq(
                    "unseen"
                ),
                "psi_contribution",
            ].sum()
        )

        rare_contribution = float(
            contribution.loc[
                contribution["bucket_type"].eq(
                    "rare"
                ),
                "psi_contribution",
            ].sum()
        )

        observed_contribution = float(
            contribution.loc[
                contribution["bucket_type"].eq(
                    "observed"
                ),
                "psi_contribution",
            ].sum()
        )

        if total > 0:
            missing_share = (
                missing_contribution / total
            )

            unseen_share = (
                unseen_contribution / total
            )
        else:
            missing_share = np.nan
            unseen_share = np.nan

        top_bucket = contribution.iloc[0]

        summaries.append(
            {
                "feature": feature,
                "comparison": comparison,
                "field_type": row["field_type"],
                "method": row["method"],
                "psi_reported": row["psi"],
                "psi_recomputed": total,
                "absolute_missing_rate_shift": row[
                    "absolute_missing_rate_shift"
                ],
                "unseen_bucket_rate": row[
                    "unseen_bucket_rate"
                ],
                "missing_psi_contribution": (
                    missing_contribution
                ),
                "missing_contribution_share": (
                    missing_share
                ),
                "unseen_psi_contribution": (
                    unseen_contribution
                ),
                "unseen_contribution_share": (
                    unseen_share
                ),
                "rare_psi_contribution": (
                    rare_contribution
                ),
                "observed_psi_contribution": (
                    observed_contribution
                ),
                "top_bucket": top_bucket["bucket"],
                "top_bucket_type": top_bucket[
                    "bucket_type"
                ],
                "top_bucket_reference_proportion": (
                    top_bucket[
                        "reference_proportion"
                    ]
                ),
                "top_bucket_comparison_proportion": (
                    top_bucket[
                        "comparison_proportion"
                    ]
                ),
                "top_bucket_psi_contribution": (
                    top_bucket[
                        "psi_contribution"
                    ]
                ),
            }
        )

        top_output = contribution.head(5).copy()

        top_output.insert(
            0,
            "feature_total_psi",
            row["psi"],
        )

        top_bucket_frames.append(top_output)

    summary = pd.DataFrame(summaries)

    if not np.allclose(
        summary["psi_reported"],
        summary["psi_recomputed"],
        rtol=0,
        atol=1e-10,
    ):
        mismatch = summary.loc[
            ~np.isclose(
                summary["psi_reported"],
                summary["psi_recomputed"],
                rtol=0,
                atol=1e-10,
            )
        ]

        raise ValueError(
            "PSI reconstruction mismatch:\n"
            + mismatch.to_string(index=False)
        )

    top_buckets = pd.concat(
        top_bucket_frames,
        ignore_index=True,
    )

    summary = summary.sort_values(
        [
            "comparison",
            "psi_reported",
        ],
        ascending=[True, False],
    )

    summary.to_csv(
        output_dir / "psi_contribution_summary.csv",
        index=False,
    )

    top_buckets.to_csv(
        output_dir / "top5_bucket_contributions.csv",
        index=False,
    )

    missing_driven = summary.loc[
        summary[
            "missing_contribution_share"
        ].ge(0.50)
    ].copy()

    missing_driven.to_csv(
        output_dir / "missing_driven_drift.csv",
        index=False,
    )

    unseen_driven = summary.loc[
        summary[
            "unseen_contribution_share"
        ].ge(0.25)
    ].copy()

    unseen_driven.to_csv(
        output_dir / "unseen_driven_drift.csv",
        index=False,
    )

    feature_groups = {
        "rating_agency_representation": [
            "RATING_AGENCY_CODE",
            "RATING_AGENCY_NAME",
        ],
        "bank_cash_level_representation": [
            "BALANCE_BANK_CASH",
            "BALANCE_BANK_CASH_SLOG",
        ],
        "bank_cash_change_and_cash_ratio": [
            "BALANCE_BANK_CASH_YOY_GROWTH",
            "CASH_RATIO_YOY_CHANGE",
            "CASH_RATIO",
        ],
        "rating_age_representation": [
            "RATING_AGE_DAYS",
            "RATING_AGE_DAYS_YOY_CHANGE",
        ],
        "balance_sheet_growth_family": [
            "BALANCE_ASSETS_TOT_YOY_GROWTH",
            "BALANCE_LA_TOT_YOY_GROWTH",
            "BALANCE_EQUITY_DERIVED_YOY_GROWTH",
        ],
        "capital_structure_change_family": [
            "LEV_RATIO_YOY_CHANGE",
            "EQUITY_RATIO_YOY_CHANGE",
        ],
    }

    group_rows = []

    for group_name, features in feature_groups.items():
        available = matrix.loc[
            matrix["feature"].isin(features)
        ].copy()

        group_rows.append(
            {
                "group": group_name,
                "expected_features": "|".join(features),
                "available_feature_count": len(available),
                "available_features": "|".join(
                    available["feature"].tolist()
                ),
                "maximum_train_test_psi": (
                    float(
                        available[
                            "train_to_test"
                        ].max()
                    )
                    if not available.empty
                    else np.nan
                ),
                "mean_train_test_psi": (
                    float(
                        available[
                            "train_to_test"
                        ].mean()
                    )
                    if not available.empty
                    else np.nan
                ),
                "maximum_valid_test_psi": (
                    float(
                        available[
                            "valid_to_test"
                        ].max()
                    )
                    if not available.empty
                    else np.nan
                ),
            }
        )

    group_frame = pd.DataFrame(group_rows)

    group_frame.to_csv(
        output_dir / "related_feature_groups.csv",
        index=False,
    )

    train_test_summary = summary.loc[
        summary["comparison"].eq(
            "train_to_test"
        )
    ].copy()

    valid_test_summary = summary.loc[
        summary["comparison"].eq(
            "valid_to_test"
        )
    ].copy()

    result_summary = {
        "analysis": "PHASE2_PSI_CONTRIBUTION_AUDIT",
        "hidden_test_labels_accessed": False,
        "model_predictions_accessed": False,
        "drift_file_sha256": sha256(
            drift_path
        ),
        "bucket_detail_sha256": sha256(
            detail_path
        ),
        "audited_feature_comparison_pairs": int(
            len(summary)
        ),
        "train_test_pairs_audited": int(
            len(train_test_summary)
        ),
        "valid_test_pairs_audited": int(
            len(valid_test_summary)
        ),
        "missing_driven_pair_count": int(
            len(missing_driven)
        ),
        "unseen_driven_pair_count": int(
            len(unseen_driven)
        ),
        "train_test_high_psi_missing_driven_count": int(
            (
                train_test_summary[
                    "psi_reported"
                ].ge(0.25)
                & train_test_summary[
                    "missing_contribution_share"
                ].ge(0.50)
            ).sum()
        ),
        "interpretation_rule": (
            "Related transformations are summarized "
            "as feature families and are not counted "
            "as independent drift mechanisms."
        ),
    }

    with (
        output_dir / "contribution_audit_summary.json"
    ).open("w", encoding="utf-8") as stream:
        json.dump(
            result_summary,
            stream,
            ensure_ascii=False,
            indent=2,
        )

    print("CONTRIBUTION AUDIT SUMMARY")

    for key, value in result_summary.items():
        print(f"{key}={value}")

    print()
    print("TRAIN-TEST CONTRIBUTION SUMMARY")

    show_columns = [
        "feature",
        "psi_reported",
        "absolute_missing_rate_shift",
        "missing_contribution_share",
        "unseen_bucket_rate",
        "unseen_contribution_share",
        "top_bucket",
        "top_bucket_type",
        "top_bucket_reference_proportion",
        "top_bucket_comparison_proportion",
        "top_bucket_psi_contribution",
    ]

    print(
        train_test_summary[
            show_columns
        ].head(25).to_string(index=False)
    )

    print()
    print("VALID-TEST CONTRIBUTION SUMMARY")

    print(
        valid_test_summary[
            show_columns
        ].head(20).to_string(index=False)
    )

    print()
    print("RELATED FEATURE GROUPS")

    print(
        group_frame.to_string(index=False)
    )

    print()
    print("PHASE2_PSI_CONTRIBUTION_AUDIT=PASS")


if __name__ == "__main__":
    main()
