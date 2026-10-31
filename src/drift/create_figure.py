from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXCLUDED_EQUIVALENT_FEATURES = {
    "RATING_AGENCY_NAME": (
        "Duplicate representation of RATING_AGENCY_CODE"
    ),
    "BALANCE_BANK_CASH_SLOG": (
        "Monotonic transformed representation of "
        "BALANCE_BANK_CASH"
    ),
    "EQUITY_RATIO_YOY_CHANGE": (
        "Algebraically related to LEV_RATIO_YOY_CHANGE"
    ),
}


DISPLAY_NAMES = {
    "BALANCE_BANK_CASH_YOY_GROWTH":
        "Bank cash YoY growth",
    "CASH_RATIO_YOY_CHANGE":
        "Cash ratio YoY change",
    "BALANCE_BANK_CASH":
        "Bank cash level",
    "CASH_RATIO":
        "Cash ratio",
    "RATING_AGENCY_CODE":
        "Rating agency",
    "RATING_AGE_DAYS":
        "Rating age",
    "RATING_AGE_DAYS_YOY_CHANGE":
        "Rating-age YoY change",
    "BALANCE_EQUITY_DERIVED_YOY_GROWTH":
        "Equity YoY growth",
    "BALANCE_ASSETS_TOT_YOY_GROWTH":
        "Total-assets YoY growth",
    "CASHFLOW_CASH_EQU_END_PRD_YOY_GROWTH":
        "Ending-cash YoY growth",
    "BALANCE_LA_TOT_YOY_GROWTH":
        "Liabilities YoY growth",
    "LEV_RATIO_YOY_CHANGE":
        "Leverage YoY change",
    "PRETAX_ROA_ANNUALIZED_YOY_CHANGE":
        "Annualized ROA YoY change",
    "INCOME_OP_PROFIT_YOY_GROWTH":
        "Operating-profit YoY growth",
    "MEAN_FINANCIAL_AGE_DAYS":
        "Mean financial-data age",
    "INCOME_OP_INCM_TOT_YOY_GROWTH":
        "Operating-income YoY growth",
    "PRETAX_ROA_ANNUALIZED":
        "Annualized pretax ROA",
    "CURRENT_RANK":
        "Current rating",
    "BALANCE_AGE_DAYS":
        "Balance-sheet age",
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


def display_name(feature: str) -> str:
    return DISPLAY_NAMES.get(
        feature,
        feature.replace("_", " ").title(),
    )


def prepare_panel(
    summary: pd.DataFrame,
    comparison: str,
    top_n: int,
) -> pd.DataFrame:
    panel = summary.loc[
        summary["comparison"].eq(comparison)
        & ~summary["feature"].isin(
            EXCLUDED_EQUIVALENT_FEATURES
        )
    ].copy()

    panel = (
        panel
        .sort_values(
            "psi_reported",
            ascending=False,
        )
        .head(top_n)
        .copy()
    )

    panel["missing_component"] = (
        panel["missing_psi_contribution"]
    )

    panel["other_component"] = (
        panel["psi_reported"]
        - panel["missing_component"]
    ).clip(lower=0)

    panel["display_name"] = (
        panel["feature"].map(display_name)
    )

    return panel.sort_values(
        "psi_reported",
        ascending=True,
    )


def draw_panel(
    axis,
    frame: pd.DataFrame,
    title: str,
    x_limit: float,
) -> None:
    y = np.arange(len(frame))

    missing = frame[
        "missing_component"
    ].to_numpy()

    other = frame[
        "other_component"
    ].to_numpy()

    axis.barh(
        y,
        other,
        color="#4C78A8",
        label="Observed-value/category contribution",
    )

    axis.barh(
        y,
        missing,
        left=other,
        color="#F58518",
        label="Missing-bucket contribution",
    )

    axis.set_yticks(y)
    axis.set_yticklabels(frame["display_name"])
    axis.set_xlim(0, x_limit)
    axis.set_xlabel("Population Stability Index (PSI)")
    axis.set_title(title, fontweight="bold")

    axis.axvline(
        0.10,
        color="#666666",
        linewidth=1,
        linestyle="--",
    )

    axis.axvline(
        0.25,
        color="#333333",
        linewidth=1,
        linestyle=":",
    )

    axis.grid(
        axis="x",
        alpha=0.20,
        linewidth=0.8,
    )

    axis.set_axisbelow(True)

    for position, (_, row) in enumerate(
        frame.iterrows()
    ):
        axis.text(
            row["psi_reported"]
            + x_limit * 0.012,
            position,
            f"{row['psi_reported']:.2f}",
            va="center",
            fontsize=8,
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--contribution-dir",
        required=True,
    )

    parser.add_argument(
        "--figure-dir",
        required=True,
    )

    parser.add_argument(
        "--result-dir",
        required=True,
    )

    args = parser.parse_args()

    contribution_dir = Path(
        args.contribution_dir
    )

    figure_dir = Path(args.figure_dir)
    result_dir = Path(args.result_dir)

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    summary_path = (
        contribution_dir
        / "psi_contribution_summary.csv"
    )

    summary = pd.read_csv(summary_path)

    train_test = prepare_panel(
        summary,
        comparison="train_to_test",
        top_n=12,
    )

    valid_test = prepare_panel(
        summary,
        comparison="valid_to_test",
        top_n=10,
    )

    train_test.to_csv(
        result_dir
        / "figure_train_test_features.csv",
        index=False,
    )

    valid_test.to_csv(
        result_dir
        / "figure_valid_test_features.csv",
        index=False,
    )

    exclusions = pd.DataFrame(
        [
            {
                "feature": feature,
                "reason": reason,
            }
            for feature, reason
            in EXCLUDED_EQUIVALENT_FEATURES.items()
        ]
    )

    exclusions.to_csv(
        result_dir
        / "figure_equivalent_feature_exclusions.csv",
        index=False,
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    figure, axes = plt.subplots(
        ncols=2,
        figsize=(15, 7.6),
        gridspec_kw={
            "width_ratios": [1.35, 1.0],
            "wspace": 0.48,
        },
    )

    train_limit = max(
        0.5,
        float(
            train_test[
                "psi_reported"
            ].max()
        ) * 1.15,
    )

    valid_limit = max(
        0.5,
        float(
            valid_test[
                "psi_reported"
            ].max()
        ) * 1.18,
    )

    draw_panel(
        axes[0],
        train_test,
        "A. Long-horizon drift: TRAIN to TEST",
        train_limit,
    )

    draw_panel(
        axes[1],
        valid_test,
        "B. Recent drift: VALID to TEST",
        valid_limit,
    )

    handles, labels = axes[0].get_legend_handles_labels()

    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )

    figure.suptitle(
        "Feature-distribution drift and its PSI components",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )

    figure.text(
        0.5,
        0.045,
        (
            "Dashed line: PSI = 0.10; "
            "dotted line: PSI = 0.25. "
            "Equivalent feature representations "
            "are consolidated."
        ),
        ha="center",
        fontsize=9,
        color="#444444",
    )

    figure.subplots_adjust(
        top=0.91,
        bottom=0.14,
        left=0.20,
        right=0.97,
    )

    png_path = (
        figure_dir
        / "feature_drift_psi_components.png"
    )

    pdf_path = (
        figure_dir
        / "feature_drift_psi_components.pdf"
    )

    figure.savefig(
        png_path,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)

    metadata = {
        "figure": (
            "Feature-distribution drift and "
            "its PSI components"
        ),
        "source_sha256": sha256(summary_path),
        "png_path": str(png_path),
        "png_sha256": sha256(png_path),
        "pdf_path": str(pdf_path),
        "pdf_sha256": sha256(pdf_path),
        "train_test_features_shown": (
            train_test["feature"].tolist()
        ),
        "valid_test_features_shown": (
            valid_test["feature"].tolist()
        ),
        "excluded_equivalent_features": (
            EXCLUDED_EQUIVALENT_FEATURES
        ),
        "hidden_test_labels_accessed": False,
        "model_predictions_accessed": False,
    }

    with (
        result_dir / "figure_metadata.json"
    ).open("w", encoding="utf-8") as stream:
        json.dump(
            metadata,
            stream,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "PNG_CREATED =",
        png_path,
    )

    print(
        "PDF_CREATED =",
        pdf_path,
    )

    print(
        "TRAIN_TEST_FEATURES_SHOWN =",
        len(train_test),
    )

    print(
        "VALID_TEST_FEATURES_SHOWN =",
        len(valid_test),
    )

    print(
        "EXCLUDED_EQUIVALENT_FEATURES =",
        sorted(
            EXCLUDED_EQUIVALENT_FEATURES
        ),
    )

    print("PHASE2_DRIFT_FIGURE=PASS")


if __name__ == "__main__":
    main()
