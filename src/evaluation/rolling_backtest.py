from pathlib import Path
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "train.csv"
OUTPUT_DIR = PROJECT_ROOT / "run_artifacts/rolling_time_backtest"

TARGET = "DOWNGRADE_6M"

CONTROL_FIELDS = {
    "ROW_ID",
    "SPLIT",
    TARGET,
    "SNAPSHOT_DT",
    "SNAPSHOT_YEAR",
    "SNAPSHOT_QUARTER",
    "INST_CODE",
    "INST_NAME",
}

FIXED_CATEGORICAL_COLUMNS = [
    "CURRENT_RANK",
    "PREVIOUS_RANK",
    "RATING_OUTLOOK",
    "RATING_AGENCY_NAME",
    "OUTLOOK_BUCKET",
]

FOLDS = [
    {
        "fold": "fold_1",
        "train_end": "2018Q4",
        "validation_start": "2019Q1",
        "validation_end": "2020Q4",
    },
    {
        "fold": "fold_2",
        "train_end": "2020Q4",
        "validation_start": "2021Q1",
        "validation_end": "2022Q4",
    },
]

FIXED_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.01,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "is_unbalance": True,
    "verbosity": -1,
    "n_jobs": -1,
    "seed": 42,
}

NUM_BOOST_ROUND = 6000
EARLY_STOPPING_ROUNDS = 300


def quarter_number(value):
    value = str(value).strip().upper()
    year = int(value[:4])
    quarter = int(value[-1])
    return year * 4 + quarter


def encode_categorical_train_only(x_train, x_valid, categorical_columns):
    audit_rows = []

    for column in categorical_columns:
        train_values = (
            x_train[column]
            .astype("string")
            .fillna("**MISSING**")
        )
        valid_values = (
            x_valid[column]
            .astype("string")
            .fillna("**MISSING**")
        )

        vocabulary = sorted(train_values.unique().tolist())

        for reserved in ["**MISSING**", "**UNKNOWN**"]:
            if reserved not in vocabulary:
                vocabulary.append(reserved)

        unknown_mask = ~valid_values.isin(vocabulary)
        unknown_count = int(unknown_mask.sum())

        valid_values = valid_values.where(
            ~unknown_mask,
            "**UNKNOWN**",
        )

        category_dtype = pd.CategoricalDtype(
            categories=vocabulary,
            ordered=False,
        )

        x_train[column] = train_values.astype(category_dtype)
        x_valid[column] = valid_values.astype(category_dtype)

        audit_rows.append({
            "column": column,
            "train_vocabulary_size": len(vocabulary),
            "validation_unknown_count": unknown_count,
            "validation_unknown_pct": (
                100.0 * unknown_count / len(x_valid)
                if len(x_valid)
                else 0.0
            ),
        })

    return pd.DataFrame(audit_rows)


def prepare_numeric_columns(x_train, x_valid, categorical_columns):
    for column in x_train.columns:
        if column in categorical_columns:
            continue

        if (
            pd.api.types.is_object_dtype(x_train[column])
            or pd.api.types.is_string_dtype(x_train[column])
        ):
            raise RuntimeError(
                f"Unexpected non-numeric model feature: {column}"
            )

        x_train[column] = pd.to_numeric(
            x_train[column],
            errors="coerce",
        )
        x_valid[column] = pd.to_numeric(
            x_valid[column],
            errors="coerce",
        )

    x_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    x_valid.replace([np.inf, -np.inf], np.nan, inplace=True)


def calibration_table(predictions, labels):
    frame = pd.DataFrame({
        "prediction": predictions,
        "label": labels,
    })

    frame["calibration_bin"] = pd.qcut(
        frame["prediction"],
        q=10,
        duplicates="drop",
    )

    result = (
        frame.groupby(
            "calibration_bin",
            observed=True,
        )
        .agg(
            row_count=("label", "size"),
            prediction_mean=("prediction", "mean"),
            observed_event_rate=("label", "mean"),
            prediction_min=("prediction", "min"),
            prediction_max=("prediction", "max"),
        )
        .reset_index()
    )

    result["calibration_bin"] = result[
        "calibration_bin"
    ].astype(str)

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {DATA_FILE}")
    data = pd.read_csv(DATA_FILE, low_memory=False)

    required = {
        "ROW_ID",
        "SPLIT",
        "SNAPSHOT_YEAR",
        "SNAPSHOT_QUARTER",
        TARGET,
    }

    missing = required - set(data.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    data["_QUARTER_NUMBER"] = (
        pd.to_numeric(data["SNAPSHOT_YEAR"], errors="raise").astype(int) * 4
        + pd.to_numeric(
            data["SNAPSHOT_QUARTER"],
            errors="raise",
        ).astype(int)
    )

    feature_columns = [
        column
        for column in data.columns
        if column not in CONTROL_FIELDS
        and column != "_QUARTER_NUMBER"
    ]

    categorical_columns = [
        column
        for column in FIXED_CATEGORICAL_COLUMNS
        if column in feature_columns
    ]

    print("Model feature count:", len(feature_columns))
    print("Categorical features:", categorical_columns)

    result_rows = []
    oof_frames = []

    for fold in FOLDS:
        fold_name = fold["fold"]

        train_end = quarter_number(fold["train_end"])
        valid_start = quarter_number(fold["validation_start"])
        valid_end = quarter_number(fold["validation_end"])

        train_mask = data["_QUARTER_NUMBER"] <= train_end
        valid_mask = (
            (data["_QUARTER_NUMBER"] >= valid_start)
            & (data["_QUARTER_NUMBER"] <= valid_end)
        )

        train_frame = data.loc[train_mask].copy()
        valid_frame = data.loc[valid_mask].copy()

        if train_frame.empty or valid_frame.empty:
            raise RuntimeError(f"{fold_name}: empty TRAIN or VALID")

        if train_frame["_QUARTER_NUMBER"].max() >= valid_frame[
            "_QUARTER_NUMBER"
        ].min():
            raise RuntimeError(
                f"{fold_name}: temporal boundary violation"
            )

        x_train = train_frame[feature_columns].copy()
        x_valid = valid_frame[feature_columns].copy()

        y_train = train_frame[TARGET].astype(int).to_numpy()
        y_valid = valid_frame[TARGET].astype(int).to_numpy()

        category_audit = encode_categorical_train_only(
            x_train,
            x_valid,
            categorical_columns,
        )
        category_audit.insert(0, "fold", fold_name)
        category_audit.to_csv(
            OUTPUT_DIR / f"{fold_name}_categorical_audit.csv",
            index=False,
        )

        prepare_numeric_columns(
            x_train,
            x_valid,
            categorical_columns,
        )

        train_dataset = lgb.Dataset(
            x_train,
            label=y_train,
            categorical_feature=categorical_columns,
            free_raw_data=False,
        )

        valid_dataset = lgb.Dataset(
            x_valid,
            label=y_valid,
            categorical_feature=categorical_columns,
            reference=train_dataset,
            free_raw_data=False,
        )

        print()
        print(f"=== {fold_name} ===")
        print(
            f"TRAIN: through {fold['train_end']}, "
            f"rows={len(train_frame)}, positives={int(y_train.sum())}"
        )
        print(
            f"VALID: {fold['validation_start']}–"
            f"{fold['validation_end']}, "
            f"rows={len(valid_frame)}, positives={int(y_valid.sum())}"
        )

        model = lgb.train(
            FIXED_PARAMS,
            train_dataset,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[valid_dataset],
            valid_names=["VALID"],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=EARLY_STOPPING_ROUNDS,
                    verbose=False,
                ),
                lgb.log_evaluation(period=200),
            ],
        )

        best_iteration = (
            model.best_iteration
            if model.best_iteration
            else NUM_BOOST_ROUND
        )

        predictions = model.predict(
            x_valid,
            num_iteration=best_iteration,
        )
        predictions = np.clip(predictions, 1e-9, 1 - 1e-9)

        pr_auc = average_precision_score(y_valid, predictions)
        prevalence = float(np.mean(y_valid))
        lift = pr_auc / prevalence if prevalence > 0 else np.nan
        brier = brier_score_loss(y_valid, predictions)
        prediction_mean = float(np.mean(predictions))

        result_rows.append({
            "fold": fold_name,
            "train_end": fold["train_end"],
            "validation_start": fold["validation_start"],
            "validation_end": fold["validation_end"],
            "train_rows": len(train_frame),
            "train_positives": int(y_train.sum()),
            "validation_rows": len(valid_frame),
            "validation_positives": int(y_valid.sum()),
            "validation_prevalence": prevalence,
            "pr_auc": pr_auc,
            "lift_over_prevalence": lift,
            "brier_score": brier,
            "prediction_mean": prediction_mean,
            "calibration_gap": prediction_mean - prevalence,
            "prediction_min": float(np.min(predictions)),
            "prediction_max": float(np.max(predictions)),
            "best_iteration": int(best_iteration),
        })

        prediction_frame = pd.DataFrame({
            "fold": fold_name,
            "ROW_ID": valid_frame["ROW_ID"].astype(str).to_numpy(),
            "SNAPSHOT_YEAR": valid_frame[
                "SNAPSHOT_YEAR"
            ].to_numpy(),
            "SNAPSHOT_QUARTER": valid_frame[
                "SNAPSHOT_QUARTER"
            ].to_numpy(),
            "y_true": y_valid,
            "y_pred": predictions,
        })

        prediction_frame.to_csv(
            OUTPUT_DIR / f"{fold_name}_validation_predictions.csv",
            index=False,
        )
        oof_frames.append(prediction_frame)

        calibration_table(predictions, y_valid).to_csv(
            OUTPUT_DIR / f"{fold_name}_calibration.csv",
            index=False,
        )

        print(f"PR-AUC: {pr_auc:.12f}")
        print(f"Prevalence: {prevalence:.12f}")
        print(f"Lift over prevalence: {lift:.4f}")
        print(f"Brier score: {brier:.12f}")
        print(f"Best iteration: {best_iteration}")

    results = pd.DataFrame(result_rows)
    results.to_csv(
        OUTPUT_DIR / "rolling_backtest_results.csv",
        index=False,
    )

    oof = pd.concat(oof_frames, ignore_index=True)
    oof.to_csv(
        OUTPUT_DIR / "rolling_oof_predictions.csv",
        index=False,
    )

    overall_pr_auc = average_precision_score(
        oof["y_true"],
        oof["y_pred"],
    )
    overall_prevalence = float(oof["y_true"].mean())
    overall_lift = overall_pr_auc / overall_prevalence
    overall_brier = brier_score_loss(
        oof["y_true"],
        oof["y_pred"],
    )

    summary = {
        "stage": "ROLLING_TIME_BACKTEST",
        "model": "FIXED_HUMAN_CONTROL_LIGHTGBM",
        "executed_folds": ["fold_1", "fold_2"],
        "deferred_fold": "fold_3",
        "hidden_test_labels_accessed": False,
        "oof_rows": int(len(oof)),
        "oof_positives": int(oof["y_true"].sum()),
        "oof_prevalence": overall_prevalence,
        "oof_pr_auc": float(overall_pr_auc),
        "oof_lift_over_prevalence": float(overall_lift),
        "oof_brier_score": float(overall_brier),
    }

    (OUTPUT_DIR / "rolling_backtest_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    summary_lines = [
        f"{key}={value}"
        for key, value in summary.items()
    ]
    (OUTPUT_DIR / "rolling_backtest_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=== Rolling backtest results ===")
    print(results.to_string(index=False))
    print()
    print("=== Combined out-of-fold summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
