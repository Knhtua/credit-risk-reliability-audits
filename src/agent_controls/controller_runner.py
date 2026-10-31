#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
import json
import math
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


TARGET = "DOWNGRADE_6M"
ROW_ID = "ROW_ID"


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def write_json(path, value):
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def validate_recipe_or_stop(
    recipe_path,
    schema_path,
    allowlist_path,
    validator_path,
):
    validator = load_module(
        validator_path,
        "phase3c_recipe_validator",
    )

    recipe = read_json(recipe_path)
    schema = read_json(schema_path)
    allowlist = read_json(allowlist_path)

    errors = validator.validate_recipe(
        recipe,
        schema,
        allowlist,
    )

    if errors:
        raise ValueError(
            "Recipe validation failed: "
            + " | ".join(errors)
        )

    return recipe, allowlist


def exact_supervised_columns(base_features):
    return [
        ROW_ID,
        *base_features,
        TARGET,
    ]


def exact_prediction_columns(base_features):
    return [
        ROW_ID,
        *base_features,
    ]


def load_supervised(path, base_features):
    frame = pd.read_csv(
        path,
        encoding="utf-8",
    )

    expected = exact_supervised_columns(
        base_features
    )

    if list(frame.columns) != expected:
        raise ValueError(
            "Supervised column order mismatch"
        )

    if frame[TARGET].isna().any():
        raise ValueError(
            "Missing supervised labels"
        )

    values = sorted(
        frame[TARGET].unique().tolist()
    )

    if values != [0, 1]:
        raise ValueError(
            "Supervised labels are not binary"
        )

    return frame


def load_prediction(path, base_features):
    header = pd.read_csv(
        path,
        nrows=0,
        encoding="utf-8",
    )

    expected = exact_prediction_columns(
        base_features
    )

    if list(header.columns) != expected:
        raise ValueError(
            "Prediction column order mismatch"
        )

    if TARGET in header.columns:
        raise ValueError(
            "Prediction input contains target"
        )

    return pd.read_csv(
        path,
        encoding="utf-8",
    )


def build_matrix(
    frame,
    recipe,
    base_features,
    feature_engine_path,
):
    feature_engine = load_module(
        feature_engine_path,
        "phase3c_feature_engine",
    )

    engineered = (
        feature_engine.apply_recipe_features(
            frame,
            recipe,
            base_features,
        )
    )

    return engineered.to_numpy(
        dtype=np.float32,
        copy=True,
    ), list(engineered.columns)


def class_weight_value(mode, labels):
    if mode == "none":
        return None

    if mode != "train_ratio":
        raise ValueError(
            "Unsupported class_weight_mode"
        )

    positive = int(np.sum(labels == 1))
    negative = int(np.sum(labels == 0))

    if positive == 0 or negative == 0:
        raise ValueError(
            "Both classes required for train_ratio"
        )

    return float(negative / positive)


def build_parameters(
    recipe,
    labels,
    num_threads,
):
    params = dict(
        recipe["lightgbm_params"]
    )

    params.update(
        {
            "objective": "binary",
            "metric": "None",
            "seed": 42,
            "data_random_seed": 42,
            "feature_fraction_seed": 42,
            "bagging_seed": 42,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
            "num_threads": int(num_threads),
        }
    )

    weight = class_weight_value(
        recipe["class_weight_mode"],
        labels,
    )

    if weight is not None:
        params["scale_pos_weight"] = weight

    forbidden_duplicates = [
        key
        for key in (
            "is_unbalance",
            "class_weight",
        )
        if key in params
    ]

    if forbidden_duplicates:
        raise ValueError(
            "Duplicate class-weight controls"
        )

    return params, weight


def pr_auc_metric(predictions, dataset):
    labels = dataset.get_label()

    return (
        "pr_auc",
        average_precision_score(
            labels,
            predictions,
        ),
        True,
    )


def run_select(args):
    recipe, allowlist = validate_recipe_or_stop(
        args.recipe,
        args.schema,
        args.allowlist,
        args.validator,
    )

    base_features = allowlist[
        "base_features_in_position_order"
    ]

    fit_frame = load_supervised(
        args.fit_data,
        base_features,
    )

    stopping_frame = load_supervised(
        args.stopping_data,
        base_features,
    )

    fit_labels = (
        fit_frame[TARGET]
        .to_numpy(dtype=np.int8)
    )

    stopping_labels = (
        stopping_frame[TARGET]
        .to_numpy(dtype=np.int8)
    )

    fit_matrix, feature_names = build_matrix(
        fit_frame,
        recipe,
        base_features,
        args.feature_engine,
    )

    stopping_matrix, stop_names = build_matrix(
        stopping_frame,
        recipe,
        base_features,
        args.feature_engine,
    )

    if feature_names != stop_names:
        raise AssertionError(
            "Feature names differ across TRAIN periods"
        )

    params, weight = build_parameters(
        recipe,
        fit_labels,
        args.num_threads,
    )

    config = read_json(args.runner_config)

    if args.synthetic_test_mode:
        maximum_rounds = config[
            "synthetic_self_test"
        ]["maximum_boosting_rounds"]

        stopping_rounds = config[
            "synthetic_self_test"
        ]["early_stopping_rounds"]
    else:
        maximum_rounds = config[
            "maximum_boosting_rounds"
        ]

        stopping_rounds = config[
            "early_stopping_rounds"
        ]

    train_dataset = lgb.Dataset(
        fit_matrix,
        label=fit_labels,
        feature_name=feature_names,
        free_raw_data=False,
    )

    stopping_dataset = lgb.Dataset(
        stopping_matrix,
        label=stopping_labels,
        feature_name=feature_names,
        reference=train_dataset,
        free_raw_data=False,
    )

    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=maximum_rounds,
        valid_sets=[stopping_dataset],
        valid_names=["internal_stopping"],
        feval=pr_auc_metric,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=stopping_rounds,
                first_metric_only=True,
                verbose=False,
                min_delta=config[
                    "early_stopping_min_delta"
                ],
            )
        ],
    )

    best_iteration = int(
        model.best_iteration
    )

    predictions = model.predict(
        stopping_matrix,
        num_iteration=best_iteration,
    )

    internal_pr_auc = float(
        average_precision_score(
            stopping_labels,
            predictions,
        )
    )

    if best_iteration < 1:
        raise AssertionError(
            "Invalid best_iteration"
        )

    record = {
        "record_type": (
            "phase3c_internal_time_selection_record"
        ),
        "created_at": (
            datetime.now().astimezone().isoformat()
        ),
        "recipe_sha256": sha256(args.recipe),
        "fit_data_sha256": sha256(
            args.fit_data
        ),
        "stopping_data_sha256": sha256(
            args.stopping_data
        ),
        "fit_rows": int(len(fit_frame)),
        "fit_events": int(
            fit_labels.sum()
        ),
        "stopping_rows": int(
            len(stopping_frame)
        ),
        "stopping_events": int(
            stopping_labels.sum()
        ),
        "feature_count": len(feature_names),
        "best_iteration": best_iteration,
        "internal_stopping_pr_auc": (
            internal_pr_auc
        ),
        "class_weight_mode": recipe[
            "class_weight_mode"
        ],
        "fit_scale_pos_weight": weight,
        "selection_model_is_final_model": False,
        "hyperparameter_changes_after_score": False,
        "recipe_changes_after_score": False,
        "synthetic_test_mode": bool(
            args.synthetic_test_mode
        ),
        "validation_features_accessed": False,
        "validation_labels_accessed": False,
        "pseudo_test_features_accessed": False,
        "pseudo_test_labels_accessed": False,
    }

    write_json(
        args.selection_record,
        record,
    )

    print(
        f"BEST_ITERATION={best_iteration}"
    )
    print(
        "INTERNAL_STOPPING_PR_AUC="
        f"{internal_pr_auc:.8f}"
    )
    print(
        f"SELECTION_RECORD={args.selection_record}"
    )


def run_fit(args):
    recipe, allowlist = validate_recipe_or_stop(
        args.recipe,
        args.schema,
        args.allowlist,
        args.validator,
    )

    base_features = allowlist[
        "base_features_in_position_order"
    ]

    selection = read_json(
        args.selection_record
    )

    if selection["recipe_sha256"] != sha256(
        args.recipe
    ):
        raise ValueError(
            "Selection record recipe hash mismatch"
        )

    best_iteration = int(
        selection["best_iteration"]
    )

    config = read_json(args.runner_config)

    maximum_allowed = (
        config["synthetic_self_test"][
            "maximum_boosting_rounds"
        ]
        if selection["synthetic_test_mode"]
        else config["maximum_boosting_rounds"]
    )

    if not 1 <= best_iteration <= maximum_allowed:
        raise ValueError(
            "Selection round outside locked bounds"
        )

    final_frame = load_supervised(
        args.final_train_data,
        base_features,
    )

    labels = (
        final_frame[TARGET]
        .to_numpy(dtype=np.int8)
    )

    matrix, feature_names = build_matrix(
        final_frame,
        recipe,
        base_features,
        args.feature_engine,
    )

    params, weight = build_parameters(
        recipe,
        labels,
        args.num_threads,
    )

    dataset = lgb.Dataset(
        matrix,
        label=labels,
        feature_name=feature_names,
        free_raw_data=False,
    )

    final_model = lgb.train(
        params,
        dataset,
        num_boost_round=best_iteration,
        valid_sets=None,
        callbacks=None,
    )

    args.model_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_model.save_model(
        str(args.model_output),
        num_iteration=best_iteration,
    )

    manifest = {
        "record_type": (
            "phase3c_final_single_model_manifest"
        ),
        "created_at": (
            datetime.now().astimezone().isoformat()
        ),
        "recipe_sha256": sha256(args.recipe),
        "selection_record_sha256": sha256(
            args.selection_record
        ),
        "final_train_data_sha256": sha256(
            args.final_train_data
        ),
        "final_train_rows": int(
            len(final_frame)
        ),
        "final_train_events": int(
            labels.sum()
        ),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "best_iteration": best_iteration,
        "final_model_count": 1,
        "prediction_averaging": False,
        "seed": 42,
        "class_weight_mode": recipe[
            "class_weight_mode"
        ],
        "final_scale_pos_weight": weight,
        "is_unbalance_used": False,
        "calibration_used": False,
        "model_sha256": sha256(
            args.model_output
        ),
        "validation_features_accessed": False,
        "validation_labels_accessed": False,
        "pseudo_test_features_accessed": False,
        "pseudo_test_labels_accessed": False,
    }

    write_json(
        args.model_manifest,
        manifest,
    )

    print(
        f"FINAL_MODEL={args.model_output}"
    )
    print(
        f"FINAL_MODEL_SHA256={manifest['model_sha256']}"
    )
    print("FINAL_MODEL_COUNT=1")
    print("PREDICTION_AVERAGING=False")


def run_predict(args):
    recipe, allowlist = validate_recipe_or_stop(
        args.recipe,
        args.schema,
        args.allowlist,
        args.validator,
    )

    base_features = allowlist[
        "base_features_in_position_order"
    ]

    manifest = read_json(
        args.model_manifest
    )

    if manifest["recipe_sha256"] != sha256(
        args.recipe
    ):
        raise ValueError(
            "Model manifest recipe hash mismatch"
        )

    if manifest["model_sha256"] != sha256(
        args.model_input
    ):
        raise ValueError(
            "Model hash mismatch"
        )

    if manifest["final_model_count"] != 1:
        raise ValueError(
            "Final model count is not one"
        )

    if manifest["prediction_averaging"]:
        raise ValueError(
            "Prediction averaging is prohibited"
        )

    prediction_frame = load_prediction(
        args.prediction_data,
        base_features,
    )

    row_ids = (
        prediction_frame[ROW_ID]
        .astype(str)
        .to_numpy(copy=True)
    )

    matrix, feature_names = build_matrix(
        prediction_frame,
        recipe,
        base_features,
        args.feature_engine,
    )

    if feature_names != manifest[
        "feature_names"
    ]:
        raise ValueError(
            "Prediction feature order mismatch"
        )

    model = lgb.Booster(
        model_file=str(args.model_input)
    )

    predictions = model.predict(
        matrix,
        num_iteration=int(
            manifest["best_iteration"]
        ),
    )

    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    )

    if len(predictions) != len(row_ids):
        raise AssertionError(
            "Prediction row count mismatch"
        )

    if not np.isfinite(predictions).all():
        raise ValueError(
            "Non-finite prediction"
        )

    if not (
        (predictions >= 0)
        & (predictions <= 1)
    ).all():
        raise ValueError(
            "Prediction outside [0, 1]"
        )

    output = pd.DataFrame(
        {
            ROW_ID: row_ids,
            TARGET: predictions,
        }
    )

    if list(output.columns) != [
        ROW_ID,
        TARGET,
    ]:
        raise AssertionError(
            "Prediction output column mismatch"
        )

    output.to_csv(
        args.prediction_output,
        index=False,
        encoding="utf-8",
    )

    print(
        f"PREDICTION_OUTPUT={args.prediction_output}"
    )
    print(
        f"PREDICTION_ROWS={len(output)}"
    )
    print("PREDICTION_MODEL_COUNT=1")
    print("PREDICTION_AVERAGING=False")
    print("PREDICTION_LABEL_ACCESS=False")


def build_parser():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    common = argparse.ArgumentParser(
        add_help=False
    )

    common.add_argument(
        "--recipe",
        required=True,
        type=Path,
    )
    common.add_argument(
        "--schema",
        required=True,
        type=Path,
    )
    common.add_argument(
        "--allowlist",
        required=True,
        type=Path,
    )
    common.add_argument(
        "--validator",
        required=True,
        type=Path,
    )
    common.add_argument(
        "--feature-engine",
        required=True,
        type=Path,
    )
    common.add_argument(
        "--runner-config",
        required=True,
        type=Path,
    )
    common.add_argument(
        "--num-threads",
        type=int,
        default=1,
    )

    select_parser = subparsers.add_parser(
        "select",
        parents=[common],
    )
    select_parser.add_argument(
        "--fit-data",
        required=True,
        type=Path,
    )
    select_parser.add_argument(
        "--stopping-data",
        required=True,
        type=Path,
    )
    select_parser.add_argument(
        "--selection-record",
        required=True,
        type=Path,
    )
    select_parser.add_argument(
        "--synthetic-test-mode",
        action="store_true",
    )

    fit_parser = subparsers.add_parser(
        "fit",
        parents=[common],
    )
    fit_parser.add_argument(
        "--final-train-data",
        required=True,
        type=Path,
    )
    fit_parser.add_argument(
        "--selection-record",
        required=True,
        type=Path,
    )
    fit_parser.add_argument(
        "--model-output",
        required=True,
        type=Path,
    )
    fit_parser.add_argument(
        "--model-manifest",
        required=True,
        type=Path,
    )

    predict_parser = subparsers.add_parser(
        "predict",
        parents=[common],
    )
    predict_parser.add_argument(
        "--prediction-data",
        required=True,
        type=Path,
    )
    predict_parser.add_argument(
        "--model-input",
        required=True,
        type=Path,
    )
    predict_parser.add_argument(
        "--model-manifest",
        required=True,
        type=Path,
    )
    predict_parser.add_argument(
        "--prediction-output",
        required=True,
        type=Path,
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.num_threads < 1:
        raise ValueError(
            "num_threads must be positive"
        )

    if args.command == "select":
        run_select(args)
    elif args.command == "fit":
        run_fit(args)
    elif args.command == "predict":
        run_predict(args)
    else:
        raise AssertionError(
            "Unknown command"
        )


if __name__ == "__main__":
    main()
