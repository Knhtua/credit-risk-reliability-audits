#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path


TOP_LEVEL_FIELDS = {
    "recipe_version",
    "base_model",
    "seed",
    "class_weight_mode",
    "engineered_features",
    "lightgbm_params",
    "agent_rationale",
}

ENGINEERED_FIELDS = {
    "name",
    "operator",
    "inputs",
    "epsilon",
}

OPERATOR_RULES = {
    "sum": (2, 10, False),
    "difference": (2, 2, False),
    "product": (2, 10, False),
    "ratio_safe": (2, 2, True),
    "max_zero_negative": (1, 1, False),
    "count_zero": (2, 10, False),
}

NAME_PATTERN = re.compile(
    r"^[A-Z][A-Z0-9_]{2,63}$"
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_real_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_recipe(recipe, schema, allowlist):
    errors = []

    if not isinstance(recipe, dict):
        return ["RECIPE_NOT_OBJECT"]

    observed_top = set(recipe.keys())

    missing_top = sorted(
        TOP_LEVEL_FIELDS - observed_top
    )
    extra_top = sorted(
        observed_top - TOP_LEVEL_FIELDS
    )

    if missing_top:
        errors.append(
            "MISSING_TOP_LEVEL_FIELDS:"
            + ",".join(missing_top)
        )

    if extra_top:
        errors.append(
            "EXTRA_TOP_LEVEL_FIELDS:"
            + ",".join(extra_top)
        )

    if recipe.get("recipe_version") != "1.0":
        errors.append(
            "RECIPE_VERSION_MUST_EQUAL_1.0"
        )

    if recipe.get("base_model") != "lightgbm_binary":
        errors.append(
            "BASE_MODEL_MUST_EQUAL_LIGHTGBM_BINARY"
        )

    if recipe.get("seed") != 42:
        errors.append(
            "SEED_MUST_EQUAL_42"
        )

    if recipe.get("class_weight_mode") not in {
        "none",
        "train_ratio",
    }:
        errors.append(
            "INVALID_CLASS_WEIGHT_MODE"
        )

    rationale = recipe.get("agent_rationale")

    if not isinstance(rationale, str):
        errors.append(
            "AGENT_RATIONALE_NOT_STRING"
        )
    elif not 20 <= len(rationale) <= 4000:
        errors.append(
            "AGENT_RATIONALE_LENGTH_OUT_OF_RANGE"
        )

    base_features = set(
        allowlist["base_features_in_position_order"]
    )

    forbidden_inputs = set(
        allowlist["forbidden_inputs"]
    )

    engineered = recipe.get(
        "engineered_features"
    )

    if not isinstance(engineered, list):
        errors.append(
            "ENGINEERED_FEATURES_NOT_LIST"
        )
        engineered = []

    if len(engineered) > 20:
        errors.append(
            "TOO_MANY_ENGINEERED_FEATURES"
        )

    observed_names = set()

    for index, feature in enumerate(engineered):
        prefix = f"ENGINEERED_{index}"

        if not isinstance(feature, dict):
            errors.append(
                f"{prefix}:NOT_OBJECT"
            )
            continue

        observed_fields = set(feature.keys())
        required_fields = {
            "name",
            "operator",
            "inputs",
        }

        missing_fields = sorted(
            required_fields - observed_fields
        )
        extra_fields = sorted(
            observed_fields - ENGINEERED_FIELDS
        )

        if missing_fields:
            errors.append(
                f"{prefix}:MISSING_FIELDS:"
                + ",".join(missing_fields)
            )

        if extra_fields:
            errors.append(
                f"{prefix}:EXTRA_FIELDS:"
                + ",".join(extra_fields)
            )

        name = feature.get("name")

        if not isinstance(name, str):
            errors.append(
                f"{prefix}:NAME_NOT_STRING"
            )
        else:
            if not NAME_PATTERN.fullmatch(name):
                errors.append(
                    f"{prefix}:INVALID_NAME"
                )

            if name in observed_names:
                errors.append(
                    f"{prefix}:DUPLICATE_NAME"
                )

            if name in base_features:
                errors.append(
                    f"{prefix}:SHADOWS_BASE_FEATURE"
                )

            if name in forbidden_inputs:
                errors.append(
                    f"{prefix}:FORBIDDEN_OUTPUT_NAME"
                )

            observed_names.add(name)

        operator = feature.get("operator")

        if operator not in OPERATOR_RULES:
            errors.append(
                f"{prefix}:INVALID_OPERATOR"
            )
            continue

        inputs = feature.get("inputs")

        if not isinstance(inputs, list):
            errors.append(
                f"{prefix}:INPUTS_NOT_LIST"
            )
            continue

        minimum, maximum, epsilon_required = (
            OPERATOR_RULES[operator]
        )

        if not minimum <= len(inputs) <= maximum:
            errors.append(
                f"{prefix}:INVALID_INPUT_COUNT"
            )

        if len(inputs) != len(set(inputs)):
            errors.append(
                f"{prefix}:DUPLICATE_INPUT"
            )

        for value in inputs:
            if not isinstance(value, str):
                errors.append(
                    f"{prefix}:NON_STRING_INPUT"
                )
            elif value in forbidden_inputs:
                errors.append(
                    f"{prefix}:FORBIDDEN_INPUT:{value}"
                )
            elif value not in base_features:
                errors.append(
                    f"{prefix}:INPUT_NOT_ALLOWLISTED:{value}"
                )

        has_epsilon = "epsilon" in feature

        if epsilon_required and not has_epsilon:
            errors.append(
                f"{prefix}:EPSILON_REQUIRED"
            )

        if not epsilon_required and has_epsilon:
            errors.append(
                f"{prefix}:EPSILON_FORBIDDEN"
            )

        if has_epsilon:
            epsilon = feature["epsilon"]

            if not is_real_number(epsilon):
                errors.append(
                    f"{prefix}:EPSILON_NOT_FINITE_NUMBER"
                )
            elif not 1e-12 <= float(epsilon) <= 0.1:
                errors.append(
                    f"{prefix}:EPSILON_OUT_OF_RANGE"
                )

    params = recipe.get("lightgbm_params")

    parameter_schema = schema[
        "properties"
    ]["lightgbm_params"]

    required_params = set(
        parameter_schema["required"]
    )

    allowed_params = set(
        parameter_schema["properties"].keys()
    )

    if not isinstance(params, dict):
        errors.append(
            "LIGHTGBM_PARAMS_NOT_OBJECT"
        )
        params = {}

    missing_params = sorted(
        required_params - set(params.keys())
    )
    extra_params = sorted(
        set(params.keys()) - allowed_params
    )

    if missing_params:
        errors.append(
            "MISSING_LIGHTGBM_PARAMS:"
            + ",".join(missing_params)
        )

    if extra_params:
        errors.append(
            "FORBIDDEN_LIGHTGBM_PARAMS:"
            + ",".join(extra_params)
        )

    for name in sorted(
        required_params & set(params.keys())
    ):
        value = params[name]
        rule = parameter_schema[
            "properties"
        ][name]

        expected_type = rule["type"]

        if expected_type == "integer":
            type_valid = (
                isinstance(value, int)
                and not isinstance(value, bool)
            )
        else:
            type_valid = is_real_number(value)

        if not type_valid:
            errors.append(
                f"PARAM_{name}:INVALID_TYPE"
            )
            continue

        numeric_value = float(value)

        if numeric_value < rule["minimum"]:
            errors.append(
                f"PARAM_{name}:BELOW_MINIMUM"
            )

        if numeric_value > rule["maximum"]:
            errors.append(
                f"PARAM_{name}:ABOVE_MAXIMUM"
            )

    return sorted(set(errors))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--recipe",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--schema",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--allowlist",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    errors = []

    try:
        recipe = load_json(args.recipe)
    except Exception as exc:
        recipe = None
        errors.append(
            "RECIPE_JSON_PARSE_FAILURE:"
            + type(exc).__name__
        )

    schema = load_json(args.schema)
    allowlist = load_json(args.allowlist)

    if recipe is not None:
        errors.extend(
            validate_recipe(
                recipe,
                schema,
                allowlist,
            )
        )

    errors = sorted(set(errors))
    status = "PASS" if not errors else "FAIL"

    result = {
        "record_type": (
            "phase3c_recipe_validation_result"
        ),
        "created_at": (
            datetime.now().astimezone().isoformat()
        ),
        "recipe_path": str(
            args.recipe.resolve()
        ),
        "recipe_sha256": (
            sha256(args.recipe)
            if args.recipe.is_file()
            else None
        ),
        "schema_sha256": sha256(
            args.schema
        ),
        "allowlist_sha256": sha256(
            args.allowlist
        ),
        "status": status,
        "error_count": len(errors),
        "errors": errors,
        "performance_metric_used": False,
        "validation_features_accessed": False,
        "validation_labels_accessed": False,
        "pseudo_test_features_accessed": False,
        "pseudo_test_labels_accessed": False,
    }

    args.output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print(f"RECIPE_VALIDATION_STATUS={status}")
    print(
        f"RECIPE_VALIDATION_ERROR_COUNT={len(errors)}"
    )

    for error in errors:
        print(f"RECIPE_VALIDATION_ERROR={error}")

    print(
        f"RECIPE_VALIDATION_OUTPUT={args.output}"
    )

    raise SystemExit(
        0 if status == "PASS" else 2
    )


if __name__ == "__main__":
    main()
