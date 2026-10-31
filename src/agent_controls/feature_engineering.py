#!/usr/bin/env python3

from typing import Iterable

import numpy as np
import pandas as pd


ALLOWED_OPERATORS = {
    "sum",
    "difference",
    "product",
    "ratio_safe",
    "max_zero_negative",
    "count_zero",
}

FORBIDDEN_INPUTS = {
    "ROW_ID",
    "DOWNGRADE_6M",
}


def _require_numeric(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> None:
    non_numeric = [
        column
        for column in columns
        if not pd.api.types.is_numeric_dtype(
            frame[column]
        )
    ]

    if non_numeric:
        raise TypeError(
            "Non-numeric base features: "
            + ",".join(non_numeric)
        )


def _as_float32(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    return frame.loc[:, columns].astype(
        np.float32,
        copy=True,
    )


def _sum_values(
    values: list[np.ndarray],
) -> np.ndarray:
    result = values[0].copy()

    for value in values[1:]:
        result = result + value

    return result


def _product_values(
    values: list[np.ndarray],
) -> np.ndarray:
    result = values[0].copy()

    for value in values[1:]:
        result = result * value

    return result


def _safe_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    denominator = denominator.copy()

    small = np.abs(denominator) < epsilon

    replacement = np.where(
        denominator < 0,
        -epsilon,
        epsilon,
    ).astype(np.float32)

    safe_denominator = np.where(
        small,
        replacement,
        denominator,
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore",
    ):
        result = numerator / safe_denominator

    return result


def apply_recipe_features(
    frame: pd.DataFrame,
    recipe: dict,
    base_features: list[str],
) -> pd.DataFrame:
    """
    Apply locked row-wise feature formulas.

    This function:
    - does not read ROW_ID or target;
    - does not compute cross-row statistics;
    - does not fit preprocessing;
    - returns base features followed by recipe features;
    - does not mutate the input DataFrame.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            "frame must be a pandas DataFrame"
        )

    if not isinstance(recipe, dict):
        raise TypeError(
            "recipe must be a dict"
        )

    if not isinstance(base_features, list):
        raise TypeError(
            "base_features must be a list"
        )

    if len(base_features) != len(
        set(base_features)
    ):
        raise ValueError(
            "base_features contains duplicates"
        )

    forbidden = sorted(
        FORBIDDEN_INPUTS & set(base_features)
    )

    if forbidden:
        raise ValueError(
            "Forbidden base features: "
            + ",".join(forbidden)
        )

    missing = [
        column
        for column in base_features
        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            "Missing base features: "
            + ",".join(missing)
        )

    _require_numeric(
        frame,
        base_features,
    )

    result = _as_float32(
        frame,
        base_features,
    )

    engineered = recipe.get(
        "engineered_features"
    )

    if not isinstance(engineered, list):
        raise TypeError(
            "engineered_features must be a list"
        )

    output_names = []

    for index, feature in enumerate(engineered):
        if not isinstance(feature, dict):
            raise TypeError(
                f"Engineered feature {index} is not an object"
            )

        name = feature.get("name")
        operator = feature.get("operator")
        inputs = feature.get("inputs")

        if not isinstance(name, str):
            raise TypeError(
                f"Engineered feature {index} has invalid name"
            )

        if name in result.columns:
            raise ValueError(
                f"Engineered feature shadows existing column: {name}"
            )

        if name in output_names:
            raise ValueError(
                f"Duplicate engineered feature: {name}"
            )

        if operator not in ALLOWED_OPERATORS:
            raise ValueError(
                f"Invalid operator: {operator}"
            )

        if not isinstance(inputs, list):
            raise TypeError(
                f"Inputs for {name} must be a list"
            )

        invalid_inputs = [
            column
            for column in inputs
            if column not in base_features
        ]

        if invalid_inputs:
            raise ValueError(
                f"Inputs for {name} are not base allowlisted: "
                + ",".join(invalid_inputs)
            )

        arrays = [
            result[column].to_numpy(
                dtype=np.float32,
                copy=True,
            )
            for column in inputs
        ]

        if operator == "sum":
            if not 2 <= len(arrays) <= 10:
                raise ValueError(
                    f"Invalid input count for {name}"
                )
            values = _sum_values(arrays)

        elif operator == "difference":
            if len(arrays) != 2:
                raise ValueError(
                    f"Invalid input count for {name}"
                )
            values = arrays[0] - arrays[1]

        elif operator == "product":
            if not 2 <= len(arrays) <= 10:
                raise ValueError(
                    f"Invalid input count for {name}"
                )
            values = _product_values(arrays)

        elif operator == "ratio_safe":
            if len(arrays) != 2:
                raise ValueError(
                    f"Invalid input count for {name}"
                )

            epsilon = feature.get("epsilon")

            if (
                not isinstance(epsilon, (int, float))
                or isinstance(epsilon, bool)
                or not np.isfinite(epsilon)
                or not 1e-12 <= float(epsilon) <= 0.1
            ):
                raise ValueError(
                    f"Invalid epsilon for {name}"
                )

            values = _safe_ratio(
                arrays[0],
                arrays[1],
                float(epsilon),
            )

        elif operator == "max_zero_negative":
            if len(arrays) != 1:
                raise ValueError(
                    f"Invalid input count for {name}"
                )
            values = np.maximum(
                np.float32(0.0),
                -arrays[0],
            )

        elif operator == "count_zero":
            if not 2 <= len(arrays) <= 10:
                raise ValueError(
                    f"Invalid input count for {name}"
                )

            values = np.zeros(
                len(result),
                dtype=np.float32,
            )

            for array in arrays:
                values += (
                    array == 0
                ).astype(np.float32)

        else:
            raise AssertionError(
                "Unreachable operator branch"
            )

        result[name] = np.asarray(
            values,
            dtype=np.float32,
        )

        output_names.append(name)

    expected_columns = (
        base_features + output_names
    )

    if list(result.columns) != expected_columns:
        raise AssertionError(
            "Feature output order mismatch"
        )

    if any(
        dtype != np.dtype("float32")
        for dtype in result.dtypes
    ):
        raise AssertionError(
            "Output contains non-float32 columns"
        )

    if len(result) != len(frame):
        raise AssertionError(
            "Feature engineering changed row count"
        )

    return result
