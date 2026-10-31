#!/usr/bin/env python3

import math
import re
import sys
from pathlib import Path

import pandas as pd


REQUIRED_FILES = [
    "main.py",
    "scores.csv",
    "submission.csv",
    "protocol_audit.txt",
]

REQUIRED_AUDIT_VALUES = {
    "CANDIDATE_STATUS": "PROTOCOL_VALID",
    "TEST_SCORING_ALLOWED": "YES",
    "FORBIDDEN_MODEL_FEATURES_FOUND": "NO",
    "TEST_USED_FOR_PREPROCESSING": "NO",
    "TEST_USED_FOR_FEATURE_ENGINEERING": "NO",
    "ROW_ID_PARSED": "NO",
    "PREPROCESSING_FIT_SPLIT": "TRAIN",
}

REQUIRED_AUDIT_KEYS = [
    *REQUIRED_AUDIT_VALUES,
    "SELECTED_CANDIDATE",
    "SELECTED_VALID_PR_AUC",
]

EXPECTED_TEST_ROWS = 114603
EXPECTED_COLUMNS = ["ROW_ID", "DOWNGRADE_6M"]


def strip_full_line_comments(text):
    return "\n".join(
        line for line in text.splitlines()
        if not line.lstrip().startswith("#")
    )


def read_audit(path):
    values = {}

    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()

    return values


def inspect_submission(path, failures):
    try:
        submission = pd.read_csv(path)
    except Exception as exc:
        failures.append(f"submission.csv cannot be read: {exc}")
        return

    if list(submission.columns) != EXPECTED_COLUMNS:
        failures.append(
            "submission.csv columns must be exactly "
            f"{EXPECTED_COLUMNS}; found {list(submission.columns)}"
        )

    if len(submission) != EXPECTED_TEST_ROWS:
        failures.append(
            f"submission.csv has {len(submission)} rows; "
            f"expected {EXPECTED_TEST_ROWS}"
        )

    if "ROW_ID" in submission.columns:
        if submission["ROW_ID"].isna().any():
            failures.append("submission.csv contains missing ROW_ID values")

        if submission["ROW_ID"].duplicated().any():
            failures.append("submission.csv contains duplicate ROW_ID values")

    if "DOWNGRADE_6M" in submission.columns:
        predictions = pd.to_numeric(
            submission["DOWNGRADE_6M"], errors="coerce"
        )

        if predictions.isna().any():
            failures.append(
                "DOWNGRADE_6M contains missing or non-numeric values"
            )
        else:
            finite = predictions.map(math.isfinite)

            if not finite.all():
                failures.append(
                    "DOWNGRADE_6M contains non-finite values"
                )

            if ((predictions < 0) | (predictions > 1)).any():
                failures.append(
                    "DOWNGRADE_6M contains values outside [0, 1]"
                )


def inspect_source(path, failures, warnings):
    source = path.read_text(errors="replace")
    code = strip_full_line_comments(source)
    normalized = re.sub(r"\s+", " ", code)

    row_id_patterns = [
        r"""(?:train|test)\s*\[\s*["']ROW_ID["']\s*\]\s*"""
        r"""\.astype\s*\(\s*str\s*\)\s*\.str\s*"""
        r"""\.(?:split|extract|slice)""",

        r"""(?:train|test)\s*\[\s*["']ROW_ID["']\s*\]\s*"""
        r"""\.str\s*\.(?:split|extract|slice)""",

        r"""(?:train|test)\s*\[\s*["']ROW_ID["']\s*\]"""
        r""".{0,200}\.apply\s*\(.{0,200}"""
        r"""(?:split|partition|extract)""",

        r"""_ISSUER_ID\s*=.{0,200}(?:train|test)\s*"""
        r"""\[\s*["']ROW_ID["']\s*\]""",
    ]

    if any(re.search(p, normalized, flags=re.I) for p in row_id_patterns):
        failures.append(
            "ROW_ID parsing or issuer reconstruction detected"
        )

    concat_patterns = [
        r"pd\.concat\s*\(\s*\[\s*train\s*,\s*test\s*\]",
        r"pd\.concat\s*\(\s*\[\s*train_df\s*,\s*test_df\s*\]",
        r"all_df\s*=\s*pd\.concat",
        r"all_vals\s*=\s*pd\.concat",
        r"combined\s*=\s*pd\.concat",
    ]

    if any(re.search(p, normalized, flags=re.I) for p in concat_patterns):
        failures.append(
            "TRAIN and TEST appear to be concatenated for preprocessing "
            "or feature engineering"
        )

    if re.search(
        r"SNAPSHOT_YEAR.{0,250}(sample_weight|weight)|"
        r"(sample_weight|weight).{0,250}SNAPSHOT_YEAR",
        normalized,
        flags=re.I,
    ):
        failures.append(
            "SNAPSHOT_YEAR appears to be used for sample weighting"
        )

    # A derived temporal field is acceptable only when it is retained
    # strictly for auditing and explicitly excluded from model features.
    drop_match = re.search(
        r"drop_cols\s*=\s*\[(.*?)\]\s*\+\s*IDENTIFIER_COLS",
        normalized,
        flags=re.I | re.S,
    )
    drop_text = drop_match.group(1) if drop_match else ""

    temporal_feature_failure = False

    # Continuous calendar/trend features are always forbidden.
    always_forbidden_time_patterns = [
        r"SNAPSHOT_DT_YEAR_FRAC",
        r"YEAR_FRAC",
        r"SNAPSHOT_MONTH\s*=",
        r"SNAPSHOT_DAY\s*=",
    ]

    if any(
        re.search(p, normalized, flags=re.I)
        for p in always_forbidden_time_patterns
    ):
        temporal_feature_failure = True

    quarter_created = bool(
        re.search(
            r"SNAPSHOT_QUARTER\s*=|"
            r"\[['\"]SNAPSHOT_QUARTER['\"]\]\s*=|"
            r"\.dt\.quarter",
            normalized,
            flags=re.I,
        )
    )
    quarter_excluded = bool(
        re.search(r"['\"]SNAPSHOT_QUARTER['\"]", drop_text, flags=re.I)
    )

    year_created = bool(
        re.search(
            r"SNAPSHOT_YEAR\s*=|"
            r"\[['\"]SNAPSHOT_YEAR['\"]\]\s*=|"
            r"\.dt\.year",
            normalized,
            flags=re.I,
        )
    )
    year_excluded = bool(
        re.search(r"['\"]SNAPSHOT_YEAR['\"]", drop_text, flags=re.I)
    )

    if quarter_created and not quarter_excluded:
        temporal_feature_failure = True

    if year_created and not year_excluded:
        temporal_feature_failure = True

    # Direct insertion into a model-feature list remains forbidden even if
    # another occurrence appears in drop_cols.
    direct_time_feature_patterns = [
        r"(?:feature_cols|model_features)\s*\.append\s*\("
        r"\s*['\"]SNAPSHOT_(?:YEAR|QUARTER)['\"]",
        r"(?:feature_cols|model_features)\s*\+=.{0,150}"
        r"SNAPSHOT_(?:YEAR|QUARTER)",
    ]

    if any(
        re.search(p, normalized, flags=re.I)
        for p in direct_time_feature_patterns
    ):
        temporal_feature_failure = True

    if temporal_feature_failure:
        failures.append(
            "A temporal control field appears to be converted into "
            "a predictive feature"
        )

    calibration_patterns = [
        r"IsotonicRegression\s*\(",
        r"(?:calibrator|calibration_model)\s*\.fit\s*\([^)]*valid",
        r"\.fit\s*\(\s*(?:valid_pred|val_pred|y_valid)",
    ]

    if any(
        re.search(p, normalized, flags=re.I)
        for p in calibration_patterns
    ):
        failures.append(
            "VALID appears to be used to fit a calibration model"
        )

    train_valid_patterns = [
        r"X_train_valid\s*=",
        r"y_train_valid\s*=",
        r"train_valid\s*=",
        r"X_full\s*=\s*(?:pd\.)?concat.{0,250}X_valid",
        r"y_full\s*=\s*(?:pd\.)?concat.{0,250}y_valid",
        r"(?:final_model|selected_model)\.fit\s*\("
        r".{0,250}(?:X_valid|y_valid|train_valid)",
    ]

    if any(
        re.search(p, normalized, flags=re.I)
        for p in train_valid_patterns
    ):
        failures.append(
            "The final model appears to be refitted using TRAIN plus VALID"
        )

    output_path_patterns = [
        r"OUTPUT_DIR\s*=.{0,100}workspace_output",
        r"workspace_output.{0,150}(?:submission|scores|protocol_audit)",
    ]

    if any(
        re.search(p, normalized, flags=re.I)
        for p in output_path_patterns
    ):
        failures.append(
            "Required outputs appear to be written under workspace_output "
            "instead of the workspace root"
        )

    if "INST_CODE" in code:
        warnings.append(
            "INST_CODE is referenced; manually verify that it is used only "
            "as a grouping/control key and never supplied to the model"
        )

    if re.search(r"(shift|rolling|expanding)\s*\(", normalized, flags=re.I):
        warnings.append(
            "Historical feature operations detected; manually verify that "
            "every feature uses records strictly earlier than the snapshot"
        )


def inspect_audit(path, failures):
    audit = read_audit(path)

    for key in REQUIRED_AUDIT_KEYS:
        if key not in audit:
            failures.append(f"protocol_audit.txt missing key: {key}")

    for key, expected in REQUIRED_AUDIT_VALUES.items():
        actual = audit.get(key)

        if actual is not None and actual.upper() != expected:
            failures.append(
                f"audit declaration failed: {key}={actual}; "
                f"expected {expected}"
            )

    candidate = audit.get("SELECTED_CANDIDATE", "").strip()

    if candidate.upper() in {"", "NONE", "NULL", "N/A"}:
        failures.append(
            "SELECTED_CANDIDATE does not identify a selected model"
        )

    raw_score = audit.get("SELECTED_VALID_PR_AUC")

    if raw_score is not None:
        try:
            score = float(raw_score)

            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError
        except ValueError:
            failures.append(
                "SELECTED_VALID_PR_AUC must be a finite number in [0, 1]"
            )


def main():
    if len(sys.argv) != 2:
        print(
            "Usage: check_protocol_v2_2.py CANDIDATE_DIR",
            file=sys.stderr,
        )
        return 2

    candidate_dir = Path(sys.argv[1]).expanduser().resolve()
    failures = []
    warnings = []

    print(f"CANDIDATE_DIR={candidate_dir}")

    if not candidate_dir.is_dir():
        print("PROTOCOL_RESULT=FAIL")
        print("TEST_SCORING_ALLOWED=NO")
        print("FAIL: candidate directory does not exist")
        return 1

    for filename in REQUIRED_FILES:
        path = candidate_dir / filename

        if not path.is_file():
            failures.append(
                f"required file missing from workspace root: {filename}"
            )

    main_path = candidate_dir / "main.py"
    submission_path = candidate_dir / "submission.csv"
    audit_path = candidate_dir / "protocol_audit.txt"

    if main_path.is_file():
        inspect_source(main_path, failures, warnings)

    if submission_path.is_file():
        inspect_submission(submission_path, failures)

    if audit_path.is_file():
        inspect_audit(audit_path, failures)

    print("\n=== WARNINGS ===")
    if warnings:
        for warning in warnings:
            print(f"WARNING: {warning}")
    else:
        print("None")

    print("\n=== FAILURES ===")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")

        print("\nPROTOCOL_RESULT=FAIL")
        print("TEST_SCORING_ALLOWED=NO")
        return 1

    print("None")
    print("\nPROTOCOL_RESULT=PASS")
    print("TEST_SCORING_ALLOWED=YES")
    print(
        "MANUAL_REVIEW_REQUIRED=YES"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
