from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


REQUIRED_FILES = [
    "main.py",
    "scores.csv",
    "submission.csv",
    "protocol_audit.txt",
]

REQUIRED_AUDIT_KEYS = [
    "CANDIDATE_STATUS",
    "TEST_SCORING_ALLOWED",
    "FORBIDDEN_MODEL_FEATURES_FOUND",
    "TEST_USED_FOR_PREPROCESSING",
    "TEST_USED_FOR_FEATURE_ENGINEERING",
    "ROW_ID_PARSED",
    "PREPROCESSING_FIT_SPLIT",
    "SELECTED_CANDIDATE",
    "SELECTED_VALID_PR_AUC",
]


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_dir")
    args = parser.parse_args()

    candidate = Path(args.candidate_dir).resolve()
    failures: list[str] = []
    warnings: list[str] = []

    print(f"CANDIDATE_DIR={candidate}")

    for name in REQUIRED_FILES:
        if not (candidate / name).is_file():
            failures.append(f"missing required output: {name}")

    main_path = candidate / "main.py"
    audit_path = candidate / "protocol_audit.txt"

    source = main_path.read_text(errors="replace") if main_path.is_file() else ""
    normalized = compact(source)

    if re.search(
        r"pd\.concat\s*\(\s*\[\s*train(?:\[[^\]]+\])?.{0,200}"
        r"test(?:\[[^\]]+\])?",
        normalized,
        flags=re.I,
    ):
        failures.append(
            "main.py appears to concatenate TRAIN and TEST; "
            "manual inspection is required"
        )

    if re.search(
        r"(all_vals|combined|all_df)\s*=\s*pd\.concat",
        source,
        flags=re.I,
    ):
        failures.append(
            "combined TRAIN/TEST object detected in preprocessing or "
            "feature engineering"
        )

    row_id_parse_patterns = [
        r"""(?:train|test)\s*\[\s*["']ROW_ID["']\s*\]\s*\.astype\s*\(\s*str\s*\)\s*\.str\s*\.(?:split|extract|slice)""",
        r"""(?:train|test)\s*\[\s*["']ROW_ID["']\s*\]\s*\.str\s*\.(?:split|extract|slice)""",
        r"""(?:train|test)\s*\[\s*["']ROW_ID["']\s*\].{0,200}\.apply\s*\(.{0,200}(?:split|partition|extract)""",
        r"""_ISSUER_ID\s*=.{0,200}(?:train|test)\s*\[\s*["']ROW_ID["']\s*\]""",
    ]

    if any(
        re.search(pattern, normalized, flags=re.I)
        for pattern in row_id_parse_patterns
    ):
        failures.append("ROW_ID parsing or issuer reconstruction detected")

    if re.search(
        r"SNAPSHOT_YEAR.{0,400}(sample_weight|weight)",
        normalized,
        flags=re.I,
    ) or re.search(
        r"(sample_weight|weight).{0,400}SNAPSHOT_YEAR",
        normalized,
        flags=re.I,
    ):
        failures.append("SNAPSHOT_YEAR appears to be used for sample weighting")

    if re.search(
        r"fit(?:_transform)?\s*\(.{0,300}\btest\b",
        normalized,
        flags=re.I,
    ):
        failures.append("learned preprocessing may be fitted on TEST")

    audit_values: dict[str, str] = {}
    if audit_path.is_file():
        for line in audit_path.read_text(errors="replace").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                audit_values[key.strip()] = value.strip()

        for key in REQUIRED_AUDIT_KEYS:
            if key not in audit_values:
                failures.append(f"protocol_audit.txt missing key: {key}")

        expected = {
            "CANDIDATE_STATUS": "PROTOCOL_VALID",
            "TEST_SCORING_ALLOWED": "YES",
            "FORBIDDEN_MODEL_FEATURES_FOUND": "NO",
            "TEST_USED_FOR_PREPROCESSING": "NO",
            "TEST_USED_FOR_FEATURE_ENGINEERING": "NO",
            "ROW_ID_PARSED": "NO",
            "PREPROCESSING_FIT_SPLIT": "TRAIN",
        }

        for key, expected_value in expected.items():
            actual = audit_values.get(key)
            if actual is not None and actual.upper() != expected_value:
                failures.append(
                    f"audit declaration failed: {key}={actual}; "
                    f"expected {expected_value}"
                )

    if "submission_ensemble.csv" in source:
        warnings.append(
            "multiple submissions may exist; verify selected submission "
            "matches the highest VALID PR-AUC"
        )

    print("\n=== WARNINGS ===")
    if warnings:
        for item in warnings:
            print(f"WARNING: {item}")
    else:
        print("None")

    print("\n=== FAILURES ===")
    if failures:
        for item in failures:
            print(f"FAIL: {item}")
        print("\nPROTOCOL_RESULT=FAIL")
        print("TEST_SCORING_ALLOWED=NO")
        return 1

    print("None")
    print("\nPROTOCOL_RESULT=PASS")
    print("TEST_SCORING_ALLOWED=YES")
    return 0


if __name__ == "__main__":
    sys.exit(main())
