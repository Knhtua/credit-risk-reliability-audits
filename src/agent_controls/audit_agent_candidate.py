#!/usr/bin/env python3

import ast
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path


main_path = Path(sys.argv[1]).resolve()
protocol_path = Path(sys.argv[2]).resolve()
run_log_path = Path(sys.argv[3]).resolve()
audit_json_path = Path(sys.argv[4]).resolve()
audit_text_path = Path(sys.argv[5]).resolve()

source = main_path.read_text(encoding="utf-8")
protocol = json.loads(
    protocol_path.read_text(encoding="utf-8")
)
run_log = run_log_path.read_text(
    encoding="utf-8",
    errors="replace",
)

tree = ast.parse(source)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def add_check(
    checks,
    check_id,
    mandatory,
    status,
    evidence,
    interpretation,
):
    checks.append(
        {
            "check_id": check_id,
            "mandatory": mandatory,
            "status": status,
            "evidence": evidence,
            "interpretation": interpretation,
        }
    )


checks = []

# 1. Frozen identity
add_check(
    checks,
    "FROZEN_CODE_IDENTITY",
    True,
    "PASS",
    {
        "main_sha256": sha256(main_path),
        "protocol_sha256": sha256(protocol_path),
        "run_log_sha256": sha256(run_log_path),
    },
    "The exact failed code, protocol, and run log are locked.",
)

# 2. Candidate outputs
submission_generated = (
    "submission.csv: shape=" in run_log
)
scores_generated = (
    "[OUTPUT] scores.csv" in run_log
)

add_check(
    checks,
    "VALID_CANDIDATE_OUTPUT_EXISTS",
    True,
    (
        "PASS"
        if submission_generated and scores_generated
        else "FAIL"
    ),
    {
        "submission_generated": submission_generated,
        "scores_generated": scores_generated,
        "log_contains_submission_not_found": (
            "Error: submission.csv not found" in run_log
        ),
    },
    (
        "No valid submission and scores were generated; "
        "there is no candidate eligible for scoring."
    ),
)

# 3. Hardcoded row count
hardcoded_rows = bool(
    re.search(
        r"\bTEST_N_ROWS\s*=\s*60702\b",
        source,
    )
)

hardcoded_assertions = [
    line_number
    for line_number, line in enumerate(
        source.splitlines(),
        start=1,
    )
    if "TEST_N_ROWS" in line
    and "assert" in line
]

add_check(
    checks,
    "ARBITRARY_SAMPLE_SIZE_COMPATIBILITY",
    True,
    "FAIL" if hardcoded_rows else "PASS",
    {
        "hardcoded_full_test_row_count": hardcoded_rows,
        "hardcoded_assertion_lines": hardcoded_assertions,
        "observed_failure": (
            "AssertionError: Test row count: "
            "200 != 60702"
            in run_log
        ),
    },
    (
        "The code hardcodes the full validation row count and "
        "cannot run on the controller-provided sample."
    ),
)

# 4. Validation/test feature EDA
forbidden_test_patterns = {
    "test_missingness_summary": (
        r"\btest\.isna\(\)\.sum\(\)"
    ),
    "test_row_id_nunique": (
        r"""test\["ROW_ID"\]\.nunique\(\)"""
    ),
    "test_row_id_sample": (
        r"""test\["ROW_ID"\]\.iloc\[0\]"""
    ),
    "test_shape_reporting": (
        r"\btest\.shape\b"
    ),
    "test_dtype_selection": (
        r"\btest\.select_dtypes\("
    ),
}

observed_test_eda = {
    name: bool(re.search(pattern, source))
    for name, pattern in forbidden_test_patterns.items()
}

test_eda_failed = any(observed_test_eda.values())

add_check(
    checks,
    "NO_VALIDATION_FEATURE_EDA",
    True,
    "FAIL" if test_eda_failed else "PASS",
    observed_test_eda,
    (
        "The candidate performs prohibited summaries and "
        "inspection of validation features."
    ),
)

# 5. ROW_ID restricted to output alignment
row_id_inspection_patterns = {
    "row_id_nunique": (
        r"""test\["ROW_ID"\]\.nunique\(\)"""
    ),
    "row_id_first_value": (
        r"""test\["ROW_ID"\]\.iloc\[0\]"""
    ),
}

observed_row_id_inspection = {
    name: bool(re.search(pattern, source))
    for name, pattern in row_id_inspection_patterns.items()
}

row_id_failed = any(
    observed_row_id_inspection.values()
)

add_check(
    checks,
    "ROW_ID_ALIGNMENT_ONLY",
    True,
    "FAIL" if row_id_failed else "PASS",
    observed_row_id_inspection,
    (
        "ROW_ID is inspected beyond preservation and output "
        "alignment."
    ),
)

# 6. Multi-model averaging
multi_model_evidence = {
    "five_fold_cv": bool(
        re.search(
            r"\bn_folds\s*=\s*5\b",
            source,
        )
    ),
    "test_prediction_accumulation": (
        "test_sum += bst.predict" in source
    ),
    "test_prediction_average": bool(
        re.search(
            r"\btest_preds\s*=\s*test_sum\s*/\s*n_folds",
            source,
        )
    ),
}

multi_model_average = all(
    multi_model_evidence.values()
)

add_check(
    checks,
    "SINGLE_MODEL_NO_PREDICTION_AVERAGING",
    True,
    (
        "FAIL"
        if multi_model_average
        else "PASS"
    ),
    multi_model_evidence,
    (
        "Final validation predictions average five fitted fold "
        "models, violating the single-model requirement."
    ),
)

# 7. Seed rule
integer_seed_assignments = []

for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and "seed" in target.id.lower()
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, int)
            ):
                integer_seed_assignments.append(
                    {
                        "name": target.id,
                        "value": node.value.value,
                        "line": node.lineno,
                    }
                )

seed_values = sorted(
    {
        item["value"]
        for item in integer_seed_assignments
    }
)

seed_pass = (
    not seed_values
    or seed_values == [42]
)

add_check(
    checks,
    "SINGLE_ALLOWED_SEED",
    True,
    "PASS" if seed_pass else "FAIL",
    {
        "integer_seed_assignments": (
            integer_seed_assignments
        ),
        "observed_seed_values": seed_values,
        "allowed_seed": 42,
    },
    (
        "No disallowed fixed model seed was detected."
        if seed_pass
        else "A seed other than 42 was detected."
    ),
)

# 8. Validation labels
test_label_access_patterns = {
    "test_target_column": bool(
        re.search(
            r"""test\[['"]DOWNGRADE_6M['"]\]""",
            source,
        )
    ),
    "validation_label_file": bool(
        re.search(
            r"validation_labels\.csv",
            source,
            flags=re.IGNORECASE,
        )
    ),
    "pseudo_test_label_file": bool(
        re.search(
            r"pseudo_test_labels\.csv",
            source,
            flags=re.IGNORECASE,
        )
    ),
}

label_access_failed = any(
    test_label_access_patterns.values()
)

add_check(
    checks,
    "NO_EXTERNAL_LABEL_ACCESS",
    True,
    (
        "FAIL"
        if label_access_failed
        else "PASS"
    ),
    test_label_access_patterns,
    (
        "No validation or pseudo-test label access was detected."
        if not label_access_failed
        else "External label access pattern was detected."
    ),
)

# 9. Sample submission access by candidate code
sample_submission_access = bool(
    re.search(
        r"sample_submission",
        source,
        flags=re.IGNORECASE,
    )
)

add_check(
    checks,
    "NO_SAMPLE_SUBMISSION_ACCESS_BY_CANDIDATE",
    True,
    (
        "FAIL"
        if sample_submission_access
        else "PASS"
    ),
    {
        "candidate_code_mentions_sample_submission": (
            sample_submission_access
        ),
        "framework_fallback_seen_after_candidate_failure": (
            "Using sample submission file: "
            "sample_submission.csv"
            in run_log
        ),
    },
    (
        "Candidate code does not access sample_submission.csv. "
        "The log occurrence comes from the framework's fallback "
        "checker after candidate execution had already failed."
    ),
)

# 10. Calibration
calibration_patterns = {
    "isotonic": bool(
        re.search(
            r"\bisotonic\b",
            source,
            flags=re.IGNORECASE,
        )
    ),
    "calibrated_classifier": bool(
        re.search(
            r"CalibratedClassifier",
            source,
            flags=re.IGNORECASE,
        )
    ),
    "platt": bool(
        re.search(
            r"\bplatt\b",
            source,
            flags=re.IGNORECASE,
        )
    ),
}

calibration_failed = any(
    calibration_patterns.values()
)

add_check(
    checks,
    "NO_CALIBRATION",
    True,
    (
        "FAIL"
        if calibration_failed
        else "PASS"
    ),
    calibration_patterns,
    (
        "No calibration procedure was detected."
        if not calibration_failed
        else "A calibration procedure was detected."
    ),
)

# 11. Numeric feature boundary
raw_category_tokens = [
    "CURRENT_RANK",
    "PREVIOUS_RANK",
    "LATEST_RT_CHG",
    "RATING_OUTLOOK",
    "RATING_AGENCY_CODE",
    "RATING_AGENCY_NAME",
    "OUTLOOK_BUCKET",
]

raw_category_mentions = [
    token
    for token in raw_category_tokens
    if re.search(
        rf"\b{re.escape(token)}\b",
        source,
    )
]

add_check(
    checks,
    "RAW_CATEGORICAL_FIELDS_ABSENT",
    True,
    (
        "PASS"
        if not raw_category_mentions
        else "FAIL"
    ),
    {
        "raw_categorical_mentions": (
            raw_category_mentions
        )
    },
    (
        "No raw categorical feature name was detected."
        if not raw_category_mentions
        else "Raw categorical feature names were detected."
    ),
)

# 12. Method warning: duplicated imbalance controls
duplicated_imbalance_control = (
    '"is_unbalance": True' in source
    and '["scale_pos_weight"]' in source
)

add_check(
    checks,
    "NON_DUPLICATED_CLASS_WEIGHTING",
    False,
    (
        "WARN"
        if duplicated_imbalance_control
        else "PASS"
    ),
    {
        "is_unbalance_true": (
            '"is_unbalance": True' in source
        ),
        "scale_pos_weight_assigned": (
            '["scale_pos_weight"]' in source
        ),
    },
    (
        "Both is_unbalance and scale_pos_weight are set, which "
        "can duplicate imbalance weighting."
    ),
)

mandatory_failures = [
    item["check_id"]
    for item in checks
    if item["mandatory"]
    and item["status"] == "FAIL"
]

warnings = [
    item["check_id"]
    for item in checks
    if item["status"] == "WARN"
]

audit_status = (
    "PERMANENTLY_BLOCKED"
    if mandatory_failures
    else "PASS"
)

record = {
    "record_type": (
        "phase3b_failed_candidate_independent_static_audit"
    ),
    "created_at": (
        datetime.now().astimezone().isoformat()
    ),
    "candidate_id": "phase3b_failed_candidate_v1",
    "protocol": protocol.get("protocol"),
    "protocol_status": protocol.get("status"),
    "authorized_run_count": (
        protocol
        .get("candidate_generation", {})
        .get("authorized_run_count")
    ),
    "authorized_run_consumed": True,
    "technical_run_status": (
        "COMPLETED_WITH_TECHNICAL_FAILURE"
    ),
    "candidate_status": "NO_VALID_CANDIDATE",
    "audit_status": audit_status,
    "mandatory_failure_count": len(
        mandatory_failures
    ),
    "mandatory_failures": mandatory_failures,
    "warning_count": len(warnings),
    "warnings": warnings,
    "checks": checks,
    "decision": {
        "phase3b_candidate_may_be_modified": False,
        "phase3b_candidate_may_be_rescued": False,
        "phase3b_additional_run_allowed_under_v1_1": False,
        "external_validation_scoring_allowed": False,
        "pseudo_test_scoring_allowed": False,
        "new_protocol_required_for_any_new_experiment": True,
    },
    "access_record": {
        "validation_labels_accessed": False,
        "pseudo_test_features_accessed": False,
        "pseudo_test_labels_accessed": False,
        "scoring_performed": False,
    },
}

audit_json_path.write_text(
    json.dumps(
        record,
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)

lines = [
    "Phase 3B Failed Candidate v1 — Independent Static Audit",
    "",
    f"Candidate: {record['candidate_id']}",
    f"Protocol: {record['protocol']}",
    f"Technical run status: {record['technical_run_status']}",
    f"Candidate status: {record['candidate_status']}",
    f"Audit status: {record['audit_status']}",
    (
        "Mandatory failure count: "
        f"{record['mandatory_failure_count']}"
    ),
    (
        "Mandatory failures: "
        f"{record['mandatory_failures']}"
    ),
    f"Warnings: {record['warnings']}",
    "",
    "Checks:",
]

for item in checks:
    lines.append(
        f"{item['status']:4s} | "
        f"mandatory={item['mandatory']} | "
        f"{item['check_id']}"
    )

lines.extend(
    [
        "",
        "Decision:",
        (
            "  Phase 3B code modification allowed: False"
        ),
        (
            "  Additional run under Protocol v1.1: False"
        ),
        (
            "  External validation scoring allowed: False"
        ),
        (
            "  Pseudo-test scoring allowed: False"
        ),
        (
            "  A new locked protocol is required for any "
            "subsequent experiment."
        ),
        "",
        "Interpretation:",
        (
            "The run failed before producing a valid candidate. "
            "Even after a hypothetical row-count repair, the "
            "generated code would remain protocol-invalid because "
            "it performs validation-feature EDA, inspects ROW_ID "
            "beyond alignment, and averages predictions from five "
            "fold models."
        ),
    ]
)

audit_text_path.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print(
    f"AUDIT_JSON={audit_json_path}"
)
print(
    f"AUDIT_TEXT={audit_text_path}"
)
print(
    f"AUDIT_STATUS={audit_status}"
)
print(
    "MANDATORY_FAILURE_COUNT="
    f"{len(mandatory_failures)}"
)
print(
    "MANDATORY_FAILURES="
    f"{mandatory_failures}"
)
print(
    f"WARNINGS={warnings}"
)
print(
    "EXTERNAL_VALIDATION_SCORING_ALLOWED=False"
)
print(
    "PSEUDO_TEST_SCORING_ALLOWED=False"
)
print(
    "NEW_PROTOCOL_REQUIRED=True"
)
