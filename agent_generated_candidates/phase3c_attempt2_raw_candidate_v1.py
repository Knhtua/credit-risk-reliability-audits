#!/usr/bin/env python3
"""Phase 3C LightGBM Recipe Pipeline.

Produces four artifacts:
  - agent_model_recipe.json (primary research artifact)
  - agent_model_recipe.md   (identical persistence duplicate)
  - submission.csv          (canary with DOWNGRADE_6M=0.5)
  - scores.csv              (local proxy ROC-AUC + ensemble row)

Hypotheses:
  H1 - Domain-driven constrained feature engineering (20 slots, 6 operators).
  H2 - Class-imbalance / threshold-independent metric alignment.
  H3 - Temporal robustness through scale-invariant operators.
"""

import json
import os
import re
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# =============================================================
# Hardcoded constants
# =============================================================
SEED = 42
RECIPE_VERSION = "1.0"
BASE_MODEL = "lightgbm_binary"
DEBUG_MODE = False  # set True for 10-min debug cap

INPUT_DIR = "./workspace_input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
MANIFEST_PATH = os.path.join(INPUT_DIR, "modeling_feature_manifest.csv")
ALLOWLIST_PATH = os.path.join(INPUT_DIR, "phase3c_base_feature_allowlist.json")
SCHEMA_PATH = os.path.join(INPUT_DIR, "agent_model_recipe.schema.json")

RECIPE_JSON_PATH = "agent_model_recipe.json"
RECIPE_MD_PATH = "agent_model_recipe.md"
SUBMISSION_PATH = "submission.csv"
SCORES_PATH = "scores.csv"

FORBIDDEN_EXPECTED = {"ROW_ID", "DOWNGRADE_6M"}
ALLOWED_OPS = {"sum", "difference", "product", "ratio_safe",
               "max_zero_negative", "count_zero"}
MAX_FEATURES = 20
RATIO_EPS = 0.01
NAME_REGEX = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")
EPS_RANGE = (1e-12, 0.1)

TARGET_COL = "DOWNGRADE_6M"
ID_COL = "ROW_ID"

# The seven required recipe fields (must always be present in the JSON).
# These are the 7 mandatory fields per the task spec; the controller relies
# on every one of them (recipe_version, base_model, seed, class_weight_mode,
# engineered_features, agent_rationale, model_hyperparameters).
REQUIRED_RECIPE_FIELDS = {
    "recipe_version",
    "base_model",
    "seed",
    "class_weight_mode",
    "engineered_features",
    "agent_rationale",
    "model_hyperparameters",
}

# Hyperparameters for the recipe (H2 alignment)
MODEL_HYPERPARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "n_estimators": 2000,
    "min_child_samples": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 20,
    "verbosity": -1,
}

METRIC_COL = ("Not explicitly disclosed (canary submission is unscored; "
              "controller-internal binary-classification metric, "
              "conventionally ROC-AUC / Gini for credit downgrade tasks)")

# =============================================================
# Operator helpers (vectorized, deterministic)
# =============================================================
def op_sum(arrs):
    if not arrs:
        return np.zeros(0, dtype=np.float64)
    out = np.asarray(arrs[0], dtype=np.float64).copy()
    for a in arrs[1:]:
        out = out + np.asarray(a, dtype=np.float64)
    return out

def op_product(arrs):
    if not arrs:
        return np.zeros(0, dtype=np.float64)
    out = np.ones_like(np.asarray(arrs[0]), dtype=np.float64)
    for a in arrs:
        out = out * np.asarray(a, dtype=np.float64)
    return out

def op_difference(arrs):
    if not arrs:
        return np.zeros(0, dtype=np.float64)
    out = np.asarray(arrs[0], dtype=np.float64).copy()
    for a in arrs[1:]:
        out = out - np.asarray(a, dtype=np.float64)
    return out

def op_ratio_safe(a, b, eps):
    return np.asarray(a, dtype=np.float64) / (np.asarray(b, dtype=np.float64) + eps)

def op_max_zero_negative(arrs):
    if not arrs:
        return np.zeros(0, dtype=np.float64)
    clipped = [np.maximum(np.asarray(a, dtype=np.float64), 0.0) for a in arrs]
    out = clipped[0].copy()
    for a in clipped[1:]:
        out = np.maximum(out, a)
    return out

def op_count_zero(arrs):
    if not arrs:
        return np.zeros(0, dtype=np.float64)
    out = np.zeros_like(np.asarray(arrs[0]), dtype=np.float64)
    for a in arrs:
        out = out + (np.asarray(a) == 0).astype(np.float64)
    return out

# =============================================================
# Engineered feature definitions (20 slots, 6 operators)
# =============================================================
def define_engineered_features():
    eps = RATIO_EPS
    return [
        # Leverage-stress ratios (3)
        {"name": "LEV_TO_EQUITY", "operator": "ratio_safe",
         "inputs": ["LEV_RATIO", "EQUITY_RATIO"], "epsilon": eps},
        {"name": "INT_DEBT_LEV", "operator": "ratio_safe",
         "inputs": ["INTEREST_DEBT_ASSET_RATIO", "LEV_RATIO"], "epsilon": eps},
        {"name": "CURR_LA_RATIO", "operator": "ratio_safe",
         "inputs": ["BALANCE_CUR_LA_TOT", "BALANCE_CUR_ASSETS_TOT"], "epsilon": eps},
        # Liquidity-coverage ratios (3)
        {"name": "LIQ_CURR_CASH", "operator": "ratio_safe",
         "inputs": ["CURRENT_RATIO", "CASH_RATIO"], "epsilon": eps},
        {"name": "CASH_TO_CURR", "operator": "ratio_safe",
         "inputs": ["CASH_RATIO", "CURRENT_RATIO"], "epsilon": eps},
        {"name": "WC_INTENSITY", "operator": "ratio_safe",
         "inputs": ["BALANCE_CUR_ASSETS_TOT", "BALANCE_CUR_LA_TOT"], "epsilon": eps},
        # Profitability-to-size ratios (4)
        {"name": "ROA_SLOG_ASSETS", "operator": "ratio_safe",
         "inputs": ["PRETAX_ROA_ANNUALIZED", "BALANCE_ASSETS_TOT_SLOG"], "epsilon": eps},
        {"name": "OPCASH_SLOG_INC", "operator": "ratio_safe",
         "inputs": ["OP_CASHFLOW_ASSET_ANNUALIZED", "INCOME_OP_INCM_TOT_SLOG"], "epsilon": eps},
        {"name": "ROA_SLOG_EQUITY", "operator": "ratio_safe",
         "inputs": ["PRETAX_ROA_YTD", "BALANCE_EQUITY_DERIVED_SLOG"], "epsilon": eps},
        {"name": "REVCFO_RATIO", "operator": "ratio_safe",
         "inputs": ["REVENUE_ASSET_YTD", "OP_CASHFLOW_ASSET_YTD"], "epsilon": eps},
        # Rating-momentum one-sided transforms (4)
        {"name": "RATING_UP_ONLY", "operator": "max_zero_negative",
         "inputs": ["RATING_SCORE_CHANGE"]},
        {"name": "RATING_AGE_INC", "operator": "max_zero_negative",
         "inputs": ["RATING_AGE_DAYS_YOY_CHANGE"]},
        {"name": "POS_OP_CASH", "operator": "max_zero_negative",
         "inputs": ["OP_CASHFLOW_ASSET_YTD"]},
        {"name": "POS_EQUITY_GROWTH", "operator": "max_zero_negative",
         "inputs": ["EQUITY_RATIO_YOY_CHANGE"]},
        # Outlook composites (3)
        {"name": "OUTLOOK_DUAL", "operator": "product",
         "inputs": ["NEGATIVE_OUTLOOK_FLAG", "WATCH_OUTLOOK_FLAG"]},
        {"name": "OUTLOOK_SUM", "operator": "sum",
         "inputs": ["NEGATIVE_OUTLOOK_FLAG", "WATCH_OUTLOOK_FLAG"]},
        {"name": "OUTLOOK_CLEAN", "operator": "count_zero",
         "inputs": ["NEGATIVE_OUTLOOK_FLAG", "WATCH_OUTLOOK_FLAG", "LATEST_DOWNGRADE_FLAG"]},
        # Availability-weighted financial signals (3)
        {"name": "BAL_AVAIL", "operator": "product",
         "inputs": ["HAS_BALANCE", "BALANCE_ASSETS_TOT_SLOG"]},
        {"name": "INC_AVAIL", "operator": "product",
         "inputs": ["HAS_INCOME", "INCOME_OP_INCM_TOT_SLOG"]},
        {"name": "CF_AVAIL", "operator": "product",
         "inputs": ["HAS_CASHFLOW", "CASHFLOW_NET_CF_OP_SLOG"]},
    ]

# =============================================================
# I/O helpers
# =============================================================
def resolve_input_dir():
    """Resolve the Phase 3C input directory.

    Strict resolution: prefers ./workspace_input relative to cwd, then falls
    back to script_dir/workspace_input. Does NOT search parent directories
    or alternate names (input/, phase3c_input/) to avoid ambiguity with
    external test scripts that also search for the input directory and
    may crash if multiple candidates are found.

    Returns the hardcoded './workspace_input' path as a last-resort fallback
    so downstream file-loading code raises a clear FileNotFoundError if no
    valid directory exists.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = ""
    if not script_dir:
        script_dir = os.getcwd()
    cwd = os.getcwd()

    expected = ["train.csv", "test.csv", "phase3c_base_feature_allowlist.json"]

    # Priority 1: ./workspace_input relative to cwd (canonical Kaggle path).
    candidate = os.path.join(cwd, "workspace_input")
    if os.path.isdir(candidate):
        if all(os.path.isfile(os.path.join(candidate, f)) for f in expected):
            return candidate

    # Priority 2: workspace_input next to main.py.
    candidate = os.path.join(script_dir, "workspace_input")
    if os.path.isdir(candidate):
        if all(os.path.isfile(os.path.join(candidate, f)) for f in expected):
            return candidate

    # Last-resort fallback: hardcoded relative path (will raise clear error
    # downstream if files are missing).
    return "./workspace_input"

def load_inputs():
    input_dir = resolve_input_dir()
    print(f"  Resolved input directory: {os.path.abspath(input_dir)}")
    train = pd.read_csv(os.path.join(input_dir, "train.csv"))
    test = pd.read_csv(os.path.join(input_dir, "test.csv"))
    manifest = pd.read_csv(os.path.join(input_dir, "modeling_feature_manifest.csv"))
    with open(os.path.join(input_dir, "phase3c_base_feature_allowlist.json"), "r", encoding="utf-8") as f:
        allowlist = json.load(f)
    with open(os.path.join(input_dir, "agent_model_recipe.schema.json"), "r", encoding="utf-8") as f:
        schema = json.load(f)
    return train, test, manifest, allowlist, schema

def validate_allowlist(allowlist):
    if "base_features_in_position_order" not in allowlist:
        raise ValueError("allowlist missing 'base_features_in_position_order'")
    if "forbidden_inputs" not in allowlist:
        raise ValueError("allowlist missing 'forbidden_inputs'")
    allowed = set(allowlist["base_features_in_position_order"])
    forbidden = set(allowlist["forbidden_inputs"])
    if len(allowed) != 230:
        print(f"WARNING: expected 230 base features, got {len(allowed)}")
    if forbidden != FORBIDDEN_EXPECTED:
        print(f"WARNING: forbidden_inputs {forbidden} != expected {FORBIDDEN_EXPECTED}")
    if "allowed_engineered_operators" in allowlist:
        ops = set(allowlist["allowed_engineered_operators"])
        if ops != ALLOWED_OPS:
            print(f"WARNING: ops {ops} != {ALLOWED_OPS}")
    return allowed, forbidden

def validate_features(features, allowed, forbidden):
    if len(features) > MAX_FEATURES:
        raise ValueError(f"Too many features: {len(features)} > {MAX_FEATURES}")
    ops_used = set()
    for f in features:
        name = f["name"]
        if not NAME_REGEX.match(name):
            raise ValueError(f"Bad name regex: {name!r}")
        op = f["operator"]
        if op not in ALLOWED_OPS:
            raise ValueError(f"Operator {op!r} for {name!r} not in {ALLOWED_OPS}")
        ops_used.add(op)
        inputs = f["inputs"]
        if not inputs:
            raise ValueError(f"Feature {name!r} has empty inputs list")
        for inp in inputs:
            if inp in forbidden:
                raise ValueError(f"Feature {name!r}: input {inp!r} is forbidden")
            if inp not in allowed:
                raise ValueError(f"Feature {name!r}: input {inp!r} not in 230 allowed base features")
        if op == "ratio_safe":
            eps = f.get("epsilon")
            if eps is None:
                raise ValueError(f"Feature {name!r}: ratio_safe missing epsilon")
            lo, hi = EPS_RANGE
            if eps < lo or eps > hi:
                raise ValueError(f"Feature {name!r}: epsilon {eps} outside [{lo}, {hi}]")
    if not ops_used.issubset(ALLOWED_OPS):
        raise ValueError(f"Unexpected operators: {ops_used - ALLOWED_OPS}")
    return ops_used

# =============================================================
# Engineered feature computation
# =============================================================
def compute_engineered(df, features):
    new_cols = {}
    for f in features:
        op = f["operator"]
        inputs = f["inputs"]
        cols = [np.asarray(df[c].to_numpy(), dtype=np.float64) for c in inputs]
        if op == "sum":
            arr = op_sum(cols)
        elif op == "product":
            arr = op_product(cols)
        elif op == "difference":
            arr = op_difference(cols)
        elif op == "ratio_safe":
            arr = op_ratio_safe(cols[0], cols[1], float(f["epsilon"]))
        elif op == "max_zero_negative":
            arr = op_max_zero_negative(cols)
        elif op == "count_zero":
            arr = op_count_zero(cols)
        else:
            raise ValueError(f"Unknown operator: {op!r}")
        arr = np.where(np.isinf(arr), np.nan, arr)
        new_cols[f["name"]] = arr
    return pd.DataFrame(new_cols, index=df.index)

# =============================================================
# Recipe builders
# =============================================================
def build_recipe(features):
    # Task spec calls for a 50-1000 character narrative.
    rationale = (
        "Phase 3C credit-downgrade recipe emits 20 engineered features across 5 domain "
        "groups: leverage-stress (LEV/equity, INT/LEV, CL/CA), liquidity-coverage "
        "(current/cash, CA/CL), profitability-to-size (ROA/OpCash/RevCFO over signed-log "
        "bases), rating-momentum (one-sided max_zero_negative), outlook composites "
        "(product/sum/count_zero), and availability-weighted signals "
        "(product HAS_* x slog). Operators are scale-invariant (ratio_safe eps=0.01), "
        "one-sided, count-based, non-chained, and restricted to the 230 permitted base "
        "features for temporal robustness across 2013-2018. class_weight_mode=train_ratio "
        "with objective=binary and metric=auc optimizes ROC-AUC/Gini under rare-downgrade "
        "imbalance. LightGBM hyperparameters (lr=0.03, num_leaves=31, n_estimators=2000, "
        "min_child_samples=50, reg_alpha=0.1, reg_lambda=0.1, feature_fraction=0.8, "
        "bagging_fraction=0.8, bagging_freq=5, min_data_in_leaf=20, seed=42) provide "
        "moderate capacity with regularization for the 111k-row training matrix."
    )
    print(f"  rationale length: {len(rationale)} chars")
    if not (50 <= len(rationale) <= 1000):
        print(f"WARNING: rationale length {len(rationale)} outside task range [50, 1000]")

    return {
        "recipe_version": RECIPE_VERSION,
        "base_model": BASE_MODEL,
        "seed": SEED,
        "class_weight_mode": "train_ratio",
        "engineered_features": features,
        "agent_rationale": rationale,
        "model_hyperparameters": dict(MODEL_HYPERPARAMS),
    }

def clean_recipe_to_schema(recipe, schema):
    """Defensive schema cleaning.

    Always retain the 7 required recipe fields (recipe_version, base_model,
    seed, class_weight_mode, engineered_features, agent_rationale,
    model_hyperparameters) regardless of whether the loaded schema's properties
    dict lists them. Additionally retain any field that is explicitly listed
    in the schema's 'properties' dict. Only drop fields that are neither
    required nor in the schema's properties.
    """
    allowed_keys = set(REQUIRED_RECIPE_FIELDS)
    if "properties" in schema and isinstance(schema["properties"], dict):
        allowed_keys |= set(schema["properties"].keys())
    cleaned = {k: v for k, v in recipe.items() if k in allowed_keys}
    missing = REQUIRED_RECIPE_FIELDS - set(cleaned.keys())
    if missing:
        print(f"WARNING: recipe still missing required fields: {sorted(missing)}")
    else:
        print(f"  all 7 required fields present: {sorted(REQUIRED_RECIPE_FIELDS)}")
    return cleaned

# =============================================================
# Proxy training
# =============================================================
def train_proxy(X, y, scale_pos_weight):
    n = len(X)

    if DEBUG_MODE:
        n_sample = max(int(n * 0.1), 1000)
        X = X.iloc[:n_sample].reset_index(drop=True)
        y = y.iloc[:n_sample].reset_index(drop=True)
        n = n_sample
        print(f"  DEBUG_MODE: using first {n} rows for proxy training")

    try:
        tr_idx, va_idx = train_test_split(
            np.arange(n), test_size=0.2, random_state=SEED, stratify=y
        )
    except ValueError:
        print("  Stratify split failed, falling back to plain split")
        tr_idx, va_idx = train_test_split(
            np.arange(n), test_size=0.2, random_state=SEED
        )

    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    print(f"  train: {len(X_tr)} rows ({int(y_tr.sum())} pos), "
          f"val: {len(X_va)} rows ({int(y_va.sum())} pos)")

    # NOTE: We do NOT pass scale_pos_weight to the proxy LightGBM.
    # The recipe already declares class_weight_mode='train_ratio', which
    # conveys the imbalance treatment to the controller. Applying
    # scale_pos_weight (~76.5 = neg/pos) in addition causes the positive
    # class to be weighted ~76.5x by LightGBM AND treated as rare by the
    # eval metric, producing a degenerate first iteration that satisfies
    # the validation AUC almost immediately (best_iteration_=1). Without
    # scale_pos_weight the booster can actually converge over many rounds
    # and produce a meaningful best_iteration / ROC-AUC estimate.
    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "random_state": SEED,
        "n_jobs": -1,
        "learning_rate": MODEL_HYPERPARAMS["learning_rate"],
        "num_leaves": MODEL_HYPERPARAMS["num_leaves"],
        "n_estimators": MODEL_HYPERPARAMS["n_estimators"],
        "min_child_samples": MODEL_HYPERPARAMS["min_child_samples"],
        "reg_alpha": MODEL_HYPERPARAMS["reg_alpha"],
        "reg_lambda": MODEL_HYPERPARAMS["reg_lambda"],
        "feature_fraction": MODEL_HYPERPARAMS["feature_fraction"],
        "bagging_fraction": MODEL_HYPERPARAMS["bagging_fraction"],
        "bagging_freq": MODEL_HYPERPARAMS["bagging_freq"],
        "min_data_in_leaf": MODEL_HYPERPARAMS["min_data_in_leaf"],
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(**params)
    print("  Fitting LightGBM with early stopping (patience=100, eval=auc)...")
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    best_iter = getattr(model, "best_iteration_", None)
    val_pred = model.predict_proba(X_va)[:, 1]
    auc = float(roc_auc_score(y_va, val_pred))
    print(f"  best_iteration: {best_iter}")
    print(f"  Proxy ROC-AUC (validation): {auc:.6f}")
    return auc, best_iter

# =============================================================
# Artifact writers
# =============================================================
def write_recipe(recipe):
    text = json.dumps(recipe, indent=2, ensure_ascii=False)
    with open(RECIPE_JSON_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    with open(RECIPE_MD_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  wrote {RECIPE_JSON_PATH} ({len(text)} bytes)")
    print(f"  wrote {RECIPE_MD_PATH} (identical content)")

def write_submission(test):
    sub = pd.DataFrame({
        ID_COL: test[ID_COL].values,
        TARGET_COL: np.full(len(test), 0.5, dtype=np.float64),
    })
    if len(sub) != len(test):
        raise ValueError(f"Submission length {len(sub)} != test length {len(test)}")
    if sub[TARGET_COL].isna().any() or np.isinf(sub[TARGET_COL]).any():
        raise ValueError("Submission has NaN/inf in DOWNGRADE_6M")
    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"  wrote {SUBMISSION_PATH} ({len(sub)} rows, all DOWNGRADE_6M=0.5)")

def write_scores(auc):
    df = pd.DataFrame(
        {METRIC_COL: [auc, auc]},
        index=pd.Index(["lightgbm_binary_recipe", "ensemble"]),
    )
    df.to_csv(SCORES_PATH, index=True)
    print(f"  wrote {SCORES_PATH}")

# =============================================================
# Main
# =============================================================
def main():
    print("=== Phase 3C LightGBM Recipe Pipeline ===")
    print(f"SEED={SEED}, DEBUG_MODE={DEBUG_MODE}")
    print(f"INTERPRETER: {sys.version.split()[0]}")

    # ---------- Stage 2: Load inputs ----------
    train, test, manifest, allowlist, schema = load_inputs()
    print(f"[Stage 2] train.csv shape: {train.shape}")
    print(f"[Stage 2] test.csv shape: {test.shape}")
    print(f"[Stage 2] modeling_feature_manifest.csv shape: {manifest.shape}")

    # ---------- Stage 3: Validate allowlist ----------
    allowed_set, forbidden_set = validate_allowlist(allowlist)
    print(f"[Stage 3] allowed base features: {len(allowed_set)}")
    print(f"[Stage 3] forbidden inputs: {sorted(forbidden_set)}")

    # ---------- Stage 4: Define + validate engineered features ----------
    features = define_engineered_features()
    ops_used = validate_features(features, allowed_set, forbidden_set)
    print(f"[Stage 4] {len(features)} engineered features validated")
    print(f"[Stage 4] operators used: {sorted(ops_used)}")

    # ---------- EDA block ----------
    print()
    print("=== Start of EDA part ===")
    target = train[TARGET_COL]
    pos_n = int(target.sum())
    neg_n = int(len(target) - pos_n)
    print(f"Train rows: {len(train)}, test rows: {len(test)}")
    print(f"Train columns: {train.shape[1]}, test columns: {test.shape[1]}")
    print(f"Target DOWNGRADE_6M: positive={pos_n}, negative={neg_n}, "
          f"ratio={pos_n / len(target):.6f}")
    print(f"Base features (allowlist): {len(allowed_set)}")
    print(f"Engineered features: {len(features)}")
    print(f"Engineered operators: {sorted(ops_used)}")
    print(f"Engineered feature names: {[f['name'] for f in features]}")
    print(f"Train dtypes count: float64={(train.dtypes == 'float64').sum()}, "
          f"int64={(train.dtypes == 'int64').sum()}, "
          f"object={(train.dtypes == 'object').sum()}")
    print(f"Test dtypes count: float64={(test.dtypes == 'float64').sum()}, "
          f"int64={(test.dtypes == 'int64').sum()}, "
          f"object={(test.dtypes == 'object').sum()}")
    eng_inputs = sorted({inp for f in features for inp in f["inputs"]})
    print(f"Distinct engineered-feature input cols: {len(eng_inputs)}")
    miss_train = int(train[eng_inputs].isna().sum().sum())
    miss_test = int(test[eng_inputs].isna().sum().sum())
    print(f"Missing values across engineered-feature input cols (train): {miss_train}")
    print(f"Missing values across engineered-feature input cols (test): {miss_test}")
    print(f"Manifest columns: {list(manifest.columns)}")
    print(f"Schema required fields: {schema.get('required', 'N/A')}")
    print(f"Schema property keys: {sorted(schema.get('properties', {}).keys())}")
    print("=== End of EDA part ===")
    print()

    # ---------- Stage 5: Compute engineered features ----------
    print("[Stage 5] Computing engineered features on train and test ...")
    eng_train = compute_engineered(train, features)
    eng_test = compute_engineered(test, features)
    print(f"  engineered train shape: {eng_train.shape}")
    print(f"  engineered test  shape: {eng_test.shape}")
    for col in eng_train.columns:
        arr = eng_train[col].to_numpy()
        print(f"  {col}: mean={np.nanmean(arr):.4f}, std={np.nanstd(arr):.4f}, "
              f"nan_count={int(np.isnan(arr).sum())}")

    # ---------- Stage 6: Class imbalance statistics ----------
    print("[Stage 6] Class imbalance statistics ...")
    pos_count = int(train[TARGET_COL].sum())
    neg_count = int(len(train) - pos_count)
    train_ratio = pos_count / len(train)
    scale_pos_weight = neg_count / max(pos_count, 1)
    print(f"  pos: {pos_count}, neg: {neg_count}, ratio: {train_ratio:.6f}")
    print(f"  scale_pos_weight (for reference only, NOT applied to proxy): "
          f"{scale_pos_weight:.4f}")

    # ---------- Stage 7: Build + clean recipe ----------
    print("[Stage 7] Building + cleaning recipe JSON ...")
    recipe = build_recipe(features)
    recipe = clean_recipe_to_schema(recipe, schema)
    print(f"  Recipe keys: {sorted(recipe.keys())}")
    print(f"  Recipe engineered_features: {len(recipe['engineered_features'])}")
    if recipe["recipe_version"] != "1.0":
        print(f"WARNING: recipe_version {recipe['recipe_version']!r} != '1.0'")
    if recipe["base_model"] != "lightgbm_binary":
        print(f"WARNING: base_model {recipe['base_model']!r} != 'lightgbm_binary'")
    if recipe["seed"] != 42:
        print(f"WARNING: seed {recipe['seed']} != 42")
    if recipe["class_weight_mode"] not in {"none", "train_ratio"}:
        print(f"WARNING: class_weight_mode {recipe['class_weight_mode']!r} not in allowed")
    if "model_hyperparameters" not in recipe:
        print(f"CRITICAL WARNING: model_hyperparameters missing from final recipe")

    # ---------- Stage 8: Proxy training ----------
    print("[Stage 8] Building X, y; training proxy LightGBM ...")
    base_cols = [c for c in train.columns if c not in (ID_COL, TARGET_COL)]
    feat_names = [f["name"] for f in features]
    X_full = pd.concat(
        [train[base_cols].reset_index(drop=True),
         eng_train.reset_index(drop=True)],
        axis=1,
    )
    y_full = train[TARGET_COL].reset_index(drop=True)
    print(f"  X shape: {X_full.shape} (230 base + 20 engineered)")
    print(f"  y shape: {y_full.shape}")

    proxy_auc, best_iter = train_proxy(X_full, y_full, scale_pos_weight)

    # ---------- Stage 9: Write artifacts ----------
    print("[Stage 9] Writing artifacts ...")
    write_recipe(recipe)
    write_submission(test)
    write_scores(proxy_auc)

    # ---------- Stage 10: Final summary ----------
    print()
    print("=== Stage 10: Final Summary ===")
    print(f"  engineered_features count: {len(recipe['engineered_features'])}")
    print(f"  class_weight_mode: {recipe['class_weight_mode']}")
    print(f"  hyperparameters summary: lr={MODEL_HYPERPARAMS['learning_rate']}, "
          f"num_leaves={MODEL_HYPERPARAMS['num_leaves']}, "
          f"n_est={MODEL_HYPERPARAMS['n_estimators']}, "
          f"min_child={MODEL_HYPERPARAMS['min_child_samples']}, "
          f"reg_alpha={MODEL_HYPERPARAMS['reg_alpha']}, "
          f"reg_lambda={MODEL_HYPERPARAMS['reg_lambda']}")
    print(f"  proxy ROC-AUC: {proxy_auc:.6f}")
    print(f"  best_iteration: {best_iter}")
    print(f"  submission rows: {len(test)}")

    sub_check = pd.read_csv(SUBMISSION_PATH)
    unique_vals = sorted(sub_check[TARGET_COL].unique().tolist())
    print(f"  submission verify: rows={len(sub_check)}, "
          f"unique DOWNGRADE_6M={unique_vals}, "
          f"NaN count={int(sub_check[TARGET_COL].isna().sum())}")

    print("=== Pipeline complete ===")

if __name__ == "__main__":
    main()