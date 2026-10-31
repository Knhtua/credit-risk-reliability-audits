"""
main.py - LightGBM with PR-AUC Optimization, Monotonic Constraints, and Domain-Informed Interaction Features

A 5-fold StratifiedKFold LightGBM pipeline that:
  * directly optimizes PR-AUC on the rare positive class,
  * applies credit-domain monotonic constraints for robustness,
  * enriches the feature set with three deterministic interaction features,
  * ensembles multiple random seeds for variance reduction.
All preprocessing is fitted exclusively on train.csv folds; test.csv is used
only for inference to comply with the structural constraints.
"""
import os
import sys
import time
import gc
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import average_precision_score

# ============================================================================
# 1. Configuration and Constants
# ============================================================================
TRAIN_PATH = "./workspace_input/train.csv"
TEST_PATH = "./workspace_input/test.csv"

SUBMISSION_PATH = "./submission.csv"
SCORES_PATH = "./scores.csv"
CANDIDATE_PRED_DIR = "./candidate_predictions"
CANDIDATE_PRED_PATH = os.path.join(CANDIDATE_PRED_DIR, "lgbm_monotonic.csv")
PROTOCOL_AUDIT_PATH = "./protocol_audit.txt"

TARGET_COL = "DOWNGRADE_6M"
ID_COL = "ROW_ID"
RANDOM_SEED = 42
SEEDS = [42, 123, 2024]  # multi-seed ensemble for variance reduction (3 seeds, ~33 min, well within budget)

DEBUG = False
N_FOLDS = 5

LGB_PARAMS_BASE = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,        # lowered from 0.03 for finer convergence
    "num_leaves": 127,            # increased from 63 for richer trees
    "max_depth": -1,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "n_jobs": -1,
    "verbose": -1,
    "seed": RANDOM_SEED,
    "deterministic": True,
    "force_col_wise": True,
}

NUM_BOOST_ROUND = 8000           # increased cap to accommodate lower lr
EARLY_STOPPING_ROUNDS = 200      # increased patience for lower lr
EPSILON_FLOOR = 1e-6

POSITIVE_CONSTRAINT_FEATURES = [
    "LEV_RATIO", "NEGATIVE_OUTLOOK_FLAG",
    "LATEST_DOWNGRADE_FLAG", "RATING_SCORE_CHANGE",
]
NEGATIVE_CONSTRAINT_FEATURES = [
    "EQUITY_RATIO", "CURRENT_RATIO", "PRETAX_ROA_ANNUALIZED",
    "CURRENT_RATING_SCORE", "PREVIOUS_RATING_SCORE",
]
ENGINEERED_FEATURE_NAMES = ["LEV_x_ROA", "RATING_x_OUTLOOK", "CF_LIAB"]

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def mem_mb(df):
    return float(df.memory_usage(deep=True).sum() / (1024 ** 2))

# ============================================================================
# 2. Data Loading
# ============================================================================
def load_data():
    log("STEP 1: Loading data")
    log(f"  train: {TRAIN_PATH}")
    train = pd.read_csv(TRAIN_PATH)
    log(f"    shape={train.shape}, memory={mem_mb(train):.1f} MB")

    log(f"  test:  {TEST_PATH}")
    test = pd.read_csv(TEST_PATH)
    log(f"    shape={test.shape}, memory={mem_mb(test):.1f} MB")

    if TARGET_COL not in train.columns:
        raise ValueError(f"Target column {TARGET_COL} not in train.csv")
    if TARGET_COL in test.columns:
        raise ValueError(f"Target column {TARGET_COL} unexpectedly in test.csv")

    cls = train[TARGET_COL].value_counts().to_dict()
    log(f"  class distribution: {cls}")
    pos = int(cls.get(1, 0))
    neg = int(cls.get(0, 0))
    if (pos + neg) > 0:
        log(f"  positive rate = {pos / (pos + neg):.4%}")
    return train, test

# ============================================================================
# 3. Feature Engineering — Three Domain-Informed Interactions
# ============================================================================
def engineer_features(df):
    """Apply three domain-informed deterministic interactions.

    All formulas guard against divide-by-zero via np.maximum floors.
    Non-finite results (Inf from division by zero) are explicitly replaced
    with NaN so LightGBM can route them through its native missing-value
    handling.
    """
    out = df

    # 1. LEV_x_ROA = LEV_RATIO / max(PRETAX_ROA_ANNUALIZED, 0.01)
    a = out["LEV_RATIO"].astype("float32")
    b = np.maximum(out["PRETAX_ROA_ANNUALIZED"].astype("float32"), np.float32(0.01))
    out["LEV_x_ROA"] = (a / b).astype("float32")

    # 2. RATING_x_OUTLOOK = RATING_SCORE_CHANGE * (1 + NEGATIVE_OUTLOOK_FLAG)
    a = out["RATING_SCORE_CHANGE"].astype("float32")
    flag = out["NEGATIVE_OUTLOOK_FLAG"].astype("float32")
    out["RATING_x_OUTLOOK"] = (a * (np.float32(1.0) + flag)).astype("float32")

    # 3. CF_LIAB = CASHFLOW_NET_CF_OP / max(BALANCE_LA_TOT, 1.0)
    a = out["CASHFLOW_NET_CF_OP"].astype("float32")
    b = np.maximum(out["BALANCE_LA_TOT"].astype("float32"), np.float32(1.0))
    out["CF_LIAB"] = (a / b).astype("float32")

    # Replace any +/-Inf that arose from guarded divisions with NaN so
    # LightGBM handles them uniformly with other missing values.
    for c in ENGINEERED_FEATURE_NAMES:
        if c in out.columns:
            arr = out[c].to_numpy(dtype="float32")
            out[c] = np.where(np.isfinite(arr), arr, np.nan).astype("float32")

    return out

# ============================================================================
# 4. Feature Assembly + Type Enforcement
# ============================================================================
def assemble_and_cast(train, test):
    """Build X, y, X_test with proper dtypes (float32 numeric, categorical for encoded)."""
    log("STEP 2-3: Feature assembly & dtype enforcement")

    feature_cols = [c for c in train.columns if c not in (ID_COL, TARGET_COL)]
    log(f"  total feature columns (after engineering): {len(feature_cols)}")

    cat_cols = [c for c in feature_cols if c.endswith("__PHASE3_TRAIN_ENC")]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    log(f"  numeric={len(num_cols)}, categorical={len(cat_cols)}")
    log(f"  categorical: {cat_cols}")

    # Numeric -> float32
    for c in num_cols:
        if train[c].dtype != np.float32:
            train[c] = train[c].astype("float32")
        if c in test.columns and test[c].dtype != np.float32:
            test[c] = test[c].astype("float32")

    # Categorical -> pandas Categorical with union of codes
    for c in cat_cols:
        tr_vals = pd.Series(train[c]).dropna().astype("int64").unique().tolist()
        te_vals = pd.Series(test[c]).dropna().astype("int64").unique().tolist()
        union = sorted(set(int(v) for v in tr_vals) | set(int(v) for v in te_vals))
        cats = pd.Index(union, dtype="int64")
        train[c] = pd.Categorical(train[c].astype("int64"), categories=cats)
        test[c] = pd.Categorical(test[c].astype("int64"), categories=cats)
        log(f"    {c}: categories={list(cats)}")

    # Sanity: non-finite check on engineered columns
    for c in ENGINEERED_FEATURE_NAMES:
        if c in train.columns:
            arr_tr = train[c].to_numpy(dtype="float32")
            arr_te = test[c].to_numpy(dtype="float32")
            n_inf_tr = int((~np.isfinite(arr_tr)).sum())
            n_inf_te = int((~np.isfinite(arr_te)).sum())
            if n_inf_tr or n_inf_te:
                log(f"  WARNING non-finite in {c}: train={n_inf_tr}, test={n_inf_te}")

    y = train[TARGET_COL].astype("int32")
    X = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    del train  # free original train df
    gc.collect()

    return X, y, X_test, feature_cols, cat_cols

# ============================================================================
# 5. Monotonic Constraint Vector Construction
# ============================================================================
def build_monotone_constraints(feature_cols):
    pos_set = set(POSITIVE_CONSTRAINT_FEATURES)
    neg_set = set(NEGATIVE_CONSTRAINT_FEATURES)
    vec, mapping = [], []
    for i, col in enumerate(feature_cols):
        if col in pos_set:
            vec.append(1); mapping.append((i, col, 1))
        elif col in neg_set:
            vec.append(-1); mapping.append((i, col, -1))
        else:
            vec.append(0)
    return vec, mapping

# ============================================================================
# 6. CV Splits (with fallback)
# ============================================================================
def get_cv_splits(y, n_folds, seed):
    try:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(skf.split(np.zeros(len(y)), y))
        ok = True
        for _, vi in splits:
            if int(y.iloc[vi].sum()) == 0:
                ok = False; break
        if ok:
            log(f"  Using StratifiedKFold n_splits={n_folds}, seed={seed}.")
            return splits
    except Exception as e:
        log(f"  WARNING StratifiedKFold failed: {e!r}")

    log("  Fallback: 80/20 stratified single split.")
    tr, va = train_test_split(np.arange(len(y)), test_size=0.20,
                              stratify=y, random_state=seed)
    return [(tr, va)]

# ============================================================================
# 7. Model Training (per fold)
# ============================================================================
def train_one_fold(X_tr, y_tr, X_val, y_val, X_test, cat_cols,
                   mono_constraints, fold_id, num_round, seed):
    n_pos_tr = int((y_tr == 1).sum())
    n_neg_tr = int((y_tr == 0).sum())
    spw = n_neg_tr / max(n_pos_tr, 1)
    log(f"  Fold {fold_id}: pos={n_pos_tr} neg={n_neg_tr} spw={spw:.4f}")

    params = dict(LGB_PARAMS_BASE)
    params["scale_pos_weight"] = spw
    params["monotone_constraints"] = mono_constraints
    params["monotone_constraints_method"] = "advanced"
    # explicit per-fold seeds for ensemble diversity
    params["seed"] = seed
    params["bagging_seed"] = seed
    params["feature_fraction_seed"] = seed

    train_set = lgb.Dataset(X_tr, label=y_tr,
                            categorical_feature=cat_cols,
                            free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val,
                          categorical_feature=cat_cols,
                          reference=train_set,
                          free_raw_data=False)

    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS,
                           first_metric_only=True, verbose=False),
        lgb.log_evaluation(period=500),
    ]

    booster = lgb.train(
        params, train_set,
        num_boost_round=num_round,
        valid_sets=[val_set], valid_names=["val"],
        callbacks=callbacks,
    )

    best_iter = booster.best_iteration or 1
    try:
        best_score = float(booster.best_score["val"]["average_precision"])
    except Exception:
        best_score = float(average_precision_score(y_val,
            booster.predict(X_val, num_iteration=best_iter)))

    log(f"    best_iter={best_iter} best_pr_auc={best_score:.6f}")

    oof_pred = booster.predict(X_val, num_iteration=best_iter)
    test_pred = booster.predict(X_test, num_iteration=best_iter)

    # safety
    oof_pred = np.nan_to_num(oof_pred, nan=0.5, posinf=1.0, neginf=0.0)
    test_pred = np.nan_to_num(test_pred, nan=0.5, posinf=1.0, neginf=0.0)

    del train_set, val_set, booster
    gc.collect()

    return oof_pred, test_pred, best_iter, best_score

# ============================================================================
# 8. Run CV
# ============================================================================
def run_cv(X, y, X_test, cat_cols, mono_constraints, n_folds, seed, debug=False):
    log(f"STEP 4: Cross-validation training (seed={seed})")

    if debug:
        rng = np.random.default_rng(seed)
        n_sub = max(int(0.10 * len(y)), 1000)
        idx_sub = np.sort(rng.choice(len(y), size=n_sub, replace=False))
        X = X.iloc[idx_sub].reset_index(drop=True)
        y = y.iloc[idx_sub].reset_index(drop=True)
        log(f"  DEBUG: subsampled {n_sub} rows (10%)")

    splits = get_cv_splits(y, n_folds, seed)
    log(f"  CV folds: {len(splits)}")

    oof_pred = np.zeros(len(y), dtype="float64")
    test_preds = []
    fold_scores = []

    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        log(f"  -- Fold {fold_id + 1}/{len(splits)} --")
        num_round = 600 if debug else NUM_BOOST_ROUND

        oof_p, te_p, best_iter, best_score = train_one_fold(
            X.iloc[tr_idx], y.iloc[tr_idx],
            X.iloc[va_idx], y.iloc[va_idx],
            X_test, cat_cols, mono_constraints,
            fold_id=fold_id + 1, num_round=num_round, seed=seed,
        )
        oof_pred[np.asarray(va_idx)] = oof_p
        test_preds.append(te_p)
        fold_scores.append(best_score)

    test_avg = np.mean(np.vstack(test_preds), axis=0) if test_preds else np.full(len(X_test), 0.5)
    overall_score = float(average_precision_score(y, oof_pred))
    fold_mean = float(np.mean(fold_scores))
    fold_std = float(np.std(fold_scores))

    log(f"  Seed {seed} Overall OOF PR-AUC: {overall_score:.6f}")
    log(f"  Seed {seed} Fold mean +/- std:  {fold_mean:.6f} +/- {fold_std:.6f}")
    log(f"  Seed {seed} Per-fold scores:   {[round(s, 6) for s in fold_scores]}")
    return oof_pred, test_avg, overall_score, fold_scores, fold_mean, fold_std

# ============================================================================
# Output helpers
# ============================================================================
def clip_preds(p, lo=EPSILON_FLOOR, hi=1.0 - EPSILON_FLOOR):
    p = np.asarray(p, dtype="float64")
    p = np.where(np.isfinite(p), p, 0.5)
    return np.clip(p, lo, hi)

def save_submission(row_ids, preds, path):
    preds = clip_preds(preds)
    df = pd.DataFrame({ID_COL: row_ids, TARGET_COL: preds})
    df.to_csv(path, index=False)
    log(f"  saved {path} rows={len(df)} "
        f"min={float(preds.min()):.6f} max={float(preds.max()):.6f} mean={float(preds.mean()):.6f}")

def save_scores(scores_dict, path):
    rows = [{"Model": n, "PR-AUC (Average Precision)": float(v)}
            for n, v in scores_dict.items()]
    df = pd.DataFrame(rows).set_index("Model")
    df.to_csv(path)
    log(f"  saved {path}")
    print(df.to_string(), flush=True)

def save_audit(out_path, train_rows, test_rows, cls_dist, fold_scores,
               fold_mean, fold_std, overall_score, mono_mapping,
               feature_cols, cat_cols, lgb_params, per_seed_scores=None):
    lines = []
    lines.append("Protocol Audit - LightGBM Monotonic PR-AUC Pipeline (Multi-Seed Ensemble)")
    lines.append(f"Timestamp: {datetime.now().isoformat()}")
    lines.append("")
    lines.append("== Data Dimensions ==")
    lines.append(f"train.csv rows: {train_rows} (after engineering: cols={99 + len(ENGINEERED_FEATURE_NAMES)})")
    lines.append(f"test.csv  rows: {test_rows} (after engineering: cols={98 + len(ENGINEERED_FEATURE_NAMES)})")
    lines.append(f"Modeling features: total={len(feature_cols)} numeric={len(feature_cols)-len(cat_cols)} categorical={len(cat_cols)}")
    lines.append(f"Class distribution in train: {cls_dist}")
    lines.append("")
    lines.append("== Cross-Validation Results ==")
    lines.append(f"Folds: {len(fold_scores)}")
    lines.append(f"Fold-wise PR-AUC (avg across seeds): {[round(float(s), 6) for s in fold_scores]}")
    lines.append(f"Fold mean +/- std: {fold_mean:.6f} +/- {fold_std:.6f}")
    lines.append(f"Ensemble OOF PR-AUC: {overall_score:.6f}")
    if per_seed_scores:
        lines.append("Per-seed OOF PR-AUC:")
        for s, sc in per_seed_scores.items():
            lines.append(f"  seed={s}: {sc:.6f}")
    lines.append("")
    lines.append("== LightGBM Hyperparameters ==")
    for k, v in lgb_params.items():
        if k == "monotone_constraints":
            continue
        lines.append(f"  {k}: {v}")
    lines.append(f"  num_boost_round cap: {NUM_BOOST_ROUND}")
    lines.append(f"  early_stopping_rounds: {EARLY_STOPPING_ROUNDS}")
    lines.append(f"  seeds in ensemble: {SEEDS}")
    lines.append("")
    lines.append("== Monotonic Constraints ==")
    lines.append(f"Positive (+1) features: {POSITIVE_CONSTRAINT_FEATURES}")
    lines.append(f"Negative (-1) features: {NEGATIVE_CONSTRAINT_FEATURES}")
    lines.append("Mapped indices:")
    for i, col, c in mono_mapping:
        lines.append(f"  idx={i:3d} col={col:40s} constraint={c:+d}")
    lines.append("")
    lines.append("== Engineered Features ==")
    lines.append("  LEV_x_ROA        = LEV_RATIO / max(PRETAX_ROA_ANNUALIZED, 0.01)")
    lines.append("  RATING_x_OUTLOOK = RATING_SCORE_CHANGE * (1 + NEGATIVE_OUTLOOK_FLAG)")
    lines.append("  CF_LIAB          = CASHFLOW_NET_CF_OP / max(BALANCE_LA_TOT, 1)")
    lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    log(f"  saved {out_path}")

# ============================================================================
# EDA section (between markers)
# ============================================================================
def print_eda_section(train, test, feature_cols, cat_cols):
    lines = []
    lines.append("=== Start of EDA part ===")
    lines.append(f"Train shape: {train.shape}  Test shape: {test.shape}")
    lines.append(f"Total feature columns (after engineering): {len(feature_cols)}")
    lines.append(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

    counts = train[TARGET_COL].value_counts().to_dict()
    pos = int(counts.get(1, 0)); neg = int(counts.get(0, 0))
    total = pos + neg
    lines.append(f"Target DOWNGRADE_6M: 0={neg}, 1={pos}, positive_rate={pos/total:.4%}")

    miss_tr = train[feature_cols].isna().sum().sort_values(ascending=False).head(15)
    lines.append("Top-15 features by missing count in train:")
    for col, cnt in miss_tr.items():
        pct = cnt / len(train) * 100
        lines.append(f"  {col:42s}: {int(cnt):7d} ({pct:5.2f}%)")

    miss_te = test[feature_cols].isna().sum().sort_values(ascending=False).head(5)
    lines.append("Top-5 features by missing count in test:")
    for col, cnt in miss_te.items():
        pct = cnt / len(test) * 100
        lines.append(f"  {col:42s}: {int(cnt):7d} ({pct:5.2f}%)")

    lines.append("Categorical cardinality (train / test):")
    for c in cat_cols:
        try:
            k_tr = int(pd.Series(train[c]).nunique(dropna=True))
            k_te = int(pd.Series(test[c]).nunique(dropna=True))
            lines.append(f"  {c:42s} train={k_tr}, test={k_te}")
        except Exception:
            lines.append(f"  {c:42s} (inspect failed)")

    sample_feats = feature_cols[:5]
    lines.append("Describe (first 5 features, train):")
    desc = train[sample_feats].astype("float32").describe().T
    for col in sample_feats:
        r = desc.loc[col]
        lines.append(f"  {col:42s} mean={r['mean']:.4g}, std={r['std']:.4g}, "
                     f"min={r['min']:.4g}, max={r['max']:.4g}")

    lines.append(f"Train rows: {len(train)}  Test rows: {len(test)}")
    if len(train) > 0:
        lines.append(f"ROW_ID prefix train example: {train[ID_COL].iloc[0][:6]}")
    if len(test) > 0:
        lines.append(f"ROW_ID prefix test  example: {test[ID_COL].iloc[0][:6]}")

    # Engineered feature spot check
    lines.append("Engineered feature summary (train):")
    for c in ENGINEERED_FEATURE_NAMES:
        if c in train.columns:
            arr = train[c].astype("float32").to_numpy()
            lines.append(f"  {c:42s} mean={arr.mean():.4g} std={arr.std():.4g} "
                         f"min={arr.min():.4g} max={arr.max():.4g}")

    lines.append("=== End of EDA part ===")
    print("\n".join(lines), flush=True)

# ============================================================================
# Main
# ============================================================================
def main():
    wall_start = time.time()
    try:
        # ---- Data loading ----
        train, test = load_data()
        train_rows_orig = train.shape[0]
        test_rows_orig = test.shape[0]

        # ---- Feature engineering (deterministic, both) ----
        log("STEP 2: Engineering 3 interaction features")
        train = engineer_features(train)
        test = engineer_features(test)

        # Capture test ROW_ID ordering (in test.csv original order)
        test_row_ids = test[ID_COL].tolist()
        log(f"  test_row_ids captured: {len(test_row_ids)}")

        # ---- EDA between markers ----
        preview_features = [c for c in train.columns if c not in (ID_COL, TARGET_COL)]
        preview_cats = [c for c in preview_features if c.endswith("__PHASE3_TRAIN_ENC")]
        print_eda_section(train, test, preview_features, preview_cats)

        # ---- Assembly + cast ----
        X, y, X_test, feature_cols, cat_cols = assemble_and_cast(train, test)
        del test
        gc.collect()

        cls_dist = {0: int((y == 0).sum()), 1: int((y == 1).sum())}

        log(f"  X shape: {X.shape}  X_test shape: {X_test.shape}")
        log(f"  X memory: {mem_mb(X):.1f} MB  X_test memory: {mem_mb(X_test):.1f} MB")

        # ---- Monotonic constraints ----
        log("STEP 5: Building monotonic constraint vector")
        mono_vec, mono_map = build_monotone_constraints(feature_cols)
        if len(mono_vec) != len(feature_cols):
            raise ValueError(
                f"Monotonic vector length ({len(mono_vec)}) != feature count ({len(feature_cols)})"
            )
        log(f"  Constraint vector length = {len(mono_vec)} (matches feature count {len(feature_cols)})")
        for i, col, c in mono_map:
            log(f"    idx={i:3d} {col:42s} -> {c:+d}")

        # ---- CV training (multi-seed ensemble) ----
        n_folds = N_FOLDS if not DEBUG else 1
        log(f"STEP 6: CV training (DEBUG={DEBUG}, n_folds={n_folds})")
        log(f"  num_boost_round cap = {NUM_BOOST_ROUND}, early_stopping = {EARLY_STOPPING_ROUNDS}")
        log(f"  Seeds for ensemble: {SEEDS}")

        all_oof_preds = []
        all_test_preds = []
        all_fold_scores_per_seed = []
        per_seed_scores = {}

        for seed in SEEDS:
            log(f"=== Training seed={seed} ===")
            oof_pred, test_avg, _, fold_scores, _, _ = run_cv(
                X, y, X_test, cat_cols, mono_vec, n_folds=n_folds, seed=seed, debug=DEBUG,
            )
            all_oof_preds.append(oof_pred)
            all_test_preds.append(test_avg)
            all_fold_scores_per_seed.append(fold_scores)
            seed_score = float(average_precision_score(y, oof_pred))
            per_seed_scores[seed] = seed_score
            log(f"  seed={seed} final OOF PR-AUC: {seed_score:.6f}")

        # Average across seeds
        final_oof = np.mean(np.vstack(all_oof_preds), axis=0)
        final_test = np.mean(np.vstack(all_test_preds), axis=0)
        overall_score = float(average_precision_score(y, final_oof))

        # Per-fold reporting: average each fold across seeds
        fold_scores = list(np.mean(np.array(all_fold_scores_per_seed), axis=0))
        fold_mean = float(np.mean(fold_scores))
        fold_std = float(np.std(fold_scores))

        log(f"  === Ensemble (avg of {len(SEEDS)} seeds) OOF PR-AUC: {overall_score:.6f} ===")
        log(f"  Ensemble Fold mean +/- std:  {fold_mean:.6f} +/- {fold_std:.6f}")
        log(f"  Ensemble Per-fold scores:   {[round(s, 6) for s in fold_scores]}")

        test_avg = final_test  # for consistency with downstream

        # Sanity: alignment of test predictions
        if len(test_row_ids) != len(test_avg):
            raise ValueError(
                f"Alignment error: test_row_ids={len(test_row_ids)} vs test_avg={len(test_avg)}"
            )

        # ---- Outputs ----
        log("STEP 7: Saving outputs")
        os.makedirs(CANDIDATE_PRED_DIR, exist_ok=True)

        save_submission(test_row_ids, test_avg, SUBMISSION_PATH)
        save_submission(test_row_ids, test_avg, CANDIDATE_PRED_PATH)

        scores_dict = {"lgbm_monotonic": overall_score, "ensemble": overall_score}
        save_scores(scores_dict, SCORES_PATH)

        params_for_audit = dict(LGB_PARAMS_BASE)
        save_audit(
            PROTOCOL_AUDIT_PATH,
            train_rows=train_rows_orig, test_rows=test_rows_orig,
            cls_dist=cls_dist, fold_scores=fold_scores,
            fold_mean=fold_mean, fold_std=fold_std,
            overall_score=overall_score, mono_mapping=mono_map,
            feature_cols=feature_cols, cat_cols=cat_cols,
            lgb_params=params_for_audit,
            per_seed_scores=per_seed_scores,
        )

        elapsed = time.time() - wall_start
        log(f"=== DONE in {elapsed:.1f}s | Ensemble OOF PR-AUC = {overall_score:.6f} ===")
        return 0

    except Exception as e:
        log(f"FATAL: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())