#!/usr/bin/env python3
"""
Phase 3B Bond Credit Risk Task - Single LightGBM pipeline.
Predicts bond downgrade within 6 months using PR-AUC metric.
"""

import os
import sys
import time
import gc
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import average_precision_score, log_loss

warnings.filterwarnings("ignore")

# ============== Hardcoded Configuration ==============
DEBUG_MODE = False  # True: 10% subsample, 1 fold, faster
WORKDIR = "./workspace_input"
SEED = 42
TEST_N_ROWS = 60702  # From test.csv (NOT from sample_submission.csv)


def print_eda(train, test, manifest):
    """Print EDA between markers for automatic extraction."""
    print("=== Start of EDA part ===")
    print(f"Train: {train.shape}, Test: {test.shape}, Manifest: {manifest.shape}")
    print(f"Memory: train={train.memory_usage(deep=True).sum()/1e6:.1f}MB, test={test.memory_usage(deep=True).sum()/1e6:.1f}MB")
    print()
    print("Manifest roles:")
    print(manifest["role"].value_counts().to_string())
    print()
    print("Manifest dtypes:")
    print(manifest["dtype"].value_counts().to_string())
    print()
    print("Manifest source_scope:")
    print(manifest["source_scope"].value_counts().to_string())
    print()
    tgt = train["DOWNGRADE_6M"]
    print(f"Target: pos={int(tgt.sum())}, neg={int((1-tgt).sum())}, prev={tgt.mean():.4f}")
    miss_tr = train.isna().sum().sort_values(ascending=False)
    miss_tr = miss_tr[miss_tr > 0]
    print(f"Train NaN cols: {len(miss_tr)}; top10:\n{miss_tr.head(10).to_string()}")
    miss_te = test.isna().sum().sort_values(ascending=False)
    miss_te = miss_te[miss_te > 0]
    print(f"Test NaN cols: {len(miss_te)}; top10:\n{miss_te.head(10).to_string()}")
    print(f"Train unique ROW_ID: {train['ROW_ID'].nunique()}")
    print(f"Test unique ROW_ID: {test['ROW_ID'].nunique()}")
    print(f"Train ROW_ID sample: {train['ROW_ID'].iloc[0]}")
    print(f"Test ROW_ID sample: {test['ROW_ID'].iloc[0]}")
    print(f"Numeric cols train: {len(train.select_dtypes(include=[np.number]).columns)}")
    print(f"Numeric cols test:  {len(test.select_dtypes(include=[np.number]).columns)}")
    print("=== End of EDA part ===")


def add_engineered(df):
    """Add 4 row-wise engineered features. NaN propagates naturally."""
    df = df.copy()
    has_cols = [c for c in df.columns if c.startswith("HAS_")]
    # MISSING_COUNT: count HAS_ features that are zero (NOT *_UNKNOWN)
    df["MISSING_COUNT"] = (df[has_cols] == 0).sum(axis=1).astype(np.float32)

    neg = df["NEGATIVE_OUTLOOK_FLAG"].values
    watch = df["WATCH_OUTLOOK_FLAG"].values
    drop = df["LATEST_DOWNGRADE_FLAG"].values
    rsc = df["RATING_SCORE_CHANGE"].values
    crs = df["CURRENT_RATING_SCORE"].values

    # RATING_STRESS_SCORE: NaN if RATING_SCORE_CHANGE is NaN
    rsc_part = np.maximum(0.0, -rsc)
    df["RATING_STRESS_SCORE"] = (neg + watch + drop + rsc_part).astype(np.float32)

    # OUTLOOK_X_SCORE
    df["OUTLOOK_X_SCORE"] = ((neg + watch) * crs).astype(np.float32)

    # NEGATIVE_OUTLOOK_X_DROP_FLAG
    df["NEGATIVE_OUTLOOK_X_DROP_FLAG"] = (neg * drop).astype(np.float32)

    return df


def make_log_cb(X_tr, y_tr, X_va, y_va, every):
    """Logging callback for training progress."""
    def cb(env):
        if (env.iteration + 1) % every == 0 or env.iteration == 0:
            tr_p = env.model.predict(X_tr)
            va_p = env.model.predict(X_va)
            tr_loss = log_loss(y_tr, np.clip(tr_p, 1e-7, 1 - 1e-7))
            va_loss = log_loss(y_va, np.clip(va_p, 1e-7, 1 - 1e-7))
            va_pr = average_precision_score(y_va, va_p)
            print(f"    [R{env.iteration+1}] tr_loss={tr_loss:.5f} va_loss={va_loss:.5f} va_pr_auc={va_pr:.5f}")
    return cb


def lgb_pr_auc(y_pred, ds):
    """PR-AUC custom metric for LightGBM (higher is better)."""
    return "pr_auc", average_precision_score(ds.get_label(), y_pred), True


def main():
    t0 = time.time()
    print("=" * 70)
    print(f"[CFG] DEBUG_MODE={DEBUG_MODE} SEED={SEED}")
    print("=" * 70)

    # ============ Load data ============
    print("\n[1/8] Loading data...")
    train = pd.read_csv(f"{WORKDIR}/train.csv", encoding="utf-8")
    test = pd.read_csv(f"{WORKDIR}/test.csv", encoding="utf-8")
    manifest = pd.read_csv(f"{WORKDIR}/modeling_feature_manifest.csv", encoding="utf-8")
    with open(f"{WORKDIR}/description.md", "r", encoding="utf-8") as f:
        _ = f.read()

    print(f"[INFO] train={train.shape}, test={test.shape}, manifest={manifest.shape}")
    assert len(test) == TEST_N_ROWS, f"Test row count: {len(test)} != {TEST_N_ROWS}"

    # ============ Process manifest ============
    print("\n[2/8] Processing manifest...")
    manifest = manifest.sort_values("position").reset_index(drop=True)
    feature_cols = [c for c in manifest["column"].tolist() if c != "DOWNGRADE_6M"]
    assert len(feature_cols) == 230, f"Expected 230 features, got {len(feature_cols)}"

    missing_tr = [c for c in feature_cols if c not in train.columns]
    missing_te = [c for c in feature_cols if c not in test.columns]
    assert not missing_tr and not missing_te, f"Missing: tr={missing_tr}, te={missing_te}"
    print(f"[INFO] All {len(feature_cols)} manifest features validated")

    # ============ EDA ============
    print("\n[3/8] EDA...")
    print_eda(train, test, manifest)

    # ============ Alignment DataFrame ============
    print("\n[4/8] Building alignment DataFrame...")
    n_train = len(train)
    align_df = pd.DataFrame({
        "id": pd.concat([train["ROW_ID"].astype(str), test["ROW_ID"].astype(str)], ignore_index=True),
        "path": [f"{WORKDIR}/train.csv"] * n_train + [f"{WORKDIR}/test.csv"] * len(test),
        "fold": -1,
    })
    print(f"[INFO] align_df shape={align_df.shape}, columns={list(align_df.columns)}")

    # ============ Feature matrices ============
    print("\n[5/8] Building features (float32)...")
    y_full = train["DOWNGRADE_6M"].astype(np.int8).values
    print(f"[INFO] Target: pos={int(y_full.sum())} neg={len(y_full)-int(y_full.sum())} prev={y_full.mean():.4f}")

    X_train_df = train[feature_cols].astype(np.float32).copy()
    X_test_df = test[feature_cols].astype(np.float32).copy()

    # ============ Engineered features ============
    print("\n[6/8] Adding engineered features...")
    X_train_df = add_engineered(X_train_df)
    X_test_df = add_engineered(X_test_df)

    eng_cols = ["MISSING_COUNT", "RATING_STRESS_SCORE", "OUTLOOK_X_SCORE", "NEGATIVE_OUTLOOK_X_DROP_FLAG"]
    all_feature_cols = feature_cols + eng_cols
    print(f"[INFO] manifest={len(feature_cols)} + engineered={len(eng_cols)} = {len(all_feature_cols)}")

    X_train = X_train_df.values.astype(np.float32)
    X_test = X_test_df.values.astype(np.float32)
    test_row_ids = test["ROW_ID"].astype(str).values

    del X_train_df, X_test_df, train, test
    gc.collect()
    print(f"[INFO] X_train={X_train.shape} {X_train.dtype} ({X_train.nbytes/1e6:.1f}MB)")
    print(f"[INFO] X_test={X_test.shape} {X_test.dtype} ({X_test.nbytes/1e6:.1f}MB)")

    # ============ CV setup ============
    print("\n[7/8] CV setup...")
    y = y_full
    n_folds = 5
    fold_indices = None
    selected = None  # For debug mode index mapping

    if DEBUG_MODE:
        rng = np.random.RandomState(SEED)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        n_pos = max(1, int(0.1 * len(pos_idx)))
        n_neg = max(1, int(0.1 * len(neg_idx)))
        sel_pos = rng.choice(pos_idx, size=n_pos, replace=False)
        sel_neg = rng.choice(neg_idx, size=n_neg, replace=False)
        selected = np.concatenate([sel_pos, sel_neg])
        rng.shuffle(selected)
        X_train = X_train[selected]
        y = y[selected]
        print(f"[DEBUG] 10% subsample: rows={len(y)} pos={int(y.sum())} prev={y.mean():.4f}")

        tr_idx, va_idx = train_test_split(
            np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED
        )
        fold_indices = [(tr_idx, va_idx)]
        n_folds = 1
    else:
        try:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            fold_indices = list(skf.split(np.zeros(len(y)), y))
            for i, (tr, va) in enumerate(fold_indices):
                pt = int(y[tr].sum()); nt = len(tr) - pt
                pv = int(y[va].sum()); nv = len(va) - pv
                if pt == 0 or nt == 0 or pv == 0 or nv == 0:
                    raise ValueError(f"Fold {i} lacks minority class")
            print(f"[INFO] Stratified 5-fold CV established")
        except Exception as e:
            print(f"[WARN] Stratified fallback: {e}")
            tr_idx, va_idx = train_test_split(
                np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED
            )
            fold_indices = [(tr_idx, va_idx)]
            n_folds = 1

    # Update align_df with fold assignments
    for fi, (tr, va) in enumerate(fold_indices):
        if DEBUG_MODE and selected is not None:
            align_df.iloc[selected[va], align_df.columns.get_loc("fold")] = fi
        else:
            align_df.iloc[va, align_df.columns.get_loc("fold")] = fi

    print(f"[INFO] n_folds={n_folds}")
    for fi, (tr, va) in enumerate(fold_indices):
        pt = int(y[tr].sum()); nt = len(tr) - pt
        scale = nt / max(1, pt)
        print(f"[INFO] Fold {fi}: tr={len(tr)} (pos={pt}, neg={nt}, scale_pos_weight={scale:.4f}), va={len(va)} (pos={int(y[va].sum())})")

    # ============ LightGBM config ============
    if DEBUG_MODE:
        max_rounds = 800; patience = 60; print_every = 25
    else:
        max_rounds = 3000; patience = 120; print_every = 50

    n_threads = os.cpu_count() or 4
    lgb_params = {
        "objective": "binary",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_bin": 255,
        "min_child_samples": 100,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.90,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 2.0,
        "deterministic": True,
        "verbose": -1,
        "force_col_wise": True,
        "seed": SEED,
        "data_random_seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "num_threads": n_threads,
        "is_unbalance": True,
    }
    print(f"[INFO] LGB threads={n_threads}, max_rounds={max_rounds}, patience={patience}")

    # ============ Phase 1: Initial fit + permutation importance ============
    print("\n[8/8] Phase 1: Initial fit + permutation importance...")
    n_feat = X_train.shape[1]
    feat_dec = np.zeros(n_feat, dtype=np.float64)
    init_oof = np.zeros(len(y), dtype=np.float32)

    for fi, (tr_idx, va_idx) in enumerate(fold_indices):
        X_tr, y_tr = X_train[tr_idx], y[tr_idx]
        X_va, y_va = X_train[va_idx], y[va_idx]
        pt = int(y_tr.sum()); nt = len(y_tr) - pt
        scale = nt / max(1, pt)
        lgb_params["scale_pos_weight"] = scale
        print(f"\n  -- Initial Fold {fi+1}/{n_folds} --")
        print(f"  [INFO] tr={len(tr_idx)} (pos={pt}, neg={nt}, scale={scale:.4f}), va={len(va_idx)} (pos={int(y_va.sum())})")

        ds_tr = lgb.Dataset(X_tr, label=y_tr)
        ds_va = lgb.Dataset(X_va, label=y_va, reference=ds_tr)
        cbs = [
            lgb.early_stopping(stopping_rounds=patience, verbose=False, min_delta=1e-5),
            make_log_cb(X_tr, y_tr, X_va, y_va, print_every),
        ]
        bst = lgb.train(
            lgb_params, ds_tr,
            num_boost_round=max_rounds,
            valid_sets=[ds_va], valid_names=["val"],
            feval=lgb_pr_auc, callbacks=cbs,
        )
        bi = bst.best_iteration
        bs = bst.best_score["val"]["pr_auc"]
        print(f"  [INFO] best_iter={bi} best_val_pr_auc={bs:.5f}")
        init_oof[va_idx] = bst.predict(X_va, num_iteration=bi)

        # Permutation importance
        print(f"  [INFO] Running permutation importance on val fold...")
        rng_f = np.random.RandomState(SEED + fi)
        pvi = np.where(y_va == 1)[0]
        nvi = np.where(y_va == 0)[0]

        if len(pvi) < 20:
            sidx = np.arange(len(y_va))
            print(f"  [INFO] Full val used (pos<20): {len(sidx)} samples")
        else:
            nps = min(len(pvi), 2500)
            nns = min(len(nvi), 5000 - nps)
            sp = rng_f.choice(pvi, nps, replace=False)
            sn = rng_f.choice(nvi, nns, replace=False)
            sidx = np.concatenate([sp, sn])
            rng_f.shuffle(sidx)
            print(f"  [INFO] Balanced sample: {len(sidx)} (pos={nps}, neg={nns})")

        Xs = X_va[sidx].copy()
        ys = y_va[sidx]
        bpr = average_precision_score(ys, bst.predict(Xs, num_iteration=bi))
        print(f"  [INFO] Baseline PR-AUC: {bpr:.5f}")

        for fk in range(n_feat):
            Xp = Xs.copy()
            Xp[:, fk] = rng_f.permutation(Xp[:, fk])
            npr = average_precision_score(ys, bst.predict(Xp, num_iteration=bi))
            feat_dec[fk] += (bpr - npr) / n_folds

        del bst, ds_tr, ds_va
        gc.collect()

    # Initial OOF PR-AUC
    if n_folds == 1:
        vai = fold_indices[0][1]
        init_pr = average_precision_score(y[vai], init_oof[vai])
    else:
        init_pr = average_precision_score(y, init_oof)
    print(f"\n  [INFO] Initial OOF PR-AUC: {init_pr:.5f}")

    # ============ Feature selection ============
    print("\n  -- Feature selection --")
    fi_df = pd.DataFrame({
        "feature": all_feature_cols,
        "importance": feat_dec,
        "position": range(len(all_feature_cols)),
    }).sort_values(["importance", "position"], ascending=[False, True]).reset_index(drop=True)

    n_remove = 24  # bottom 10% of 234
    n_keep = max(len(all_feature_cols) - n_remove, 32)
    sel = fi_df.head(n_keep)["feature"].tolist()
    rem = fi_df.iloc[n_keep:]["feature"].tolist()
    print(f"  [INFO] total={len(all_feature_cols)} selected={len(sel)} removed={len(rem)}")
    print(f"  [INFO] Top 20: {fi_df.head(20)['feature'].tolist()}")
    print(f"  [INFO] Bottom 20: {fi_df.tail(20)['feature'].tolist()}")
    print(f"  [INFO] Removed features: {rem}")

    # ============ Phase 2: Final fit with selected features ============
    print("\n[8/8] Phase 2: Final fit with selected features...")
    sel_idx = [all_feature_cols.index(f) for f in sel]
    X_tr_s = X_train[:, sel_idx]
    X_te_s = X_test[:, sel_idx]
    print(f"  [INFO] X_train_sel={X_tr_s.shape}, X_test_sel={X_te_s.shape}")

    final_oof = np.zeros(len(y), dtype=np.float32)
    test_sum = np.zeros(len(X_te_s), dtype=np.float64)
    final_val_scores = []
    final_best_rounds = []

    for fi, (tr_idx, va_idx) in enumerate(fold_indices):
        X_tr, y_tr = X_tr_s[tr_idx], y[tr_idx]
        X_va, y_va = X_tr_s[va_idx], y[va_idx]
        pt = int(y_tr.sum()); nt = len(y_tr) - pt
        scale = nt / max(1, pt)
        lgb_params["scale_pos_weight"] = scale
        print(f"\n  -- Final Fold {fi+1}/{n_folds} --")
        print(f"  [INFO] tr={len(tr_idx)} (pos={pt}, neg={nt}, scale={scale:.4f}), va={len(va_idx)} (pos={int(y_va.sum())})")

        ds_tr = lgb.Dataset(X_tr, label=y_tr)
        ds_va = lgb.Dataset(X_va, label=y_va, reference=ds_tr)
        cbs = [
            lgb.early_stopping(stopping_rounds=patience, verbose=False, min_delta=1e-5),
            make_log_cb(X_tr, y_tr, X_va, y_va, print_every),
        ]
        bst = lgb.train(
            lgb_params, ds_tr,
            num_boost_round=max_rounds,
            valid_sets=[ds_va], valid_names=["val"],
            feval=lgb_pr_auc, callbacks=cbs,
        )
        bi = bst.best_iteration
        bs = bst.best_score["val"]["pr_auc"]
        final_best_rounds.append(bi)
        final_val_scores.append(bs)
        print(f"  [INFO] best_iter={bi} best_val_pr_auc={bs:.5f}")

        final_oof[va_idx] = bst.predict(X_va, num_iteration=bi)
        test_sum += bst.predict(X_te_s, num_iteration=bi)

        del bst, ds_tr, ds_va
        gc.collect()

    test_preds = test_sum / n_folds

    # Final OOF PR-AUC
    if n_folds == 1:
        vai = fold_indices[0][1]
        final_pr = average_precision_score(y[vai], final_oof[vai])
    else:
        final_pr = average_precision_score(y, final_oof)

    print(f"\n  [INFO] Final OOF PR-AUC: {final_pr:.5f}")
    print(f"  [INFO] Test pred range: min={test_preds.min():.6f} max={test_preds.max():.6f} mean={test_preds.mean():.6f}")
    print(f"  [INFO] Final best rounds per fold: {final_best_rounds}")
    print(f"  [INFO] Final val PR-AUC per fold: {final_val_scores}")

    # ============ Save outputs ============
    print("\n[OUTPUT] Saving files...")
    sub = pd.DataFrame({"ROW_ID": test_row_ids, "DOWNGRADE_6M": test_preds})

    # Validate submission per task contract
    assert len(sub) == TEST_N_ROWS, f"Submission rows: {len(sub)} != {TEST_N_ROWS}"
    assert list(sub.columns) == ["ROW_ID", "DOWNGRADE_6M"], f"Bad columns: {list(sub.columns)}"
    assert sub["ROW_ID"].tolist() == test_row_ids.tolist(), "ROW_ID order mismatch"
    assert np.isfinite(sub["DOWNGRADE_6M"].values).all(), "Non-finite predictions"
    assert (sub["DOWNGRADE_6M"].between(0, 1)).all(), "Predictions outside [0, 1]"

    sub.to_csv("submission.csv", index=False)
    print(f"  [OUTPUT] submission.csv: shape={sub.shape}")
    print(f"  [OUTPUT]   cols={list(sub.columns)}")
    print(f"  [OUTPUT]   dtypes={dict(sub.dtypes)}")
    print(f"  [OUTPUT]   sample head:\n{sub.head(3).to_string()}")

    sc = pd.DataFrame(
        [final_pr, final_pr],
        index=pd.Index(["LightGBM_selected", "ensemble"], name="Model"),
        columns=["PR-AUC (Precision-Recall Area Under the Curve)"],
    )
    sc.to_csv("scores.csv")
    print(f"  [OUTPUT] scores.csv:\n{sc.to_string()}")

    print(f"\n[INFO] Total time: {time.time()-t0:.2f}s")
    print("[INFO] Output paths:")
    print("  - submission.csv")
    print("  - scores.csv")


if __name__ == "__main__":
    main()