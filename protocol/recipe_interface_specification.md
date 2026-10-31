# Phase 3C RD-Agent Recipe Interface v1.0

## Purpose

RD-Agent proposes a bounded machine-readable recipe. The recipe is not executable candidate code and does not directly produce final validation predictions.

## Required output

`agent_model_recipe.json` encoded as UTF-8 JSON.

## Base model

- Exactly `lightgbm_binary`.
- Seed is exactly `42`.
- One final fitted model.
- No fold-model averaging, stacking, or calibration.

## Engineered features

- At most 20 row-wise engineered features.
- Inputs must come from the locked 230-feature controller allowlist.
- `ROW_ID` and `DOWNGRADE_6M` are forbidden.
- An engineered feature cannot depend on another engineered feature.
- Operators: sum, difference, product, ratio_safe, max_zero_negative, count_zero.
- Only ratio_safe may contain epsilon; epsilon must be between 1e-12 and 0.1.

## Class imbalance

- `class_weight_mode` is either `none` or `train_ratio`.
- RD-Agent must not provide is_unbalance, scale_pos_weight, or class_weight parameters.
- When train_ratio is selected, the controller computes exactly one scale_pos_weight from the relevant TRAIN labels.

## Attempt selection

- The first structurally valid recipe is locked.
- A second attempt is allowed only after a technical or structural failure, never because of model score.

## Data boundary

- RD-Agent sees only Phase 3 TRAIN data and a TRAIN-derived compatibility sample.
- VALID and PSEUDO-TEST features and labels remain invisible during recipe generation.
