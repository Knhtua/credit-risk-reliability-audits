# Bond Issuer Credit Rating Downgrade Early-Warning

## Task Description

Build a binary classifier that predicts whether a bond issuer will experience at least one long-term issuer credit-rating downgrade during the six months after each quarter-end snapshot.

Each row represents one issuer at one quarter-end. The feature table was built point-in-time: a financial statement is eligible only when its actual announcement date and report date are no later than the snapshot date.

## Target

`DOWNGRADE_6M` is 1 when a downgrade occurs in the next six months and 0 otherwise.

## Files

- `train.csv`: TRAIN and VALID periods, with labels.
- `test.csv`: chronologically later TEST period, without labels.
- `modeling_feature_manifest.csv`: source-panel column roles and missingness.

`modeling_feature_manifest.csv` is retained as the source-level contract used
by the raw RD-Agent candidates and the drift audit. It includes audit-only
fields that are deliberately omitted from the released panels; `ROW_ID` is a
release-only identifier. It is therefore not an exact header specification for
`train.csv` or `test.csv`.

## Mandatory Time Protocol

Do not randomly mix TRAIN and VALID rows.

- Rows whose `SPLIT` is `TRAIN`: 2013-06-30 through 2020-12-31. Fit preprocessing and models only on these rows.
- Rows whose `SPLIT` is `VALID`: 2021-03-31 through 2022-12-31. Use only for model selection, threshold selection and calibration.
- `test.csv`: 2023-03-31 through 2025-09-30.

`SPLIT` and `ROW_ID` must never be used as predictive features. `SNAPSHOT_YEAR` should also be excluded to prevent learning the secular decline in event frequency. Any imputation, scaling, encoding, feature selection, sampling and calibration must be fitted on TRAIN only.

## Evaluation

Primary metric: area under the precision-recall curve (average precision / PR-AUC).

Also report ROC-AUC, Brier score, and precision/recall among the highest-risk 1% and 5% of rows. Accuracy is not an appropriate main metric because the positive class is below 1%.

## Baseline

A class-weighted logistic regression achieved:

- VALID PR-AUC 0.1079 and ROC-AUC 0.8792.
- TEST PR-AUC 0.0527 and ROC-AUC 0.8685.
- TEST top-1% recall 26.97% with 9.24% precision.

Proposed experiments must be compared with this baseline under the same chronological split.

## Submission Format

Create `submission.csv` with exactly two columns:

```text
ROW_ID,DOWNGRADE_6M
20230331_example,0.0123
```

`DOWNGRADE_6M` must contain probabilities between 0 and 1, not hard class labels.
