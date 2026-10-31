# Credit-Risk Reliability Audits with RD-Agent

An auditable research project on time-consistent bond credit-risk early warning and the reliability of RD-Agent-generated modeling candidates.

## Project Overview

This repository contains the implementation and research artifacts for a study of time-aware issuer credit-risk modeling and the reliability of RD-Agent-generated modeling candidates. The workflow covers source-level data auditing, point-in-time panel construction, time-consistent labeling, rolling out-of-time evaluation, uncertainty analysis, feature-drift analysis, and structural audits of generated candidates.

## Key Findings

- A controlled LightGBM benchmark achieved validation PR-AUC of 0.2764 and held-out 2023Q1-2025Q3 PR-AUC of 0.0866. The latter was 25.27 times the test event-rate baseline of 0.3429%, illustrating substantial temporal performance decay while retaining signal above baseline.
- The held-out diagnostic analysis used a model frozen before test-label access; it did not use test labels for fitting, feature selection, calibration, or candidate selection.
- In a repeated blind evaluation of RD-Agent's first raw candidates, 3 of 15 passed the predefined structural-compliance gate. This endpoint measures engineering and protocol compliance, not predictive accuracy.

## Reports and Artifacts

- [Final English research report](reports/bond-credit-risk-reliability-report.pdf)
- [Final English presentation](reports/bond-credit-risk-reliability-presentation.pptx)

The report and presentation are frozen public-release artifacts. Their SHA-256 checksums are recorded in [`reports/final-release/release-manifest.json`](reports/final-release/release-manifest.json). Verify them with:

```bash
python scripts/verify_public_release.py
```

## Data and Reproduction Boundary

This repository is published with permission from the internship project supervisor. The underlying study used public-market data, and the project materials are authorized for release on GitHub. No credentials or access instructions for internal systems are included.

The project data are tracked with Git LFS:

- `data/train.csv` contains the labeled development panel, including `DOWNGRADE_6M`.
- `data/test.csv` contains the held-out panel without that target column.
- `data/description.md` records the time protocol and evaluation boundary.
- `data/modeling_feature_manifest.csv` records source-panel feature roles and audit exclusions. It intentionally includes audit-only fields omitted from the released panels; see `data/description.md` for the release-schema boundary.

Column names are in English. Selected categorical values, including source rating-agency names and rating-outlook labels, remain in their original Chinese form to preserve source coding and reproducibility.

After cloning, install Git LFS and retrieve the data with `git lfs pull`. Readers who only need the documentation or source code can clone without downloading LFS objects by setting `GIT_LFS_SKIP_SMUDGE=1` before cloning.

The held-out test labels and test-set prediction files are not publicly released. Accordingly, the reported held-out metrics can be inspected in the retained report and aggregate artifacts, but cannot be recomputed from the released test panel alone. The released panels are research artifacts, not investment advice or a production credit-decision system. Their use remains subject to applicable source-data terms; see [DATA_USE.md](DATA_USE.md).

## Repository Structure

- `sql/`: source-data, data-quality, label-maturity, and point-in-time panel construction SQL
- `src/evaluation/`: out-of-time diagnostics, rolling backtests, uncertainty analysis, and stability figures
- `src/drift/`: preflight checks, population-stability-index analysis, contribution audits, and figure generation
- `src/agent_controls/`: deterministic feature engineering, recipe validation, controlled model execution, and static auditing of generated candidates
- `src/protocol/`: checks for time-consistency and evaluation-protocol compliance
- `protocol/`: recipe schema, feature allowlist, and interface specification used to constrain candidate generation
- `evaluation_contract/`: validation and grading interfaces used in the study
- `agent_generated_candidates/`: RD-Agent-generated candidate code and retained recipes used for auditability
- `reports/`: final research report and presentation
- `reports/final-release/`: frozen-release manifest and Phase 3E result record corresponding to the final report and presentation

## Setup and Verification

```bash
python -m pip install -r requirements.txt
git lfs pull
```

The published runtime dependencies are pinned to versions recorded from the
Python 3.10.20 RD-Agent environment used for this release. The requirements
file intentionally excludes unrelated environment-level packages.

The following entry points operate on repository-relative paths:

```bash
python -m src.evaluation.rolling_backtest
python -m src.evaluation.uncertainty_analysis
python -m src.evaluation.stability_figures
python scripts/verify_public_release.py
```

The scripts under `src/drift/`, `src/protocol/`, and `src/agent_controls/` support the data-quality, drift, protocol, and candidate-audit stages described above. Generated outputs are written to `run_artifacts/`, which is intentionally excluded from version control.

The raw candidate scripts retain their original `workspace_input` contract. Their task description, feature manifest, schema, allowlist, interface specification, and generated recipe artifacts are included for auditability rather than rewritten into a new candidate runner.

## RD-Agent Provenance

The files under `agent_generated_candidates/` were generated by RD-Agent and are not presented as independently authored models. They are included because this project audits their structural compliance and empirical reliability. The upstream RD-Agent codebase is not redistributed; refer to the official Microsoft RD-Agent repository for the framework itself.

## Development and validation

I designed the experiment, reviewed and revised the final implementation and analyses, and validated the reported results. AI assisted with early code scaffolding.

## Limitations

The released data panels support the Python-based modeling, diagnostic, and audit stages, but reproducing the complete source-level SQL workflow requires a compatible database environment. The reported held-out metrics cannot be recomputed from the public test panel because labels and prediction files remain withheld. Results should therefore be interpreted as research artifacts rather than as investment advice or an endorsement by any organization.

## License and Material Use

Project-authored code is available under the [MIT License](LICENSE). The [notice](NOTICE.md) defines the separate treatment of reports, presentations, datasets, and third-party materials.
