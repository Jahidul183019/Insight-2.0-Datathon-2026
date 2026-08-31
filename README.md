# Insight 2.0 Datathon 2026

This project predicts cancer-patient `vital_status` from the organizer-provided
tabular data. Valid output labels are `Dead` and `Alive`.

## Final notebook

Submit **`insight_2_0_consolidated.ipynb`** as the single notebook deliverable.
It trains Submission 6 from `train.csv` and `test.csv` and writes the final
`submission.csv` without loading saved test predictions.

Verified output:

- Rows: 36,000
- `Dead`: 30,411
- `Alive`: 5,589
- Public leaderboard: `0.877258`
- CSV SHA-256:
  `fd7cca1ee4a7654757adb78934baf42a07ae264dc581217df3e7863b552ef477`

A clean-kernel audit executed all five code cells without errors. The generated
CSV and probability vector matched the archived Submission 6 references.

## Model summary

Submission 6 combines two feature views:

1. LightGBM, XGBoost, and CatBoost without target encoding.
2. The same model families with fold-safe target encoding.

The workflow uses three repeated five-fold splits, blends the two views, adds
high-confidence model-generated pseudo labels once, retrains the six component
models, and averages the original and pseudo-trained probabilities. A stable
rank rule selects exactly 30,411 `Dead` predictions.

The notebook includes EDA, preprocessing, feature engineering, fixed tuned
parameters, local evaluation, a representative feature-importance plot, an
age-band audit, and limitations.

## Metric reporting

The competition description calls the metric weighted F1 and also identifies
`Dead` as the positive class. To avoid mixing definitions, current local model
comparisons use support-weighted F1 and historical Dead-class F1 values are
labelled explicitly. Kaggle scores are reported only when verified from the
submissions page.

## Submission 10 reconstruction

`insight_2_0_submission10.ipynb` is a complete modelling audit plus exact
prediction reproduction for historical Submission 10. It performs EDA,
feature engineering, full Submission 6 training, five-fold NN training with
new checkpoints, validation, interpretation, and then reconstructs the scored
CSV from:

- `archive/probs_v6_final.npy`
- `archive/probs_nn.npy`
- an 80% tree / 20% neural-network blend
- a stable top-30,411 decision rule

For one-file artifact portability, the two immutable `.npy` references are
compressed and embedded in the notebook. The full run requires the
organizer-provided `train.csv` and `test.csv`; it verifies the original
artifact hashes after decompression.

Running the replay writes `submission.csv` and matches
`archive/submission10.csv` with SHA-256
`333af97cfbc16ffdcc2d9f910000664c443694239c60e67d6504af18687e86f1`.
The recovered project snapshot contains the original NN source and probability
vectors, but not the five original fold checkpoints or an environment lockfile.
Fresh NN retraining therefore does not reproduce the historical probabilities
exactly. The audited full run saved five new checkpoints and differed from the
historical submission on 108 labels; this is reported separately rather than
substituted for the exact historical replay.

## Reproducibility

- The final notebook requires `train.csv`, `test.csv`, NumPy, pandas,
  scikit-learn, LightGBM, XGBoost, CatBoost, and Matplotlib.
- Model parameters, folds, feature engineering, and ensemble weights are fixed
  in the notebook.
- Schema, patient IDs, labels, row count, missing values, and class count are
  checked strictly.
- Historical hashes are diagnostic references. A cross-platform byte mismatch
  produces a warning rather than failing an otherwise valid prediction file.
- `pipeline_v6.py` contains the maintained Submission 6 training workflow.
- `SUBMISSION_HISTORY.md` records scored submissions and supporting evidence.
- `UNSUBMITTED_EXPERIMENTS.md` records rejected local experiments.

## Competition compliance

The final workflow uses manually specified preprocessing, features, models,
folds, and ensemble rules. It does not use an AutoML pipeline generator,
external data, manual test-set labels, or row-level leaderboard probing.
Pseudo labels are generated only from model probabilities. Team members remain
responsible for following organizer guidance and crediting any external sources
used during development.

## Important files

- `insight_2_0_consolidated.ipynb`: final executable Submission 6 notebook
- `insight_2_0_submission10.ipynb`: complete Submission 10 recipe audit and exact historical prediction replay
- `submission.csv`: current Submission 6 output
- `submission6.csv`: verified Submission 6 reference
- `archive/submission6.csv`: archived byte-identical Submission 6
- `archive/submission10.csv`: historical Submission 10
- `archive/submission12.csv`: highest verified Public-LB artifact (`0.877460`)
- `SUBMISSION_HISTORY.md`: scored-submission record
- `diagnostic_outputs/`: validation and subgroup reports

Historical submissions and superseded pipelines remain under `archive/` for
traceability.
