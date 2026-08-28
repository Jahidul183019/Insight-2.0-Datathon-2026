# Insight 2.0

This workspace predicts lung-cancer vital status (`Dead` is the positive class,
scored with F1). The canonical upload file is `submission.csv`.

## Current candidate

- Recipe: 80% frozen Submission 6 probabilities + 20% neural-network probabilities
- Decision rule: stable top-30,411 ranking
- Output: `submission.csv` (36,000 rows, 30,411 `Dead`)
- Generator: `pipeline_mega_ensemble.py`
- Status: prepared locally; leaderboard score pending

Regenerate and validate the upload file:

```bash
python3 pipeline_mega_ensemble.py
```

The generator requires `test.csv`, `submission6.csv`,
`archive/probs_v6_final.npy`, and `archive/probs_nn.npy`. It validates shapes,
IDs, labels, and finite probabilities before writing the output.

## Reproducibility

- `insight_2_0_consolidated.ipynb` documents the workflow and validates the
  frozen and pending artifacts.
- `pipeline_v6.py` contains the original tree/pseudo-label training workflow.
  It is expensive and now writes only under `artifacts/v6_rerun/`; it cannot
  overwrite the canonical root submission.
- `pipeline_nn.py` contains the neural-network training workflow.
- `step1_cv_diagnostic.py` regenerates the cached tree OOF diagnostics.
- `validation_harness.py` runs the saved-OOF 40/60 stability audit and writes
  results under `diagnostic_outputs/validation_harness/`.
- `SUBMISSION_HISTORY.md` maps scored submissions to their canonical files.

The current notebook is artifact-reproducible: it verifies frozen probability
vectors and submissions byte-for-byte. A fresh v6 training run can vary slightly
and must not be described as an exact replay unless its output matches the frozen
Submission 6 artifact.

## Important artifacts

- `submission6.csv`: highest verified public-LB submission (`0.877258`)
- `submission.csv`: pending upload candidate
- `archive/submission10_candidate.csv`: frozen copy of the pending candidate
- `archive/probs_v6_final.npy`: exact probability source for Submission 6
- `archive/probs_v6_blend.npy`: pre-pseudo v6 teacher probabilities
- `archive/probs_nn.npy`: NN test probabilities used by the pending blend
- `oof_step1.npy`, `archive/oof_nn.npy`, `y_step1.npy`: local validation inputs

Historical pipelines and submissions are retained in `archive/` for provenance.
