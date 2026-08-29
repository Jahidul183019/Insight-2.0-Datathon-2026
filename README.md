# Insight 2.0

This workspace predicts lung-cancer vital status. The official competition
metric is support-weighted F1; `Dead` is the designated positive class. The
canonical upload file is `submission.csv`.

## Metric correction

Earlier local audits used scikit-learn's default binary F1, which measures only
the `Dead` class. That is not the official support-weighted F1 and explains the
apparent roughly 0.05 gap between local scores near 0.928 and Public LB scores
near 0.877. Corrected weighted-F1 results are the primary evidence from this
point forward; legacy binary results are retained only for provenance.

## Current upload candidate

- Candidate: Submission 12
- Recipe: 90% archived Submission 6 probabilities + 10% NN probabilities
- Decision rule: stable top-30,411 ranking
- Output: `submission.csv` (36,000 rows, 30,411 `Dead`)
- Generator: `pipeline_submission12_nn10.py`
- Canonical nested validation: weighted F1 `0.877259`, versus `0.877004`
  for the Submission 6 nested-equivalent reference (`+0.000255`)
- Difference from Submission 6 / 10 / 11: 156 / 136 / 300 labels
- SHA-256: `4e4011c6a70a7a907685fa6a88b33023846529aa0b9beaeeb302c2bad64c3d11`
- Status: Submitted and verified on Public LB (`0.877460` - New Personal Best)

Regenerate and validate the upload file:

```bash
python3 pipeline_submission12_nn10.py
```

The generator requires `test.csv`, `archive/probs_v6_final.npy`,
`archive/probs_nn.npy`, and archived Submissions 6, 10, and 11. It writes
isolated artifacts under `artifacts/submission12_nn10/` and promotes the
validated candidate to `submission.csv` for upload and `archive/submission12.csv`
for archival preservation. Submission 11 remains preserved in `archive/submission11.csv`.

## Reproducibility

- `insight_2_0_consolidated.ipynb` documents the workflow and validates the
  frozen Submission 10 and verified Submission 11 artifacts.
- `pseudo_label_nested_validation.py` performs the five-fold outer-holdout gate
  comparing 90% pseudo-labeling with a rebuilt 95% control.
- `pipeline_v6.py` contains the original tree/pseudo-label training workflow.
  It is expensive and now writes only under `artifacts/v6_rerun/`; it cannot
  overwrite the canonical root submission.
- `pipeline_nn.py` contains the neural-network training workflow.
- `pipeline_submission12_nn10.py` deterministically reconstructs the prepared
  Submission 12 upload candidate from frozen probabilities.
- `step1_cv_diagnostic.py` regenerates the cached tree OOF diagnostics.
- `validation_harness.py` runs the saved-OOF 40/60 stability audit and writes
  results under `diagnostic_outputs/validation_harness/`.
- `submission6_nested_reconstruction.py` is the resumable, leakage-safe
  five-fold reconstruction of the Submission 6 recipe. It writes checkpoints,
  OOF predictions, metrics, and replay diagnostics under
  `diagnostic_outputs/submission6_nested/`; it never writes a submission.
- `submission6_practical_reference_gate.py` independently validates the saved
  reconstruction and writes its non-training acceptance audit under
  `diagnostic_outputs/submission6_reference_gate/`.
- `age55_subgroup_investigation.py` reruns the age-55–59 diagnostic against the
  approved nested reference and writes only under
  `diagnostic_outputs/age55_investigation/`.
- `SUBMISSION_HISTORY.md` maps scored submissions to their canonical files.

The current notebook is artifact-reproducible: it verifies frozen probability
vectors and submissions byte-for-byte. A fresh v6 training run can vary slightly
and must not be described as an exact replay unless its output matches the frozen
Submission 6 artifact. It does not yet demonstrate a clean end-to-end retraining
of every historical submitted model; that regeneration path must be completed
and verified before the final competition notebook is handed in.

## Competition compliance

- The model workflow is manually specified; no AutoML library is used.
- The rules also prohibit "automated pipeline-generation systems." Because
  AI-assisted code may fall within that wording, obtain written confirmation
  from the organizers before relying on AI-authored modelling code in the
  submitted notebook; do not assume that "not AutoML" resolves this ambiguity.
- Do not infer, probe, or manually assign hidden test labels from leaderboard
  feedback.
- Keep restricted competition data, artifacts, and solution code within the
  official team during the competition.
- The final notebook must reproduce the submitted workflow and outputs; frozen
  artifact validation alone is not a substitute for that requirement.

## Important artifacts

- `submission.csv`: current verified highest Public LB upload file (`0.877460`)
- `archive/submission12.csv`: byte-identical archived snapshot of Submission 12
- `artifacts/submission12_nn10/submission12_nn10.csv`: byte-identical generated
  Submission 12 candidate
- `artifacts/submission12_nn10/validation_summary.json`: Submission 12 format,
  provenance, hash, and disagreement audit
- `submission6.csv`: verified baseline public-LB submission (`0.877258`)
- `archive/submission10.csv`: byte-identical archived Submission 10 snapshot
- `artifacts/pseudo90/submission11_pseudo90.csv`: byte-identical generated
  Submission 11 candidate
- `archive/submission11.csv`: byte-identical archived Submission 11 upload
- `artifacts/pseudo90/probs_pseudo90.npy`: Submission 11 probabilities
- `archive/probs_v6_final.npy`: exact probability source for Submission 6
- `archive/probs_v6_blend.npy`: pre-pseudo v6 teacher probabilities
- `archive/probs_nn.npy`: NN test probabilities used by Submission 10
- `oof_step1.npy`, `archive/oof_nn.npy`, `y_step1.npy`: local validation inputs
- `diagnostic_outputs/submission6_nested/nested_oof_predictions.npz`: completed
  Submission 6 nested-recipe OOF vector
- `diagnostic_outputs/submission6_reference_gate/submission6_practical_reference_acceptance.json`:
  practical-vs-strict canonical-reference decision
- `diagnostic_outputs/age55_investigation/age55_investigation_summary.json` and
  `age55_investigation_report.md`: canonical age-55 follow-up and Step 3 stop
  decision

Historical pipelines and submissions are retained in `archive/` for provenance.
