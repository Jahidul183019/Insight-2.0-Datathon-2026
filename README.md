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

## Final reproducible deliverable

- Final notebook model: Submission 6
- Recipe: manually specified v4/v5 LightGBM, XGBoost, and CatBoost blend,
  followed by one-pass pseudo-label augmentation
- Output: `submission.csv` (36,000 rows, 30,411 `Dead`)
- Public LB score: `0.877258`
- SHA-256: `fd7cca1ee4a7654757adb78934baf42a07ae264dc581217df3e7863b552ef477`
- Status: regenerated end-to-end and byte-identical to `archive/submission6.csv`

Open `insight_2_0_consolidated.ipynb` and run all cells from a clean kernel.
The notebook reads only `train.csv` and `test.csv` to train the final model and
promotes the fully validated result to `submission.csv`.

Submission 12 remains the highest Public-LB artifact (`0.877460`) and is safely
preserved at `archive/submission12.csv`. It is not presented as the final
reproducible model because the original per-fold neural-network checkpoints
were not saved; fresh NN retraining is hardware-sensitive and does not
reproduce that CSV exactly.

The planned Kaggle final pair is Submission 10 plus Submission 6, while the
notebook demonstrates Submission 6 only. Submission 10 is supported by
saved-OOF screening evidence but is not claimed as a clean-runtime retrain.
Written organizer acceptance of this notebook/selection asymmetry remains an
open compliance item until a response is retained.

## Reproducibility

- `insight_2_0_consolidated.ipynb` embeds and executes the complete Submission 6
  workflow without loading cached model probabilities.
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

The notebook has been executed from top to bottom. Its fresh probability vector
and submission match the archived Submission 6 artifacts byte-for-byte. The
saved notebook includes execution counts, EDA tables and plots, training logs,
and the final SHA-256 verification.

Exact probability and CSV hashes are reported as diagnostics rather than hard
cross-platform gates. Machine-independent checks for schema, patient IDs,
probability validity, labels, row count, and class count remain strict. If a
different supported environment introduces floating-point drift, the notebook
warns and reports historical label disagreement instead of crashing solely on
a byte mismatch.

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

- `submission.csv`: final reproducible Submission 6 output (`0.877258`)
- `artifacts/final_notebook/submission.csv`: byte-identical notebook output
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
- `archive/oof_nn.npy` and the versioned diagnostic outputs: historical local
  validation inputs; they are not required by the final notebook
- `diagnostic_outputs/submission6_nested/nested_oof_predictions.npz`: completed
  Submission 6 nested-recipe OOF vector
- `diagnostic_outputs/submission6_reference_gate/submission6_practical_reference_acceptance.json`:
  practical-vs-strict canonical-reference decision
- `diagnostic_outputs/age55_investigation/age55_investigation_summary.json` and
  `age55_investigation_report.md`: canonical age-55 follow-up and Step 3 stop
  decision

Historical pipelines and submissions are retained in `archive/` for provenance.
