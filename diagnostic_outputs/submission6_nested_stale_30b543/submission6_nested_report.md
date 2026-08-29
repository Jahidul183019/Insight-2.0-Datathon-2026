# Submission 6 Nested-Equivalent Reconstruction

**Status:** IN PROGRESS (0/5 outer folds complete)

No completed-fold score is available yet.

## Design

- Official metric: support-weighted F1 across Alive and Dead.
- Decision policy: deterministic top 84.5% within each reported population.
- Outer validation: 5-fold stratified, shuffled, seed 42.
- Teacher: production v5/v4 families, 3x5 early-stopped fold average,
  55% v5 + 45% v4.
- Pseudo labels: competition-test teacher scores >=0.95 or <=0.05; outer
  validation is never used.
- Student: production six-model family, five folds, seed 1041, early stopping
  80; final prediction is 50% teacher + 50% student.
- Completed outer folds: none.
- Mean pseudo-label count across completed folds: nan.

## Interpretation boundary

This is a nested-equivalent validation reference, not a byte-for-byte replay of
the submitted full-data model.  Each fold necessarily trains on 80% of labelled
rows, and its encoder/target encodings exclude the outer validation fold.  See
`recipe_equivalence_audit.csv` for the component-by-component audit.

The vector is approved as the canonical future comparison only after all five
folds finish, the predeclared absolute weighted-F1 gap to the public LB is at
most 0.005, and an opt-in full-data production replay
verifies the archived probability artifacts.  Until then,
`canonical_reference_approved` is false in `global_metrics.csv`.

No test submission and no final candidate test probability vector were created.
Model-level checkpoints under `checkpoints/` are resumable and are bound to the
input/recipe hash in `run_manifest.json`.
