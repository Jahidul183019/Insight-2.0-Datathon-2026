# Insight 2.0 Datathon - Unsubmitted Experiments & Diagnostics

This document stores records of experimental candidates and diagnostics that were evaluated locally but did not proceed to a Kaggle submission. They are kept here for future reference and enhancement.

## tree80_nn20 Experiment (Gate Failed; Not Submitted)
- **Candidate:** 80% pseudo90 probabilities plus 20% neural-network
  probabilities, followed by the same fixed 84.5%-rate decision policy.
- **Weighted OOF F1:** `0.876155`, versus `0.874797` for pseudo90 alone
  (`+0.001358`).
- **Uncertainty:** The paired bootstrap 95% interval for the improvement was
  `[-0.000255, +0.002631]`, which includes zero.
- **Fold Gate:** Improved in only 3 of 5 outer folds, below the required 4 of
  5.
- **Simulated-Private Gate:** Best in 0 of 50 simulated 40/60 splits across the
  full candidate set; `tree80_nn20` was best in all 50. This is a saved-OOF
  stability screen, not a reconstruction of Kaggle's hidden public/private
  split.
- **Subgroup Gate:** Weighted F1 for the already weak age-55–59 slice regressed
  by `-0.0023` versus pseudo90.
- **Submission 6 Gate (at evaluation time):** A leakage-safe nested-equivalent
  reconstruction of the actual Submission 6 training recipe was then
  unavailable, so the required direct comparison could not be passed. That
  infrastructure was completed later and is documented below.
- **Decision:** **No submission.** No CSV was generated. The candidate failed local evaluation gates.

## Submission 6 Nested Reference Reconstruction (Completed; No Submission)
- **Purpose:** Build the missing leakage-safe reference for the actual
  Submission 6 recipe so future candidates can be compared with the verified
  best workflow rather than only with pseudo95, pseudo90, or frozen-tree
  proxies.
- **Recipe:** Five outer folds. Within every outer-training partition, the
  teacher reproduces the 55% v5 / 45% v4 blend; the v4 target encodings exclude
  the outer-validation rows; test pseudo-labels are selected at ≥95% or ≤5%;
  the six-model student is refit with those pseudo-labels; and the final OOF
  probability is 50% teacher plus 50% student.
- **Completed Work:** All five folds, exactly 90 teacher and 30 student model
  fits per fold (`600` nested fits total), with every one of the 24,000 labelled
  rows predicted exactly once.
- **Official Weighted F1 at Fixed 84.5% Rate:** `0.877004`.
- **Fold Weighted F1:** `0.876919`, `0.879890`, `0.877344`, `0.874797`,
  `0.874797`.
- **ROC AUC:** `0.900666`.
- **Scale Check:** `-0.000254` versus Submission 6's Public LB `0.877258`,
  passing the predeclared absolute-gap limit of `0.005`.
- **Production Replay:** Recovered exactly 17,535 pseudo-label rows (17,234
  Dead and 301 Alive), threshold `0.684`, exactly 30,411 Dead predictions, and
  zero hard-label changes versus archived Submission 6. Probability drift was
  approximately `1e-9`, so strict probability/artifact equivalence remains
  false rather than being relabelled.
- **Canonical Status:** The independent practical gate passed `21/21` checks
  with no failures or pending checks and approved
  `submission6_nested_recipe_reference` as the canonical nested-equivalent
  comparison vector. The original `global_metrics.csv` strict-artifact field
  remains false by design; the authority for practical canonical approval is
  `diagnostic_outputs/submission6_reference_gate/submission6_practical_reference_acceptance.json`.
- **Harness Status:** `validation_harness.py` now loads this vector only after
  verifying its approval, SHA-256, run signature, labels, fold coverage, and
  finiteness. It is therefore available as the Submission 6 baseline for
  future candidate gates.
- **Decision:** Infrastructure accepted. This reconstruction generated no test
  candidate and no submission CSV.

## Age 55–59 Canonical Follow-up (No Targeted Fix; No Submission)
- **Reference:** The approved Submission 6 nested-equivalent OOF vector, not the
  earlier frozen-tree proxy.
- **Subgroup Result:** 1,821 rows; weighted F1 `0.838062` versus `0.877004`
  globally and `0.880201` outside the band. ROC AUC is `0.862474`; the band is
  below both adjacent age bands in only 3 of 5 canonical outer folds for both
  weighted F1 and AUC.
- **Class Balance:** Dead prevalence is `79.24%`, versus `83.31%` elsewhere;
  the global fixed-rate policy predicts `82.04%` Dead in the band, producing
  169 false positives and 118 false negatives.
- **Targeted Replication:** The EOD-all-blank × adenocarcinoma-family and raw
  histology-8140 cells retain BH-significant **total-error** associations
  (`q=0.006233` and `q=0.006456`), but total error is confounded by their
  different Dead prevalence.
- **Actionable Error Test:** No cluster passes the class-conditional
  denominator, effect-size, and BH gates. For the family cell, the excess FPR
  and FNR are only `+0.0117` and `+0.0331`; for raw 8140 they are `+0.0079` and
  `+0.0265`, all below the predeclared five-percentage-point effect gate.
- **Other Causes Checked:** The age band generally has less—not more—semantic
  unavailability; no material categorical association was found; and v6
  already contains categorical age, numeric age midpoint, and age interactions.
- **Step 2 Verdict:** The weak subgroup replicates, but the proposed actionable
  EOD-unavailable × adenocarcinoma error mechanism does not. It is therefore
  treated as a prevalence/case-mix association rather than evidence for a new
  interaction feature.
- **Step 3 Decision:** Not entered. No targeted feature was trained because its
  prerequisite failed; consequently no candidate or submission CSV was
  generated.
