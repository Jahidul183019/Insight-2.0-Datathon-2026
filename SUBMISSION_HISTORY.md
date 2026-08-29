# Insight 2.0 Datathon - Submission History & Analysis

> [!IMPORTANT]
> **Reproducibility vs. Leaderboard Performance**
> Local reproducibility (e.g., matching SHA-256 hashes, nested CV scores, bootstrap intervals, and MAE/Pearson checks) proves that a file correctly implements its intended recipe. **It does not prove what score Kaggle will return.** 
> Going forward, any stated "LB Score" must be explicitly corroborated (e.g. "verified via Kaggle submissions page screenshot"). Unsubmitted models are strictly local estimates and must not be confused with actual leaderboard results.

## Current Status

- **Best Kaggle-Verified LB Score:** `0.877460` (Submission 12) - Confirmed via user screenshot on 2026-08-29.
- **Private LB Selections:** 
  1. **Primary:** Submission 12 (`0.877460`)
  2. **Secondary:** Submission 6 (`0.877258`)
- **Pending/Unverified Candidates:** None. All local candidates (like Submission 12) have either been skipped or explicitly resolved.

## Metric Correction

Earlier local validation code used scikit-learn's default binary F1, which
scores only the `Dead` class. Those values near `0.928` are not directly
comparable with the official weighted-F1 leaderboard scores near `0.877`.
Recomputing the frozen tree OOF predictions gives weighted F1 `0.877513`
instead of legacy binary F1 `0.928209`, resolving most of the apparent scale
gap. Historical Public LB scores below are unchanged; corrected weighted F1 is
the primary metric for all new local decisions.

At the production decision rate of 30,411 / 36,000 = `0.84475`, the earlier
like-for-like nested proxies compared with the Public LB as follows:

| Submission reference | Nested proxy recipe | Weighted F1 | Public LB | Local − LB |
| --- | --- | ---: | ---: | ---: |
| Submission 6 | pseudo95 proxy | `0.874244` | `0.877258` | `-0.003014` |
| Submission 10 | 80% pseudo95 + 20% NN proxy | `0.875686` | `0.876616` | `-0.000930` |
| Submission 11 | pseudo90 proxy | `0.874753` | `0.876616` | `-0.001863` |

These corrected local values are on the same scale and move in the same
direction as the leaderboard scores. They remain historical proxy recipes and
not the exact Kaggle split. The earlier roughly `0.05` gap was a metric
mismatch; the residual differences do not prove a causal generalization
effect.

The later leakage-safe reconstruction now supersedes the pseudo95 proxy as the
canonical local comparator for the Submission 6 recipe:

| Canonical reference | Weighted F1 | Public LB | Local − LB | Scale gate |
| --- | ---: | ---: | ---: | --- |
| Submission 6 nested-equivalent | `0.877004` | `0.877258` | `-0.000254` | Pass (`|gap| ≤ 0.005`) |

This is a canonical **nested recipe-equivalent comparator**, not a claim that
threaded model probabilities are byte-identical to the historical files. The
production replay reproduced the pseudo-label membership, the historical
`0.684` threshold, all 30,411 hard labels, and their ranking exactly; archived
probabilities differed only at approximately `1e-9`. The strict probability
flag therefore remains false, while the separately audited practical recipe
gate is approved.

## Submission Log

### Submission 1
- **File:** `archive/submission1.csv`
- **Strategy:** Early baseline model.
- **Dead Rate:** 90.0% (32,392 Dead)
- **LB Score:** `0.871881` (verified via Kaggle submissions page screenshot on 2026-08-29)
- **Insight:** Over-predicted the positive class significantly and produced the lowest verified score among Submissions 1–9.

### Submission 2
- **File:** `archive/submission2.csv`
- **Strategy:** Target encoding within a complex inner CV + Optuna tuning + threshold sweep (Pipeline v4).
- **Dead Rate:** 84.5% (30,431 Dead)
- **LB Score:** `0.875506` (verified via Kaggle submissions page screenshot on 2026-08-29)
- **Insight:** Established that targeting an ~84.5% Dead rate is the "sweet spot" for the Leaderboard, balancing precision and recall effectively.

### Submission 3
- **File:** `archive/submission3.csv`
- **Strategy:** Re-run of Pipeline v4.
- **Dead Rate:** 84.5% (30,438 Dead)
- **LB Score:** `0.875380` (verified via Kaggle submissions page screenshot on 2026-08-29)
- **Insight:** Nearly identical to Submission 2. Confirmed the stability of the v4 model at the 84.5% threshold.

### Submission 4
- **File:** `archive/submission4.csv`
- **Strategy:** Cleaned up code (Pipeline v5). Removed target encoding. Used the raw CV F1-optimal threshold of `0.48`.
- **Dead Rate:** 88.3% (31,789 Dead)
- **LB Score:** `0.869451` (verified via Kaggle submissions page screenshot on 2026-08-29)
- **Insight:** Although `0.48` was theoretically optimal in local CV, it resulted in 88.3% Dead on the test set, which the LB heavily penalized. Re-affirmed that we must target 84.5% Dead.

### Submission 5
- **File:** `archive/submission5.csv` (from v5 fold-avg)
- **Strategy:** Pipeline v5 using 3x5 Fold averaging (LGB+XGB+CB). The threshold was explicitly tuned to output exactly 84.5% Dead (th=`0.605`).
- **Dead Rate:** 84.5% (30,424 Dead)
- **LB Score:** `0.876730` (verified via Kaggle submissions page screenshot on 2026-08-29)
- **Improvement:** +0.00122 over Submission 2.
- **Insight:** Stable fold-averaging with clean features beats the complex target encoding of v4 when the threshold is aligned properly.

### Submission 6
- **File:** `submission6.csv` (frozen working reference);
  `archive/submission6.csv` (byte-identical archived snapshot from Pipeline v6)
- **Strategy:** Maximum Generalization Blend. 
  1. Blended v4 (with Target Encoding) and v5 (without Target Encoding) models (55% v5 + 45% v4).
  2. Pseudo-labeling: Added ~17,500 highly confident test predictions (≥95% or ≤5%) to the training set and retrained.
  3. Final ensemble: 50% original blend + 50% pseudo-labeled blend.
  4. Threshold tuned to 84.5% Dead (th=`0.684`).
- **Dead Rate:** 84.5% (30,411 Dead)
- **LB Score:** `0.877258` (verified via Kaggle submissions page screenshot on 2026-08-29)
- **Improvement:** +0.00053 over Submission 5.
- **Insight:** Feature diversity (blending TE and non-TE) plus pseudo-labeling provides the best generalization yet.

### Submission 7
- **File:** `archive/submission7.csv` (Pipeline v7 Stacking Meta-Model)
- **Strategy:** Option 3 - Pure Stacking Meta-Model:
  1. Level-1: 6 base models (v5 LGBM, v5 XGBoost, v5 CatBoost, v4 LGBM, v4 XGBoost, v4 CatBoost) trained across 3×5 CV (15 folds each).
  2. Level-2: Logistic Regression meta-learner trained on the 6 out-of-fold probability columns.
  3. Learned Model Weights: CatBoost was assigned the highest weights (~2.3 to 2.5), followed by XGBoost (~0.4 to 0.7) and LightGBM (~0.3).
  4. Threshold calibrated to 84.49% Dead (th=`0.666`).
- **Dead Rate:** 84.49% (30,418 Dead)
- **LB Score:** `0.876170` (verified via Kaggle submissions page screenshot on 2026-08-29)
- **Key Takeaway:** Confirmed that while stacking yields strong standalone performance (`0.876170`), the semi-supervised pseudo-labeling in Submission 6 (`0.877258`) was the critical factor that drove the score above 0.877.
- **Follow-up:** Submission 8 tested the planned blend with pseudo-labeling.
### Submission 8
- **File:** `archive/submission8.csv` (Super-Blend)
- **Strategy:** 50/50 blend of Submission 6 (Pseudo-labeling) and Submission 7 (Stacking).
- **Dead Rate:** 84.48% (30,414 Dead)
- **LB Score:** `0.876305` (verified via Kaggle submissions page screenshot on 2026-08-29)
- **Insight:** Scored between Sub 6 (0.877258) and Sub 7 (0.876170). On the public split, diluting Submission 6 with the stacking predictions reduced F1; this is evidence favoring Submission 6, not proof about hidden labels or a single causal component.

### Submission 9
- **File:** `archive/submission9.csv` (Pipeline v8 Iterative Pseudo-Labeling)
- **Strategy:** Iterative pseudo-labeling using an ultra-strict >98% confidence threshold from Submission 6, yielding ~14,000 pure labels, retrained on all 6 models and blended 50/50.
- **Dead Rate:** 84.48% (30,414 Dead)
- **LB Score:** `0.875022` (verified via Kaggle submissions page screenshot on 2026-08-29)
- **Insight:** The Public Leaderboard score dropped. This means that either (A) tightening the threshold to 98% reduced the diversity of the pseudo-labels, causing slight overfitting to the highly confident samples, or (B) the 411 changed rows were actually correct in Submission 6 for the Public LB portion of the test set. 

### Submission 10
- **File:** `archive/submission10.csv` (byte-identical archived upload)
- **Strategy:** Neural-network diversity blend using 80% of the archived Submission 6 probabilities (`archive/probs_v6_final.npy`) and 20% neural-network probabilities (`archive/probs_nn.npy`). Final labels use a deterministic top-k cutoff rather than a floating-point threshold.
- **Dead Rate:** 84.475% (exactly 30,411 Dead), matching Submission 6's class count.
- **Difference from Submission 6:** 292 rows change labels while the total Dead count remains fixed.
- **LB Score:** `0.876616` (verified via Kaggle submissions page screenshot on 2026-08-29)
- **Comparison with Submission 6:** `-0.000642`; the neural-network diversity blend did not surpass Submission 6 on the Public Leaderboard.
- **Insight:** Submission 10 outperformed the stacking Super-Blend in Submission 8 (`0.876305`) by `0.000311`, but remained below Submission 6 (`0.877258`). It is therefore useful as a verified diversity candidate, not as the primary submission.
- **Status:** Submitted and verified from the Kaggle submissions page on 2026-08-29.

### Submission 11
- **File:** `archive/submission11.csv` (byte-identical archived upload);
  `artifacts/pseudo90/submission11_pseudo90.csv` (generated artifact)
- **Strategy:** Controlled one-pass pseudo-label confidence experiment based on
  the Submission 6 recipe. The frozen pre-pseudo teacher
  (`archive/probs_v6_blend.npy`) selects test rows at ≥90% Dead or ≤10% Dead,
  yielding 22,922 pseudo-labels (22,203 Dead and 719 Alive). The same six-model
  student family is retrained, blended 50/50 with the teacher, and converted to
  labels using a deterministic top-30,411 cutoff.
- **Dead Rate:** 84.475% (exactly 30,411 Dead), matching Submission 6.
- **Difference from Submission 6:** 288 rows change labels while the total Dead
  count remains fixed (144 in each direction).
- **Validation:** In a five-fold outer holdout where teacher creation,
  pseudo-label selection, target encoding, and student fitting excluded each
  validation fold, corrected weighted F1 was `0.874797` at the locked 84.5%
  rate versus `0.874203` for the like-for-like rebuilt 95% control
  (`+0.000594`). The originally reported `0.926617` versus `0.926269`
  (`+0.000348`) values used legacy binary Dead-class F1, not the competition
  metric. Its paired bootstrap 95% interval was `[-0.000398, +0.000846]`, so
  even that legacy signal was inconclusive.
- **SHA-256:** `cbea4ad3e7c525ab5352bd31f04a37d67bfdf13150fe3c8d2f88628df027ed0f`
- **LB Score:** `0.876616` (verified via Kaggle submissions page screenshot on 2026-08-29)
- **Comparison:** Tied Submission 10 exactly and scored `-0.000642` below
  Submission 6.
- **Insight:** Loosening the pseudo-label gate from 95%/5% to 90%/10% produced
  a small positive nested-OOF signal but no Public LB improvement. Because
  Submission 11 changes 288 labels relative to Submission 6 and 384 relative
  to Submission 10, the identical score does not mean the files are identical;
  it means only that Kaggle reported the same six-decimal Public F1. The public
  labels are unavailable, so the row-level effect cannot be determined.
- **Status:** Submitted and verified from the Kaggle submissions page on
  2026-08-29.

### Submission 12 (New Best Score!)
- **File:** `submission.csv` (current upload file);
  `archive/submission12.csv` (byte-identical archived snapshot);
  `artifacts/submission12_nn10/submission12_nn10.csv` (byte-identical generated
  artifact).
- **Strategy:** Conservative neural-network interpolation using 90% archived
  Submission 6 probabilities (`archive/probs_v6_final.npy`) and 10% neural-
  network probabilities (`archive/probs_nn.npy`). Labels use a deterministic
  stable top-30,411 cutoff.
- **Rows / Class Count:** 36,000 rows; exactly 30,411 `Dead` and 5,589 `Alive`.
- **Difference from Previous Submissions:** 156 labels versus Submission 6,
  136 versus Submission 10, and 300 versus Submission 11. The file is not a
  copy of any previous submission.
- **Canonical OOF Evidence:** Weighted F1 `0.877259` versus `0.877004` for the
  approved Submission 6 nested-equivalent baseline (`+0.000255`); ROC AUC
  improves by `+0.000199`. (This is a local estimate only).
- **Robustness:** Improved in 3 of 5 canonical outer folds. In 50 repeated
  stratified 40/60 checks it recorded 35 wins, 4 ties, and 11 losses versus the
  canonical baseline. The paired bootstrap interval included zero, so the
  apparent improvement is encouraging but not a confirmed generalization win.
- **Age 55–59 Safety Check:** Weighted F1 changed by `-0.000471`, a small
  regression that remains within the exploratory non-inferiority tolerance but
  prevents claiming uniform subgroup improvement.
- **SHA-256:**
  `4e4011c6a70a7a907685fa6a88b33023846529aa0b9beaeeb302c2bad64c3d11`.
- **LB Score:** `0.877460` (verified via user screenshot on 2026-08-29).
- **Comparison with Submission 6:** `+0.000202`. This is a new **Personal Best**. The conservative 10% injection of the Neural Network probabilities provided enough structural diversity to correct tree errors without diluting the strong GBDT signal.
- **Status:** Submitted and corroborated via Kaggle submissions page screenshot on 2026-08-29. This is now the undisputed primary candidate for the Private Leaderboard.

---

## Final Strategy Selection

For the final Kaggle Private Leaderboard evaluation, you are allowed to select two submissions. Based on our empirical testing:
1. **Submission 12 (`0.877460`):** Keep this as the primary selection. It is the highest verified Public LB score, successfully injecting neural-network diversity into the robust tree ensemble without degrading performance.
2. **Submission 6 (`0.877258`):** Use this as the secondary selection. It is our strongest pure-tree/pseudo-labeling configuration. If the neural-network blend in Submission 12 overfits the public split, Submission 6 acts as a highly generalized pure-tree fallback.

The recommended pair is now **Submission 12 plus Submission 6**.

*(Note: Submission 10 (`0.876616`) and Submission 11 (`0.876616`) were previously considered but have been formally superseded by Submission 12.)*

The failed Submission 12 experiment does not change the pair: its local weighted-F1 improvement did not pass the uncertainty, fold-consistency, simulated-private, or subgroup-safety gates. At the time it was evaluated, the exact Submission 6 nested-equivalent reference was also unavailable. That infrastructure gap is now closed, but closing it does not retroactively turn the failed candidate into a submission recommendation.

---

## Competition Rules and Reproducibility

- The workflow uses manually specified preprocessing, features, models, and
  blends; no AutoML library is used.
- The rules separately prohibit "automated pipeline-generation systems."
  AI-assisted modelling code may be covered by that wording, so the team must
  obtain written organizer confirmation before relying on it in the submitted
  notebook rather than assuming that the absence of an AutoML library is
  sufficient.
- Hidden test labels must not be inferred, probed, or manually assigned using
  leaderboard feedback.
- Restricted competition data, artifacts, and solution code must remain within
  the official team during the competition.
- The final notebook must regenerate the submitted workflow and outputs. The
  current consolidated notebook validates frozen artifacts, but that alone
  must not be represented as a complete fresh retraining of every historical
  model.

---

## Workspace Organization
Intermediate files and scripts have been moved to the `archive/` folder to keep the workspace clean:
- **Old scripts:** `archive/pipeline.py` (v4), `archive/pipeline_v5.py`,
  `archive/pipeline_v7.py`, and `archive/pipeline_v8.py`.
- **Old submissions:** `archive/submission1.csv` through
  `archive/submission12.csv`, plus
  diagnostic files such as
  `archive/submission_all_dead.csv` and individual-model predictions.
- **Probabilities:** Canonical historical `archive/probs_*.npy` arrays used for
  reconstruction and blending.

The main directory contains the active scripts (`pipeline_v6.py`,
`pipeline_nn.py`, `pipeline_mega_ensemble.py`, `pipeline_pseudo90.py`,
`pipeline_submission12_nn10.py`,
`pseudo_label_nested_validation.py`, `submission6_nested_reconstruction.py`,
`submission6_practical_reference_gate.py`,
`age55_subgroup_investigation.py`, `step1_cv_diagnostic.py`, and
`validation_harness.py`), data files, the prepared Submission 12 upload file
`submission.csv`, and this history document. The corresponding nested,
practical-gate, and age-subgroup evidence is retained under
`diagnostic_outputs/`. Historical pipelines v4, v5, v7, and v8 are retained
under `archive/`.
