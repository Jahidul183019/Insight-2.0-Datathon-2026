# Insight 2.0 Datathon - Submission History & Analysis

This document tracks the evolution of our submissions for the lung-cancer vital status prediction task. The primary objective is to maximize generalization and private/hidden-test performance. The evaluation metric is **F1-Score** (Dead = 1). Public LB scores for Submissions 1–9 were verified against the Kaggle submissions page in chronological order on 2026-08-29.

## Submission Log

### Submission 1
- **File:** `archive/submission1.csv`
- **Strategy:** Early baseline model.
- **Dead Rate:** 90.0% (32,392 Dead)
- **LB Score:** `0.871881`
- **Insight:** Over-predicted the positive class significantly and produced the lowest verified score among Submissions 1–9.

### Submission 2
- **File:** `archive/submission2.csv`
- **Strategy:** Target encoding within a complex inner CV + Optuna tuning + threshold sweep (Pipeline v4).
- **Dead Rate:** 84.5% (30,431 Dead)
- **LB Score:** `0.875506`
- **Insight:** Established that targeting an ~84.5% Dead rate is the "sweet spot" for the Leaderboard, balancing precision and recall effectively.

### Submission 3
- **File:** `archive/submission3.csv`
- **Strategy:** Re-run of Pipeline v4.
- **Dead Rate:** 84.5% (30,438 Dead)
- **LB Score:** `0.875380`
- **Insight:** Nearly identical to Submission 2. Confirmed the stability of the v4 model at the 84.5% threshold.

### Submission 4
- **File:** `archive/submission4.csv`
- **Strategy:** Cleaned up code (Pipeline v5). Removed target encoding. Used the raw CV F1-optimal threshold of `0.48`.
- **Dead Rate:** 88.3% (31,789 Dead)
- **LB Score:** `0.869451`
- **Insight:** Although `0.48` was theoretically optimal in local CV, it resulted in 88.3% Dead on the test set, which the LB heavily penalized. Re-affirmed that we must target 84.5% Dead.

### Submission 5
- **File:** `archive/submission5.csv` (from v5 fold-avg)
- **Strategy:** Pipeline v5 using 3x5 Fold averaging (LGB+XGB+CB). The threshold was explicitly tuned to output exactly 84.5% Dead (th=`0.605`).
- **Dead Rate:** 84.5% (30,424 Dead)
- **LB Score:** `0.876730`
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
- **LB Score:** `0.877258`
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
- **LB Score:** `0.876170`
- **Key Takeaway:** Confirmed that while stacking yields strong standalone performance (`0.876170`), the semi-supervised pseudo-labeling in Submission 6 (`0.877258`) was the critical factor that drove the score above 0.877.
- **Follow-up:** Submission 8 tested the planned blend with pseudo-labeling.
### Submission 8
- **File:** `archive/submission8.csv` (Super-Blend)
- **Strategy:** 50/50 blend of Submission 6 (Pseudo-labeling) and Submission 7 (Stacking).
- **Dead Rate:** 84.48% (30,414 Dead)
- **LB Score:** `0.876305`
- **Insight:** Scored between Sub 6 (0.877258) and Sub 7 (0.876170). On the public split, diluting Submission 6 with the stacking predictions reduced F1; this is evidence favoring Submission 6, not proof about hidden labels or a single causal component.

### Submission 9
- **File:** `archive/submission9.csv` (Pipeline v8 Iterative Pseudo-Labeling)
- **Strategy:** Iterative pseudo-labeling using an ultra-strict >98% confidence threshold from Submission 6, yielding ~14,000 pure labels, retrained on all 6 models and blended 50/50.
- **Dead Rate:** 84.48% (30,414 Dead)
- **LB Score:** `0.875022`
- **Insight:** The Public Leaderboard score dropped. This means that either (A) tightening the threshold to 98% reduced the diversity of the pseudo-labels, causing slight overfitting to the highly confident samples, or (B) the 411 changed rows were actually correct in Submission 6 for the Public LB portion of the test set. 

### Submission 10
- **File:** `submission.csv` (upload file); `archive/submission10.csv` (byte-identical archived snapshot)
- **Strategy:** Neural-network diversity blend using 80% of the archived Submission 6 probabilities (`archive/probs_v6_final.npy`) and 20% neural-network probabilities (`archive/probs_nn.npy`). Final labels use a deterministic top-k cutoff rather than a floating-point threshold.
- **Dead Rate:** 84.475% (exactly 30,411 Dead), matching Submission 6's class count.
- **Difference from Submission 6:** 292 rows change labels while the total Dead count remains fixed.
- **LB Score:** `0.876616`
- **Comparison with Submission 6:** `-0.000642`; the neural-network diversity blend did not surpass Submission 6 on the Public Leaderboard.
- **Insight:** Submission 10 outperformed the stacking Super-Blend in Submission 8 (`0.876305`) by `0.000311`, but remained below Submission 6 (`0.877258`). It is therefore useful as a verified diversity candidate, not as the primary submission.
- **Status:** Submitted and verified from the Kaggle submissions page on 2026-08-29.

---

## Final Strategy Selection
For the final Kaggle Private Leaderboard evaluation, you are allowed to select two submissions. Based on our empirical testing:
1. **Submission 6 (`0.877258`):** Keep this as the primary selection for now. It is the highest verified Public LB score and the strongest observed pseudo-labeling configuration; the hidden/private-set result remains unknown.
2. **Submission 10 (`0.876616`):** Use this as the secondary selection. It adds neural-network diversity, keeps the same class count as Submission 6, and has a higher verified Public LB score than Submission 8.

The recommended pair is now Submission 6 plus Submission 10. Submission 6 remains the primary choice because it has the highest verified Public LB score; Submission 10 provides a different model blend for private-set diversification.

---

## Workspace Organization
Intermediate files and scripts have been moved to the `archive/` folder to keep the workspace clean:
- **Old scripts:** `archive/pipeline.py` (v4), `archive/pipeline_v5.py`,
  `archive/pipeline_v7.py`, and `archive/pipeline_v8.py`.
- **Old submissions:** `archive/submission1.csv` through
  `archive/submission10.csv`, plus
  diagnostic files such as
  `archive/submission_all_dead.csv` and individual-model predictions.
- **Probabilities:** Canonical historical `archive/probs_*.npy` arrays used for
  reconstruction and blending.

The main directory contains the active scripts (`pipeline_v6.py`,
`pipeline_nn.py`, `pipeline_mega_ensemble.py`, `step1_cv_diagnostic.py`, and
`validation_harness.py`), data files, the verified Submission 10 upload file
`submission.csv`, and this
history document. Historical pipelines v4, v5, v7, and v8 are retained under
`archive/`.
