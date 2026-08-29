# Validation-harness report

## Scope

This is a prediction-level audit using 24,000 labelled training rows, 50 repeated stratified 40%/60% splits, and already-saved OOF probabilities. It performs no model training and never reads Kaggle public/private labels.

The words `public` and `private` in CSV column names mean only the synthetic 40/60 selection and holdout partitions. They are not Kaggle partitions.

The primary metric is the competition's support-weighted F1 across Alive and Dead. Explicit `*_dead_class_f1_legacy` columns retain the earlier binary Dead-class calculation for historical comparison. Unqualified `*_f1` columns are deprecated compatibility aliases; see `metric_schema_version=2_weighted_f1_primary`.

## Direct findings

- At a fixed 84.5% positive rate on full OOF data, the best tested OOF candidate was `tree80_nn20` (support-weighted F1 0.878447).
- Across repeated 40/60 splits under that fixed-rate policy, `tree80_nn20` had the highest mean holdout support-weighted F1 (0.878170).
- The current 80% tree / 20% NN candidate changed mean holdout support-weighted F1 by +0.000846 relative to the saved tree OOF proxy. This comparison does not include pseudo-label retraining.
- The reported leaderboard order is `submission_6 > submission_8 > submission_7 > submission_9`, but this harness cannot honestly reproduce or refute that order from the available files.
- The closest hard-submission pair is `submission_7` vs `submission_8` with 260 changed rows (0.72%). Disagreement is a diversity/risk proxy, not evidence that either side is correct.

## Why Submissions 7--9 cannot yet be locally ranked with Submission 6

The independently approved nested-equivalent reference makes Submission 6 locally scoreable under its locked 84.5% OOF policy. Recipe-equivalent OOF vectors remain unavailable for Submissions 7--9, and hidden test labels remain unavailable for every submission.
Reported leaderboard scores are included only as historical metadata; they are not validation targets.

A valid cross-submission ranking still requires regenerating the remaining candidate recipes inside the same outer folds, including pseudo-label creation using only information allowed within each fold.
Comparing current hard test files against training labels would be invalid because the rows do not correspond.

## Important limitations

- The split is applied to locked OOF predictions, not to end-to-end model training. It tests selection and threshold stability, not full training-set shift.
- `oof_step1.npy` already contains a globally selected tree blend, so its estimate is not fully nested and can be mildly optimistic.
- The NN and tree OOF vectors were produced by different CV schemes. Their blend comparison is useful as a screening proxy, not a submission guarantee.
- Repeated splits overlap. Their standard deviations describe observed split dispersion and are not independent-sample confidence intervals.
- A fixed positive-rate policy depends only on ranking; it does not evaluate probability calibration.
- The 84.5% rate is a historically selected policy, not an independently derived value in this harness. It must not be used as evidence from an independent validation set or refined through leaderboard probing.

## Output guide

- `oof_global_candidate_metrics.csv`: full-OOF screening metrics.
- `oof_40_60_split_results.csv`: every candidate/policy/split result.
- `oof_40_60_candidate_summary.csv`: mean and dispersion by candidate.
- `oof_40_60_selection_stability.csv`: winner/rank transfer per split.
- `oof_40_60_selection_summary.csv`: aggregate selection stability.
- `submission_6_to_9_artifact_audit.csv`: format, rate, and score provenance.
- `submission_6_to_9_pairwise_disagreement.csv`: hard-prediction diversity.

The detailed selection-policy summary follows:

```csv
policy,metric_schema_version,ranking_metric,n_splits,winner_exact_match_rate,rank_correlation_mean,rank_correlation_std,private_weighted_f1_regret_mean,private_weighted_f1_regret_max,private_f1_regret_mean,private_f1_regret_max,most_frequent_public_winner,most_frequent_public_winner_count,most_frequent_private_winner,most_frequent_private_winner_count
fixed_rank_rate_0.845,2_weighted_f1_primary,official_support_weighted_f1,50,0.220000,0.615411,0.251466,0.000600,0.002688,0.000600,0.002688,tree70_nn30_previous_candidate,20,tree80_nn20,24
legacy_dead_class_public_selected_probability_threshold,2_weighted_f1_primary,official_support_weighted_f1,50,0.620000,0.886963,0.135102,0.000276,0.002545,0.000276,0.002545,submission6_nested_recipe_reference,17,submission6_nested_recipe_reference,16
legacy_dead_class_public_selected_rank_rate_0.80_to_0.92,2_weighted_f1_primary,official_support_weighted_f1,50,0.700000,0.899438,0.131866,0.000188,0.003475,0.000188,0.003475,submission6_nested_recipe_reference,17,submission6_nested_recipe_reference,18
official_weighted_public_selected_probability_threshold,2_weighted_f1_primary,official_support_weighted_f1,50,0.180000,0.589286,0.300306,0.000696,0.003885,0.000696,0.003885,tree_oof,16,tree_oof,26
official_weighted_public_selected_rank_rate_0.80_to_0.92,2_weighted_f1_primary,official_support_weighted_f1,50,0.180000,0.509604,0.263168,0.000725,0.002552,0.000725,0.002552,tree95_nn05,14,tree95_nn05,18
```
