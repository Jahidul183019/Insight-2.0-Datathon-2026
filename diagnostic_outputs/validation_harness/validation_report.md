# Validation-harness report

## Scope

This is a prediction-level audit using 24,000 labelled training rows, 50 repeated stratified 40%/60% splits, and already-saved OOF probabilities. It performs no model training and never reads Kaggle public/private labels.

The words `public` and `private` in CSV column names mean only the synthetic 40/60 selection and holdout partitions. They are not Kaggle partitions.

## Direct findings

- At a fixed 84.5% positive rate on full OOF data, the best tested tree/NN candidate was `tree80_nn20` (F1 0.928756).
- Across repeated 40/60 splits under that fixed-rate policy, `tree80_nn20` had the highest mean holdout F1 (0.928594).
- The current 80% tree / 20% NN candidate changed mean holdout F1 by +0.000496 relative to the saved tree OOF proxy. This comparison does not include pseudo-label retraining.
- The reported leaderboard order is `submission_6 > submission_8 > submission_7 > submission_9`, but this harness cannot honestly reproduce or refute that order from the available files.
- The closest hard-submission pair is `submission_7` vs `submission_8` with 260 changed rows (0.72%). Disagreement is a diversity/risk proxy, not evidence that either side is correct.

## Why Submissions 6--9 cannot be locally ranked

Only their test probabilities or hard labels are available. There are no recipe-equivalent OOF vectors for the 95%-pseudo-label model (Sub 6), stack (Sub 7), super-blend (Sub 8), and 98%-pseudo-label model (Sub 9). Test labels are hidden. Reported leaderboard scores are included only as historical metadata; they are not validation targets.

A valid replay requires regenerating all four candidates inside the same outer folds, including pseudo-label creation using only information allowed within each fold. Comparing their current hard test files against training labels would be invalid because the rows do not correspond.

## Important limitations

- The split is applied to locked OOF predictions, not to end-to-end model training. It tests selection and threshold stability, not full training-set shift.
- `oof_step1.npy` already contains a globally selected tree blend, so its estimate is not fully nested and can be mildly optimistic.
- The NN and tree OOF vectors were produced by different CV schemes. Their blend comparison is useful as a screening proxy, not a submission guarantee.
- Repeated splits overlap. Their standard deviations describe observed split dispersion and are not independent-sample confidence intervals.
- A fixed positive-rate policy depends only on ranking; it does not evaluate probability calibration.

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
policy,n_splits,winner_exact_match_rate,rank_correlation_mean,rank_correlation_std,private_f1_regret_mean,private_f1_regret_max,most_frequent_public_winner,most_frequent_public_winner_count,most_frequent_private_winner,most_frequent_private_winner_count
fixed_rank_rate_0.845,50,0.260000,0.560550,0.288237,0.000308,0.001078,tree80_nn20,21,tree80_nn20,25
public_selected_probability_threshold,50,0.060000,0.296142,0.303221,0.000478,0.002577,tree90_nn10,16,tree70_nn30_previous_candidate,19
public_selected_rank_rate_0.80_to_0.92,50,0.120000,0.263493,0.337894,0.000494,0.001442,tree90_nn10,13,tree80_nn20,16
```
