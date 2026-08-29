# Submission 12 gate audit

## Decision

**NO-GO: do not generate or submit Submission 12.** The candidate fails multiple predeclared gates under the official weighted-F1 metric. The legacy Dead-class binary-F1 view reaches the same decision.

This audit created diagnostics only. It did not create a submission CSV or test-probability artifact and did not alter `submission.csv`, `SUBMISSION_HISTORY.md`, or the consolidated notebook.

## Metric scope

The official evaluation page specifies support-weighted F1, so `weighted_f1` is the authoritative gate metric. Standard binary F1 with `Dead=1` is retained only as a legacy comparison with earlier workspace analyses. All hard predictions use deterministic top-k selection at exactly 84.5% predicted Dead within the evaluated partition.

## Full-OOF fixed-rate results

| candidate | predicted_dead_rate | weighted_f1 | dead_class_binary_f1 | weighted_f1_delta_vs_nested_pseudo90 | dead_f1_delta_vs_nested_pseudo90 |
| --- | --- | --- | --- | --- | --- |
| nested_pseudo95_proxy | 0.845000 | 0.874203 | 0.926269 | -0.000594 | -0.000348 |
| nested_pseudo90 | 0.845000 | 0.874797 | 0.926617 | 0.000000 | 0.000000 |
| sub10_nested_analogue_p95_nn20 | 0.845000 | 0.875476 | 0.927015 | 0.000679 | 0.000398 |
| sub12_candidate_p90_nn20 | 0.845000 | 0.876155 | 0.927413 | 0.001358 | 0.000796 |
| existing_tree80_nn20 | 0.845000 | 0.878447 | 0.928756 | 0.003650 | 0.002139 |

The candidate's full-OOF point improvement is positive, but a point estimate alone is insufficient for the predeclared gate.

## Five outer folds

| outer_fold | dead_class_binary_f1_nested_pseudo90 | dead_class_binary_f1_sub12_candidate_p90_nn20 | weighted_f1_nested_pseudo90 | weighted_f1_sub12_candidate_p90_nn20 | dead_f1_delta_vs_nested_pseudo90 | weighted_f1_delta_vs_nested_pseudo90 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.928109 | 0.928358 | 0.877344 | 0.877768 | 0.000249 | 0.000424 |
| 2 | 0.929602 | 0.929602 | 0.879890 | 0.879890 | 0.000000 | 0.000000 |
| 3 | 0.924129 | 0.925871 | 0.870553 | 0.873524 | 0.001741 | 0.002971 |
| 4 | 0.925871 | 0.927114 | 0.873524 | 0.875646 | 0.001244 | 0.002122 |
| 5 | 0.925622 | 0.924876 | 0.873100 | 0.871826 | -0.000746 | -0.001273 |

| metric | fold_mean | worst_fold_value | improved_fold_count | tied_fold_count | regressed_fold_count |
| --- | --- | --- | --- | --- | --- |
| dead_class_binary_f1 | 0.927164 | 0.924876 | 3 | 1 | 1 |
| weighted_f1 | 0.875731 | 0.871826 | 3 | 1 | 1 |

Fold-local top-k is used here. Slicing one globally selected hard vector would not preserve 84.5% within each fold and is not used to pass the fold gate.

## Paired stratified bootstrap

| metric | full_oof_point_delta | bootstrap_mean_delta | bootstrap_95pct_lower | bootstrap_95pct_upper | fraction_strictly_above_zero | ci_lower_strictly_above_zero |
| --- | --- | --- | --- | --- | --- | --- |
| weighted_f1 | 0.001358 | 0.001195 | -0.000255 | 0.002631 | 0.947000 | False |
| dead_class_binary_f1 | 0.000796 | 0.000701 | -0.000149 | 0.001542 | 0.947000 | False |

The official weighted-F1 lower confidence bound crosses zero, so Gate 1 fails. The legacy binary-F1 interval also crosses zero.

## Repeated stratified 40/60 audit

The table below uses the synthetic 60% holdout (`private`) partition and the same fixed-rate policy and seeds as `validation_harness.py`.

| candidate | dead_class_binary_f1_mean | dead_class_binary_f1_std | weighted_f1_mean | weighted_f1_std |
| --- | --- | --- | --- | --- |
| existing_tree80_nn20 | 0.928594 | 0.000896 | 0.878170 | 0.001528 |
| sub12_candidate_p90_nn20 | 0.927235 | 0.001058 | 0.875853 | 0.001805 |
| sub10_nested_analogue_p95_nn20 | 0.926905 | 0.000959 | 0.875289 | 0.001636 |
| nested_pseudo90 | 0.926499 | 0.001078 | 0.874596 | 0.001840 |
| nested_pseudo95_proxy | 0.926323 | 0.001090 | 0.874296 | 0.001860 |

| metric | candidate | harness_selected_winner_count | inclusive_co_winner_count | unique_winner_count | clear_majority_selected_winner |
| --- | --- | --- | --- | --- | --- |
| weighted_f1 | existing_tree80_nn20 | 50 | 50 | 50 | True |
| weighted_f1 | nested_pseudo90 | 0 | 0 | 0 | False |
| weighted_f1 | nested_pseudo95_proxy | 0 | 0 | 0 | False |
| weighted_f1 | sub10_nested_analogue_p95_nn20 | 0 | 0 | 0 | False |
| weighted_f1 | sub12_candidate_p90_nn20 | 0 | 0 | 0 | False |
| dead_class_binary_f1 | existing_tree80_nn20 | 50 | 50 | 50 | True |
| dead_class_binary_f1 | nested_pseudo90 | 0 | 0 | 0 | False |
| dead_class_binary_f1 | nested_pseudo95_proxy | 0 | 0 | 0 | False |
| dead_class_binary_f1 | sub10_nested_analogue_p95_nn20 | 0 | 0 | 0 | False |
| dead_class_binary_f1 | sub12_candidate_p90_nn20 | 0 | 0 | 0 | False |

Pairwise Sub12-versus-pseudo90 stability is directionally favorable:

| metric | mean_delta | minimum_delta | maximum_delta | improved_split_count | tied_split_count | regressed_split_count |
| --- | --- | --- | --- | --- | --- | --- |
| weighted_f1 | 0.001256 | -0.000283 | 0.002688 | 49 | 0 | 1 |
| dead_class_binary_f1 | 0.000736 | -0.000166 | 0.001575 | 49 | 0 | 1 |

However, Gate 3 asks whether Sub12 is the best candidate, not merely whether it beats pseudo90 pairwise. Existing tree80/NN20 wins the full comparison set.

## Recipe-equivalence limitations

| item | status | exact_match | disagreement_count | detail |
| --- | --- | --- | --- | --- |
| actual_submission6_hard_reconstruction | exact | True | 0 | Top-30,411 of archive/probs_v6_final.npy reproduces archive/submission6.csv in memory. |
| actual_submission10_hard_reconstruction | exact | True | 0 | Top-30,411 of 0.80*v6 + 0.20*NN reproduces archive/submission10.csv in memory. |
| nested_pseudo95_vs_actual_submission6 | not_recipe_equivalent | False |  | The nested control uses one full 1500-round fit per model family without inner early stopping or fold averaging. Actual Submission 6 uses a 3x5 early-stopped teacher and a five-fold early-stopped pseudo student. It is a like-for-like pseudo90 control, not an exact nested replay of Submission 6. |
| oof_step1_vs_actual_submission6 | not_recipe_equivalent | False |  | oof_step1/frozen_tree_oof is the pre-pseudo teacher proxy and does not contain Submission 6's final pseudo-student component. |
| nn_oof_outer_fold_comparability | screening_proxy_only | False |  | pipeline_nn.py uses the same five-fold StratifiedKFold seed 42, but preprocessing is fit globally and each validation fold selects the best NN epoch. The blend is aligned OOF screening, not the same untouched-outer-fold standard as the nested pseudo audit. |

Consequently, the saved pseudo95 nested vector is a useful paired control but cannot satisfy the prompt's exact Submission 6 nested-equivalence requirement. An exact replay would require end-to-end nested retraining of the original 3x5 teacher and five-fold pseudo-student recipe.

## Gate result

| gate | metric | metric_role | observed | required | passed | note |
| --- | --- | --- | --- | --- | --- | --- |
| 1_bootstrap_ci_lower_strictly_above_zero | weighted_f1 | official_primary | -0.000255 | > 0 | False | Sub12 minus nested pseudo90 paired stratified bootstrap. |
| 2_outer_fold_improvement_count | weighted_f1 | official_primary | 3 | >= 4 of 5 | False | Each fold independently uses an exact 84.5% top-rate policy. |
| 3_simulated_private_clear_majority | weighted_f1 | official_primary | 0 | > 25 of 50 | False | Winner comparison includes all five requested OOF candidates. |
| 1_bootstrap_ci_lower_strictly_above_zero | dead_class_binary_f1 | legacy_comparison | -0.000149 | > 0 | False | Sub12 minus nested pseudo90 paired stratified bootstrap. |
| 2_outer_fold_improvement_count | dead_class_binary_f1 | legacy_comparison | 3 | >= 4 of 5 | False | Each fold independently uses an exact 84.5% top-rate policy. |
| 3_simulated_private_clear_majority | dead_class_binary_f1 | legacy_comparison | 0 | > 25 of 50 | False | Winner comparison includes all five requested OOF candidates. |
| 4_no_meaningful_subgroup_regression | external | separate_audit |  | separate subgroup audit must pass |  | Handled by the separate exact subgroup audit, not this script. |
| 5_directionally_better_than_actual_submission6_nested | weighted_f1 | official_primary |  | exact recipe-equivalent nested Submission 6 OOF | False | Unavailable: nested pseudo95 is not an exact replay. Sub12 is +0.001952315 versus the pseudo95 proxy but -0.002291848 versus existing tree80_nn20. |

Official-metric decisive failed gate rows: 4. Gate 4 is deliberately left to the separate subgroup audit; its result cannot rescue failures in Gates 1–3.

## Reproduction

```bash
python3 submission12_gate_audit.py
```

Seeds: outer folds `42`; bootstrap `9042`; simulated splits `20260829` through `20260878`.
