# Age 55–59 subgroup investigation

## Decision

**No model change and no submission.** The band has lower Dead prevalence and a full-OOF F1/AUC trough, but this canonical practical-reference follow-up audit does not isolate a class-conditional error, missingness, tumor-size, stage, histology, or treatment defect that justifies reweighting or a targeted feature. Case-mix and registry-era associations remain descriptive leads, not evidence that a model change would generalize.

The approved practical Submission 6 nested recipe OOF vector is used as the primary reference. Practical recipe replay is accepted; strict historical probability equivalence remains `false` because of approximately 1e-9 numerical drift and is not being reinterpreted as true.

Every hard prediction is selected once from the full OOF vector using stable top-k ranking at `0.845` predicted Dead. Support-weighted F1 is primary; Dead-class F1 is secondary.

## Main evidence

- Age 55–59 contains `1,821` rows and has a Dead rate of `79.24%`, versus `83.31%` elsewhere (`-4.07%`). This targeted follow-up was selected after prior subgroup scans, so its p-value is descriptive rather than an independently pre-specified confirmatory test.
- Under `submission6_nested_equivalent`, its weighted F1 is `0.838062`, versus `0.877004` globally and `0.880201` in the rest.
- Its OOF ROC AUC is `0.862474`. The neighboring 50–54 and 60–64 bands are shown below; the ranking signal also dips at 55–59, so this is not only a single global cutoff mismatch.
- It has `169` false positives and `118` false negatives. Its predicted Dead rate (`82.04%`) exceeds its observed Dead rate (`79.24%`).
- The local-neighbor trough check is `present`. Across the five canonical outer folds, 55–59 is below both neighbors in only `3/5` for weighted F1 and `3/5` for ROC AUC.
- Material semantic-unavailable differences after BH control: `15` (`13` less unavailable in 55–59, `2` more unavailable); material categorical effects: `0`; material numeric effects: `4`. There are `3` prevalence-confounded overall-error flags but `0` class-conditional signals after denominator and BH gates.

## Class balance

| focal_rows | rest_rows | focal_rate | rest_rate | rate_difference | difference_ci95_lower | difference_ci95_upper | binary_smd | p_value | dead_rate_ratio | dead_odds_ratio_haldane |
|---|---|---|---|---|---|---|---|---|---|---|
| 1821 | 22179 | 0.792422 | 0.833085 | -0.040664 | -0.059927 | -0.021400 | -0.104378 | 0.000009 | 0.951189 | 0.764192 |

## Adjacent age bands

| age_band | rows | actual_dead_rate | predicted_dead_rate | weighted_f1 | dead_class_f1 | roc_auc | mean_probability | false_positive | false_negative | error_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 50-54 years | 789 | 0.769328 | 0.793409 | 0.851355 | 0.906732 | 0.890418 | 0.782378 | 67 | 48 | 0.145754 |
| 55-59 years | 1821 | 0.792422 | 0.820428 | 0.838062 | 0.902281 | 0.862474 | 0.802476 | 169 | 118 | 0.157606 |
| 60-64 years | 2925 | 0.802735 | 0.810598 | 0.858401 | 0.912905 | 0.886035 | 0.803055 | 217 | 194 | 0.140513 |

Broader trend:

| age_band | rows | actual_dead_rate | predicted_dead_rate | weighted_f1 | roc_auc | mean_probability | error_rate |
|---|---|---|---|---|---|---|---|
| 40-44 years | 158 | 0.696203 | 0.708861 | 0.847170 | 0.928030 | 0.699355 | 0.151899 |
| 45-49 years | 341 | 0.730205 | 0.753666 | 0.851199 | 0.907194 | 0.734644 | 0.146628 |
| 50-54 years | 789 | 0.769328 | 0.793409 | 0.851355 | 0.890418 | 0.782378 | 0.145754 |
| 55-59 years | 1821 | 0.792422 | 0.820428 | 0.838062 | 0.862474 | 0.802476 | 0.157606 |
| 60-64 years | 2925 | 0.802735 | 0.810598 | 0.858401 | 0.886035 | 0.803055 | 0.140513 |
| 65-69 years | 3851 | 0.798754 | 0.805505 | 0.848422 | 0.880092 | 0.802452 | 0.150610 |
| 70-74 years | 4233 | 0.809119 | 0.824238 | 0.872311 | 0.898823 | 0.812239 | 0.125679 |

Five-fold stability view (fold-local 84.5% policy):

| age_band | weighted_f1_mean | weighted_f1_std | weighted_f1_min | weighted_f1_max | roc_auc_mean | roc_auc_std |
|---|---|---|---|---|---|---|
| 40-44 years | 0.831485 | 0.081492 | 0.748252 | 0.952381 | 0.908797 | 0.061744 |
| 45-49 years | 0.851535 | 0.037343 | 0.827729 | 0.916835 | 0.911249 | 0.024964 |
| 50-54 years | 0.848791 | 0.043604 | 0.771800 | 0.877173 | 0.891369 | 0.032107 |
| 55-59 years | 0.838562 | 0.025779 | 0.810514 | 0.869617 | 0.863412 | 0.036251 |
| 60-64 years | 0.859046 | 0.018898 | 0.828778 | 0.876599 | 0.886051 | 0.018116 |
| 65-69 years | 0.848905 | 0.008838 | 0.839188 | 0.861356 | 0.880601 | 0.011656 |
| 70-74 years | 0.871612 | 0.008260 | 0.863672 | 0.883446 | 0.899531 | 0.006765 |

| outer_fold | age50_54_weighted_f1 | age55_59_weighted_f1 | age60_64_weighted_f1 | age55_f1_below_both_neighbors | age50_54_roc_auc | age55_59_roc_auc | age60_64_roc_auc | age55_auc_below_both_neighbors |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.864042 | 0.869617 | 0.828778 | False | 0.906982 | 0.897653 | 0.889942 | False |
| 2 | 0.871907 | 0.846729 | 0.876599 | True | 0.929724 | 0.878944 | 0.911939 | True |
| 3 | 0.877173 | 0.810514 | 0.859457 | True | 0.900436 | 0.813068 | 0.867114 | True |
| 4 | 0.771800 | 0.813251 | 0.873211 | False | 0.846820 | 0.838122 | 0.890880 | True |
| 5 | 0.859033 | 0.852697 | 0.857185 | True | 0.872881 | 0.889273 | 0.870381 | False |

These rows use approved canonical outer folds. They remain subgroup diagnostics; a future intervention would still need its own candidate-versus-control fold deltas.

## Missingness and unavailable-code audit

`physical_na` and `semantic_unavailable` are separate in the CSV. The second is a heuristic registry audit covering values such as Blank(s), Unknown, EOD 88/TX/NX/MX, tumor-size 999, and regional-nodes-positive >=98. It is **not** an exact per-feature reconstruction of model missing-value handling, and these states are not assumed clinically equivalent.

| column | focal_rate | rest_rate | rate_difference | binary_smd | p_value | bh_q_value |
|---|---|---|---|---|---|---|
| rx_summ_surgradseq | 0.833059 | 0.905677 | -0.072618 | -0.216746 | 0.000000 | 0.000000 |
| regional_nodes_positive | 0.606260 | 0.690157 | -0.083897 | -0.176372 | 0.000000 | 0.000000 |
| diagnostic_confirmation | 0.010983 | 0.036070 | -0.025087 | -0.166087 | 0.000000 | 0.000000 |
| rx_summ_scope_reglnsur2003 | 0.637562 | 0.712476 | -0.074914 | -0.160461 | 0.000000 | 0.000000 |
| seer_combined_metsatdxbrain2010 | 0.034596 | 0.069029 | -0.034433 | -0.155820 | 0.000000 | 0.000000 |
| seer_combined_metsatdxbone2010 | 0.034047 | 0.067451 | -0.033404 | -0.152635 | 0.000000 | 0.000000 |
| seer_combined_metsatdxliver2010 | 0.034596 | 0.067767 | -0.033170 | -0.150951 | 0.000000 | 0.000000 |
| radiation_recode | 0.526634 | 0.599215 | -0.072582 | -0.146720 | 0.000000 | 0.000000 |
| seer_combined_metsatdxlung2010 | 0.039539 | 0.072952 | -0.033413 | -0.145409 | 0.000000 | 0.000000 |
| summary_stage | 0.465678 | 0.536724 | -0.071046 | -0.142452 | 0.000000 | 0.000000 |
| rx_summ_surgothregdis2003 | 0.012081 | 0.033094 | -0.021013 | -0.141776 | 0.000001 | 0.000003 |
| reason_nocancer_directed_surgery | 0.026359 | 0.050318 | -0.023959 | -0.125021 | 0.000005 | 0.000014 |
| rx_summ_surgprim_site20232023 | 0.959363 | 0.932233 | 0.027130 | 0.120039 | 0.000007 | 0.000018 |
| rx_summ_surgprim_site19982022 | 0.040637 | 0.067767 | -0.027130 | -0.120039 | 0.000007 | 0.000018 |
| derived_eod2018m_recode2018 | 0.577155 | 0.527075 | 0.050080 | 0.100837 | 0.000038 | 0.000090 |

Effects require both BH q<0.05 and |binary SMD|>=0.10. Small p-values without a material effect size are not treated as causes.

Crucially, `13` of `15` material differences point to *less* unavailable information in age 55–59, not more. The main exceptions include the EOD-M unavailable code and the mutually era-specific surgery-site fields. This does not support a targeted missingness-indicator fix.

## Stage, histology, treatment, and tumor-size composition

| feature_family | feature | categories_after_rare_pooling | cramers_v | chi_square_p_value | bh_q_value | material_effect | fraction_expected_cells_under_5 |
|---|---|---|---|---|---|---|---|
| histology | histologic_type_icdo3 | 28 | 0.076575 | 0.000000 | 0.000000 | False | 0.125000 |
| histology | histology_family | 7 | 0.070358 | 0.000000 | 0.000000 | False | 0.000000 |
| stage | summary_stage | 5 | 0.067021 | 0.000000 | 0.000000 | False | 0.000000 |
| data_quality | diagnostic_confirmation | 7 | 0.059113 | 0.000000 | 0.000000 | False | 0.142857 |
| treatment | rx_summ_surgprim_site19982022 | 11 | 0.053935 | 0.000000 | 0.000000 | False | 0.090909 |
| stage | derived_eod_composite_stage | 7 | 0.052296 | 0.000000 | 0.000000 | False | 0.000000 |
| treatment | rx_summ_scope_reglnsur2003 | 7 | 0.049376 | 0.000000 | 0.000000 | False | 0.071429 |
| treatment | rx_summ_surgothregdis2003 | 7 | 0.045044 | 0.000000 | 0.000000 | False | 0.071429 |
| treatment | reason_nocancer_directed_surgery | 8 | 0.044273 | 0.000000 | 0.000000 | False | 0.125000 |
| treatment | radiation_recode | 6 | 0.043949 | 0.000000 | 0.000000 | False | 0.166667 |

Largest individual level shifts:

| feature_family | feature | level | focal_count | focal_prevalence | rest_prevalence | prevalence_difference | focal_dead_rate | rest_dead_rate |
|---|---|---|---|---|---|---|---|---|
| stage | summary_stage | Distant | 634 | 0.34816 | 0.26038 | 0.08778 | 0.94795 | 0.97870 |
| treatment | radiation_recode | Beam radiation | 818 | 0.44920 | 0.37391 | 0.07529 | 0.82274 | 0.83142 |
| treatment | radiation_recode | None/Unknown | 944 | 0.51840 | 0.59236 | -0.07397 | 0.76165 | 0.82859 |
| histology | histologic_type_icdo3 | 8000 | 69 | 0.03789 | 0.10194 | -0.06405 | 0.95652 | 0.96196 |
| treatment | rx_summ_scope_reglnsur2003 | <NA> | 1119 | 0.61450 | 0.66915 | -0.05465 | 0.87310 | 0.89745 |
| stage | summary_stage | Blank(s) | 823 | 0.45195 | 0.50525 | -0.05330 | 0.70474 | 0.74853 |
| treatment | rx_summ_scope_reglnsur2003 | Biopsy or aspiration of regional lymph node, NOS | 349 | 0.19165 | 0.14324 | 0.04841 | 0.83954 | 0.83286 |
| histology | histologic_type_icdo3 | 8041 | 284 | 0.15596 | 0.10835 | 0.04761 | 0.91901 | 0.93009 |
| stage | summary_stage | Localized | 113 | 0.06205 | 0.10208 | -0.04002 | 0.63717 | 0.77871 |
| histology | histologic_type_icdo3 | 8140 | 734 | 0.40308 | 0.37003 | 0.03304 | 0.79564 | 0.82783 |
| histology | histologic_type_icdo3 | 8070 | 248 | 0.13619 | 0.16719 | -0.03100 | 0.84677 | 0.87406 |
| stage | summary_stage | Regional | 226 | 0.12411 | 0.10082 | 0.02329 | 0.73451 | 0.88775 |

Numeric profiles (quantitative node values exclude registry sentinel codes >=90):

| feature | focal_nonmissing | rest_nonmissing | focal_mean | rest_mean | standardized_mean_difference | rank_biserial | bh_q_value | material_effect |
|---|---|---|---|---|---|---|---|---|
| regional_nodes_positive_quantitative | 365 | 4113 | 0.909589 | 0.548991 | 0.202510 | 0.147349 | 0.000000 | True |
| year_of_diagnosis | 1821 | 22179 | 2017.261944 | 2017.686686 | -0.141040 | -0.076799 | 0.000000 | True |
| tumor_size_overtime | 1400 | 17184 | 44.084286 | 41.209032 | 0.105875 | 0.054210 | 0.000850 | True |
| tumor_size_summary | 983 | 12323 | 47.472024 | 42.986042 | 0.104353 | 0.072908 | 0.000243 | True |
| tumor_size_best | 1434 | 17386 | 45.974198 | 42.602784 | 0.087127 | 0.055820 | 0.000606 | False |
| regional_nodes_examined_quantitative | 1383 | 17447 | 2.456255 | 2.079899 | 0.065697 | 0.044088 | 0.000176 | False |
| cs_tumor_size20042015 | 475 | 5242 | 64.303158 | 61.166539 | 0.023036 | 0.023631 | 0.392975 | False |

The two component tumor-size fields barely cross |SMD|=0.10, while the merged `tumor_size_best` feature is below that threshold. The node-positive comparison uses only quantitative (non-sentinel) rows, so its apparent shift may partly reflect selection into having a count. The roughly 0.43-year diagnosis-era shift is case-mix evidence, not a standalone model defect.

## Error clusters

The total-error comparison is prevalence-confounded because age 55–59 has a different Dead rate. It is retained as a descriptive lead only. Actionability is assessed separately with false-positive rate among Alive rows and false-negative rate among Dead rows.

Class-conditional signals passing denominator, effect-size, and BH gates:

_None._

Prevalence-confounded total-error flags (descriptive only; family and raw-code rows can overlap):

| cluster_type | cluster | focal_rows | rest_same_cluster_rows | focal_dead_rate | rest_dead_rate | focal_error_rate | rest_error_rate | error_rate_difference | overall_error_bh_q_value | focal_alive_denominator | rest_alive_denominator | focal_false_positive_rate | rest_false_positive_rate | focal_dead_denominator | rest_dead_denominator | focal_false_negative_rate | rest_false_negative_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| composite_stage_x_histology_code | EOD unavailable (all blank) | 8010 | 31 | 439 | 0.870968 | 0.984055 | 0.129032 | 0.013667 | 0.115365 | 0.049531 | 4 | 7 | 1.000000 | 0.857143 | 27 | 432 | 0.000000 | 0.000000 |
| composite_stage_x_histology_family | EOD unavailable (all blank) | adenocarcinoma family | 473 | 4852 | 0.830867 | 0.887675 | 0.164905 | 0.106142 | 0.058763 | 0.006233 | 80 | 545 | 0.575000 | 0.563303 | 393 | 4307 | 0.081425 | 0.048293 |
| composite_stage_x_histology_code | EOD unavailable (all blank) | 8140 | 407 | 4095 | 0.862408 | 0.917705 | 0.137592 | 0.079121 | 0.058471 | 0.006456 | 56 | 337 | 0.660714 | 0.652819 | 351 | 3758 | 0.054131 | 0.027674 |

A class-conditional signal requires at least 30 focal and 100 comparison rows of the relevant true class, an excess FPR or FNR of at least 5 percentage points, and BH q<0.05 within cluster definition and error type.

The reported q≈0.006456 for EOD-all-blank × histology 8140 belongs to the total-error test and therefore does not establish class-conditional difficulty. For the broader EOD-all-blank × adenocarcinoma family, FPR is 46/80 (0.5750) versus 307/545 (0.5633), while FNR is 32/393 (0.0814) versus 208/4307 (0.0483). Neither endpoint passes the conservative 5-point class-conditional screen. Raw 8010 has only 4 focal and 7 comparison Alive rows, so its FPR is too poorly supported for intervention.

**Step 2 canonical replication verdict:** the prevalence-confounded total-error association remains BH-significant for both targeted definitions (adenocarcinoma family q=`0.006233`; raw 8140 q=`0.006456`). The actionable class-conditional excess-error finding **does not replicate**: family FPR/FNR gaps are +0.0117/+0.0331, and raw-8140 FPR/FNR gaps are +0.0079/+0.0265. All are below the 5% effect gate, and zero targeted definitions pass the denominator/effect/BH class-conditional gates. Therefore Step 3 is not entered.

Largest family-level cells are retained in `age55_error_clusters.csv`; none may be called actionable merely because its unconditioned total-error Fisher test is small.

## Age representation check

| representation | present_in_pipeline_v6 | source_file |
|---|---|---|
| age_recode categorical | True | pipeline_v6.py |
| age_midpoint numeric | True | pipeline_v6.py |
| age × diagnosis-year interaction | True | pipeline_v6.py |
| age × stage interaction | True | pipeline_v6.py |
| age × metastasis interaction | True | pipeline_v6.py |

Pipeline v6 already uses the numeric band midpoint alongside categorical `age_recode` and age×year/stage/metastasis interactions. The source data still provides age as a band, so the midpoint cannot recover within-band age. Even so, the trough is not simply evidence that the model forgot a smooth age term, and adding another one is not justified by this audit.

## Rejected pseudo90 + NN stress-test changes

Within age 55–59, pseudo90 + 20% NN has weighted F1 `0.832225` versus `0.834485` for pseudo90 (`-0.002260`). It changes `16/1821` predictions: `6` corrected and `10` harmed. This directly supports rejecting that stress candidate, not introducing an age-specific rewrite.

| cluster | focal_age_rows | cluster_focal_rows | changed_predictions | changed_fraction_of_cluster | changed_fraction_of_focal_age | control_dead_to_stress_alive | control_alive_to_stress_dead | corrected_predictions | harmed_predictions | net_corrected |
|---|---|---|---|---|---|---|---|---|---|---|
| EOD unavailable (all blank) | adenocarcinoma family | 1821 | 473 | 2 | 0.0042 | 0.0011 | 1 | 1 | 0 | 2 | -2 |
| EOD unavailable (all blank) | other histology | 1821 | 81 | 1 | 0.0123 | 0.0005 | 0 | 1 | 0 | 1 | -1 |
| EOD unavailable (all blank) | small-cell family | 1821 | 181 | 1 | 0.0055 | 0.0005 | 0 | 1 | 0 | 1 | -1 |
| M0 N0 T0-2 localized | squamous family | 1821 | 23 | 1 | 0.0435 | 0.0005 | 1 | 0 | 0 | 1 | -1 |
| M1 distant/metastatic | NOS/other epithelial family | 1821 | 42 | 1 | 0.0238 | 0.0005 | 1 | 0 | 0 | 1 | -1 |
| M1 distant/metastatic | small-cell family | 1821 | 85 | 1 | 0.0118 | 0.0005 | 1 | 0 | 0 | 1 | -1 |
| EOD unavailable (all blank) | squamous family | 1821 | 145 | 2 | 0.0138 | 0.0011 | 2 | 0 | 1 | 1 | 0 |
| M0 N1-3 node-positive | squamous family | 1821 | 48 | 2 | 0.0417 | 0.0011 | 1 | 1 | 1 | 1 | 0 |
| M1 distant/metastatic | adenocarcinoma family | 1821 | 236 | 3 | 0.0127 | 0.0016 | 0 | 3 | 2 | 1 | 1 |
| EOD not applicable (88) | other histology | 1821 | 48 | 1 | 0.0208 | 0.0005 | 1 | 0 | 1 | 0 | 1 |
| EOD unavailable (all blank) | neuroendocrine family | 1821 | 37 | 1 | 0.0270 | 0.0005 | 0 | 1 | 1 | 0 | 1 |

## Interpretation guardrails

- This is association and OOF error analysis, not a causal study.
- BH correction is applied within each test family. It limits false discoveries but does not turn a registry-code association into a model-fix justification.
- Age 55–59 was selected after earlier subgroup scans; all inferential results here are targeted follow-up evidence, not pristine confirmatory inference.
- Total error and weighted F1 vary with class prevalence. Cluster claims therefore require the separately reported class-conditional FPR/FNR gates and adequate denominators.
- Slice F1 is prevalence-sensitive; lower Dead prevalence can reduce weighted F1 under a global fixed-rate policy even when ranking remains useful.
- Proxy fold re-slices are descriptive and cannot pass an intervention's 4-of-5 outer-fold gate.
- No age-specific threshold, post-hoc label rewrite, subgroup reweighting, training, or submission file is produced.
- If the report used the frozen-tree proxy, rerun this script after the canonical Submission 6 nested artifact exists before using these findings for a future gate.

## Reproduce

```bash
python3 age55_subgroup_investigation.py
```
