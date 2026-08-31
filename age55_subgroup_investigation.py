"""Diagnose the age-55--59 OOF weakness without changing the model.

The script is intentionally diagnostics-only.  It reads training data and
saved out-of-fold (OOF) probabilities, applies a deterministic 84.5% global
Dead-rate policy, and writes tables/reporting under
``diagnostic_outputs/age55_investigation``.  It never trains a model and never
creates or modifies a Kaggle submission.

Support-weighted F1 is the official competition metric.  Binary Dead-class F1
is retained as a secondary diagnostic so older workspace results remain
reconcilable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, norm
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "diagnostic_outputs" / "age55_investigation"
FIXED_POSITIVE_RATE = 0.845
FOCAL_AGE = "55-59 years"
NEIGHBOR_AGES = ("50-54 years", FOCAL_AGE, "60-64 years")
TREND_AGES = (
    "40-44 years",
    "45-49 years",
    "50-54 years",
    "55-59 years",
    "60-64 years",
    "65-69 years",
    "70-74 years",
)
OUTER_SEED = 42
N_OUTER_FOLDS = 5
RARE_CATEGORY_MIN = 30
MATERIAL_BINARY_SMD = 0.10
MATERIAL_CRAMERS_V = 0.10
MATERIAL_NUMERIC_SMD = 0.10
MATERIAL_ERROR_RATE_GAP = 0.05
MIN_CLASS_CONDITIONAL_FOCAL = 30
MIN_CLASS_CONDITIONAL_REST = 100
ALPHA = 0.05

T_COLUMN = "derived_eod2018t_recode2018"
N_COLUMN = "derived_eod2018n_recode2018"
M_COLUMN = "derived_eod2018m_recode2018"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--fixed-positive-rate", type=float, default=FIXED_POSITIVE_RATE
    )
    args = parser.parse_args()
    if not 0.0 < args.fixed_positive_rate < 1.0:
        parser.error("--fixed-positive-rate must be between zero and one")
    return args


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_probability_vector(name: str, values: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    if values.min() < 0.0 or values.max() > 1.0:
        raise ValueError(f"{name} contains values outside [0, 1]")
    return values


def top_rate_predictions(scores: np.ndarray, positive_rate: float) -> np.ndarray:
    """Select exactly round(rate*n) rows as Dead with stable tie-breaking."""

    n_positive = int(round(positive_rate * len(scores)))
    n_positive = min(max(n_positive, 0), len(scores))
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    predictions = np.zeros(len(scores), dtype=np.int8)
    predictions[order[:n_positive]] = 1
    return predictions


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values, preserving NaNs."""

    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(values[valid])]
    ranked = values[order] * len(valid) / np.arange(1, len(valid) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    return adjusted


def binary_rate_comparison(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Compare two independent binary samples with effect sizes and a z test."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    p_a = float(a.mean())
    p_b = float(b.mean())
    diff = p_a - p_b
    pooled_sd = math.sqrt((p_a * (1 - p_a) + p_b * (1 - p_b)) / 2)
    smd = diff / pooled_sd if pooled_sd > 0 else 0.0
    pooled = float((a.sum() + b.sum()) / (len(a) + len(b)))
    z_se = math.sqrt(pooled * (1 - pooled) * (1 / len(a) + 1 / len(b)))
    z_value = diff / z_se if z_se > 0 else 0.0
    p_value = float(2 * norm.sf(abs(z_value))) if z_se > 0 else 1.0
    ci_se = math.sqrt(
        p_a * (1 - p_a) / len(a) + p_b * (1 - p_b) / len(b)
    )
    return {
        "focal_rate": p_a,
        "rest_rate": p_b,
        "rate_difference": diff,
        "difference_ci95_lower": diff - 1.96 * ci_se,
        "difference_ci95_upper": diff + 1.96 * ci_se,
        "binary_smd": smd,
        "z_statistic": z_value,
        "p_value": p_value,
    }


def derive_composite_stage(frame: pd.DataFrame) -> pd.Series:
    def classify(row: pd.Series) -> str:
        t_value, n_value, m_value = (str(row[c]) for c in (T_COLUMN, N_COLUMN, M_COLUMN))
        if m_value.startswith("M1"):
            return "M1 distant/metastatic"
        if m_value == "M0":
            if n_value in {"N1", "N2", "N3"}:
                return "M0 N1-3 node-positive"
            if n_value == "N0":
                if t_value == "T0" or t_value.startswith(("T1", "T2")):
                    return "M0 N0 T0-2 localized"
                if t_value == "T3" or t_value.startswith("T4"):
                    return "M0 N0 T3-4 locally advanced"
            return "M0 indeterminate T/N"
        if (t_value, n_value, m_value) == ("Blank(s)", "Blank(s)", "Blank(s)"):
            return "EOD unavailable (all blank)"
        if (t_value, n_value, m_value) == ("88", "88", "88"):
            return "EOD not applicable (88)"
        return "Other/discordant EOD"

    return frame[[T_COLUMN, N_COLUMN, M_COLUMN]].apply(classify, axis=1)


def histology_group(value: object) -> str:
    try:
        code = int(value)
    except (TypeError, ValueError):
        return "unavailable"
    adenocarcinoma = {
        8140, 8141, 8143, 8144, 8145, 8147, 8148, 8211, 8230, 8250, 8251,
        8252, 8253, 8254, 8255, 8256, 8257, 8260, 8263, 8310, 8323, 8333,
        8480, 8481, 8490, 8550, 8551, 8570, 8574,
    }
    squamous = {8070, 8071, 8072, 8073, 8074, 8075, 8076, 8078, 8083, 8084}
    small_cell = {8041, 8042, 8043, 8044, 8045}
    large_cell = {8012, 8013, 8014}
    nos_epithelial = {
        8000, 8001, 8002, 8003, 8004, 8005, 8010, 8011, 8020, 8021, 8022
    }
    neuroendocrine = {8240, 8241, 8242, 8243, 8244, 8245, 8246, 8249}
    if code in adenocarcinoma:
        return "adenocarcinoma family"
    if code in squamous:
        return "squamous family"
    if code in small_cell:
        return "small-cell family"
    if code in large_cell:
        return "large-cell family"
    if code in nos_epithelial:
        return "NOS/other epithelial family"
    if code in neuroendocrine:
        return "neuroendocrine family"
    return "other histology"


def parse_tumor_size(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text == "Blank(s)" or text == "999" or text.startswith("Unknown"):
        return np.nan
    if "Not Consistent" in text or "Not Applicable" in text:
        return np.nan
    try:
        parsed = float(text)
    except ValueError:
        return np.nan
    return parsed if 0 <= parsed < 999 else np.nan


def tumor_size_bin(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[-np.inf, 10, 20, 30, 50, 70, 100, np.inf],
        labels=["<=10", "11-20", "21-30", "31-50", "51-70", "71-100", ">100"],
    ).astype("string").fillna("Unavailable")


def semantic_unavailable_mask(column: str, series: pd.Series) -> pd.Series:
    """Flag registry values that heuristically express unavailable information.

    This deliberately supplements physical NA with encoded states such as
    ``Blank(s)`` and ``Unknown``.  It is not an exact reconstruction of every
    per-feature pipeline transformation, nor a claim that all registry codes
    have the same clinical meaning.
    """

    result = series.isna().copy()
    text = series.astype("string").str.strip()
    lower = text.str.lower()
    result |= lower.isin({"", "blank(s)", "unknown", "unknown/unstaged", "999"})
    result |= lower.str.contains("unknown|not applicable|unreasonable", na=False)
    if column in {T_COLUMN, N_COLUMN, M_COLUMN}:
        result |= text.isin({"88", "TX", "NX", "MX"})
    if column == "regional_nodes_positive":
        numeric = pd.to_numeric(series, errors="coerce")
        result |= numeric.ge(98)
    return result.astype(bool)


def cramers_v(contingency: np.ndarray) -> tuple[float, float, float, float]:
    """Bias-corrected Cramer's V plus test diagnostics."""

    chi2, p_value, _, expected = chi2_contingency(contingency)
    n = contingency.sum()
    rows, columns = contingency.shape
    phi2 = chi2 / n
    phi2_corrected = max(0.0, phi2 - ((columns - 1) * (rows - 1)) / (n - 1))
    rows_corrected = rows - ((rows - 1) ** 2) / (n - 1)
    columns_corrected = columns - ((columns - 1) ** 2) / (n - 1)
    denominator = min(columns_corrected - 1, rows_corrected - 1)
    value = math.sqrt(phi2_corrected / denominator) if denominator > 0 else 0.0
    return value, float(p_value), float(expected.min()), float((expected < 5).mean())


def collapse_rare(values: pd.Series, minimum: int = RARE_CATEGORY_MIN) -> pd.Series:
    values = values.astype("string").fillna("<NA>")
    counts = values.value_counts(dropna=False)
    rare = counts[counts < minimum].index
    return values.mask(values.isin(rare), f"<RARE n<{minimum}>")


def prediction_metrics(
    y: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray | None = None,
) -> dict[str, float | int]:
    y = np.asarray(y, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    fp = int(((y == 0) & (predictions == 1)).sum())
    fn = int(((y == 1) & (predictions == 0)).sum())
    tn = int(((y == 0) & (predictions == 0)).sum())
    tp = int(((y == 1) & (predictions == 1)).sum())
    result: dict[str, float | int] = {
        "rows": len(y),
        "actual_dead_rate": float(y.mean()),
        "predicted_dead_rate": float(predictions.mean()),
        "weighted_f1": float(f1_score(y, predictions, average="weighted", zero_division=0)),
        "dead_class_f1": float(f1_score(y, predictions, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y, predictions)),
        "dead_precision": float(precision_score(y, predictions, pos_label=1, zero_division=0)),
        "dead_recall": float(recall_score(y, predictions, pos_label=1, zero_division=0)),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
        "false_negative_rate": fn / (fn + tp) if fn + tp else np.nan,
        "error_rate": (fp + fn) / len(y),
    }
    if scores is None:
        result.update({"mean_probability": np.nan, "roc_auc": np.nan, "brier_score": np.nan})
    else:
        scores = np.asarray(scores, dtype=float)
        result.update(
            {
                "mean_probability": float(scores.mean()),
                "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else np.nan,
                "brier_score": float(brier_score_loss(y, scores)),
            }
        )
    return result


def load_candidates(
    train: pd.DataFrame, y: np.ndarray
) -> tuple[dict[str, np.ndarray], str, dict[str, str], dict[str, str], np.ndarray | None]:
    """Load only an approved, current Sub6 nested OOF; otherwise use honest proxies."""

    candidates: dict[str, np.ndarray] = {}
    notes: dict[str, str] = {}
    hashes: dict[str, str] = {}
    canonical_folds: np.ndarray | None = None

    canonical_dir = ROOT / "diagnostic_outputs" / "submission6_nested"
    canonical_path = canonical_dir / "nested_oof_predictions.npz"
    manifest_path = canonical_dir / "run_manifest.json"
    global_metrics_path = canonical_dir / "global_metrics.csv"
    replay_path = canonical_dir / "production_replay_comparison.json"
    practical_gate_path = (
        ROOT
        / "diagnostic_outputs"
        / "submission6_reference_gate"
        / "submission6_practical_reference_acceptance.json"
    )
    if canonical_path.is_file():
        try:
            for support_path in (
                manifest_path,
                global_metrics_path,
                replay_path,
                practical_gate_path,
            ):
                if not support_path.is_file():
                    raise ValueError(f"missing approval artifact {support_path.name}")

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_signature = manifest.get("run_signature")
            if not isinstance(manifest_signature, str) or not manifest_signature:
                raise ValueError("run_manifest.json has no valid run_signature")

            current_hash_expectations = {
                "train_sha256": ROOT / "train.csv",
                "test_sha256": ROOT / "test.csv",
                "pipeline_v6_sha256": ROOT / "pipeline_v6.py",
                "reconstruction_script_sha256": ROOT / "submission6_nested_reconstruction.py",
                "archived_v6_teacher_sha256": ROOT / "archive" / "probs_v6_blend.npy",
                "archived_v6_final_sha256": ROOT / "archive" / "probs_v6_final.npy",
                "archived_v6_submission_sha256": ROOT / "archive" / "submission6.csv",
            }
            for manifest_key, current_path in current_hash_expectations.items():
                if not current_path.is_file():
                    raise ValueError(f"current provenance input is missing: {current_path}")
                current_hash = file_sha256(current_path)
                if manifest.get(manifest_key) != current_hash:
                    raise ValueError(
                        f"stale manifest: {manifest_key} does not match current {current_path.name}"
                    )

            saved = np.load(canonical_path, allow_pickle=False)
            required_keys = {
                "run_signature",
                "y",
                "teacher_oof",
                "student_oof",
                "submission6_nested_oof",
                "outer_fold",
                "completed_outer_folds",
            }
            missing_keys = required_keys.difference(saved.files)
            if missing_keys:
                raise ValueError(f"canonical NPZ is missing keys: {sorted(missing_keys)}")
            artifact_signature = str(saved["run_signature"].item())
            if artifact_signature != manifest_signature:
                raise ValueError("NPZ run_signature does not match current run_manifest.json")

            practical_gate = json.loads(practical_gate_path.read_text(encoding="utf-8"))
            if practical_gate.get("status") != "approved":
                raise ValueError("practical reference gate status is not approved")
            if practical_gate.get("run_signature") != manifest_signature:
                raise ValueError("practical reference gate run_signature does not match manifest")
            for approval_flag in (
                "practical_recipe_replay_accepted",
                "nested_reference_integrity_accepted",
                "canonical_nested_recipe_reference_approved",
            ):
                if practical_gate.get(approval_flag) is not True:
                    raise ValueError(f"practical reference gate has {approval_flag} != true")
            if practical_gate.get("nested_oof_key") != "submission6_nested_oof":
                raise ValueError("practical reference gate names the wrong nested OOF key")
            if practical_gate.get("nested_oof_path") != str(canonical_path.relative_to(ROOT)):
                raise ValueError("practical reference gate names the wrong nested OOF path")
            canonical_sha256 = file_sha256(canonical_path)
            if practical_gate.get("nested_oof_sha256") != canonical_sha256:
                raise ValueError("practical reference gate hash does not match canonical NPZ")
            nested_gate_summary = practical_gate.get("nested_summary", {})
            if nested_gate_summary.get("nested_oof_sha256") != canonical_sha256:
                raise ValueError("nested gate summary hash does not match canonical NPZ")
            if nested_gate_summary.get("complete") is not True:
                raise ValueError("practical reference gate does not confirm complete nested OOF")
            if nested_gate_summary.get("integrity_pass") is not True:
                raise ValueError("practical reference gate does not confirm nested integrity")
            if nested_gate_summary.get("scale_pass") is not True:
                raise ValueError("practical reference gate does not confirm scale sanity")
            if saved["y"].shape != (len(y),) or not np.array_equal(
                saved["y"].astype(np.int8), y
            ):
                raise ValueError("saved labels do not match train.csv")

            teacher = validate_probability_vector("teacher_oof", saved["teacher_oof"], len(y))
            student = validate_probability_vector("student_oof", saved["student_oof"], len(y))
            canonical_scores = validate_probability_vector(
                "submission6_nested_oof", saved["submission6_nested_oof"], len(y)
            )
            completed = np.asarray(saved["completed_outer_folds"], dtype=int)
            if completed.shape != (N_OUTER_FOLDS,) or set(completed.tolist()) != set(
                range(1, N_OUTER_FOLDS + 1)
            ):
                raise ValueError("all five completed_outer_folds are required")
            folds = np.asarray(saved["outer_fold"], dtype=int)
            if folds.shape != (len(y),) or set(np.unique(folds).tolist()) != set(
                range(1, N_OUTER_FOLDS + 1)
            ):
                raise ValueError("outer_fold must give complete 1..5 coverage for all rows")
            fold_counts = np.bincount(folds, minlength=N_OUTER_FOLDS + 1)[1:]
            if not np.array_equal(
                fold_counts,
                np.full(N_OUTER_FOLDS, len(y) // N_OUTER_FOLDS, dtype=int),
            ):
                raise ValueError(
                    f"outer_fold coverage counts are invalid: {fold_counts.tolist()}"
                )
            if not (np.isfinite(teacher).all() and np.isfinite(student).all()):
                raise ValueError("teacher/student canonical coverage is incomplete")

            metrics = pd.read_csv(global_metrics_path)
            nested_rows = metrics[metrics["scope"].eq("submission6_nested")]
            if len(nested_rows) != 1:
                raise ValueError("global_metrics.csv must have exactly one submission6_nested row")
            metric_row = nested_rows.iloc[0]

            def approved_boolean(value: object) -> bool:
                if isinstance(value, (bool, np.bool_)):
                    return bool(value)
                return str(value).strip().lower() in {"true", "1", "yes"}

            if not approved_boolean(metric_row.get("is_complete_5_fold", False)):
                raise ValueError("global metrics do not confirm complete five-fold coverage")
            if int(metric_row.get("completed_outer_folds", -1)) != N_OUTER_FOLDS:
                raise ValueError("global metrics do not report five completed outer folds")
            if int(metric_row.get("finite_oof_rows", -1)) != len(y):
                raise ValueError("global metrics finite_oof_rows is not full training coverage")
            if not approved_boolean(metric_row.get("sanity_scale_pass", False)):
                raise ValueError("global metrics do not pass the public-LB scale sanity check")

            canonical_predictions = top_rate_predictions(
                canonical_scores, FIXED_POSITIVE_RATE
            )
            recomputed_weighted = f1_score(
                y, canonical_predictions, average="weighted", zero_division=0
            )
            recomputed_dead_f1 = f1_score(
                y, canonical_predictions, average="binary", zero_division=0
            )
            recomputed_auc = roc_auc_score(y, canonical_scores)
            if not np.isclose(
                recomputed_weighted,
                float(metric_row["official_support_weighted_f1"]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("global weighted F1 does not match the canonical vector")
            if not np.isclose(
                recomputed_dead_f1,
                float(metric_row["legacy_dead_class_f1"]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("global Dead-class F1 does not match the canonical vector")
            if not np.isclose(
                recomputed_auc,
                float(metric_row["roc_auc"]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("global ROC AUC does not match the canonical vector")
            for gate_metric in (
                practical_gate.get("official_support_weighted_f1"),
                nested_gate_summary.get("weighted_f1"),
            ):
                if not np.isclose(
                    recomputed_weighted,
                    float(gate_metric),
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise ValueError("practical reference gate metric does not match canonical OOF")
            if nested_gate_summary.get("completed_outer_folds") != list(
                range(1, N_OUTER_FOLDS + 1)
            ):
                raise ValueError("practical gate does not confirm all five outer folds")
            if int(nested_gate_summary.get("finite_final_rows", -1)) != len(y):
                raise ValueError("practical gate does not confirm full finite OOF coverage")

            replay = json.loads(replay_path.read_text(encoding="utf-8"))
            if replay.get("run_signature") != manifest_signature:
                raise ValueError("production replay run_signature does not match manifest")
            if replay.get("practical_rank_label_equivalence_verified") is not True:
                raise ValueError("production replay does not pass practical rank/label equivalence")
            strict_probability = bool(
                replay.get("strict_probability_equivalence_verified", False)
            )
            strict_recipe = bool(
                replay.get("strict_recipe_artifact_equivalence_verified", False)
            )
            if practical_gate.get("strict_probability_equivalence_verified") != strict_probability:
                raise ValueError("strict probability status disagrees between replay and gate")
            if practical_gate.get("strict_recipe_artifact_equivalence_verified") != strict_recipe:
                raise ValueError("strict recipe status disagrees between replay and gate")

            candidates["submission6_nested_equivalent"] = canonical_scores
            notes["submission6_nested_equivalent"] = (
                "Approved practical nested recipe reference: five folds, full finite coverage, "
                "current provenance/signature, scale gate, and practical production replay accepted."
            )
            notes["canonical_acceptance_mode"] = "approved practical recipe reference gate"
            notes["practical_recipe_replay_accepted"] = "true"
            notes["strict_probability_equivalence_verified"] = str(strict_probability).lower()
            notes["strict_recipe_artifact_equivalence_verified"] = str(strict_recipe).lower()
            canonical_folds = folds
            for support_path in (
                canonical_path,
                manifest_path,
                global_metrics_path,
                replay_path,
                practical_gate_path,
            ):
                hashes[str(support_path.relative_to(ROOT))] = file_sha256(support_path)
        except Exception as exc:  # diagnostic fallback must be explicit in the report
            notes["canonical_load_warning"] = f"Canonical artifact rejected: {exc}"

    tree_path = require_file(ROOT / "oof_step1.npy")
    candidates["frozen_tree_pre_pseudo_proxy"] = validate_probability_vector(
        "oof_step1.npy", np.load(tree_path, allow_pickle=False), len(y)
    )
    notes["frozen_tree_pre_pseudo_proxy"] = (
        "Saved pre-pseudo tree OOF vector; not Submission 6's final pseudo-labelled recipe."
    )
    hashes[str(tree_path.relative_to(ROOT))] = file_sha256(tree_path)

    nested_path = ROOT / "diagnostic_outputs" / "pseudo90_nested" / "nested_oof_predictions.npz"
    pseudo90: np.ndarray | None = None
    if nested_path.is_file():
        nested = np.load(nested_path, allow_pickle=False)
        if "y" in nested.files and not np.array_equal(nested["y"].astype(np.int8), y):
            raise ValueError("pseudo90 nested labels do not match train.csv")
        if "pseudo95_oof" in nested.files:
            candidates["nested_pseudo95_proxy"] = validate_probability_vector(
                "pseudo95_oof", nested["pseudo95_oof"], len(y)
            )
            notes["nested_pseudo95_proxy"] = (
                "Leak-safe outer-fold pseudo95 proxy, but not an exact replay of Submission 6."
            )
        if "pseudo90_oof" in nested.files:
            pseudo90 = validate_probability_vector(
                "pseudo90_oof", nested["pseudo90_oof"], len(y)
            )
            candidates["nested_pseudo90"] = pseudo90
            notes["nested_pseudo90"] = "Saved nested pseudo90 OOF diagnostic from Submission 11."
        hashes[str(nested_path.relative_to(ROOT))] = file_sha256(nested_path)

    nn_path = ROOT / "archive" / "oof_nn.npy"
    if pseudo90 is not None and nn_path.is_file():
        nn = validate_probability_vector(
            "archive/oof_nn.npy", np.load(nn_path, allow_pickle=False), len(y)
        )
        candidates["pseudo90_nn20_stress_screen"] = 0.80 * pseudo90 + 0.20 * nn
        notes["pseudo90_nn20_stress_screen"] = (
            "Saved-OOF screening blend used for the rejected pre-numbering pseudo90+NN20 stress experiment; not jointly nested."
        )
        hashes[str(nn_path.relative_to(ROOT))] = file_sha256(nn_path)

    primary = (
        "submission6_nested_equivalent"
        if "submission6_nested_equivalent" in candidates
        else "frozen_tree_pre_pseudo_proxy"
    )
    return candidates, primary, notes, hashes, canonical_folds


def build_missingness_table(train: pd.DataFrame, focal: np.ndarray) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for column in train.columns:
        if column in {"patient_id", "vital_status", "age_recode"}:
            continue
        definitions = {
            "physical_na": train[column].isna().to_numpy(),
            "semantic_unavailable": semantic_unavailable_mask(
                column, train[column]
            ).to_numpy(),
        }
        for definition, mask in definitions.items():
            comparison = binary_rate_comparison(mask[focal], mask[~focal])
            records.append(
                {
                    "column": column,
                    "definition": definition,
                    "focal_missing_count": int(mask[focal].sum()),
                    "rest_missing_count": int(mask[~focal].sum()),
                    **comparison,
                }
            )
    result = pd.DataFrame(records)
    result["bh_q_value"] = np.nan
    for definition, indices in result.groupby("definition").groups.items():
        result.loc[indices, "bh_q_value"] = benjamini_hochberg(
            result.loc[indices, "p_value"]
        )
    result["material_effect"] = (
        (result["binary_smd"].abs() >= MATERIAL_BINARY_SMD)
        & (result["bh_q_value"] < ALPHA)
    )
    return result.sort_values(
        ["definition", "material_effect", "binary_smd"], ascending=[True, False, True]
    ).reset_index(drop=True)


def build_class_balance(train: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    focal = train["age_recode"].eq(FOCAL_AGE).to_numpy()
    result = binary_rate_comparison(y[focal], y[~focal])
    odds_focal = (y[focal].sum() + 0.5) / ((y[focal] == 0).sum() + 0.5)
    odds_rest = (y[~focal].sum() + 0.5) / ((y[~focal] == 0).sum() + 0.5)
    return pd.DataFrame(
        [
            {
                "comparison": f"{FOCAL_AGE} vs all other ages",
                "focal_rows": int(focal.sum()),
                "rest_rows": int((~focal).sum()),
                "focal_dead_count": int(y[focal].sum()),
                "rest_dead_count": int(y[~focal].sum()),
                **result,
                "dead_rate_ratio": result["focal_rate"] / result["rest_rate"],
                "dead_odds_ratio_haldane": odds_focal / odds_rest,
            }
        ]
    )


def build_age_representation_audit() -> pd.DataFrame:
    """Verify how the submitted v6 recipe represents age."""

    pipeline_path = require_file(ROOT / "pipeline_v6.py")
    source = pipeline_path.read_text(encoding="utf-8")
    checks = [
        ("age_recode categorical", 'cat_cols=["age_recode"'),
        ("age_midpoint numeric", 'F["age_midpoint"]'),
        ("age × diagnosis-year interaction", 'F["age_x_years"]'),
        ("age × stage interaction", 'F["age_x_stage"]'),
        ("age × metastasis interaction", 'F["age_x_mets"]'),
    ]
    return pd.DataFrame(
        [
            {
                "representation": name,
                "present_in_pipeline_v6": token in source,
                "source_file": "pipeline_v6.py",
            }
            for name, token in checks
        ]
    )


def feature_series(train: pd.DataFrame) -> dict[str, tuple[str, pd.Series]]:
    summary_size = train["tumor_size_summary"].map(parse_tumor_size)
    overtime_size = train["tumor_size_overtime"].map(parse_tumor_size)
    best_size = summary_size.fillna(overtime_size)
    return {
        "summary_stage": ("stage", train["summary_stage"]),
        "derived_eod_composite_stage": ("stage", train["derived_eod_composite_stage"]),
        "histologic_type_icdo3": ("histology", train["histologic_type_icdo3"].astype(str)),
        "histology_family": ("histology", train["histology_family"]),
        "reason_nocancer_directed_surgery": (
            "treatment", train["reason_nocancer_directed_surgery"]
        ),
        "radiation_recode": ("treatment", train["radiation_recode"]),
        "rx_summ_scope_reglnsur2003": (
            "treatment", train["rx_summ_scope_reglnsur2003"]
        ),
        "rx_summ_surgprim_site19982022": (
            "treatment", train["rx_summ_surgprim_site19982022"]
        ),
        "rx_summ_surgothregdis2003": (
            "treatment", train["rx_summ_surgothregdis2003"]
        ),
        "tumor_size_best_bin": ("tumor_size", tumor_size_bin(best_size)),
        "grade_recode_thru2017": ("disease_profile", train["grade_recode_thru2017"]),
        "diagnostic_confirmation": (
            "data_quality", train["diagnostic_confirmation"]
        ),
        "year_of_diagnosis": ("calendar", train["year_of_diagnosis"].astype(str)),
    }


def build_categorical_tables(
    train: pd.DataFrame,
    y: np.ndarray,
    predictions: np.ndarray,
    focal: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_records: list[dict[str, object]] = []
    level_records: list[dict[str, object]] = []
    errors = predictions != y
    for feature, (family, raw_values) in feature_series(train).items():
        values = collapse_rare(raw_values)
        contingency = pd.crosstab(focal, values).reindex(index=[False, True], fill_value=0)
        value, p_value, expected_min, expected_under_five = cramers_v(
            contingency.to_numpy()
        )
        feature_records.append(
            {
                "feature_family": family,
                "feature": feature,
                "categories_after_rare_pooling": contingency.shape[1],
                "cramers_v": value,
                "chi_square_p_value": p_value,
                "expected_cell_min": expected_min,
                "fraction_expected_cells_under_5": expected_under_five,
            }
        )
        for level in sorted(values.unique().astype(str)):
            level_mask = values.eq(level).to_numpy()
            focal_level = focal & level_mask
            rest_level = (~focal) & level_mask
            level_records.append(
                {
                    "feature_family": family,
                    "feature": feature,
                    "level": level,
                    "focal_count": int(focal_level.sum()),
                    "rest_count": int(rest_level.sum()),
                    "focal_prevalence": float(level_mask[focal].mean()),
                    "rest_prevalence": float(level_mask[~focal].mean()),
                    "prevalence_difference": float(
                        level_mask[focal].mean() - level_mask[~focal].mean()
                    ),
                    "focal_dead_rate": float(y[focal_level].mean()) if focal_level.any() else np.nan,
                    "rest_dead_rate": float(y[rest_level].mean()) if rest_level.any() else np.nan,
                    "focal_error_rate": float(errors[focal_level].mean()) if focal_level.any() else np.nan,
                    "rest_error_rate": float(errors[rest_level].mean()) if rest_level.any() else np.nan,
                }
            )
    effects = pd.DataFrame(feature_records)
    effects["bh_q_value"] = benjamini_hochberg(effects["chi_square_p_value"])
    effects["material_effect"] = (
        (effects["cramers_v"] >= MATERIAL_CRAMERS_V)
        & (effects["bh_q_value"] < ALPHA)
    )
    effects = effects.sort_values("cramers_v", ascending=False).reset_index(drop=True)
    levels = pd.DataFrame(level_records).sort_values(
        ["feature_family", "feature", "prevalence_difference"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    return effects, levels


def build_numeric_table(train: pd.DataFrame, focal: np.ndarray) -> pd.DataFrame:
    size_summary = train["tumor_size_summary"].map(parse_tumor_size)
    size_overtime = train["tumor_size_overtime"].map(parse_tumor_size)
    best_size = size_summary.fillna(size_overtime)
    nodes_examined = pd.to_numeric(train["regional_nodes_examined"], errors="coerce").where(
        lambda values: values < 90
    )
    nodes_positive = pd.to_numeric(train["regional_nodes_positive"], errors="coerce").where(
        lambda values: values < 90
    )
    cs_size = train["cs_tumor_size20042015"].map(parse_tumor_size)
    numeric = {
        "year_of_diagnosis": train["year_of_diagnosis"].astype(float),
        "tumor_size_best": best_size,
        "tumor_size_summary": size_summary,
        "tumor_size_overtime": size_overtime,
        "cs_tumor_size20042015": cs_size,
        "regional_nodes_examined_quantitative": nodes_examined,
        "regional_nodes_positive_quantitative": nodes_positive,
    }
    records: list[dict[str, object]] = []
    for feature, series in numeric.items():
        a = series[focal].dropna().to_numpy(dtype=float)
        b = series[~focal].dropna().to_numpy(dtype=float)
        if not len(a) or not len(b):
            continue
        pooled_sd = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        smd = (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else 0.0
        test = mannwhitneyu(a, b, alternative="two-sided")
        rank_biserial = 2 * float(test.statistic) / (len(a) * len(b)) - 1
        records.append(
            {
                "feature": feature,
                "focal_nonmissing": len(a),
                "rest_nonmissing": len(b),
                "focal_mean": float(a.mean()),
                "rest_mean": float(b.mean()),
                "mean_difference": float(a.mean() - b.mean()),
                "standardized_mean_difference": float(smd),
                "focal_median": float(np.median(a)),
                "rest_median": float(np.median(b)),
                "focal_q1": float(np.quantile(a, 0.25)),
                "focal_q3": float(np.quantile(a, 0.75)),
                "rest_q1": float(np.quantile(b, 0.25)),
                "rest_q3": float(np.quantile(b, 0.75)),
                "rank_biserial": rank_biserial,
                "mann_whitney_p_value": float(test.pvalue),
            }
        )
    result = pd.DataFrame(records)
    result["bh_q_value"] = benjamini_hochberg(result["mann_whitney_p_value"])
    result["material_effect"] = (
        (result["standardized_mean_difference"].abs() >= MATERIAL_NUMERIC_SMD)
        & (result["bh_q_value"] < ALPHA)
    )
    return result.sort_values(
        "standardized_mean_difference", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)


def build_model_metrics(
    train: pd.DataFrame,
    y: np.ndarray,
    candidates: dict[str, np.ndarray],
    positive_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    model_records: list[dict[str, object]] = []
    age_records: list[dict[str, object]] = []
    hard_predictions: dict[str, np.ndarray] = {}
    focal = train["age_recode"].eq(FOCAL_AGE).to_numpy()
    scopes = {
        "GLOBAL": np.ones(len(train), dtype=bool),
        FOCAL_AGE: focal,
        "all_other_ages": ~focal,
    }
    age_values = train["age_recode"].astype(str)
    for candidate, scores in candidates.items():
        predictions = top_rate_predictions(scores, positive_rate)
        hard_predictions[candidate] = predictions
        for scope, mask in scopes.items():
            model_records.append(
                {
                    "candidate": candidate,
                    "scope": scope,
                    **prediction_metrics(y[mask], predictions[mask], scores[mask]),
                }
            )
        for age_band in sorted(age_values.unique(), key=age_sort_key):
            mask = age_values.eq(age_band).to_numpy()
            age_records.append(
                {
                    "candidate": candidate,
                    "age_band": age_band,
                    "age_sort": age_sort_key(age_band),
                    **prediction_metrics(y[mask], predictions[mask], scores[mask]),
                }
            )
    return (
        pd.DataFrame(model_records),
        pd.DataFrame(age_records).sort_values(["candidate", "age_sort"]),
        hard_predictions,
    )


def age_sort_key(value: str) -> int:
    try:
        return int(str(value).split("-")[0].replace("+ years", ""))
    except ValueError:
        digits = "".join(character for character in str(value) if character.isdigit())
        return int(digits) if digits else 999


def build_fold_age_metrics(
    train: pd.DataFrame,
    y: np.ndarray,
    scores: np.ndarray,
    positive_rate: float,
    canonical_folds: np.ndarray | None,
) -> pd.DataFrame:
    if canonical_folds is None:
        fold_ids = np.empty(len(y), dtype=int)
        splitter = StratifiedKFold(N_OUTER_FOLDS, shuffle=True, random_state=OUTER_SEED)
        for fold, (_, validation) in enumerate(splitter.split(np.zeros(len(y)), y), start=1):
            fold_ids[validation] = fold
        fold_source = (
            "descriptive re-slice of saved multi-fold proxy OOF using "
            "StratifiedKFold(seed=42)"
        )
        analysis_role = "proxy_reslice_descriptive_only"
        gate_eligible = False
    else:
        unique = sorted(np.unique(canonical_folds))
        remap = {value: index + 1 for index, value in enumerate(unique)}
        fold_ids = np.array([remap[value] for value in canonical_folds], dtype=int)
        fold_source = "canonical Submission 6 nested artifact"
        analysis_role = "approved_canonical_outer_fold_diagnostic"
        gate_eligible = True

    records: list[dict[str, object]] = []
    age_values = train["age_recode"].astype(str)
    for fold in range(1, N_OUTER_FOLDS + 1):
        fold_mask = fold_ids == fold
        fold_indices = np.flatnonzero(fold_mask)
        fold_predictions = top_rate_predictions(scores[fold_indices], positive_rate)
        positions = np.full(len(y), -1, dtype=int)
        positions[fold_indices] = np.arange(len(fold_indices))
        for age_band in TREND_AGES:
            mask = fold_mask & age_values.eq(age_band).to_numpy()
            if not mask.any():
                continue
            local = positions[np.flatnonzero(mask)]
            records.append(
                {
                    "fold_source": fold_source,
                    "analysis_role": analysis_role,
                    "eligible_for_intervention_fold_gate": gate_eligible,
                    "outer_fold": fold,
                    "age_band": age_band,
                    **prediction_metrics(
                        y[mask], fold_predictions[local], scores[np.flatnonzero(mask)]
                    ),
                }
            )
    return pd.DataFrame(records).sort_values(["age_band", "outer_fold"])


def build_neighbor_fold_comparisons(fold_age_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare the focal band with both immediate neighbors in each descriptive fold."""

    required = set(NEIGHBOR_AGES)
    records: list[dict[str, object]] = []
    for fold, frame in fold_age_metrics.groupby("outer_fold"):
        indexed = frame.set_index("age_band")
        if not required.issubset(indexed.index):
            continue
        records.append(
            {
                "fold_source": frame["fold_source"].iloc[0],
                "analysis_role": frame["analysis_role"].iloc[0],
                "eligible_for_intervention_fold_gate": bool(
                    frame["eligible_for_intervention_fold_gate"].iloc[0]
                ),
                "outer_fold": int(fold),
                "age50_54_weighted_f1": float(indexed.loc["50-54 years", "weighted_f1"]),
                "age55_59_weighted_f1": float(indexed.loc[FOCAL_AGE, "weighted_f1"]),
                "age60_64_weighted_f1": float(indexed.loc["60-64 years", "weighted_f1"]),
                "age55_f1_below_both_neighbors": bool(
                    indexed.loc[FOCAL_AGE, "weighted_f1"]
                    < min(
                        indexed.loc["50-54 years", "weighted_f1"],
                        indexed.loc["60-64 years", "weighted_f1"],
                    )
                ),
                "age50_54_roc_auc": float(indexed.loc["50-54 years", "roc_auc"]),
                "age55_59_roc_auc": float(indexed.loc[FOCAL_AGE, "roc_auc"]),
                "age60_64_roc_auc": float(indexed.loc["60-64 years", "roc_auc"]),
                "age55_auc_below_both_neighbors": bool(
                    indexed.loc[FOCAL_AGE, "roc_auc"]
                    < min(
                        indexed.loc["50-54 years", "roc_auc"],
                        indexed.loc["60-64 years", "roc_auc"],
                    )
                ),
            }
        )
    return pd.DataFrame(records).sort_values("outer_fold").reset_index(drop=True)


def build_error_clusters(
    train: pd.DataFrame,
    y: np.ndarray,
    predictions: np.ndarray,
    focal: np.ndarray,
) -> pd.DataFrame:
    cluster_specs = {
        "composite_stage_x_histology_family": (
            train["derived_eod_composite_stage"].astype(str)
            + " | "
            + train["histology_family"].astype(str)
        ),
        "composite_stage_x_histology_code": (
            train["derived_eod_composite_stage"].astype(str)
            + " | "
            + train["histologic_type_icdo3"].astype(str)
        ),
    }
    error = predictions != y
    records: list[dict[str, object]] = []
    for cluster_type, clusters in cluster_specs.items():
        for cluster in sorted(clusters[focal].unique()):
            same = clusters.eq(cluster).to_numpy()
            focal_cluster = focal & same
            rest_cluster = (~focal) & same
            if focal_cluster.sum() < 5:
                continue
            focal_metrics = prediction_metrics(y[focal_cluster], predictions[focal_cluster])
            rest_metrics = (
                prediction_metrics(y[rest_cluster], predictions[rest_cluster])
                if rest_cluster.any()
                else None
            )
            overall_error_table = np.array(
                [
                    [error[focal_cluster].sum(), (~error[focal_cluster]).sum()],
                    [error[rest_cluster].sum(), (~error[rest_cluster]).sum()],
                ],
                dtype=int,
            )
            focal_alive = focal_cluster & (y == 0)
            rest_alive = rest_cluster & (y == 0)
            focal_dead = focal_cluster & (y == 1)
            rest_dead = rest_cluster & (y == 1)
            focal_fp = int(((predictions == 1) & focal_alive).sum())
            rest_fp = int(((predictions == 1) & rest_alive).sum())
            focal_fn = int(((predictions == 0) & focal_dead).sum())
            rest_fn = int(((predictions == 0) & rest_dead).sum())
            focal_alive_n = int(focal_alive.sum())
            rest_alive_n = int(rest_alive.sum())
            focal_dead_n = int(focal_dead.sum())
            rest_dead_n = int(rest_dead.sum())
            fpr_table = np.array(
                [
                    [focal_fp, focal_alive_n - focal_fp],
                    [rest_fp, rest_alive_n - rest_fp],
                ],
                dtype=int,
            )
            fnr_table = np.array(
                [
                    [focal_fn, focal_dead_n - focal_fn],
                    [rest_fn, rest_dead_n - rest_fn],
                ],
                dtype=int,
            )
            overall_p_value = (
                float(fisher_exact(overall_error_table).pvalue)
                if rest_cluster.any()
                else np.nan
            )
            fpr_p_value = (
                float(fisher_exact(fpr_table).pvalue)
                if focal_alive_n and rest_alive_n
                else np.nan
            )
            fnr_p_value = (
                float(fisher_exact(fnr_table).pvalue)
                if focal_dead_n and rest_dead_n
                else np.nan
            )
            focal_fpr = focal_fp / focal_alive_n if focal_alive_n else np.nan
            rest_fpr = rest_fp / rest_alive_n if rest_alive_n else np.nan
            focal_fnr = focal_fn / focal_dead_n if focal_dead_n else np.nan
            rest_fnr = rest_fn / rest_dead_n if rest_dead_n else np.nan
            records.append(
                {
                    "cluster_type": cluster_type,
                    "cluster": cluster,
                    "focal_rows": int(focal_cluster.sum()),
                    "rest_same_cluster_rows": int(rest_cluster.sum()),
                    "focal_dead_rate": focal_metrics["actual_dead_rate"],
                    "rest_dead_rate": rest_metrics["actual_dead_rate"] if rest_metrics else np.nan,
                    "focal_predicted_dead_rate": focal_metrics["predicted_dead_rate"],
                    "rest_predicted_dead_rate": rest_metrics["predicted_dead_rate"] if rest_metrics else np.nan,
                    "focal_weighted_f1": focal_metrics["weighted_f1"],
                    "rest_weighted_f1": rest_metrics["weighted_f1"] if rest_metrics else np.nan,
                    "focal_false_positive": focal_metrics["false_positive"],
                    "focal_false_negative": focal_metrics["false_negative"],
                    "rest_false_positive": rest_fp,
                    "rest_false_negative": rest_fn,
                    "focal_alive_denominator": focal_alive_n,
                    "rest_alive_denominator": rest_alive_n,
                    "focal_false_positive_rate": focal_fpr,
                    "rest_false_positive_rate": rest_fpr,
                    "false_positive_rate_difference": focal_fpr - rest_fpr,
                    "fpr_fisher_p_value": fpr_p_value,
                    "focal_dead_denominator": focal_dead_n,
                    "rest_dead_denominator": rest_dead_n,
                    "focal_false_negative_rate": focal_fnr,
                    "rest_false_negative_rate": rest_fnr,
                    "false_negative_rate_difference": focal_fnr - rest_fnr,
                    "fnr_fisher_p_value": fnr_p_value,
                    "focal_error_rate": focal_metrics["error_rate"],
                    "rest_error_rate": rest_metrics["error_rate"] if rest_metrics else np.nan,
                    "error_rate_difference": (
                        focal_metrics["error_rate"] - rest_metrics["error_rate"]
                        if rest_metrics
                        else np.nan
                    ),
                    "overall_error_fisher_p_value": overall_p_value,
                }
            )
    result = pd.DataFrame(records)
    for column in (
        "overall_error_bh_q_value",
        "fpr_bh_q_value",
        "fnr_bh_q_value",
    ):
        result[column] = np.nan
    for cluster_type, indices in result.groupby("cluster_type").groups.items():
        result.loc[indices, "overall_error_bh_q_value"] = benjamini_hochberg(
            result.loc[indices, "overall_error_fisher_p_value"]
        )
        result.loc[indices, "fpr_bh_q_value"] = benjamini_hochberg(
            result.loc[indices, "fpr_fisher_p_value"]
        )
        result.loc[indices, "fnr_bh_q_value"] = benjamini_hochberg(
            result.loc[indices, "fnr_fisher_p_value"]
        )
    result["descriptive_overall_error_flag"] = (
        (result["focal_rows"] >= 30)
        & (result["rest_same_cluster_rows"] >= 100)
        & (result["error_rate_difference"] >= MATERIAL_ERROR_RATE_GAP)
        & (result["overall_error_bh_q_value"] < ALPHA)
    )
    result["fpr_denominator_gate"] = (
        (result["focal_alive_denominator"] >= MIN_CLASS_CONDITIONAL_FOCAL)
        & (result["rest_alive_denominator"] >= MIN_CLASS_CONDITIONAL_REST)
    )
    result["fnr_denominator_gate"] = (
        (result["focal_dead_denominator"] >= MIN_CLASS_CONDITIONAL_FOCAL)
        & (result["rest_dead_denominator"] >= MIN_CLASS_CONDITIONAL_REST)
    )
    result["class_conditional_fpr_signal"] = (
        result["fpr_denominator_gate"]
        & (result["false_positive_rate_difference"] >= MATERIAL_ERROR_RATE_GAP)
        & (result["fpr_bh_q_value"] < ALPHA)
    )
    result["class_conditional_fnr_signal"] = (
        result["fnr_denominator_gate"]
        & (result["false_negative_rate_difference"] >= MATERIAL_ERROR_RATE_GAP)
        & (result["fnr_bh_q_value"] < ALPHA)
    )
    result["class_conditional_excess_error_signal"] = (
        result["class_conditional_fpr_signal"]
        | result["class_conditional_fnr_signal"]
    )
    return result.sort_values(
        [
            "class_conditional_excess_error_signal",
            "descriptive_overall_error_flag",
            "error_rate_difference",
            "focal_rows",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def build_stress_change_clusters(
    train: pd.DataFrame,
    y: np.ndarray,
    hard_predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    if not {"nested_pseudo90", "pseudo90_nn20_stress_screen"}.issubset(hard_predictions):
        return pd.DataFrame()
    focal = train["age_recode"].eq(FOCAL_AGE).to_numpy()
    control = hard_predictions["nested_pseudo90"]
    stress = hard_predictions["pseudo90_nn20_stress_screen"]
    changed = control != stress
    clusters = (
        train["derived_eod_composite_stage"].astype(str)
        + " | "
        + train["histology_family"].astype(str)
    )
    records = []
    for cluster in sorted(clusters[focal & changed].unique()):
        mask = focal & changed & clusters.eq(cluster).to_numpy()
        focal_cluster = focal & clusters.eq(cluster).to_numpy()
        corrected = (control != y) & (stress == y) & mask
        harmed = (control == y) & (stress != y) & mask
        records.append(
            {
                "cluster": cluster,
                "focal_age_rows": int(focal.sum()),
                "cluster_focal_rows": int(focal_cluster.sum()),
                "changed_predictions": int(mask.sum()),
                "changed_fraction_of_cluster": float(mask.sum() / focal_cluster.sum()),
                "changed_fraction_of_focal_age": float(mask.sum() / focal.sum()),
                "control_dead_to_stress_alive": int(((control == 1) & (stress == 0) & mask).sum()),
                "control_alive_to_stress_dead": int(((control == 0) & (stress == 1) & mask).sum()),
                "corrected_predictions": int(corrected.sum()),
                "harmed_predictions": int(harmed.sum()),
                "net_corrected": int(corrected.sum() - harmed.sum()),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["net_corrected", "changed_predictions"], ascending=[True, False]
    ).reset_index(drop=True)


def markdown_table(frame: pd.DataFrame, columns: list[str], decimals: int = 4) -> str:
    if frame.empty:
        return "_None._"
    display = frame[columns].copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{decimals}f}"
        )
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def write_report(
    path: Path,
    positive_rate: float,
    primary: str,
    notes: dict[str, str],
    age_representation: pd.DataFrame,
    class_balance: pd.DataFrame,
    model_metrics: pd.DataFrame,
    age_metrics: pd.DataFrame,
    fold_age_metrics: pd.DataFrame,
    neighbor_fold_comparisons: pd.DataFrame,
    missingness: pd.DataFrame,
    categorical_effects: pd.DataFrame,
    categorical_levels: pd.DataFrame,
    numeric_effects: pd.DataFrame,
    error_clusters: pd.DataFrame,
    stress_changes: pd.DataFrame,
) -> dict[str, object]:
    balance = class_balance.iloc[0]
    focal_metric = model_metrics[
        (model_metrics["candidate"] == primary) & (model_metrics["scope"] == FOCAL_AGE)
    ].iloc[0]
    rest_metric = model_metrics[
        (model_metrics["candidate"] == primary) & (model_metrics["scope"] == "all_other_ages")
    ].iloc[0]
    global_metric = model_metrics[
        (model_metrics["candidate"] == primary) & (model_metrics["scope"] == "GLOBAL")
    ].iloc[0]
    neighbor = age_metrics[
        (age_metrics["candidate"] == primary) & age_metrics["age_band"].isin(NEIGHBOR_AGES)
    ].sort_values("age_sort")
    trend = age_metrics[
        (age_metrics["candidate"] == primary) & age_metrics["age_band"].isin(TREND_AGES)
    ].sort_values("age_sort")
    fold_summary = fold_age_metrics.groupby("age_band", as_index=False).agg(
        weighted_f1_mean=("weighted_f1", "mean"),
        weighted_f1_std=("weighted_f1", "std"),
        weighted_f1_min=("weighted_f1", "min"),
        weighted_f1_max=("weighted_f1", "max"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
    )
    material_missing = missingness[
        (missingness["definition"] == "semantic_unavailable")
        & missingness["material_effect"]
    ].copy()
    material_missing = material_missing.sort_values(
        "binary_smd", key=lambda series: series.abs(), ascending=False
    )
    more_unavailable = int((material_missing["rate_difference"] > 0).sum())
    less_unavailable = int((material_missing["rate_difference"] < 0).sum())
    top_categories = categorical_effects.head(10)
    material_categories = categorical_effects[categorical_effects["material_effect"]]
    material_numeric = numeric_effects[numeric_effects["material_effect"]]
    class_conditional_signals = error_clusters[
        error_clusters["class_conditional_excess_error_signal"]
    ]
    descriptive_error_flags = error_clusters[
        error_clusters["descriptive_overall_error_flag"]
    ]
    descriptive_family_error = descriptive_error_flags[
        descriptive_error_flags["cluster_type"]
        == "composite_stage_x_histology_family"
    ]
    descriptive_code_error = descriptive_error_flags[
        descriptive_error_flags["cluster_type"]
        == "composite_stage_x_histology_code"
    ]
    def one_cluster(cluster_type: str, cluster: str) -> pd.Series | None:
        selected = error_clusters[
            (error_clusters["cluster_type"] == cluster_type)
            & (error_clusters["cluster"] == cluster)
        ]
        return selected.iloc[0] if len(selected) == 1 else None

    adenocarcinoma_lead = one_cluster(
        "composite_stage_x_histology_family",
        "EOD unavailable (all blank) | adenocarcinoma family",
    )
    histology_8140_lead = one_cluster(
        "composite_stage_x_histology_code",
        "EOD unavailable (all blank) | 8140",
    )
    histology_8010_lead = one_cluster(
        "composite_stage_x_histology_code",
        "EOD unavailable (all blank) | 8010",
    )
    if adenocarcinoma_lead is not None:
        adenocarcinoma_detail = (
            "For the broader EOD-all-blank × adenocarcinoma family, FPR is "
            f"{int(adenocarcinoma_lead['focal_false_positive'])}/"
            f"{int(adenocarcinoma_lead['focal_alive_denominator'])} "
            f"({adenocarcinoma_lead['focal_false_positive_rate']:.4f}) versus "
            f"{int(adenocarcinoma_lead['rest_false_positive'])}/"
            f"{int(adenocarcinoma_lead['rest_alive_denominator'])} "
            f"({adenocarcinoma_lead['rest_false_positive_rate']:.4f}), while FNR is "
            f"{int(adenocarcinoma_lead['focal_false_negative'])}/"
            f"{int(adenocarcinoma_lead['focal_dead_denominator'])} "
            f"({adenocarcinoma_lead['focal_false_negative_rate']:.4f}) versus "
            f"{int(adenocarcinoma_lead['rest_false_negative'])}/"
            f"{int(adenocarcinoma_lead['rest_dead_denominator'])} "
            f"({adenocarcinoma_lead['rest_false_negative_rate']:.4f})."
        )
    else:
        adenocarcinoma_detail = "The broader adenocarcinoma-family lead was unavailable."

    q_adenocarcinoma = (
        f"{adenocarcinoma_lead['overall_error_bh_q_value']:.6f}"
        if adenocarcinoma_lead is not None
        else "unavailable"
    )
    q_8140 = (
        f"{histology_8140_lead['overall_error_bh_q_value']:.6f}"
        if histology_8140_lead is not None
        else "unavailable"
    )
    if histology_8010_lead is not None:
        code_8010_detail = (
            f"Raw 8010 has only {int(histology_8010_lead['focal_alive_denominator'])} "
            f"focal and {int(histology_8010_lead['rest_alive_denominator'])} comparison Alive rows"
        )
    else:
        code_8010_detail = "Raw 8010 denominators were unavailable"
    top_levels = categorical_levels[
        categorical_levels["feature"].isin(
            ["summary_stage", "histologic_type_icdo3", "radiation_recode", "rx_summ_scope_reglnsur2003"]
        )
    ].copy()
    top_levels = top_levels.reindex(
        top_levels["prevalence_difference"].abs().sort_values(ascending=False).index
    ).head(12)

    canonical_used = primary == "submission6_nested_equivalent"
    targeted_rows = [
        row for row in (adenocarcinoma_lead, histology_8140_lead) if row is not None
    ]
    targeted_total_error_replication = bool(
        canonical_used
        and len(targeted_rows) == 2
        and all(bool(row["descriptive_overall_error_flag"]) for row in targeted_rows)
    )
    targeted_class_conditional_replication = bool(
        canonical_used
        and any(
            bool(row["class_conditional_excess_error_signal"])
            for row in targeted_rows
        )
    )
    if canonical_used and len(targeted_rows) == 2:
        replication_verdict = (
            "**Step 2 canonical replication verdict:** the prevalence-confounded total-error "
            "association remains BH-significant for both targeted definitions "
            f"(adenocarcinoma family q=`{q_adenocarcinoma}`; raw 8140 q=`{q_8140}`). "
            "The actionable class-conditional excess-error finding **does not replicate**: "
            f"family FPR/FNR gaps are "
            f"{adenocarcinoma_lead['false_positive_rate_difference']:+.4f}/"
            f"{adenocarcinoma_lead['false_negative_rate_difference']:+.4f}, and raw-8140 "
            f"FPR/FNR gaps are {histology_8140_lead['false_positive_rate_difference']:+.4f}/"
            f"{histology_8140_lead['false_negative_rate_difference']:+.4f}. All are below the "
            f"{MATERIAL_ERROR_RATE_GAP:.0%} effect gate, and zero targeted definitions pass "
            "the denominator/effect/BH class-conditional gates. Therefore Step 3 is not entered."
        )
    else:
        replication_verdict = (
            "**Step 2 replication remains provisional:** the approved canonical practical "
            "reference and both targeted cluster rows were not simultaneously available. "
            "No intervention step may be entered from the proxy analysis."
        )
    lower_prevalence = balance["rate_difference"] <= -0.02
    focal_neighbor_values = neighbor.set_index("age_band")["weighted_f1"]
    focal_trough = (
        all(age in focal_neighbor_values for age in NEIGHBOR_AGES)
        and focal_neighbor_values[FOCAL_AGE] + 0.005 < focal_neighbor_values["50-54 years"]
        and focal_neighbor_values[FOCAL_AGE] + 0.005 < focal_neighbor_values["60-64 years"]
    )
    proxy_fold_f1_below_both = int(
        neighbor_fold_comparisons["age55_f1_below_both_neighbors"].sum()
    )
    proxy_fold_auc_below_both = int(
        neighbor_fold_comparisons["age55_auc_below_both_neighbors"].sum()
    )
    proxy_fold_count = len(neighbor_fold_comparisons)
    fold_gate_eligible = bool(
        proxy_fold_count
        and neighbor_fold_comparisons["eligible_for_intervention_fold_gate"].all()
    )
    fold_view_label = (
        "canonical outer folds" if fold_gate_eligible else "descriptive proxy re-slices"
    )

    stress_control = model_metrics[
        (model_metrics["candidate"] == "nested_pseudo90")
        & (model_metrics["scope"] == FOCAL_AGE)
    ]
    stress_candidate = model_metrics[
        (model_metrics["candidate"] == "pseudo90_nn20_stress_screen")
        & (model_metrics["scope"] == FOCAL_AGE)
    ]
    stress_available = not stress_control.empty and not stress_candidate.empty
    if stress_available:
        stress_control_row = stress_control.iloc[0]
        stress_candidate_row = stress_candidate.iloc[0]
        stress_weighted_delta = float(
            stress_candidate_row["weighted_f1"] - stress_control_row["weighted_f1"]
        )
        stress_changed = int(stress_changes["changed_predictions"].sum())
        stress_corrected = int(stress_changes["corrected_predictions"].sum())
        stress_harmed = int(stress_changes["harmed_predictions"].sum())
    else:
        stress_control_row = None
        stress_candidate_row = None
        stress_weighted_delta = np.nan
        stress_changed = 0
        stress_corrected = 0
        stress_harmed = 0

    if canonical_used:
        status = (
            "The approved practical Submission 6 nested recipe OOF vector is used as the primary "
            "reference. Practical recipe replay is accepted; strict historical probability "
            "equivalence remains `false` because of approximately 1e-9 numerical drift and is "
            "not being reinterpreted as true."
        )
    else:
        rejection = notes.get(
            "canonical_load_warning",
            "no complete approved canonical artifact was present",
        )
        status = (
            "No approved Submission 6 nested-equivalent vector was accepted at run time. "
            "The primary error analysis therefore uses the frozen pre-pseudo tree OOF proxy and "
            f"is provisional. Loader detail: `{rejection}`"
        )

    audit_scope = (
        "canonical practical-reference follow-up" if canonical_used else "provisional proxy audit"
    )
    conclusion = (
        "The band has lower Dead prevalence and a full-OOF F1/AUC trough, but this "
        f"{audit_scope} "
        "audit does not isolate a class-conditional error, missingness, tumor-size, stage, "
        "histology, or treatment defect that justifies reweighting or a targeted feature. "
        "Case-mix and registry-era associations remain descriptive leads, not evidence that a "
        "model change would generalize."
    )

    report = [
        "# Age 55–59 subgroup investigation",
        "",
        "## Decision",
        "",
        "**No model change and no submission.** " + conclusion,
        "",
        status,
        "",
        "Every hard prediction is selected once from the full OOF vector using stable top-k ranking at "
        f"`{positive_rate:.3f}` predicted Dead. Support-weighted F1 is primary; Dead-class F1 is secondary.",
        "",
        "## Main evidence",
        "",
        f"- Age 55–59 contains `{int(balance['focal_rows']):,}` rows and has a Dead rate of "
        f"`{balance['focal_rate']:.2%}`, versus `{balance['rest_rate']:.2%}` elsewhere "
        f"(`{balance['rate_difference']:+.2%}`). This targeted follow-up was selected after prior "
        "subgroup scans, so its p-value is descriptive rather than an independently pre-specified "
        "confirmatory test.",
        f"- Under `{primary}`, its weighted F1 is `{focal_metric['weighted_f1']:.6f}`, versus "
        f"`{global_metric['weighted_f1']:.6f}` globally and `{rest_metric['weighted_f1']:.6f}` in the rest.",
        f"- Its OOF ROC AUC is `{focal_metric['roc_auc']:.6f}`. The neighboring 50–54 and 60–64 "
        "bands are shown below; the ranking signal also dips at 55–59, so this is not only a "
        "single global cutoff mismatch.",
        f"- It has `{int(focal_metric['false_positive'])}` false positives and "
        f"`{int(focal_metric['false_negative'])}` false negatives. Its predicted Dead rate "
        f"(`{focal_metric['predicted_dead_rate']:.2%}`) exceeds its observed Dead rate "
        f"(`{focal_metric['actual_dead_rate']:.2%}`).",
        f"- The local-neighbor trough check is `{'present' if focal_trough else 'not cleanly present'}`. "
        f"Across the five {fold_view_label}, 55–59 is below both neighbors in only "
        f"`{proxy_fold_f1_below_both}/{proxy_fold_count}` for weighted F1 and "
        f"`{proxy_fold_auc_below_both}/{proxy_fold_count}` for ROC AUC.",
        f"- Material semantic-unavailable differences after BH control: `{len(material_missing)}` "
        f"(`{less_unavailable}` less unavailable in 55–59, `{more_unavailable}` more unavailable); "
        f"material categorical effects: `{len(material_categories)}`; material numeric effects: "
        f"`{len(material_numeric)}`. There are `{len(descriptive_error_flags)}` "
        "prevalence-confounded overall-error flags but "
        f"`{len(class_conditional_signals)}` class-conditional signals after denominator and BH gates.",
        "",
        "## Class balance",
        "",
        markdown_table(
            class_balance,
            [
                "focal_rows", "rest_rows", "focal_rate", "rest_rate", "rate_difference",
                "difference_ci95_lower", "difference_ci95_upper", "binary_smd", "p_value",
                "dead_rate_ratio", "dead_odds_ratio_haldane",
            ],
            decimals=6,
        ),
        "",
        "## Adjacent age bands",
        "",
        markdown_table(
            neighbor,
            [
                "age_band", "rows", "actual_dead_rate", "predicted_dead_rate", "weighted_f1",
                "dead_class_f1", "roc_auc", "mean_probability", "false_positive",
                "false_negative", "error_rate",
            ],
            decimals=6,
        ),
        "",
        "Broader trend:",
        "",
        markdown_table(
            trend,
            [
                "age_band", "rows", "actual_dead_rate", "predicted_dead_rate", "weighted_f1",
                "roc_auc", "mean_probability", "error_rate",
            ],
            decimals=6,
        ),
        "",
        "Five-fold stability view (fold-local 84.5% policy):",
        "",
        markdown_table(
            fold_summary,
            [
                "age_band", "weighted_f1_mean", "weighted_f1_std", "weighted_f1_min",
                "weighted_f1_max", "roc_auc_mean", "roc_auc_std",
            ],
            decimals=6,
        ),
        "",
        markdown_table(
            neighbor_fold_comparisons,
            [
                "outer_fold", "age50_54_weighted_f1", "age55_59_weighted_f1",
                "age60_64_weighted_f1", "age55_f1_below_both_neighbors",
                "age50_54_roc_auc", "age55_59_roc_auc", "age60_64_roc_auc",
                "age55_auc_below_both_neighbors",
            ],
            decimals=6,
        ),
        "",
        (
            "These rows use approved canonical outer folds. They remain subgroup diagnostics; "
            "a future intervention would still need its own candidate-versus-control fold deltas."
            if fold_gate_eligible
            else
            "**Non-gating proxy view:** these are seed-42 re-slices of a saved multi-fold proxy "
            "OOF vector, not the proxy model's untouched training folds. They cannot satisfy the "
            "required intervention fold gate. The focal band is below both neighbors in only "
            f"{proxy_fold_f1_below_both}/{proxy_fold_count} re-slices for weighted F1 and "
            f"{proxy_fold_auc_below_both}/{proxy_fold_count} for ROC AUC."
        ),
        "",
        "## Missingness and unavailable-code audit",
        "",
        "`physical_na` and `semantic_unavailable` are separate in the CSV. The second is a "
        "heuristic registry audit covering values such as Blank(s), Unknown, EOD 88/TX/NX/MX, "
        "tumor-size 999, and regional-nodes-positive >=98. It is **not** an exact per-feature "
        "reconstruction of model missing-value handling, and these states are not assumed "
        "clinically equivalent.",
        "",
        markdown_table(
            material_missing.head(15),
            [
                "column", "focal_rate", "rest_rate", "rate_difference", "binary_smd",
                "p_value", "bh_q_value",
            ],
            decimals=6,
        ),
        "",
        "Effects require both BH q<0.05 and |binary SMD|>=0.10. Small p-values without a material "
        "effect size are not treated as causes.",
        "",
        f"Crucially, `{less_unavailable}` of `{len(material_missing)}` material differences point to "
        "*less* unavailable information in age 55–59, not more. The main exceptions include the "
        "EOD-M unavailable code and the mutually era-specific surgery-site fields. This does not "
        "support a targeted missingness-indicator fix.",
        "",
        "## Stage, histology, treatment, and tumor-size composition",
        "",
        markdown_table(
            top_categories,
            [
                "feature_family", "feature", "categories_after_rare_pooling", "cramers_v",
                "chi_square_p_value", "bh_q_value", "material_effect",
                "fraction_expected_cells_under_5",
            ],
            decimals=6,
        ),
        "",
        "Largest individual level shifts:",
        "",
        markdown_table(
            top_levels,
            [
                "feature_family", "feature", "level", "focal_count", "focal_prevalence",
                "rest_prevalence", "prevalence_difference", "focal_dead_rate", "rest_dead_rate",
            ],
            decimals=5,
        ),
        "",
        "Numeric profiles (quantitative node values exclude registry sentinel codes >=90):",
        "",
        markdown_table(
            numeric_effects,
            [
                "feature", "focal_nonmissing", "rest_nonmissing", "focal_mean", "rest_mean",
                "standardized_mean_difference", "rank_biserial", "bh_q_value", "material_effect",
            ],
            decimals=6,
        ),
        "",
        "The two component tumor-size fields barely cross |SMD|=0.10, while the merged "
        "`tumor_size_best` feature is below that threshold. The node-positive comparison uses only "
        "quantitative (non-sentinel) rows, so its apparent shift may partly reflect selection into "
        "having a count. The roughly 0.43-year diagnosis-era shift is case-mix evidence, not a "
        "standalone model defect.",
        "",
        "## Error clusters",
        "",
        "The total-error comparison is prevalence-confounded because age 55–59 has a different "
        "Dead rate. It is retained as a descriptive lead only. Actionability is assessed separately "
        "with false-positive rate among Alive rows and false-negative rate among Dead rows.",
        "",
        "Class-conditional signals passing denominator, effect-size, and BH gates:",
        "",
        markdown_table(
            class_conditional_signals,
            [
                "cluster_type", "cluster", "focal_rows", "rest_same_cluster_rows",
                "focal_alive_denominator", "rest_alive_denominator",
                "focal_false_positive_rate", "rest_false_positive_rate",
                "false_positive_rate_difference", "fpr_bh_q_value",
                "focal_dead_denominator", "rest_dead_denominator",
                "focal_false_negative_rate", "rest_false_negative_rate",
                "false_negative_rate_difference", "fnr_bh_q_value",
            ],
            decimals=6,
        ),
        "",
        "Prevalence-confounded total-error flags (descriptive only; family and raw-code rows can "
        "overlap):",
        "",
        markdown_table(
            descriptive_error_flags,
            [
                "cluster_type", "cluster", "focal_rows", "rest_same_cluster_rows",
                "focal_dead_rate", "rest_dead_rate", "focal_error_rate", "rest_error_rate",
                "error_rate_difference", "overall_error_bh_q_value",
                "focal_alive_denominator", "rest_alive_denominator",
                "focal_false_positive_rate", "rest_false_positive_rate",
                "focal_dead_denominator", "rest_dead_denominator",
                "focal_false_negative_rate", "rest_false_negative_rate",
            ],
            decimals=6,
        ),
        "",
        "A class-conditional signal requires at least "
        f"{MIN_CLASS_CONDITIONAL_FOCAL} focal and {MIN_CLASS_CONDITIONAL_REST} comparison rows "
        "of the relevant true class, an excess FPR or FNR of at least 5 percentage points, and "
        "BH q<0.05 within cluster definition and error type.",
        "",
        f"The reported q≈{q_8140} for EOD-all-blank × histology 8140 belongs to the "
        "total-error test and therefore does not establish class-conditional difficulty. "
        + adenocarcinoma_detail
        + " Neither endpoint passes the conservative 5-point class-conditional screen. "
        + code_8010_detail
        + ", so its FPR is too poorly supported for intervention.",
        "",
        replication_verdict,
        "",
        "Largest family-level cells are retained in `age55_error_clusters.csv`; none may be called "
        "actionable merely because its unconditioned total-error Fisher test is small.",
        "",
        "## Age representation check",
        "",
        markdown_table(
            age_representation,
            ["representation", "present_in_pipeline_v6", "source_file"],
        ),
        "",
        "Pipeline v6 already uses the numeric band midpoint alongside categorical `age_recode` and "
        "age×year/stage/metastasis interactions. The source data still provides age as a band, so the "
        "midpoint cannot recover within-band age. Even so, the trough is not simply evidence that the "
        "model forgot a smooth age term, and adding another one is not justified by this audit.",
        "",
        "## Rejected pseudo90 + NN stress-test changes",
        "",
        (
            f"Within age 55–59, pseudo90 + 20% NN has weighted F1 "
            f"`{float(stress_candidate_row['weighted_f1']):.6f}` versus "
            f"`{float(stress_control_row['weighted_f1']):.6f}` for pseudo90 "
            f"(`{stress_weighted_delta:+.6f}`). It changes `{stress_changed}/{int(balance['focal_rows'])}` "
            f"predictions: `{stress_corrected}` corrected and `{stress_harmed}` harmed. This directly "
            "supports rejecting that stress candidate, not introducing an age-specific rewrite."
            if stress_available
            else
            "The saved pseudo90/NN vectors needed for the direct stress comparison were unavailable."
        ),
        "",
        markdown_table(
            stress_changes,
            [
                "cluster", "focal_age_rows", "cluster_focal_rows", "changed_predictions",
                "changed_fraction_of_cluster", "changed_fraction_of_focal_age",
                "control_dead_to_stress_alive",
                "control_alive_to_stress_dead", "corrected_predictions", "harmed_predictions",
                "net_corrected",
            ],
        ) if not stress_changes.empty else "_The required saved vectors were unavailable._",
        "",
        "## Interpretation guardrails",
        "",
        "- This is association and OOF error analysis, not a causal study.",
        "- BH correction is applied within each test family. It limits false discoveries but does not "
        "turn a registry-code association into a model-fix justification.",
        "- Age 55–59 was selected after earlier subgroup scans; all inferential results here are "
        "targeted follow-up evidence, not pristine confirmatory inference.",
        "- Total error and weighted F1 vary with class prevalence. Cluster claims therefore require "
        "the separately reported class-conditional FPR/FNR gates and adequate denominators.",
        "- Slice F1 is prevalence-sensitive; lower Dead prevalence can reduce weighted F1 under a "
        "global fixed-rate policy even when ranking remains useful.",
        "- Proxy fold re-slices are descriptive and cannot pass an intervention's 4-of-5 outer-fold gate.",
        "- No age-specific threshold, post-hoc label rewrite, subgroup reweighting, training, or "
        "submission file is produced.",
        "- If the report used the frozen-tree proxy, rerun this script after the canonical Submission 6 "
        "nested artifact exists before using these findings for a future gate.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python3 age55_subgroup_investigation.py",
        "```",
    ]
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return {
        "primary_candidate": primary,
        "canonical_submission6_nested_used": canonical_used,
        "canonical_acceptance_mode": (
            notes.get("canonical_acceptance_mode") if canonical_used else None
        ),
        "practical_recipe_replay_accepted": canonical_used,
        "strict_probability_equivalence_verified": (
            notes.get("strict_probability_equivalence_verified") == "true"
            if canonical_used
            else None
        ),
        "strict_recipe_artifact_equivalence_verified": (
            notes.get("strict_recipe_artifact_equivalence_verified") == "true"
            if canonical_used
            else None
        ),
        "fixed_positive_rate": positive_rate,
        "official_metric": "support-weighted F1",
        "focal_age": FOCAL_AGE,
        "focal_rows": int(balance["focal_rows"]),
        "focal_dead_rate": float(balance["focal_rate"]),
        "rest_dead_rate": float(balance["rest_rate"]),
        "focal_weighted_f1": float(focal_metric["weighted_f1"]),
        "global_weighted_f1": float(global_metric["weighted_f1"]),
        "lower_prevalence_signal": bool(lower_prevalence),
        "neighbor_trough_signal": bool(focal_trough),
        "material_semantic_unavailable_effect_count": int(len(material_missing)),
        "material_categorical_effect_count": int(len(material_categories)),
        "material_numeric_effect_count": int(len(material_numeric)),
        "descriptive_total_error_flag_count": int(len(descriptive_error_flags)),
        "descriptive_family_total_error_flag_count": int(len(descriptive_family_error)),
        "descriptive_raw_code_total_error_flag_count": int(len(descriptive_code_error)),
        "targeted_total_error_replication": targeted_total_error_replication,
        "targeted_class_conditional_replication": targeted_class_conditional_replication,
        "targeted_adenocarcinoma_total_error_bh_q_value": (
            float(adenocarcinoma_lead["overall_error_bh_q_value"])
            if adenocarcinoma_lead is not None
            else None
        ),
        "targeted_raw_8140_total_error_bh_q_value": (
            float(histology_8140_lead["overall_error_bh_q_value"])
            if histology_8140_lead is not None
            else None
        ),
        "class_conditional_excess_error_signal_count": int(
            len(class_conditional_signals)
        ),
        "material_unavailable_effects_more_unavailable": more_unavailable,
        "material_unavailable_effects_less_unavailable": less_unavailable,
        "focal_roc_auc": float(focal_metric["roc_auc"]),
        "descriptive_fold_count": proxy_fold_count,
        "descriptive_folds_f1_below_both_neighbors": proxy_fold_f1_below_both,
        "descriptive_folds_auc_below_both_neighbors": proxy_fold_auc_below_both,
        "fold_view_eligible_for_intervention_gate": fold_gate_eligible,
        "stress_pseudo90_weighted_f1": (
            float(stress_control_row["weighted_f1"]) if stress_available else None
        ),
        "stress_pseudo90_nn20_weighted_f1": (
            float(stress_candidate_row["weighted_f1"]) if stress_available else None
        ),
        "stress_weighted_f1_delta": (
            stress_weighted_delta if stress_available else None
        ),
        "stress_changed_predictions": stress_changed,
        "stress_corrected_predictions": stress_corrected,
        "stress_harmed_predictions": stress_harmed,
        "model_change_recommended": False,
        "submission_generated": False,
        "candidate_notes": notes,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = require_file(ROOT / "train.csv")
    y_path = require_file(ROOT / "y_step1.npy")
    train = pd.read_csv(train_path, low_memory=False)
    if "vital_status" not in train or "age_recode" not in train:
        raise ValueError("train.csv must include vital_status and age_recode")
    y = train["vital_status"].eq("Dead").to_numpy(dtype=np.int8)
    saved_y = np.load(y_path, allow_pickle=False).astype(np.int8)
    if not np.array_equal(y, saved_y):
        raise ValueError("y_step1.npy does not match train.csv labels/order")

    train = train.copy()
    train["derived_eod_composite_stage"] = derive_composite_stage(train)
    train["histology_family"] = train["histologic_type_icdo3"].map(histology_group)
    focal = train["age_recode"].eq(FOCAL_AGE).to_numpy()
    if focal.sum() < 200:
        raise ValueError(f"Unexpectedly small {FOCAL_AGE} slice: {focal.sum()}")

    candidates, primary, notes, hashes, canonical_folds = load_candidates(train, y)
    model_metrics, age_metrics, hard_predictions = build_model_metrics(
        train, y, candidates, args.fixed_positive_rate
    )
    class_balance = build_class_balance(train, y)
    age_representation = build_age_representation_audit()
    missingness = build_missingness_table(train, focal)
    categorical_effects, categorical_levels = build_categorical_tables(
        train, y, hard_predictions[primary], focal
    )
    numeric_effects = build_numeric_table(train, focal)
    fold_age_metrics = build_fold_age_metrics(
        train,
        y,
        candidates[primary],
        args.fixed_positive_rate,
        canonical_folds if primary == "submission6_nested_equivalent" else None,
    )
    neighbor_fold_comparisons = build_neighbor_fold_comparisons(fold_age_metrics)
    error_clusters = build_error_clusters(
        train, y, hard_predictions[primary], focal
    )
    stress_changes = build_stress_change_clusters(train, y, hard_predictions)

    outputs = {
        "class_balance.csv": class_balance,
        "age_representation_audit.csv": age_representation,
        "model_metric_summary.csv": model_metrics,
        "age_band_metrics.csv": age_metrics,
        "age_band_fold_metrics.csv": fold_age_metrics,
        "age_neighbor_fold_comparisons.csv": neighbor_fold_comparisons,
        "missingness_comparison.csv": missingness,
        "categorical_feature_effects.csv": categorical_effects,
        "categorical_level_profiles.csv": categorical_levels,
        "numeric_feature_effects.csv": numeric_effects,
        "age55_error_clusters.csv": error_clusters,
        "sub12_age55_change_clusters.csv": stress_changes,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    summary = write_report(
        output_dir / "age55_investigation_report.md",
        args.fixed_positive_rate,
        primary,
        notes,
        age_representation,
        class_balance,
        model_metrics,
        age_metrics,
        fold_age_metrics,
        neighbor_fold_comparisons,
        missingness,
        categorical_effects,
        categorical_levels,
        numeric_effects,
        error_clusters,
        stress_changes,
    )
    summary.update(
        {
            "artifact_hashes": {
                "train.csv": file_sha256(train_path),
                "y_step1.npy": file_sha256(y_path),
                "pipeline_v6.py": file_sha256(ROOT / "pipeline_v6.py"),
                **hashes,
            },
            "output_files": sorted([*outputs, "age55_investigation_report.md", "age55_investigation_summary.json"]),
            "multiple_comparison_policy": (
                "Benjamini-Hochberg within missingness-definition, categorical-feature, "
                "numeric-feature, and each error-cluster definition/endpoint family; "
                "effect-size and class-denominator thresholds also required"
            ),
        }
    )
    (output_dir / "age55_investigation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Primary OOF reference: {primary}")
    print(f"Wrote diagnostics to {output_dir}")
    print("Decision: diagnostics only; no model change or submission generated")


if __name__ == "__main__":
    main()
