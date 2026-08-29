"""Independent practical-acceptance gate for the Submission 6 nested reference.

This audit never trains a model and never writes into the active reconstruction
directory.  It independently reconstructs the full-data replay probabilities
from the saved replay checkpoints, compares them with the frozen Submission 6
artifacts, and validates a completed five-fold nested OOF vector when one is
available.

The historical strict probability-equivalence result remains authoritative and
is never rewritten by this script.  The separate practical gate is deliberately
tight and may approve the nested vector only when both the practical production
replay and every nested-integrity/scale check pass.

Outputs are written only when this script is run, under:
    diagnostic_outputs/submission6_reference_gate/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent
NESTED_DIR = ROOT / "diagnostic_outputs" / "submission6_nested"
DEFAULT_OUTPUT_DIR = ROOT / "diagnostic_outputs" / "submission6_reference_gate"

SCHEMA_VERSION = "submission6_practical_reference_gate_v1"
EXPECTED_NESTED_SCHEMA_VERSION = "submission6_nested_v1"
EXPECTED_OUTER_FOLDS = 5
EXPECTED_TRAIN_ROWS = 24_000
EXPECTED_TEST_ROWS = 36_000
EXPECTED_TEACHER_COMPLETE_SHAPE = (3, 5, 6)
EXPECTED_STUDENT_COMPLETE_SHAPE = (5, 6)
EXPECTED_SUBMISSION_DEAD_COUNT = 30_411
EXPECTED_HISTORICAL_THRESHOLD = 0.684
HISTORICAL_THRESHOLD_ABS_TOLERANCE = 1e-12
FIXED_DEAD_RATE = 0.845
SUBMISSION6_PUBLIC_LB = 0.877258
SCALE_GAP_ABS_TOLERANCE = 0.005

MAX_MAE = 1e-8
MAX_RMSE = 1e-8
MAX_ABSOLUTE_ERROR = 1e-7
MIN_PEARSON = 0.999999999
MIN_SPEARMAN = 0.999999999

TEACHER_V5_WEIGHT = 0.55
TEACHER_V4_WEIGHT = 0.45
FINAL_TEACHER_WEIGHT = 0.50
FINAL_STUDENT_WEIGHT = 0.50
PSEUDO_DEAD_THRESHOLD = 0.95
PSEUDO_ALIVE_THRESHOLD = 0.05


@dataclass
class Check:
    check: str
    section: str
    required: bool
    status: str
    criterion: str
    observed: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Separate output directory for the gate JSON, CSV, and report.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def add_check(
    checks: list[Check],
    check: str,
    section: str,
    required: bool,
    status: str,
    criterion: str,
    observed: Any,
) -> None:
    if status not in {"pass", "fail", "pending"}:
        raise ValueError(f"Invalid check status: {status}")
    if isinstance(observed, (dict, list, tuple)):
        observed_text = json.dumps(observed, sort_keys=True, default=str)
    else:
        observed_text = str(observed)
    checks.append(
        Check(
            check=check,
            section=section,
            required=required,
            status=status,
            criterion=criterion,
            observed=observed_text,
        )
    )


def boolean_check(
    checks: list[Check],
    check: str,
    section: str,
    condition: bool,
    criterion: str,
    observed: Any,
    *,
    required: bool = True,
) -> None:
    add_check(
        checks,
        check,
        section,
        required,
        "pass" if bool(condition) else "fail",
        criterion,
        observed,
    )


def pending_check(
    checks: list[Check],
    check: str,
    section: str,
    criterion: str,
    observed: Any,
) -> None:
    add_check(checks, check, section, True, "pending", criterion, observed)


def top_rate_predictions(scores: np.ndarray, rate: float = FIXED_DEAD_RATE) -> np.ndarray:
    count = int(round(len(scores) * rate))
    order = np.argsort(-scores, kind="mergesort")
    predictions = np.zeros(len(scores), dtype=np.int8)
    predictions[order[:count]] = 1
    return predictions


def historical_threshold_predictions(scores: np.ndarray) -> tuple[np.ndarray, float]:
    for threshold in np.arange(0.40, 0.75, 0.001):
        if float((scores >= threshold).mean()) <= FIXED_DEAD_RATE:
            return (scores >= threshold).astype(np.int8), float(threshold)
    raise RuntimeError("Historical threshold scan found no admissible threshold")


def probability_comparison(
    replayed: np.ndarray, archived: np.ndarray
) -> dict[str, float]:
    delta = replayed - archived
    return {
        "rows": int(len(replayed)),
        "mean_absolute_error": float(np.mean(np.abs(delta))),
        "root_mean_squared_error": float(np.sqrt(np.mean(np.square(delta)))),
        "maximum_absolute_error": float(np.max(np.abs(delta))),
        "pearson_correlation": float(np.corrcoef(replayed, archived)[0, 1]),
        "spearman_rank_correlation": float(
            pd.Series(replayed).corr(pd.Series(archived), method="spearman")
        ),
    }


def probability_vector_is_valid(values: np.ndarray, rows: int) -> bool:
    return bool(
        values.shape == (rows,)
        and np.isfinite(values).all()
        and np.all((values >= 0.0) & (values <= 1.0))
    )


def npz_signature(artifact: Any) -> str | None:
    if "run_signature" not in artifact.files:
        return None
    return str(artifact["run_signature"].item())


def audit_signatures_and_inputs(
    checks: list[Check], manifest: dict[str, Any]
) -> str:
    run_signature = str(manifest.get("run_signature", ""))
    boolean_check(
        checks,
        "manifest_schema",
        "signature",
        manifest.get("schema_version") == EXPECTED_NESTED_SCHEMA_VERSION,
        f"schema_version == {EXPECTED_NESTED_SCHEMA_VERSION}",
        manifest.get("schema_version"),
    )
    recipe_sha = stable_json_sha256(manifest.get("recipe", {}))
    boolean_check(
        checks,
        "recipe_hash_consistency",
        "signature",
        recipe_sha == manifest.get("recipe_sha256"),
        "manifest recipe hashes to recipe_sha256",
        {"computed": recipe_sha, "manifest": manifest.get("recipe_sha256")},
    )

    current_files = {
        "reconstruction_script_sha256": ROOT
        / "submission6_nested_reconstruction.py",
        "pipeline_v6_sha256": ROOT / "pipeline_v6.py",
        "train_sha256": ROOT / "train.csv",
        "test_sha256": ROOT / "test.csv",
        "archived_v6_teacher_sha256": ROOT / "archive" / "probs_v6_blend.npy",
        "archived_v6_final_sha256": ROOT / "archive" / "probs_v6_final.npy",
        "archived_v6_submission_sha256": ROOT / "archive" / "submission6.csv",
    }
    hash_observed: dict[str, Any] = {}
    all_hashes_match = True
    for key, path in current_files.items():
        current_hash = sha256_file(path) if path.is_file() else None
        expected_hash = manifest.get(key)
        hash_observed[key] = {
            "path": str(path.relative_to(ROOT)),
            "current": current_hash,
            "manifest": expected_hash,
        }
        all_hashes_match &= current_hash is not None and current_hash == expected_hash
    boolean_check(
        checks,
        "current_input_hashes_match_manifest",
        "signature",
        all_hashes_match,
        "all code/data/archive hashes equal the run manifest",
        hash_observed,
    )

    signature_keys = [
        "schema_version",
        "recipe_sha256",
        "reconstruction_script_sha256",
        "pipeline_v6_sha256",
        "train_sha256",
        "test_sha256",
        "archived_v6_teacher_sha256",
        "archived_v6_final_sha256",
        "archived_v6_submission_sha256",
        "python_runtime",
        "platform_runtime",
        "library_versions",
    ]
    signature_payload = {key: manifest.get(key) for key in signature_keys}
    recomputed_signature = stable_json_sha256(signature_payload)
    boolean_check(
        checks,
        "run_signature_recomputes",
        "signature",
        bool(run_signature) and recomputed_signature == run_signature,
        "stable hash of the signature payload equals run_signature",
        {"computed": recomputed_signature, "manifest": run_signature},
    )

    provenance_path = NESTED_DIR / "input_provenance.csv"
    provenance_ok = False
    provenance_observed: Any = "missing"
    if provenance_path.is_file():
        provenance = pd.read_csv(provenance_path)
        provenance_rows: list[dict[str, Any]] = []
        provenance_ok = True
        for row in provenance.to_dict("records"):
            path = ROOT / str(row["path"])
            current_hash = sha256_file(path) if path.is_file() else None
            recorded_hash = row.get("sha256")
            row_ok = bool(row.get("exists")) and current_hash == recorded_hash
            provenance_ok &= row_ok
            provenance_rows.append(
                {
                    "path": row["path"],
                    "recorded": recorded_hash,
                    "current": current_hash,
                    "match": row_ok,
                }
            )
        provenance_observed = provenance_rows
    boolean_check(
        checks,
        "input_provenance_is_current",
        "signature",
        provenance_ok,
        "every recorded provenance artifact still exists with the recorded SHA-256",
        provenance_observed,
    )
    return run_signature


def audit_structural_isolation(checks: list[Check]) -> None:
    structural_path = NESTED_DIR / "structural_invariants.json"
    if not structural_path.is_file():
        pending_check(
            checks,
            "structural_invariants",
            "structural",
            "all predeclared structural invariants pass",
            "file missing",
        )
    else:
        structural = json.loads(structural_path.read_text(encoding="utf-8"))
        required = {
            "outer_fold_count": EXPECTED_OUTER_FOLDS,
            "all_fit_validation_disjoint": True,
            "all_fit_validation_unions_cover_train": True,
            "oof_row_coverage_min": 1,
            "oof_row_coverage_max": 1,
            "every_oof_row_exactly_once": True,
            "expected_teacher_models_per_outer_fold": 90,
            "expected_student_models_per_outer_fold": 30,
            "v5_branch_weight_sum_pass": True,
            "v4_branch_weight_sum_pass": True,
            "teacher_family_weight_sum_pass": True,
            "final_weight_sum_pass": True,
            "teacher_family_weights_exact": True,
            "final_weights_exact": True,
            "all_structural_invariants_pass": True,
        }
        mismatches = {
            key: {"expected": expected, "observed": structural.get(key)}
            for key, expected in required.items()
            if structural.get(key) != expected
        }
        boolean_check(
            checks,
            "structural_invariants",
            "structural",
            not mismatches,
            "all outer-split/model-count/blend-weight invariants pass",
            mismatches or "all required invariants match",
        )

    outer_path = NESTED_DIR / "outer_split_audit.csv"
    outer_ok = False
    outer_observed: Any = "file missing"
    if outer_path.is_file():
        outer = pd.read_csv(outer_path)
        outer_ok = bool(
            len(outer) == EXPECTED_OUTER_FOLDS
            and outer["outer_fold"].tolist() == [1, 2, 3, 4, 5]
            and (outer["fit_rows"] == 19_200).all()
            and (outer["validation_rows"] == 4_800).all()
            and (outer["fit_validation_intersection_rows"] == 0).all()
            and (outer["fit_validation_union_rows"] == EXPECTED_TRAIN_ROWS).all()
            and outer["disjoint_pass"].all()
            and outer["full_union_pass"].all()
        )
        outer_observed = outer.to_dict("records")
    boolean_check(
        checks,
        "outer_split_isolation_audit",
        "structural",
        outer_ok,
        "five disjoint 19,200/4,800 splits, each covering all 24,000 rows",
        outer_observed,
    )

    recipe_path = NESTED_DIR / "recipe_equivalence_audit.csv"
    recipe_ok = False
    recipe_observed: Any = "file missing"
    if recipe_path.is_file():
        recipe = pd.read_csv(recipe_path).set_index("component")
        expected_statuses = {
            "outer validation": "required_nested_analogue",
            "feature builder": "strict_outer_isolation",
            "ordinal categorical encoding": "strict_outer_isolation",
            "v5 teacher": "production_equivalent",
            "v4 target encoding": "strict_outer_isolation",
            "teacher family blend": "exact_hyperparameter",
            "pseudo-label selection": "production_equivalent",
            "pseudo student": "production_equivalent",
            "final probability blend": "exact_hyperparameter",
            "decision policy": "standardized_evaluation_policy",
            "test output": "safety_guardrail",
        }
        mismatches = {}
        for component, expected in expected_statuses.items():
            observed = (
                recipe.loc[component, "status"] if component in recipe.index else None
            )
            if observed != expected:
                mismatches[component] = {"expected": expected, "observed": observed}
        recipe_ok = not mismatches
        recipe_observed = mismatches or expected_statuses
    boolean_check(
        checks,
        "recipe_isolation_audit",
        "structural",
        recipe_ok,
        "all preprocessing/TE/pseudo/student isolation statuses match the locked audit",
        recipe_observed,
    )


def audit_production_replay(
    checks: list[Check], run_signature: str
) -> dict[str, Any]:
    replay_path = NESTED_DIR / "production_replay_comparison.json"
    teacher_checkpoint_path = (
        NESTED_DIR / "checkpoints" / "production_replay_teacher.npz"
    )
    student_checkpoint_path = (
        NESTED_DIR / "checkpoints" / "production_replay_student.npz"
    )
    required_paths = [replay_path, teacher_checkpoint_path, student_checkpoint_path]
    missing = [str(path.relative_to(ROOT)) for path in required_paths if not path.is_file()]
    if missing:
        for name in (
            "production_replay_artifacts",
            "practical_probability_tolerances",
            "archived_pseudo_labels",
            "historical_submission_labels",
        ):
            pending_check(
                checks,
                name,
                "replay",
                "full-data replay JSON and both replay checkpoints are complete",
                {"missing": missing},
            )
        return {"available": False, "missing": missing}

    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    with np.load(teacher_checkpoint_path, allow_pickle=False) as teacher_cp, np.load(
        student_checkpoint_path, allow_pickle=False
    ) as student_cp:
        signatures = {
            "manifest": run_signature,
            "replay_json": replay.get("run_signature"),
            "teacher_checkpoint": npz_signature(teacher_cp),
            "student_checkpoint": npz_signature(student_cp),
        }
        completion_ok = bool(
            teacher_cp["complete"].shape == EXPECTED_TEACHER_COMPLETE_SHAPE
            and teacher_cp["complete"].all()
            and student_cp["complete"].shape == EXPECTED_STUDENT_COMPLETE_SHAPE
            and student_cp["complete"].all()
        )
        signatures_ok = bool(
            run_signature
            and replay.get("status") == "complete"
            and all(value == run_signature for value in signatures.values())
        )
        boolean_check(
            checks,
            "production_replay_artifacts",
            "replay",
            completion_ok and signatures_ok,
            "JSON is complete; 90/90 teacher and 30/30 student models share run_signature",
            {"signatures": signatures, "completion_ok": completion_ok},
        )

        v5_teacher = teacher_cp["v5_validation"].astype(np.float64)
        v4_teacher = teacher_cp["v4_validation"].astype(np.float64)
        v5_teacher_duplicate = teacher_cp["v5_test"].astype(np.float64)
        v4_teacher_duplicate = teacher_cp["v4_test"].astype(np.float64)
        v5_student = student_cp["v5_validation"].astype(np.float64)
        v4_student = student_cp["v4_validation"].astype(np.float64)
        replay_teacher = TEACHER_V5_WEIGHT * v5_teacher + TEACHER_V4_WEIGHT * v4_teacher
        replay_student = TEACHER_V5_WEIGHT * v5_student + TEACHER_V4_WEIGHT * v4_student
        replay_final = (
            FINAL_TEACHER_WEIGHT * replay_teacher
            + FINAL_STUDENT_WEIGHT * replay_student
        )
        replay_vectors_ok = bool(
            all(
                probability_vector_is_valid(vector, EXPECTED_TEST_ROWS)
                for vector in (
                    v5_teacher,
                    v4_teacher,
                    v5_teacher_duplicate,
                    v4_teacher_duplicate,
                    v5_student,
                    v4_student,
                    replay_teacher,
                    replay_student,
                    replay_final,
                )
            )
            and np.array_equal(v5_teacher, v5_teacher_duplicate)
            and np.array_equal(v4_teacher, v4_teacher_duplicate)
        )

        replay_pseudo_mask = student_cp["pseudo_mask"].astype(bool)
        replay_pseudo_y = student_cp["pseudo_y"].astype(np.int8)
        mask_hash = hashlib.sha256(replay_pseudo_mask.tobytes()).hexdigest()
        y_hash = hashlib.sha256(replay_pseudo_y.tobytes()).hexdigest()
        stored_hashes_ok = bool(
            str(student_cp["pseudo_mask_sha256"].item()) == mask_hash
            and str(student_cp["pseudo_y_sha256"].item()) == y_hash
        )

    archived_teacher = np.load(
        ROOT / "archive" / "probs_v6_blend.npy", allow_pickle=False
    ).astype(np.float64)
    archived_final = np.load(
        ROOT / "archive" / "probs_v6_final.npy", allow_pickle=False
    ).astype(np.float64)
    archived_vectors_ok = bool(
        probability_vector_is_valid(archived_teacher, EXPECTED_TEST_ROWS)
        and probability_vector_is_valid(archived_final, EXPECTED_TEST_ROWS)
    )
    boolean_check(
        checks,
        "replay_and_archive_shapes_finiteness",
        "replay",
        replay_vectors_ok and archived_vectors_ok,
        "all replay/archive vectors are finite [0,1] arrays with shape (36,000,)",
        {
            "replay_vectors_ok": replay_vectors_ok,
            "archive_vectors_ok": archived_vectors_ok,
        },
    )

    teacher_comparison = probability_comparison(replay_teacher, archived_teacher)
    final_comparison = probability_comparison(replay_final, archived_final)
    practical_tolerances_ok = bool(
        all(
            comparison["mean_absolute_error"] <= MAX_MAE
            and comparison["root_mean_squared_error"] <= MAX_RMSE
            and comparison["maximum_absolute_error"] <= MAX_ABSOLUTE_ERROR
            and comparison["pearson_correlation"] >= MIN_PEARSON
            and comparison["spearman_rank_correlation"] >= MIN_SPEARMAN
            for comparison in (teacher_comparison, final_comparison)
        )
    )
    boolean_check(
        checks,
        "practical_probability_tolerances",
        "replay",
        practical_tolerances_ok,
        (
            "teacher/final MAE and RMSE <=1e-8, max abs <=1e-7, "
            "Pearson and Spearman >=0.999999999"
        ),
        {"teacher": teacher_comparison, "final": final_comparison},
    )

    expected_mask = (archived_teacher >= PSEUDO_DEAD_THRESHOLD) | (
        archived_teacher <= PSEUDO_ALIVE_THRESHOLD
    )
    expected_pseudo_y = (archived_teacher[expected_mask] >= PSEUDO_DEAD_THRESHOLD).astype(
        np.int8
    )
    replay_derived_mask = (replay_teacher >= PSEUDO_DEAD_THRESHOLD) | (
        replay_teacher <= PSEUDO_ALIVE_THRESHOLD
    )
    replay_derived_y = (
        replay_teacher[replay_derived_mask] >= PSEUDO_DEAD_THRESHOLD
    ).astype(np.int8)
    pseudo_ok = bool(
        replay_pseudo_mask.shape == (EXPECTED_TEST_ROWS,)
        and np.array_equal(replay_pseudo_mask, expected_mask)
        and np.array_equal(replay_pseudo_y, expected_pseudo_y)
        and np.array_equal(replay_pseudo_mask, replay_derived_mask)
        and np.array_equal(replay_pseudo_y, replay_derived_y)
        and stored_hashes_ok
    )
    boolean_check(
        checks,
        "archived_pseudo_labels",
        "replay",
        pseudo_ok,
        "replay mask and labels exactly equal archived-teacher and replay-teacher rules",
        {
            "archived_count": int(expected_mask.sum()),
            "replay_count": int(replay_pseudo_mask.sum()),
            "mask_changes": int(np.sum(replay_pseudo_mask != expected_mask)),
            "label_changes": (
                int(np.sum(replay_pseudo_y != expected_pseudo_y))
                if replay_pseudo_y.shape == expected_pseudo_y.shape
                else None
            ),
            "stored_hashes_ok": stored_hashes_ok,
        },
    )

    test = pd.read_csv(ROOT / "test.csv")
    submission = pd.read_csv(ROOT / "archive" / "submission6.csv")
    submission_format_ok = bool(
        list(submission.columns) == ["patient_id", "vital_status"]
        and len(submission) == EXPECTED_TEST_ROWS
        and submission["patient_id"].equals(test["patient_id"])
        and set(submission["vital_status"].unique()) <= {"Dead", "Alive"}
    )
    archived_labels = submission["vital_status"].eq("Dead").to_numpy(dtype=np.int8)
    historical_labels, selected_threshold = historical_threshold_predictions(replay_final)
    top_k_labels = top_rate_predictions(replay_final, archived_labels.mean())
    historical_changes = int(np.sum(historical_labels != archived_labels))
    top_k_changes = int(np.sum(top_k_labels != archived_labels))
    label_checks_ok = bool(
        submission_format_ok
        and math.isclose(
            selected_threshold,
            EXPECTED_HISTORICAL_THRESHOLD,
            rel_tol=0.0,
            abs_tol=HISTORICAL_THRESHOLD_ABS_TOLERANCE,
        )
        and int(archived_labels.sum()) == EXPECTED_SUBMISSION_DEAD_COUNT
        and int(historical_labels.sum()) == EXPECTED_SUBMISSION_DEAD_COUNT
        and int(top_k_labels.sum()) == EXPECTED_SUBMISSION_DEAD_COUNT
        and historical_changes == 0
        and top_k_changes == 0
    )
    boolean_check(
        checks,
        "historical_submission_labels",
        "replay",
        label_checks_ok,
        "threshold 0.684, 30,411 Dead, zero historical/top-k label changes",
        {
            "submission_format_ok": submission_format_ok,
            "selected_threshold": selected_threshold,
            "archived_dead_count": int(archived_labels.sum()),
            "historical_dead_count": int(historical_labels.sum()),
            "top_k_dead_count": int(top_k_labels.sum()),
            "historical_label_changes": historical_changes,
            "top_k_label_changes": top_k_changes,
        },
    )

    replay_json_consistent = bool(
        replay.get("historical_threshold_scan", {}).get(
            "changed_labels_vs_archived_submission"
        )
        == historical_changes
        and replay.get("top_k_diagnostic", {}).get(
            "changed_labels_vs_archived_submission"
        )
        == top_k_changes
        and math.isclose(
            float(
                replay.get("historical_threshold_scan", {}).get(
                    "selected_threshold", float("nan")
                )
            ),
            selected_threshold,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and all(
            math.isclose(
                float(replay[f"{scope}_comparison"][metric]),
                float(comparison[metric]),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            for scope, comparison in (
                ("teacher", teacher_comparison),
                ("final", final_comparison),
            )
            for metric in comparison
        )
    )
    boolean_check(
        checks,
        "replay_json_matches_recomputation",
        "replay",
        replay_json_consistent,
        "saved replay JSON metrics/labels equal independent checkpoint recomputation",
        replay_json_consistent,
    )

    strict_probability = bool(
        replay.get("strict_probability_equivalence_verified", False)
    )
    strict_recipe = bool(
        replay.get("strict_recipe_artifact_equivalence_verified", False)
    )
    boolean_check(
        checks,
        "strict_equivalence_remains_false",
        "strict_boundary",
        not strict_probability and not strict_recipe,
        "record current strict results without converting them into practical approval",
        {
            "strict_probability_equivalence_verified": strict_probability,
            "strict_recipe_artifact_equivalence_verified": strict_recipe,
        },
        required=False,
    )
    return {
        "available": True,
        "teacher_comparison": teacher_comparison,
        "final_comparison": final_comparison,
        "selected_threshold": selected_threshold,
        "archived_dead_count": int(archived_labels.sum()),
        "historical_label_changes": historical_changes,
        "top_k_label_changes": top_k_changes,
        "strict_probability_equivalence_verified": strict_probability,
        "strict_recipe_artifact_equivalence_verified": strict_recipe,
    }


def audit_fold_artifacts(
    checks: list[Check], run_signature: str, y: np.ndarray
) -> dict[str, Any]:
    splitter = StratifiedKFold(5, shuffle=True, random_state=42)
    expected_splits = list(splitter.split(np.zeros(len(y)), y))
    assembled_teacher = np.full(len(y), np.nan, dtype=np.float64)
    assembled_student = np.full(len(y), np.nan, dtype=np.float64)
    assembled_final = np.full(len(y), np.nan, dtype=np.float64)
    assembled_fold = np.full(len(y), -1, dtype=np.int8)
    issues: list[str] = []

    for fold, (_, expected_validation_idx) in enumerate(expected_splits, start=1):
        teacher_path = (
            NESTED_DIR / "checkpoints" / f"outer_fold_{fold:02d}_teacher.npz"
        )
        student_path = (
            NESTED_DIR / "checkpoints" / f"outer_fold_{fold:02d}_student.npz"
        )
        final_path = NESTED_DIR / "folds" / f"outer_fold_{fold:02d}_final.npz"
        if not all(path.is_file() for path in (teacher_path, student_path, final_path)):
            issues.append(f"fold {fold}: one or more teacher/student/final files missing")
            continue
        try:
            with np.load(teacher_path, allow_pickle=False) as teacher_cp, np.load(
                student_path, allow_pickle=False
            ) as student_cp, np.load(final_path, allow_pickle=False) as final_artifact:
                if not all(
                    npz_signature(artifact) == run_signature
                    for artifact in (teacher_cp, student_cp, final_artifact)
                ):
                    issues.append(f"fold {fold}: run signature mismatch")
                    continue
                if not (
                    teacher_cp["complete"].shape == EXPECTED_TEACHER_COMPLETE_SHAPE
                    and teacher_cp["complete"].all()
                    and student_cp["complete"].shape == EXPECTED_STUDENT_COMPLETE_SHAPE
                    and student_cp["complete"].all()
                ):
                    issues.append(f"fold {fold}: model completion grid incomplete")
                    continue

                teacher_validation = (
                    TEACHER_V5_WEIGHT * teacher_cp["v5_validation"].astype(np.float64)
                    + TEACHER_V4_WEIGHT
                    * teacher_cp["v4_validation"].astype(np.float64)
                )
                teacher_test = (
                    TEACHER_V5_WEIGHT * teacher_cp["v5_test"].astype(np.float64)
                    + TEACHER_V4_WEIGHT * teacher_cp["v4_test"].astype(np.float64)
                )
                student_validation = (
                    TEACHER_V5_WEIGHT * student_cp["v5_validation"].astype(np.float64)
                    + TEACHER_V4_WEIGHT
                    * student_cp["v4_validation"].astype(np.float64)
                )
                expected_mask = (teacher_test >= PSEUDO_DEAD_THRESHOLD) | (
                    teacher_test <= PSEUDO_ALIVE_THRESHOLD
                )
                expected_pseudo_y = (
                    teacher_test[expected_mask] >= PSEUDO_DEAD_THRESHOLD
                ).astype(np.int8)
                mask = student_cp["pseudo_mask"].astype(bool)
                pseudo_y = student_cp["pseudo_y"].astype(np.int8)
                hashes_ok = bool(
                    str(student_cp["pseudo_mask_sha256"].item())
                    == hashlib.sha256(mask.tobytes()).hexdigest()
                    and str(student_cp["pseudo_y_sha256"].item())
                    == hashlib.sha256(pseudo_y.tobytes()).hexdigest()
                )
                if not (
                    np.array_equal(mask, expected_mask)
                    and np.array_equal(pseudo_y, expected_pseudo_y)
                    and hashes_ok
                ):
                    issues.append(f"fold {fold}: pseudo mask/labels/hash mismatch")
                    continue

                saved_idx = final_artifact["validation_idx"].astype(np.int64)
                saved_y = final_artifact["y_validation"].astype(np.int8)
                artifact_teacher = final_artifact["teacher_validation"].astype(
                    np.float64
                )
                artifact_student = final_artifact["student_validation"].astype(
                    np.float64
                )
                artifact_final = final_artifact["final_validation"].astype(np.float64)
                expected_final = (
                    FINAL_TEACHER_WEIGHT * teacher_validation
                    + FINAL_STUDENT_WEIGHT * student_validation
                )
                if not (
                    np.array_equal(saved_idx, expected_validation_idx)
                    and np.array_equal(saved_y, y[expected_validation_idx])
                    and np.array_equal(artifact_teacher, teacher_validation)
                    and np.array_equal(artifact_student, student_validation)
                    and np.array_equal(artifact_final, expected_final)
                    and all(
                        probability_vector_is_valid(vector, len(expected_validation_idx))
                        for vector in (
                            artifact_teacher,
                            artifact_student,
                            artifact_final,
                        )
                    )
                ):
                    issues.append(f"fold {fold}: validation index/y/probability mismatch")
                    continue
                assembled_teacher[saved_idx] = artifact_teacher
                assembled_student[saved_idx] = artifact_student
                assembled_final[saved_idx] = artifact_final
                assembled_fold[saved_idx] = fold
        except Exception as exc:  # audit should report a broken artifact, not hide it
            issues.append(f"fold {fold}: {type(exc).__name__}: {exc}")

    complete = bool(
        not issues
        and np.isfinite(assembled_teacher).all()
        and np.isfinite(assembled_student).all()
        and np.isfinite(assembled_final).all()
        and np.all(assembled_fold >= 1)
    )
    if complete:
        boolean_check(
            checks,
            "fold_artifact_reconstruction",
            "nested",
            True,
            "all five teacher/student/final artifacts reconstruct with isolated pseudo labels",
            "five folds reconstructed",
        )
    else:
        # This function is called only after the aggregate artifact declares
        # 5/5 completion.  Missing or inconsistent per-fold evidence is
        # therefore a failed integrity claim, not ordinary in-progress state.
        boolean_check(
            checks,
            "fold_artifact_reconstruction",
            "nested",
            False,
            "all five teacher/student/final artifacts reconstruct with isolated pseudo labels",
            issues,
        )
    return {
        "complete": complete,
        "issues": issues,
        "teacher": assembled_teacher,
        "student": assembled_student,
        "final": assembled_final,
        "fold": assembled_fold,
    }


def audit_nested_reference(
    checks: list[Check], run_signature: str
) -> dict[str, Any]:
    nested_path = NESTED_DIR / "nested_oof_predictions.npz"
    train = pd.read_csv(ROOT / "train.csv")
    y = train["vital_status"].eq("Dead").to_numpy(dtype=np.int8)
    if len(y) != EXPECTED_TRAIN_ROWS:
        boolean_check(
            checks,
            "training_label_source",
            "nested",
            False,
            "train.csv contains exactly 24,000 labels",
            len(y),
        )
        return {"complete": False}
    y_saved = np.load(ROOT / "y_step1.npy", allow_pickle=False).astype(np.int8)
    boolean_check(
        checks,
        "training_label_source",
        "nested",
        np.array_equal(y, y_saved),
        "train.csv labels exactly match y_step1.npy row order",
        {"rows": len(y), "exact_match": bool(np.array_equal(y, y_saved))},
    )

    if not nested_path.is_file():
        pending_check(
            checks,
            "nested_oof_artifact",
            "nested",
            "nested_oof_predictions.npz exists under the current run signature",
            "file missing",
        )
        pending_check(
            checks,
            "nested_five_fold_integrity",
            "nested",
            "5/5 folds, 24,000 finite aligned OOF rows, exact fold coverage",
            "nested artifact unavailable",
        )
        pending_check(
            checks,
            "nested_scale_gap",
            "nested",
            "absolute weighted-F1 gap to 0.877258 <= 0.005",
            "nested artifact unavailable",
        )
        return {"complete": False}

    with np.load(nested_path, allow_pickle=False) as nested:
        nested_signature = npz_signature(nested)
        nested_y = nested["y"].astype(np.int8)
        teacher_oof = nested["teacher_oof"].astype(np.float64)
        student_oof = nested["student_oof"].astype(np.float64)
        final_oof = nested["submission6_nested_oof"].astype(np.float64)
        outer_fold = nested["outer_fold"].astype(np.int8)
        completed = nested["completed_outer_folds"].astype(np.int8)

    boolean_check(
        checks,
        "nested_oof_artifact",
        "nested",
        nested_signature == run_signature,
        "nested vector run_signature equals current manifest",
        {"nested": nested_signature, "manifest": run_signature},
    )
    finite_final_rows = int(np.isfinite(final_oof).sum())
    declared_complete = bool(
        np.array_equal(completed, np.arange(1, 6, dtype=np.int8))
        and finite_final_rows == EXPECTED_TRAIN_ROWS
    )
    if not declared_complete:
        pending_check(
            checks,
            "nested_five_fold_integrity",
            "nested",
            "5/5 folds, 24,000 finite aligned OOF rows, exact fold coverage",
            {
                "completed_outer_folds": completed.tolist(),
                "finite_final_rows": finite_final_rows,
            },
        )
        pending_check(
            checks,
            "nested_scale_gap",
            "nested",
            "absolute weighted-F1 gap to 0.877258 <= 0.005",
            "waiting for all five folds",
        )
        return {
            "complete": False,
            "completed_outer_folds": completed.tolist(),
            "finite_final_rows": finite_final_rows,
            "nested_oof_sha256": sha256_file(nested_path),
        }

    splitter = StratifiedKFold(5, shuffle=True, random_state=42)
    expected_fold = np.full(len(y), -1, dtype=np.int8)
    for fold, (_, validation_idx) in enumerate(
        splitter.split(np.zeros(len(y)), y), start=1
    ):
        expected_fold[validation_idx] = fold
    base_integrity_ok = bool(
        nested_y.shape == (EXPECTED_TRAIN_ROWS,)
        and np.array_equal(nested_y, y)
        and all(
            probability_vector_is_valid(vector, EXPECTED_TRAIN_ROWS)
            for vector in (teacher_oof, student_oof, final_oof)
        )
        and outer_fold.shape == (EXPECTED_TRAIN_ROWS,)
        and np.array_equal(outer_fold, expected_fold)
        and all(int(np.sum(outer_fold == fold)) == 4_800 for fold in range(1, 6))
    )

    fold_audit = audit_fold_artifacts(checks, run_signature, y)
    assembly_ok = bool(
        fold_audit["complete"]
        and np.array_equal(teacher_oof, fold_audit["teacher"])
        and np.array_equal(student_oof, fold_audit["student"])
        and np.array_equal(final_oof, fold_audit["final"])
        and np.array_equal(outer_fold, fold_audit["fold"])
    )

    completion_path = NESTED_DIR / "checkpoint_completion.csv"
    completion_ok = False
    if completion_path.is_file():
        completion = pd.read_csv(completion_path)
        completion_ok = bool(
            len(completion) == 5
            and (completion["teacher_models_complete"] == 90).all()
            and completion["teacher_complete"].all()
            and (completion["student_models_complete"] == 30).all()
            and completion["student_complete"].all()
            and completion["final_fold_artifact_complete"].all()
        )
    five_fold_integrity_ok = base_integrity_ok and assembly_ok and completion_ok
    boolean_check(
        checks,
        "nested_five_fold_integrity",
        "nested",
        five_fold_integrity_ok,
        "5/5 folds, 24,000 finite aligned OOF rows, exact fold/artifact coverage",
        {
            "base_integrity_ok": base_integrity_ok,
            "fold_assembly_ok": assembly_ok,
            "checkpoint_completion_ok": completion_ok,
        },
    )

    predictions = top_rate_predictions(final_oof)
    weighted_f1 = float(
        f1_score(y, predictions, average="weighted", zero_division=0)
    )
    gap = weighted_f1 - SUBMISSION6_PUBLIC_LB
    scale_pass = bool(abs(gap) <= SCALE_GAP_ABS_TOLERANCE)
    boolean_check(
        checks,
        "nested_scale_gap",
        "nested",
        scale_pass,
        "absolute weighted-F1 gap to 0.877258 <= 0.005",
        {"weighted_f1": weighted_f1, "gap": gap},
    )

    global_metrics_path = NESTED_DIR / "global_metrics.csv"
    global_metrics_ok = False
    global_metrics_observed: Any = "missing or empty"
    if global_metrics_path.is_file() and global_metrics_path.stat().st_size > 1:
        try:
            global_metrics = pd.read_csv(global_metrics_path)
            selected = global_metrics[
                global_metrics["scope"] == "submission6_nested"
            ]
            if len(selected) == 1:
                row = selected.iloc[0]
                global_metrics_ok = bool(
                    bool(row["is_complete_5_fold"])
                    and int(row["finite_oof_rows"]) == EXPECTED_TRAIN_ROWS
                    and math.isclose(
                        float(row["official_support_weighted_f1"]),
                        weighted_f1,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                    and math.isclose(
                        float(row["nested_minus_public_lb"]),
                        gap,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                    and bool(row["sanity_scale_pass"]) == scale_pass
                    and not bool(row["production_probability_replay_verified"])
                    and not bool(row["canonical_reference_approved"])
                )
                global_metrics_observed = row.to_dict()
        except (pd.errors.EmptyDataError, KeyError, ValueError) as exc:
            global_metrics_observed = f"{type(exc).__name__}: {exc}"
    boolean_check(
        checks,
        "nested_global_metrics_consistency",
        "nested",
        global_metrics_ok,
        (
            "saved global row matches recomputation and preserves the original "
            "strict-replay approval as false"
        ),
        global_metrics_observed,
    )
    return {
        "complete": declared_complete,
        "integrity_pass": five_fold_integrity_ok and global_metrics_ok,
        "scale_pass": scale_pass,
        "weighted_f1": weighted_f1,
        "nested_minus_public_lb": gap,
        "completed_outer_folds": completed.tolist(),
        "finite_final_rows": finite_final_rows,
        "nested_oof_sha256": sha256_file(nested_path),
    }


def section_pass(checks: list[Check], sections: set[str]) -> bool:
    selected = [check for check in checks if check.required and check.section in sections]
    return bool(selected) and all(check.status == "pass" for check in selected)


def write_report(output_path: Path, result: dict[str, Any], checks: list[Check]) -> None:
    counts = pd.Series([check.status for check in checks]).value_counts()
    status = result["status"]
    if status == "approved":
        headline = "APPROVED — practical replay and five-fold nested reference passed"
    elif status.startswith("pending"):
        headline = "PENDING — canonical nested reference is not approved"
    else:
        headline = "REJECTED — canonical nested reference is not approved"

    lines = [
        "# Submission 6 Practical Reference Gate",
        "",
        f"**Status: {headline}.**",
        "",
        (
            "This is a separate practical-recipe acceptance decision. It does not "
            "replace or alter the reconstruction's strict probability-equivalence "
            "result, and it never claims byte-identical probabilities."
        ),
        "",
        f"- Practical recipe replay accepted: `{result['practical_recipe_replay_accepted']}`",
        (
            "- Canonical nested recipe reference approved: "
            f"`{result['canonical_nested_recipe_reference_approved']}`"
        ),
        (
            "- Strict probability equivalence (preserved): "
            f"`{result['strict_probability_equivalence_verified']}`"
        ),
        (
            "- Strict recipe-artifact equivalence (preserved): "
            f"`{result['strict_recipe_artifact_equivalence_verified']}`"
        ),
        f"- Checks: pass `{int(counts.get('pass', 0))}`, fail `{int(counts.get('fail', 0))}`, pending `{int(counts.get('pending', 0))}`.",
        "",
        "## Locked practical criteria",
        "",
        "- Current code/input/recipe/run signatures must agree.",
        "- Teacher and final replay MAE/RMSE must be at most 1e-8, maximum "
        "absolute error at most 1e-7, and Pearson/Spearman at least 0.999999999.",
        "- Pseudo mask and labels must exactly match the archived teacher rule.",
        "- Historical threshold must be 0.684 (floating tolerance 1e-12), with "
        "30,411 Dead and zero historical/top-k label changes.",
        "- All structural isolation audits must pass.",
        "- The nested vector must cover all 24,000 rows exactly once across five "
        "folds, remain finite/aligned, and be within 0.005 weighted F1 of 0.877258.",
        "",
        "## Check table",
        "",
        "```csv",
        pd.DataFrame(asdict(check) for check in checks).to_csv(index=False).rstrip(),
        "```",
        "",
        "No model, probability vector, or submission file was created or changed.",
        "",
    ]
    atomic_write_text(output_path, "\n".join(lines))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    checks: list[Check] = []

    manifest_path = NESTED_DIR / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Current nested run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_signature = audit_signatures_and_inputs(checks, manifest)
    audit_structural_isolation(checks)
    replay_result = audit_production_replay(checks, run_signature)
    nested_result = audit_nested_reference(checks, run_signature)

    practical_sections = {"signature", "structural", "replay"}
    practical_accepted = section_pass(checks, practical_sections)
    nested_accepted = section_pass(checks, {"nested"})
    canonical_approved = bool(practical_accepted and nested_accepted)

    if canonical_approved:
        status = "approved"
    elif any(
        check.required
        and check.status == "fail"
        and check.section in practical_sections
        for check in checks
    ):
        status = "rejected_practical_replay"
    elif any(check.required and check.status == "pending" for check in checks):
        status = "pending_nested_oof"
    else:
        status = "rejected_nested_reference"

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "run_signature": run_signature,
        "definition": (
            "separate tight practical recipe-replay acceptance plus complete nested "
            "OOF integrity/scale gate; never byte-identical probability equivalence"
        ),
        "practical_recipe_replay_accepted": practical_accepted,
        "nested_reference_integrity_accepted": nested_accepted,
        "canonical_nested_recipe_reference_approved": canonical_approved,
        "strict_probability_equivalence_verified": bool(
            replay_result.get("strict_probability_equivalence_verified", False)
        ),
        "strict_recipe_artifact_equivalence_verified": bool(
            replay_result.get("strict_recipe_artifact_equivalence_verified", False)
        ),
        "strict_equivalence_note": (
            "Preserved from production_replay_comparison.json; practical acceptance "
            "does not overwrite or reinterpret this field."
        ),
        "candidate_name_for_validation_harness": (
            "submission6_nested_recipe_reference"
        ),
        "nested_oof_path": str(
            (NESTED_DIR / "nested_oof_predictions.npz").relative_to(ROOT)
        ),
        "nested_oof_key": "submission6_nested_oof",
        "nested_oof_sha256": nested_result.get("nested_oof_sha256"),
        "official_support_weighted_f1": nested_result.get("weighted_f1"),
        "nested_minus_public_lb": nested_result.get("nested_minus_public_lb"),
        "submission6_public_lb_reference": SUBMISSION6_PUBLIC_LB,
        "scale_gap_absolute_tolerance": SCALE_GAP_ABS_TOLERANCE,
        "practical_thresholds": {
            "maximum_mean_absolute_error": MAX_MAE,
            "maximum_root_mean_squared_error": MAX_RMSE,
            "maximum_absolute_error": MAX_ABSOLUTE_ERROR,
            "minimum_pearson": MIN_PEARSON,
            "minimum_spearman": MIN_SPEARMAN,
            "historical_threshold": EXPECTED_HISTORICAL_THRESHOLD,
            "historical_threshold_absolute_tolerance": HISTORICAL_THRESHOLD_ABS_TOLERANCE,
            "required_dead_count": EXPECTED_SUBMISSION_DEAD_COUNT,
            "maximum_historical_label_changes": 0,
            "maximum_top_k_label_changes": 0,
        },
        "replay_summary": replay_result,
        "nested_summary": nested_result,
        "check_summary": {
            status_name: int(sum(check.status == status_name for check in checks))
            for status_name in ("pass", "fail", "pending")
        },
        "new_probability_vector_saved": False,
        "new_submission_saved": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        output_dir / "submission6_practical_reference_acceptance.json",
        json.dumps(result, indent=2, default=str) + "\n",
    )
    atomic_write_csv(
        pd.DataFrame(asdict(check) for check in checks),
        output_dir / "submission6_practical_reference_gate_checks.csv",
    )
    write_report(
        output_dir / "submission6_practical_reference_gate_report.md",
        result,
        checks,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
