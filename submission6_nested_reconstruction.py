"""Leakage-safe nested-equivalent reconstruction of Submission 6.

This audit reproduces the modelling recipe in ``pipeline_v6.py`` inside five
outer folds.  For every outer fold, all supervised operations exclude the
outer validation rows:

* a 3 x 5-fold, early-stopped v5/v4 teacher is fitted on the outer-training
  rows and predicts both the untouched validation rows and competition test;
* the teacher's competition-test predictions select >=95% / <=5% pseudo
  labels;
* a six-model, five-fold, early-stopped student is fitted to outer-training
  plus those pseudo labels; and
* the final validation score is the 50/50 teacher/student blend.

The official competition metric is support-weighted F1.  It is evaluated with
the repository's fixed top-84.5% decision policy.  Binary Dead-class F1 is
reported only as a legacy diagnostic.

The computation is intentionally expensive (600 boosted-tree fits for all
five outer folds).  Model-level teacher/student checkpoints make the run
resumable.  This script never writes a test submission or final candidate test
probability vector.

Recipe-equivalence boundary
---------------------------
The model families, parameters, weights, teacher repeats/folds, early stopping,
pseudo thresholds, student folds, and final blend match ``pipeline_v6.py``.
The outer split is necessarily a nested analogue: each model sees 80% of the
labelled rows rather than the production model's 100%.  The ordinal encoder and
train-domain frequency maps are therefore fitted on outer-training only;
competition test retains its production test-domain frequency features.  These
differences are recorded in ``recipe_equivalence_audit.csv``; the output is
called nested-equivalent, never an exact reproduction of the submitted test
probabilities.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import catboost
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "pipeline_v6.py"
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test.csv"
OUTPUT_DIR = ROOT / "diagnostic_outputs" / "submission6_nested"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FOLD_DIR = OUTPUT_DIR / "folds"

SEED = 42
N_OUTER_FOLDS = 5
N_TEACHER_REPEATS = 3
N_TEACHER_FOLDS = 5
N_STUDENT_FOLDS = 5
EARLY_STOPPING_ROUNDS = 80
TEACHER_V5_WEIGHT = 0.55
TEACHER_V4_WEIGHT = 0.45
FINAL_TEACHER_WEIGHT = 0.50
FINAL_STUDENT_WEIGHT = 0.50
PSEUDO_DEAD_THRESHOLD = 0.95
PSEUDO_ALIVE_THRESHOLD = 0.05
FIXED_DEAD_RATE = 0.845
SUBMISSION6_PUBLIC_LB = 0.877258
SANITY_ABS_GAP_TOLERANCE = 0.005
PRACTICAL_MIN_PEARSON = 0.9999
PRACTICAL_MIN_SPEARMAN = 0.9999
PRACTICAL_MAX_PRODUCTION_LABEL_CHANGES = 0
SCHEMA_VERSION = "submission6_nested_v1"

MODEL_LABELS = (
    "v5_lgb",
    "v5_xgb",
    "v5_cb",
    "v4_lgb",
    "v4_xgb",
    "v4_cb",
)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    branch: str
    weight: float
    constructor: Callable[..., Any]
    parameters: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outer-folds",
        default="all",
        help="Comma-separated 1-based folds to run, or 'all' (default).",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write provenance/equivalence files and validate preprocessing; fit no models.",
    )
    parser.add_argument(
        "--production-replay",
        action="store_true",
        help=(
            "Opt-in full-data v6 replay against archived teacher/final probabilities. "
            "This fits 120 models, writes comparison diagnostics, and no submission."
        ),
    )
    return parser.parse_args()


def parse_outer_folds(value: str) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(N_OUTER_FOLDS))
    folds: list[int] = []
    for token in value.split(","):
        fold = int(token.strip())
        if fold < 1 or fold > N_OUTER_FOLDS:
            raise ValueError(f"Outer fold must be in 1..{N_OUTER_FOLDS}: {fold}")
        folds.append(fold - 1)
    return sorted(set(folds))


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


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _assigned_names(node: ast.Assign) -> set[str]:
    names: set[str] = set()

    def visit(target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                visit(element)

    for target in node.targets:
        visit(target)
    return names


def load_frozen_recipe() -> dict[str, Any]:
    """Extract recipe definitions without executing the production pipeline."""

    wanted_functions = {"build_features", "smoothed_te"}
    wanted_assignments = {
        "SMOOTHING_M",
        "cat_cols",
        "te_cols",
        "v5_lgb",
        "v5_xgb",
        "v5_cb",
        "v4_lgb",
        "v4_xgb",
        "v4_cb",
        "W5_LGB",
        "W5_XGB",
        "W5_CB",
        "W4_LGB",
        "W4_XGB",
        "W4_CB",
    }
    parsed = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), SOURCE_PATH.name)
    selected: list[ast.stmt] = []
    found: set[str] = set()
    for node in parsed.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in wanted_functions:
                selected.append(node)
                found.add(node.name)
        elif isinstance(node, ast.Assign):
            names = _assigned_names(node)
            if names & wanted_assignments:
                selected.append(node)
                found.update(names & wanted_assignments)
    missing = (wanted_functions | wanted_assignments) - found
    if missing:
        raise RuntimeError(f"Could not extract recipe definitions: {sorted(missing)}")
    namespace: dict[str, Any] = {
        "np": np,
        "pd": pd,
        "LEAKAGE_CLEAN": False,
    }
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, SOURCE_PATH.name, "exec"), namespace)
    return namespace


def recipe_payload(recipe: dict[str, Any]) -> dict[str, Any]:
    names = (
        "SMOOTHING_M",
        "cat_cols",
        "te_cols",
        "v5_lgb",
        "v5_xgb",
        "v5_cb",
        "v4_lgb",
        "v4_xgb",
        "v4_cb",
        "W5_LGB",
        "W5_XGB",
        "W5_CB",
        "W4_LGB",
        "W4_XGB",
        "W4_CB",
    )
    payload = {name: recipe[name] for name in names}
    payload["nested_constants"] = {
        "outer_folds": N_OUTER_FOLDS,
        "outer_seed": SEED,
        "teacher_repeats": N_TEACHER_REPEATS,
        "teacher_folds": N_TEACHER_FOLDS,
        "teacher_repeat_seeds": [SEED + repeat * 111 for repeat in range(N_TEACHER_REPEATS)],
        "teacher_v5_weight": TEACHER_V5_WEIGHT,
        "teacher_v4_weight": TEACHER_V4_WEIGHT,
        "student_folds": N_STUDENT_FOLDS,
        "student_seed": SEED + 999,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "pseudo_dead_threshold": PSEUDO_DEAD_THRESHOLD,
        "pseudo_alive_threshold": PSEUDO_ALIVE_THRESHOLD,
        "final_teacher_weight": FINAL_TEACHER_WEIGHT,
        "final_student_weight": FINAL_STUDENT_WEIGHT,
        "fixed_dead_rate": FIXED_DEAD_RATE,
    }
    return payload


def model_specs(recipe: dict[str, Any]) -> list[ModelSpec]:
    return [
        ModelSpec("v5_lgb", "v5", float(recipe["W5_LGB"]), lgb.LGBMClassifier, dict(recipe["v5_lgb"])),
        ModelSpec("v5_xgb", "v5", float(recipe["W5_XGB"]), xgb.XGBClassifier, dict(recipe["v5_xgb"])),
        ModelSpec("v5_cb", "v5", float(recipe["W5_CB"]), CatBoostClassifier, dict(recipe["v5_cb"])),
        ModelSpec("v4_lgb", "v4", float(recipe["W4_LGB"]), lgb.LGBMClassifier, dict(recipe["v4_lgb"])),
        ModelSpec("v4_xgb", "v4", float(recipe["W4_XGB"]), xgb.XGBClassifier, dict(recipe["v4_xgb"])),
        ModelSpec("v4_cb", "v4", float(recipe["W4_CB"]), CatBoostClassifier, dict(recipe["v4_cb"])),
    ]


def as_float_matrix(frame: pd.DataFrame | np.ndarray) -> np.ndarray:
    values = frame.to_numpy() if isinstance(frame, pd.DataFrame) else frame
    return np.nan_to_num(values.astype(np.float32), nan=-999.0)


def build_outer_base_features(
    fit_raw: pd.DataFrame,
    validation_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    recipe: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build production features while fitting the encoder on outer-fit only."""

    build_features = recipe["build_features"]
    fit_numeric = build_features(fit_raw).reset_index(drop=True)
    validation_numeric = build_features(validation_raw).reset_index(drop=True)
    test_numeric = build_features(test_raw).reset_index(drop=True)

    # pipeline_v6 builds one train-domain feature matrix before any CV split.
    # Therefore a held-out labelled row must receive frequency values learned
    # from outer-fit (the leakage-safe analogue of full labelled train), not
    # frequencies recomputed within the 4,800-row validation population.
    # Competition test remains its own domain exactly as in production.
    for column in ("primary_site", "histologic_type_icdo3"):
        fit_frequency = (
            fit_raw[column].astype(str).value_counts(normalize=True)
        )
        validation_numeric[f"{column}_freq"] = (
            validation_raw[column]
            .astype(str)
            .map(fit_frequency)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

    categorical_columns = list(recipe["cat_cols"])
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    fit_categorical = fit_raw[categorical_columns].fillna("__NA__").astype(str)
    validation_categorical = validation_raw[categorical_columns].fillna("__NA__").astype(str)
    test_categorical = test_raw[categorical_columns].fillna("__NA__").astype(str)
    encoder.fit(fit_categorical)
    encoded_columns = [f"{column}_enc" for column in categorical_columns]
    fit_encoded = pd.DataFrame(encoder.transform(fit_categorical), columns=encoded_columns)
    validation_encoded = pd.DataFrame(
        encoder.transform(validation_categorical), columns=encoded_columns
    )
    test_encoded = pd.DataFrame(encoder.transform(test_categorical), columns=encoded_columns)
    fit_base = pd.concat([fit_numeric, fit_encoded], axis=1)
    validation_base = pd.concat([validation_numeric, validation_encoded], axis=1)
    test_base = pd.concat([test_numeric, test_encoded], axis=1)
    if not (fit_base.columns.equals(validation_base.columns) and fit_base.columns.equals(test_base.columns)):
        raise RuntimeError("Feature columns drifted across outer populations")
    details = {
        "base_feature_count": int(fit_base.shape[1]),
        "categorical_feature_count": len(categorical_columns),
        "unknown_validation_category_cells": int((validation_encoded == -1).sum().sum()),
        "unknown_test_category_cells": int((test_encoded == -1).sum().sum()),
    }
    return fit_base, validation_base, test_base, details


def map_target_encoding(
    train_raw: pd.DataFrame,
    y_train: np.ndarray,
    evaluation_raw: pd.DataFrame,
    recipe: dict[str, Any],
) -> pd.DataFrame:
    encoded = pd.DataFrame(index=np.arange(len(evaluation_raw)))
    for column in recipe["te_cols"]:
        _, values, _ = recipe["smoothed_te"](
            train_raw[column].reset_index(drop=True),
            y_train,
            evaluation_raw[column].reset_index(drop=True),
        )
        encoded[f"{column}_te"] = values
    return encoded


def cross_fitted_target_encoding(
    train_raw: pd.DataFrame,
    y_train: np.ndarray,
    recipe: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    encoded = pd.DataFrame(index=np.arange(len(train_raw)))
    for column in recipe["te_cols"]:
        source = train_raw[column].reset_index(drop=True)
        values = np.zeros(len(train_raw), dtype=np.float64)
        splitter = StratifiedKFold(5, shuffle=True, random_state=seed)
        for fit_idx, held_idx in splitter.split(source, y_train):
            _, held_values, _ = recipe["smoothed_te"](
                source.iloc[fit_idx], y_train[fit_idx], source.iloc[held_idx]
            )
            values[held_idx] = held_values
        encoded[f"{column}_te"] = values
    return encoded


def fit_model(
    spec: ModelSpec,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_early_stop: np.ndarray,
    y_early_stop: np.ndarray,
) -> Any:
    parameters = dict(spec.parameters)
    if spec.label.endswith("lgb"):
        model = spec.constructor(**parameters)
        model.fit(
            x_fit,
            y_fit,
            eval_set=[(x_early_stop, y_early_stop)],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
    elif spec.label.endswith("xgb"):
        model = spec.constructor(
            **parameters, early_stopping_rounds=EARLY_STOPPING_ROUNDS
        )
        model.fit(
            x_fit,
            y_fit,
            eval_set=[(x_early_stop, y_early_stop)],
            verbose=False,
        )
    else:
        model = spec.constructor(
            **parameters, early_stopping_rounds=EARLY_STOPPING_ROUNDS
        )
        model.fit(
            x_fit,
            y_fit,
            eval_set=(x_early_stop, y_early_stop),
            verbose=False,
        )
    return model


def best_iteration(model: Any, label: str) -> int:
    if label.endswith("lgb"):
        return int(getattr(model, "best_iteration_", -1))
    if label.endswith("xgb"):
        value = getattr(model, "best_iteration", -1)
        return int(value) + 1 if int(value) >= 0 else -1
    value = int(model.get_best_iteration())
    return value + 1 if value >= 0 else -1


def validate_checkpoint_signature(checkpoint: Any, run_signature: str, path: Path) -> None:
    saved = str(checkpoint["run_signature"].item())
    if saved != run_signature:
        raise RuntimeError(
            f"Checkpoint signature mismatch at {path}. Refusing to mix recipes: "
            f"saved={saved}, current={run_signature}"
        )


def train_teacher(
    outer_fold: int,
    fit_base: pd.DataFrame,
    fit_raw: pd.DataFrame,
    y_fit: np.ndarray,
    validation_base: pd.DataFrame,
    validation_raw: pd.DataFrame,
    test_base: pd.DataFrame,
    test_raw: pd.DataFrame,
    recipe: dict[str, Any],
    run_signature: str,
    checkpoint_prefix: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit/resume the production 3x5 teacher inside one outer fold."""

    prefix = checkpoint_prefix or f"outer_fold_{outer_fold + 1:02d}"
    checkpoint_path = CHECKPOINT_DIR / f"{prefix}_teacher.npz"
    shape_validation = len(validation_base)
    shape_test = len(test_base)
    complete = np.zeros((N_TEACHER_REPEATS, N_TEACHER_FOLDS, len(MODEL_LABELS)), dtype=bool)
    best_iterations = np.full(complete.shape, -1, dtype=np.int32)
    v5_validation = np.zeros(shape_validation, dtype=np.float64)
    v4_validation = np.zeros(shape_validation, dtype=np.float64)
    v5_test = np.zeros(shape_test, dtype=np.float64)
    v4_test = np.zeros(shape_test, dtype=np.float64)
    if checkpoint_path.exists():
        checkpoint = np.load(checkpoint_path, allow_pickle=False)
        validate_checkpoint_signature(checkpoint, run_signature, checkpoint_path)
        complete = checkpoint["complete"].astype(bool)
        best_iterations = checkpoint["best_iterations"].astype(np.int32)
        v5_validation = checkpoint["v5_validation"].astype(np.float64)
        v4_validation = checkpoint["v4_validation"].astype(np.float64)
        v5_test = checkpoint["v5_test"].astype(np.float64)
        v4_test = checkpoint["v4_test"].astype(np.float64)
        print(
            f"    teacher checkpoint: {int(complete.sum())}/{complete.size} models complete",
            flush=True,
        )

    specifications = model_specs(recipe)
    total_teacher_splits = N_TEACHER_REPEATS * N_TEACHER_FOLDS
    validation_v5_matrix = as_float_matrix(validation_base)
    test_v5_matrix = as_float_matrix(test_base)
    # Production predicts competition test with TE learned from all labelled
    # training rows.  Here the strict outer analogue learns it from outer-fit.
    validation_full_te = map_target_encoding(fit_raw, y_fit, validation_raw, recipe)
    test_full_te = map_target_encoding(fit_raw, y_fit, test_raw, recipe)
    validation_v4_matrix = as_float_matrix(
        np.hstack([validation_base.to_numpy(), validation_full_te.to_numpy()])
    )
    test_v4_matrix = as_float_matrix(
        np.hstack([test_base.to_numpy(), test_full_te.to_numpy()])
    )

    def save_checkpoint() -> None:
        atomic_save_npz(
            checkpoint_path,
            run_signature=np.array(run_signature),
            complete=complete,
            best_iterations=best_iterations,
            v5_validation=v5_validation,
            v4_validation=v4_validation,
            v5_test=v5_test,
            v4_test=v4_test,
        )

    for repeat in range(N_TEACHER_REPEATS):
        repeat_seed = SEED + repeat * 111
        splitter = StratifiedKFold(
            N_TEACHER_FOLDS, shuffle=True, random_state=repeat_seed
        )
        for inner_fold, (inner_fit_idx, inner_stop_idx) in enumerate(
            splitter.split(fit_base, y_fit)
        ):
            incomplete = np.flatnonzero(~complete[repeat, inner_fold])
            if len(incomplete) == 0:
                continue
            print(
                f"    teacher R{repeat + 1}/3 F{inner_fold + 1}/5: "
                f"{len(incomplete)} models pending",
                flush=True,
            )
            x_v5_fit = as_float_matrix(fit_base.iloc[inner_fit_idx])
            x_v5_stop = as_float_matrix(fit_base.iloc[inner_stop_idx])

            x_v4_fit: np.ndarray | None = None
            x_v4_stop: np.ndarray | None = None
            if np.any(incomplete >= 3):
                inner_fit_raw = fit_raw.iloc[inner_fit_idx].reset_index(drop=True)
                inner_stop_raw = fit_raw.iloc[inner_stop_idx].reset_index(drop=True)
                inner_y = y_fit[inner_fit_idx]
                fit_te = cross_fitted_target_encoding(
                    inner_fit_raw, inner_y, recipe, repeat_seed
                )
                stop_te = map_target_encoding(
                    inner_fit_raw, inner_y, inner_stop_raw, recipe
                )
                x_v4_fit = as_float_matrix(
                    np.hstack(
                        [fit_base.iloc[inner_fit_idx].to_numpy(), fit_te.to_numpy()]
                    )
                )
                x_v4_stop = as_float_matrix(
                    np.hstack(
                        [fit_base.iloc[inner_stop_idx].to_numpy(), stop_te.to_numpy()]
                    )
                )

            for model_index in incomplete:
                spec = specifications[int(model_index)]
                if spec.branch == "v5":
                    x_fit_model, x_stop_model = x_v5_fit, x_v5_stop
                    validation_matrix, test_matrix = (
                        validation_v5_matrix,
                        test_v5_matrix,
                    )
                else:
                    if x_v4_fit is None or x_v4_stop is None:
                        raise AssertionError("v4 TE matrices were not built")
                    x_fit_model, x_stop_model = x_v4_fit, x_v4_stop
                    validation_matrix, test_matrix = (
                        validation_v4_matrix,
                        test_v4_matrix,
                    )
                started = time.monotonic()
                model = fit_model(
                    spec,
                    x_fit_model,
                    y_fit[inner_fit_idx],
                    x_stop_model,
                    y_fit[inner_stop_idx],
                )
                contribution = spec.weight / total_teacher_splits
                if spec.branch == "v5":
                    v5_validation += contribution * model.predict_proba(validation_matrix)[:, 1]
                    v5_test += contribution * model.predict_proba(test_matrix)[:, 1]
                else:
                    v4_validation += contribution * model.predict_proba(validation_matrix)[:, 1]
                    v4_test += contribution * model.predict_proba(test_matrix)[:, 1]
                best_iterations[repeat, inner_fold, model_index] = best_iteration(
                    model, spec.label
                )
                complete[repeat, inner_fold, model_index] = True
                save_checkpoint()
                print(
                    f"      {spec.label}: best_iter="
                    f"{best_iterations[repeat, inner_fold, model_index]} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
                del model

    if not complete.all():
        raise RuntimeError("Teacher phase ended with incomplete model checkpoints")
    teacher_validation = (
        TEACHER_V5_WEIGHT * v5_validation + TEACHER_V4_WEIGHT * v4_validation
    )
    teacher_test = TEACHER_V5_WEIGHT * v5_test + TEACHER_V4_WEIGHT * v4_test
    return teacher_validation, teacher_test, best_iterations


def train_student(
    outer_fold: int,
    fit_base: pd.DataFrame,
    fit_raw: pd.DataFrame,
    y_fit: np.ndarray,
    validation_base: pd.DataFrame,
    validation_raw: pd.DataFrame,
    test_base: pd.DataFrame,
    test_raw: pd.DataFrame,
    teacher_test: np.ndarray,
    recipe: dict[str, Any],
    run_signature: str,
    checkpoint_prefix: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit/resume the production five-fold pseudo student."""

    pseudo_dead = teacher_test >= PSEUDO_DEAD_THRESHOLD
    pseudo_alive = teacher_test <= PSEUDO_ALIVE_THRESHOLD
    pseudo_mask = pseudo_dead | pseudo_alive
    pseudo_y = pseudo_dead[pseudo_mask].astype(np.int8)
    if len(pseudo_y) <= 500:
        raise RuntimeError(
            f"Only {len(pseudo_y)} pseudo labels in outer fold {outer_fold + 1}; "
            "production would skip student retraining"
        )

    combined_base = pd.concat(
        [fit_base.reset_index(drop=True), test_base.loc[pseudo_mask].reset_index(drop=True)],
        ignore_index=True,
    )
    combined_raw = pd.concat(
        [fit_raw.reset_index(drop=True), test_raw.loc[pseudo_mask].reset_index(drop=True)],
        ignore_index=True,
    )
    combined_y = np.concatenate([y_fit, pseudo_y])
    combined_te = cross_fitted_target_encoding(
        combined_raw, combined_y, recipe, SEED
    )
    validation_te = map_target_encoding(
        combined_raw, combined_y, validation_raw, recipe
    )
    combined_v5 = as_float_matrix(combined_base)
    combined_v4 = as_float_matrix(
        np.hstack([combined_base.to_numpy(), combined_te.to_numpy()])
    )
    validation_v5 = as_float_matrix(validation_base)
    validation_v4 = as_float_matrix(
        np.hstack([validation_base.to_numpy(), validation_te.to_numpy()])
    )

    prefix = checkpoint_prefix or f"outer_fold_{outer_fold + 1:02d}"
    checkpoint_path = CHECKPOINT_DIR / f"{prefix}_student.npz"
    complete = np.zeros((N_STUDENT_FOLDS, len(MODEL_LABELS)), dtype=bool)
    best_iterations = np.full(complete.shape, -1, dtype=np.int32)
    v5_validation = np.zeros(len(validation_base), dtype=np.float64)
    v4_validation = np.zeros(len(validation_base), dtype=np.float64)
    if checkpoint_path.exists():
        checkpoint = np.load(checkpoint_path, allow_pickle=False)
        validate_checkpoint_signature(checkpoint, run_signature, checkpoint_path)
        saved_pseudo_mask_sha = str(checkpoint["pseudo_mask_sha256"].item())
        saved_pseudo_y_sha = str(checkpoint["pseudo_y_sha256"].item())
        current_pseudo_mask_sha = hashlib.sha256(pseudo_mask.tobytes()).hexdigest()
        current_pseudo_y_sha = hashlib.sha256(pseudo_y.tobytes()).hexdigest()
        if (
            saved_pseudo_mask_sha != current_pseudo_mask_sha
            or saved_pseudo_y_sha != current_pseudo_y_sha
        ):
            raise RuntimeError(
                f"Pseudo-label mask/assignments changed for outer fold {outer_fold + 1}; "
                "refusing to resume incompatible student"
            )
        complete = checkpoint["complete"].astype(bool)
        best_iterations = checkpoint["best_iterations"].astype(np.int32)
        v5_validation = checkpoint["v5_validation"].astype(np.float64)
        v4_validation = checkpoint["v4_validation"].astype(np.float64)
        print(
            f"    student checkpoint: {int(complete.sum())}/{complete.size} models complete",
            flush=True,
        )

    pseudo_mask_sha = hashlib.sha256(pseudo_mask.tobytes()).hexdigest()
    pseudo_y_sha = hashlib.sha256(pseudo_y.tobytes()).hexdigest()

    def save_checkpoint() -> None:
        atomic_save_npz(
            checkpoint_path,
            run_signature=np.array(run_signature),
            pseudo_mask_sha256=np.array(pseudo_mask_sha),
            pseudo_y_sha256=np.array(pseudo_y_sha),
            pseudo_mask=pseudo_mask,
            pseudo_y=pseudo_y,
            complete=complete,
            best_iterations=best_iterations,
            v5_validation=v5_validation,
            v4_validation=v4_validation,
        )

    specifications = model_specs(recipe)
    splitter = StratifiedKFold(
        N_STUDENT_FOLDS, shuffle=True, random_state=SEED + 999
    )
    for student_fold, (student_fit_idx, student_stop_idx) in enumerate(
        splitter.split(combined_v5, combined_y)
    ):
        incomplete = np.flatnonzero(~complete[student_fold])
        if len(incomplete) == 0:
            continue
        print(
            f"    student F{student_fold + 1}/5: {len(incomplete)} models pending",
            flush=True,
        )
        for model_index in incomplete:
            spec = specifications[int(model_index)]
            if spec.branch == "v5":
                x_matrix, validation_matrix = combined_v5, validation_v5
            else:
                x_matrix, validation_matrix = combined_v4, validation_v4
            started = time.monotonic()
            model = fit_model(
                spec,
                x_matrix[student_fit_idx],
                combined_y[student_fit_idx],
                x_matrix[student_stop_idx],
                combined_y[student_stop_idx],
            )
            contribution = spec.weight / N_STUDENT_FOLDS
            if spec.branch == "v5":
                v5_validation += contribution * model.predict_proba(validation_matrix)[:, 1]
            else:
                v4_validation += contribution * model.predict_proba(validation_matrix)[:, 1]
            best_iterations[student_fold, model_index] = best_iteration(model, spec.label)
            complete[student_fold, model_index] = True
            save_checkpoint()
            print(
                f"      {spec.label}: best_iter="
                f"{best_iterations[student_fold, model_index]} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
            del model

    if not complete.all():
        raise RuntimeError("Student phase ended with incomplete model checkpoints")
    student_validation = (
        TEACHER_V5_WEIGHT * v5_validation + TEACHER_V4_WEIGHT * v4_validation
    )
    pseudo_counts = np.array(
        [len(pseudo_y), int(pseudo_y.sum()), int((pseudo_y == 0).sum())],
        dtype=np.int64,
    )
    return student_validation, pseudo_counts, best_iterations


def top_rate_predictions(scores: np.ndarray, rate: float = FIXED_DEAD_RATE) -> np.ndarray:
    count = int(round(len(scores) * rate))
    if count <= 0 or count > len(scores):
        raise ValueError(f"Invalid top-rate count {count} for {len(scores)} scores")
    # Match the canonical policy used by the corrected validation audits:
    # descending stable sort, so an exact tie favors the earlier row.
    order = np.argsort(-scores, kind="mergesort")
    predictions = np.zeros(len(scores), dtype=np.int8)
    predictions[order[:count]] = 1
    return predictions


def historical_v6_threshold_predictions(
    scores: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Mirror pipeline_v6.py's submitted-label threshold scan exactly."""

    selected_threshold: float | None = None
    for threshold in np.arange(0.40, 0.75, 0.001):
        if float((scores >= threshold).mean()) <= FIXED_DEAD_RATE:
            selected_threshold = float(threshold)
            break
    if selected_threshold is None:
        raise RuntimeError("Historical v6 threshold scan did not find a threshold")
    return (scores >= selected_threshold).astype(np.int8), selected_threshold


def metric_row(
    scope: str,
    scores: np.ndarray,
    y_true: np.ndarray,
    completed_outer_folds: int,
) -> dict[str, Any]:
    predictions = top_rate_predictions(scores)
    return {
        "scope": scope,
        "rows": len(y_true),
        "completed_outer_folds": completed_outer_folds,
        "fixed_dead_rate_target": FIXED_DEAD_RATE,
        "fixed_dead_rate_actual": float(predictions.mean()),
        "official_support_weighted_f1": float(
            f1_score(y_true, predictions, average="weighted", zero_division=0)
        ),
        "legacy_dead_class_f1": float(
            f1_score(y_true, predictions, average="binary", zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }


def write_provenance(recipe: dict[str, Any]) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    candidate_inputs = [
        Path(__file__).resolve(),
        SOURCE_PATH,
        TRAIN_PATH,
        TEST_PATH,
        ROOT / "submission6.csv",
        ROOT / "archive" / "submission6.csv",
        ROOT / "archive" / "probs_v6_blend.npy",
        ROOT / "archive" / "probs_v6_final.npy",
    ]
    provenance_rows = []
    for path in candidate_inputs:
        provenance_rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
                "sha256": sha256_file(path) if path.exists() else None,
            }
        )
    atomic_write_csv(pd.DataFrame(provenance_rows), OUTPUT_DIR / "input_provenance.csv")

    recipe_data = recipe_payload(recipe)
    recipe_sha = stable_json_sha256(recipe_data)
    versions = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lgb.__version__,
        "xgboost": xgb.__version__,
        "catboost": catboost.__version__,
    }
    signature_payload = {
        "schema_version": SCHEMA_VERSION,
        "recipe_sha256": recipe_sha,
        "reconstruction_script_sha256": sha256_file(Path(__file__).resolve()),
        "pipeline_v6_sha256": sha256_file(SOURCE_PATH),
        "train_sha256": sha256_file(TRAIN_PATH),
        "test_sha256": sha256_file(TEST_PATH),
        "archived_v6_teacher_sha256": sha256_file(
            ROOT / "archive" / "probs_v6_blend.npy"
        ),
        "archived_v6_final_sha256": sha256_file(
            ROOT / "archive" / "probs_v6_final.npy"
        ),
        "archived_v6_submission_sha256": sha256_file(
            ROOT / "archive" / "submission6.csv"
        ),
        "python_runtime": sys.version,
        "platform_runtime": platform.platform(),
        "library_versions": versions,
    }
    run_signature = stable_json_sha256(signature_payload)
    manifest = {
        **signature_payload,
        "run_signature": run_signature,
        "official_metric": "support-weighted F1 across Alive and Dead",
        "legacy_metric": "binary Dead-class F1 (diagnostic only)",
        "python": sys.version,
        "platform": platform.platform(),
        "versions": versions,
        "recipe": recipe_data,
    }
    manifest_path = OUTPUT_DIR / "run_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_signature = previous.get("run_signature")
        if previous_signature != run_signature:
            model_artifacts = list(CHECKPOINT_DIR.glob("*.npz")) + list(
                FOLD_DIR.glob("*.npz")
            )
            if model_artifacts:
                raise RuntimeError(
                    "Existing Submission 6 nested model outputs use a different "
                    f"code/input/recipe/environment signature ({previous_signature}); "
                    f"refusing to mix them with {run_signature}"
                )
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
    return run_signature


def write_recipe_equivalence_audit(run_signature: str | None = None) -> None:
    replay_path = OUTPUT_DIR / "production_replay_comparison.json"
    replay_verified = False
    replay_status = "unverified_until_opt_in_replay"
    replay_limitation = (
        "Nested OOF must not be called canonical until all folds finish, "
        "the scale sanity check passes, and the opt-in replay verifies"
    )
    if replay_path.exists():
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        replay_signature_matches = bool(
            run_signature is not None
            and replay.get("run_signature") == run_signature
        )
        if replay_signature_matches:
            replay_verified = bool(
                replay.get("strict_recipe_artifact_equivalence_verified", False)
            )
            replay_status = (
                "verified_against_archived_probabilities"
                if replay_verified
                else "replay_completed_probability_mismatch"
            )
            replay_limitation = (
                "No limitation: strict archived probability and production-label checks passed"
                if replay_verified
                else "Replay ran but strict archived probability equivalence failed; inspect comparison JSON"
            )
        else:
            replay_status = "stale_replay_signature_ignored"
            replay_limitation = (
                "Replay JSON does not match the current run signature and cannot approve the reference"
            )
    rows = [
        {
            "component": "outer validation",
            "production_recipe": "No outer validation; fit all labelled rows",
            "nested_reconstruction": "5-fold StratifiedKFold, shuffle=True, seed=42",
            "status": "required_nested_analogue",
            "limitation": "Each nested model has 80% rather than 100% of labelled rows",
        },
        {
            "component": "feature builder",
            "production_recipe": "Train-domain frequencies for labelled CV rows; test-domain frequencies for test",
            "nested_reconstruction": "Outer-fit frequencies mapped to validation (unseen=0); test keeps test-domain frequencies",
            "status": "strict_outer_isolation",
            "limitation": "Outer-fit is the leakage-safe analogue of the full labelled train domain",
        },
        {
            "component": "ordinal categorical encoding",
            "production_recipe": "Fit full labelled train; transform competition test",
            "nested_reconstruction": "Fit outer-training only; transform outer-validation and test",
            "status": "strict_outer_isolation",
            "limitation": "Necessary nested analogue, so category vocabularies can differ",
        },
        {
            "component": "v5 teacher",
            "production_recipe": "3x5 early-stopped fold average; weights .30/.15/.55",
            "nested_reconstruction": "Same within every outer-training fold",
            "status": "production_equivalent",
            "limitation": "Training population reduced by outer holdout",
        },
        {
            "component": "v4 target encoding",
            "production_recipe": "Fold-safe training TE; full-labelled TE for test",
            "nested_reconstruction": "Fold-safe teacher-fit TE; full outer-fit TE for validation/test",
            "status": "strict_outer_isolation",
            "limitation": "Full outer-fit TE is the leakage-safe test analogue",
        },
        {
            "component": "teacher family blend",
            "production_recipe": "55% v5 + 45% v4",
            "nested_reconstruction": "55% v5 + 45% v4 (frozen; no outer-label tuning)",
            "status": "exact_hyperparameter",
            "limitation": "Historical 55/45 choice originated from full-data OOF sweep",
        },
        {
            "component": "pseudo-label selection",
            "production_recipe": "Competition test teacher >=.95 or <=.05",
            "nested_reconstruction": "Outer-fit-only teacher on same competition test, same cutoffs",
            "status": "production_equivalent",
            "limitation": "Pseudo set varies by outer fold as required for nested validation",
        },
        {
            "component": "pseudo student",
            "production_recipe": "Six models, 5-fold seed=1041, early stopping 80",
            "nested_reconstruction": "Same within each augmented outer-training set",
            "status": "production_equivalent",
            "limitation": "Training population and pseudo set vary by outer fold",
        },
        {
            "component": "final probability blend",
            "production_recipe": "50% teacher + 50% pseudo student",
            "nested_reconstruction": "50% teacher + 50% pseudo student",
            "status": "exact_hyperparameter",
            "limitation": "None beyond nested training populations",
        },
        {
            "component": "decision policy",
            "production_recipe": "Threshold scan targeting approximately 84.5% Dead",
            "nested_reconstruction": "Deterministic top 84.5% within fold/global OOF",
            "status": "standardized_evaluation_policy",
            "limitation": "Rank-equivalent policy, not the historical floating threshold",
        },
        {
            "component": "test output",
            "production_recipe": "Writes test probabilities and submission",
            "nested_reconstruction": "Writes nested OOF only; no submission/test candidate",
            "status": "safety_guardrail",
            "limitation": "Cannot compare submitted labels byte-for-byte from this audit",
        },
        {
            "component": "production probability replay",
            "production_recipe": "Archived probs_v6_blend.npy and probs_v6_final.npy",
            "nested_reconstruction": (
                "Opt-in full-data replay compared with archived arrays"
                if replay_path.exists()
                else "Static recipe matched; full-data replay not run by this command"
            ),
            "status": replay_status,
            "limitation": replay_limitation,
        },
    ]
    atomic_write_csv(pd.DataFrame(rows), OUTPUT_DIR / "recipe_equivalence_audit.csv")


def vector_comparison(replayed: np.ndarray, archived: np.ndarray) -> dict[str, float]:
    if replayed.shape != archived.shape:
        raise ValueError(
            f"Probability shape mismatch: replay={replayed.shape}, archive={archived.shape}"
        )
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
        "allclose_rtol_1e_10_atol_1e_12": bool(
            np.allclose(replayed, archived, rtol=1e-10, atol=1e-12)
        ),
    }


def run_production_replay(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    recipe: dict[str, Any],
    run_signature: str,
) -> None:
    """Replay full-data v6 and compare, without writing a new submission."""

    print(
        "\nPRODUCTION REPLAY: 3x5 teacher + 5-fold pseudo student (120 fits)",
        flush=True,
    )
    # The third return value is the production test-domain feature matrix.
    train_base, _, test_base, feature_details = build_outer_base_features(
        train.reset_index(drop=True),
        test.reset_index(drop=True),
        test.reset_index(drop=True),
        recipe,
    )
    teacher_test, teacher_test_duplicate, teacher_iterations = train_teacher(
        -1,
        train_base,
        train.reset_index(drop=True),
        y,
        test_base,
        test.reset_index(drop=True),
        test_base,
        test.reset_index(drop=True),
        recipe,
        run_signature,
        checkpoint_prefix="production_replay",
    )
    if not np.allclose(teacher_test, teacher_test_duplicate, rtol=0, atol=0):
        raise RuntimeError("Production replay duplicate teacher predictions diverged")
    student_test, pseudo_counts, student_iterations = train_student(
        -1,
        train_base,
        train.reset_index(drop=True),
        y,
        test_base,
        test.reset_index(drop=True),
        test_base,
        test.reset_index(drop=True),
        teacher_test,
        recipe,
        run_signature,
        checkpoint_prefix="production_replay",
    )
    final_test = (
        FINAL_TEACHER_WEIGHT * teacher_test
        + FINAL_STUDENT_WEIGHT * student_test
    )
    archived_teacher = np.load(
        ROOT / "archive" / "probs_v6_blend.npy", allow_pickle=False
    )
    archived_final = np.load(
        ROOT / "archive" / "probs_v6_final.npy", allow_pickle=False
    )
    archived_submission = pd.read_csv(ROOT / "archive" / "submission6.csv")
    if not archived_submission["patient_id"].equals(test["patient_id"]):
        raise ValueError("Archived Submission 6 patient IDs/order do not match test.csv")
    archived_labels = (
        archived_submission["vital_status"] == "Dead"
    ).to_numpy(dtype=np.int8)
    replay_production_labels, replay_threshold = historical_v6_threshold_predictions(
        final_test
    )
    replay_top_k_labels = top_rate_predictions(
        final_test, float(archived_labels.mean())
    )
    teacher_comparison = vector_comparison(teacher_test, archived_teacher)
    final_comparison = vector_comparison(final_test, archived_final)
    production_changed_labels = int(
        np.sum(replay_production_labels != archived_labels)
    )
    top_k_changed_labels = int(np.sum(replay_top_k_labels != archived_labels))
    strict_probability_verified = bool(
        teacher_comparison["allclose_rtol_1e_10_atol_1e_12"]
        and final_comparison["allclose_rtol_1e_10_atol_1e_12"]
    )
    production_label_verified = production_changed_labels == 0
    strict_recipe_artifact_verified = bool(
        strict_probability_verified and production_label_verified
    )
    practical_rank_label_verified = bool(
        teacher_comparison["pearson_correlation"] >= PRACTICAL_MIN_PEARSON
        and teacher_comparison["spearman_rank_correlation"]
        >= PRACTICAL_MIN_SPEARMAN
        and final_comparison["pearson_correlation"] >= PRACTICAL_MIN_PEARSON
        and final_comparison["spearman_rank_correlation"]
        >= PRACTICAL_MIN_SPEARMAN
        and production_changed_labels <= PRACTICAL_MAX_PRODUCTION_LABEL_CHANGES
    )
    comparison = {
        "status": "complete",
        "run_signature": run_signature,
        "definition": "full-data replay of pipeline_v6.py; no submission written",
        "teacher_comparison": teacher_comparison,
        "final_comparison": final_comparison,
        "replay_pseudo_counts": {
            "total": int(pseudo_counts[0]),
            "dead": int(pseudo_counts[1]),
            "alive": int(pseudo_counts[2]),
        },
        "archived_teacher_pseudo_count": int(
            (
                (archived_teacher >= PSEUDO_DEAD_THRESHOLD)
                | (archived_teacher <= PSEUDO_ALIVE_THRESHOLD)
            ).sum()
        ),
        "archived_submission_dead_count": int(archived_labels.sum()),
        "historical_threshold_scan": {
            "start": 0.40,
            "stop_exclusive": 0.75,
            "step": 0.001,
            "first_rate_at_most": FIXED_DEAD_RATE,
            "selected_threshold": replay_threshold,
            "replay_dead_count": int(replay_production_labels.sum()),
            "changed_labels_vs_archived_submission": production_changed_labels,
        },
        "top_k_diagnostic": {
            "dead_count": int(replay_top_k_labels.sum()),
            "changed_labels_vs_archived_submission": top_k_changed_labels,
            "note": "Diagnostic only; pipeline_v6.py submitted the threshold-scan labels",
        },
        "strict_probability_equivalence_verified": strict_probability_verified,
        "production_label_equivalence_verified": production_label_verified,
        "strict_recipe_artifact_equivalence_verified": strict_recipe_artifact_verified,
        "practical_rank_label_equivalence_verified": practical_rank_label_verified,
        "strict_probability_tolerance": {"rtol": 1e-10, "atol": 1e-12},
        "practical_equivalence_thresholds": {
            "minimum_pearson": PRACTICAL_MIN_PEARSON,
            "minimum_spearman": PRACTICAL_MIN_SPEARMAN,
            "maximum_production_label_changes": PRACTICAL_MAX_PRODUCTION_LABEL_CHANGES,
        },
        "feature_details": feature_details,
        "teacher_best_iteration_min": int(teacher_iterations.min()),
        "teacher_best_iteration_max": int(teacher_iterations.max()),
        "student_best_iteration_min": int(student_iterations.min()),
        "student_best_iteration_max": int(student_iterations.max()),
        "new_probability_vector_saved": False,
        "new_submission_saved": False,
    }
    atomic_write_text(
        OUTPUT_DIR / "production_replay_comparison.json",
        json.dumps(comparison, indent=2) + "\n",
    )
    write_recipe_equivalence_audit(run_signature)
    print(json.dumps(comparison, indent=2), flush=True)


def write_structural_audit(
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    recipe: dict[str, Any],
) -> None:
    """Assert and persist the outer-fold and blend invariants."""

    coverage = np.zeros(len(y), dtype=np.int16)
    rows: list[dict[str, Any]] = []
    for fold, (fit_idx, validation_idx) in enumerate(outer_splits, start=1):
        intersection = np.intersect1d(fit_idx, validation_idx)
        union = np.union1d(fit_idx, validation_idx)
        coverage[validation_idx] += 1
        row = {
            "outer_fold": fold,
            "fit_rows": len(fit_idx),
            "validation_rows": len(validation_idx),
            "fit_validation_intersection_rows": len(intersection),
            "fit_validation_union_rows": len(union),
            "fit_dead_rate": float(y[fit_idx].mean()),
            "validation_dead_rate": float(y[validation_idx].mean()),
            "disjoint_pass": len(intersection) == 0,
            "full_union_pass": len(union) == len(y),
        }
        rows.append(row)
    split_frame = pd.DataFrame(rows)
    atomic_write_csv(split_frame, OUTPUT_DIR / "outer_split_audit.csv")

    v5_weight_sum = float(
        recipe["W5_LGB"] + recipe["W5_XGB"] + recipe["W5_CB"]
    )
    v4_weight_sum = float(
        recipe["W4_LGB"] + recipe["W4_XGB"] + recipe["W4_CB"]
    )
    invariants = {
        "outer_fold_count": len(outer_splits),
        "all_fit_validation_disjoint": bool(split_frame["disjoint_pass"].all()),
        "all_fit_validation_unions_cover_train": bool(split_frame["full_union_pass"].all()),
        "oof_row_coverage_min": int(coverage.min()),
        "oof_row_coverage_max": int(coverage.max()),
        "every_oof_row_exactly_once": bool(np.all(coverage == 1)),
        "expected_teacher_models_per_outer_fold": N_TEACHER_REPEATS
        * N_TEACHER_FOLDS
        * len(MODEL_LABELS),
        "expected_student_models_per_outer_fold": N_STUDENT_FOLDS
        * len(MODEL_LABELS),
        "v5_branch_weight_sum": v5_weight_sum,
        "v4_branch_weight_sum": v4_weight_sum,
        "teacher_family_weight_sum": TEACHER_V5_WEIGHT + TEACHER_V4_WEIGHT,
        "final_weight_sum": FINAL_TEACHER_WEIGHT + FINAL_STUDENT_WEIGHT,
        "v5_branch_weight_sum_pass": bool(np.isclose(v5_weight_sum, 1.0)),
        "v4_branch_weight_sum_pass": bool(np.isclose(v4_weight_sum, 1.0)),
        "teacher_family_weight_sum_pass": bool(
            np.isclose(TEACHER_V5_WEIGHT + TEACHER_V4_WEIGHT, 1.0)
        ),
        "final_weight_sum_pass": bool(
            np.isclose(FINAL_TEACHER_WEIGHT + FINAL_STUDENT_WEIGHT, 1.0)
        ),
        "teacher_family_weights_exact": bool(
            TEACHER_V5_WEIGHT == 0.55 and TEACHER_V4_WEIGHT == 0.45
        ),
        "final_weights_exact": bool(
            FINAL_TEACHER_WEIGHT == 0.5 and FINAL_STUDENT_WEIGHT == 0.5
        ),
    }
    required = [
        invariants["outer_fold_count"] == N_OUTER_FOLDS,
        invariants["all_fit_validation_disjoint"],
        invariants["all_fit_validation_unions_cover_train"],
        invariants["every_oof_row_exactly_once"],
        invariants["expected_teacher_models_per_outer_fold"] == 90,
        invariants["expected_student_models_per_outer_fold"] == 30,
        invariants["v5_branch_weight_sum_pass"],
        invariants["v4_branch_weight_sum_pass"],
        invariants["teacher_family_weight_sum_pass"],
        invariants["final_weight_sum_pass"],
        invariants["teacher_family_weights_exact"],
        invariants["final_weights_exact"],
    ]
    invariants["all_structural_invariants_pass"] = bool(all(required))
    atomic_write_text(
        OUTPUT_DIR / "structural_invariants.json",
        json.dumps(invariants, indent=2) + "\n",
    )
    if not invariants["all_structural_invariants_pass"]:
        raise RuntimeError("Submission 6 nested structural invariant audit failed")


def write_checkpoint_completion(run_signature: str) -> None:
    rows: list[dict[str, Any]] = []
    for fold in range(N_OUTER_FOLDS):
        teacher_path = CHECKPOINT_DIR / f"outer_fold_{fold + 1:02d}_teacher.npz"
        student_path = CHECKPOINT_DIR / f"outer_fold_{fold + 1:02d}_student.npz"
        final_path = FOLD_DIR / f"outer_fold_{fold + 1:02d}_final.npz"
        teacher_complete = 0
        student_complete = 0
        if teacher_path.exists():
            checkpoint = np.load(teacher_path, allow_pickle=False)
            validate_checkpoint_signature(checkpoint, run_signature, teacher_path)
            teacher_complete = int(checkpoint["complete"].sum())
        if student_path.exists():
            checkpoint = np.load(student_path, allow_pickle=False)
            validate_checkpoint_signature(checkpoint, run_signature, student_path)
            student_complete = int(checkpoint["complete"].sum())
        rows.append(
            {
                "outer_fold": fold + 1,
                "teacher_models_complete": teacher_complete,
                "teacher_models_expected": 90,
                "teacher_complete": teacher_complete == 90,
                "student_models_complete": student_complete,
                "student_models_expected": 30,
                "student_complete": student_complete == 30,
                "final_fold_artifact_complete": final_path.exists(),
            }
        )
    atomic_write_csv(pd.DataFrame(rows), OUTPUT_DIR / "checkpoint_completion.csv")


def collect_outputs(
    y: np.ndarray,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    run_signature: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    write_checkpoint_completion(run_signature)
    teacher_oof = np.full(len(y), np.nan, dtype=np.float64)
    student_oof = np.full(len(y), np.nan, dtype=np.float64)
    final_oof = np.full(len(y), np.nan, dtype=np.float64)
    outer_fold_index = np.full(len(y), -1, dtype=np.int8)
    pseudo_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []

    for fold, (_, validation_idx) in enumerate(outer_splits):
        path = FOLD_DIR / f"outer_fold_{fold + 1:02d}_final.npz"
        if not path.exists():
            continue
        artifact = np.load(path, allow_pickle=False)
        validate_checkpoint_signature(artifact, run_signature, path)
        saved_validation_idx = artifact["validation_idx"].astype(np.int64)
        if not np.array_equal(saved_validation_idx, validation_idx):
            raise RuntimeError(f"Outer validation indices changed for {path}")
        teacher = artifact["teacher_validation"].astype(np.float64)
        student = artifact["student_validation"].astype(np.float64)
        final = artifact["final_validation"].astype(np.float64)
        expected_shape = (len(validation_idx),)
        for label, values in (
            ("teacher_validation", teacher),
            ("student_validation", student),
            ("final_validation", final),
        ):
            if values.shape != expected_shape:
                raise RuntimeError(
                    f"{label} shape mismatch in {path}: "
                    f"{values.shape} != {expected_shape}"
                )
            if not np.isfinite(values).all():
                raise RuntimeError(f"{label} contains non-finite values in {path}")
        saved_y = artifact["y_validation"].astype(np.int8)
        if saved_y.shape != expected_shape or not np.array_equal(
            saved_y, y[validation_idx]
        ):
            raise RuntimeError(f"Saved validation labels are invalid/misaligned in {path}")
        teacher_oof[validation_idx] = teacher
        student_oof[validation_idx] = student
        final_oof[validation_idx] = final
        outer_fold_index[validation_idx] = fold + 1
        counts = artifact["pseudo_counts"].astype(np.int64)
        if counts.shape != (3,) or np.any(counts < 0) or counts[0] != counts[1] + counts[2]:
            raise RuntimeError(f"Invalid pseudo-label counts in {path}: {counts}")
        pseudo_rows.append(
            {
                "outer_fold": fold + 1,
                "pseudo_total": int(counts[0]),
                "pseudo_dead": int(counts[1]),
                "pseudo_alive": int(counts[2]),
                "test_rows": 36_000,
                "pseudo_fraction": float(counts[0] / 36_000),
            }
        )
        for scope, scores in (
            ("teacher", teacher),
            ("student", student),
            ("submission6_nested", final),
        ):
            row = metric_row(scope, scores, y[validation_idx], 1)
            row["outer_fold"] = fold + 1
            fold_metric_rows.append(row)

    atomic_write_csv(pd.DataFrame(pseudo_rows), OUTPUT_DIR / "pseudo_counts.csv")
    atomic_write_csv(pd.DataFrame(fold_metric_rows), OUTPUT_DIR / "fold_metrics.csv")
    completed = sorted(set(int(row["outer_fold"]) for row in pseudo_rows))
    atomic_save_npz(
        OUTPUT_DIR / "nested_oof_predictions.npz",
        run_signature=np.array(run_signature),
        y=y,
        teacher_oof=teacher_oof,
        student_oof=student_oof,
        submission6_nested_oof=final_oof,
        outer_fold=outer_fold_index,
        completed_outer_folds=np.asarray(completed, dtype=np.int8),
    )

    global_rows: list[dict[str, Any]] = []
    complete_mask = np.isfinite(final_oof)
    completed_count = len(completed)
    is_complete_5_fold = bool(
        completed_count == N_OUTER_FOLDS
        and int(complete_mask.sum()) == len(y)
        and np.isfinite(teacher_oof).all()
        and np.isfinite(student_oof).all()
        and np.isfinite(final_oof).all()
        and np.all(outer_fold_index >= 1)
    )
    if completed_count == N_OUTER_FOLDS and not is_complete_5_fold:
        raise RuntimeError(
            "Five fold artifacts exist but canonical OOF coverage/integrity is not 24,000/24,000"
        )
    if complete_mask.any():
        for scope, scores in (
            ("teacher", teacher_oof[complete_mask]),
            ("student", student_oof[complete_mask]),
            ("submission6_nested", final_oof[complete_mask]),
        ):
            row = metric_row(scope, scores, y[complete_mask], completed_count)
            row["is_complete_5_fold"] = is_complete_5_fold
            row["finite_oof_rows"] = int(complete_mask.sum())
            if scope == "submission6_nested":
                row["submission6_public_lb_reference"] = SUBMISSION6_PUBLIC_LB
                row["nested_minus_public_lb"] = (
                    row["official_support_weighted_f1"] - SUBMISSION6_PUBLIC_LB
                )
                row["sanity_absolute_gap_tolerance"] = SANITY_ABS_GAP_TOLERANCE
                row["sanity_scale_pass"] = bool(
                    is_complete_5_fold
                    and abs(row["nested_minus_public_lb"])
                    <= SANITY_ABS_GAP_TOLERANCE
                )
                replay_path = OUTPUT_DIR / "production_replay_comparison.json"
                replay_verified = False
                if replay_path.exists():
                    replay = json.loads(replay_path.read_text(encoding="utf-8"))
                    replay_verified = bool(
                        replay.get("run_signature") == run_signature
                        and replay.get(
                            "strict_recipe_artifact_equivalence_verified", False
                        )
                    )
                row["production_probability_replay_verified"] = replay_verified
                row["canonical_reference_approved"] = bool(
                    is_complete_5_fold
                    and row["sanity_scale_pass"]
                    and replay_verified
                )
            global_rows.append(row)
    atomic_write_csv(pd.DataFrame(global_rows), OUTPUT_DIR / "global_metrics.csv")
    write_report(global_rows, pseudo_rows, completed)
    return teacher_oof, student_oof, final_oof, outer_fold_index


def write_report(
    global_rows: list[dict[str, Any]],
    pseudo_rows: list[dict[str, Any]],
    completed: list[int],
) -> None:
    nested_row = next(
        (row for row in global_rows if row["scope"] == "submission6_nested"), None
    )
    if len(completed) != N_OUTER_FOLDS:
        status = f"IN PROGRESS ({len(completed)}/{N_OUTER_FOLDS} outer folds complete)"
    elif nested_row is None or not bool(nested_row.get("sanity_scale_pass", False)):
        status = "COMPLETE — SANITY REVIEW REQUIRED"
    elif not bool(nested_row.get("production_probability_replay_verified", False)):
        status = "COMPLETE — SCALE PASS, PRODUCTION REPLAY UNVERIFIED"
    else:
        status = "CANONICAL REFERENCE APPROVED"
    if nested_row is None:
        metric_text = "No completed-fold score is available yet."
    else:
        metric_text = (
            f"Weighted F1 at fixed 84.5%: "
            f"`{nested_row['official_support_weighted_f1']:.6f}`"
        )
        if len(completed) == N_OUTER_FOLDS:
            gap = float(nested_row["nested_minus_public_lb"])
            closeness = (
                "close"
                if abs(gap) <= SANITY_ABS_GAP_TOLERANCE
                else "not close"
            )
            metric_text += (
                f". Gap versus Submission 6 public LB `0.877258`: `{gap:+.6f}` "
                f"({closeness} under the predeclared "
                f"±{SANITY_ABS_GAP_TOLERANCE:.3f} sanity band)."
            )
        else:
            metric_text += ". This partial-fold score is not a canonical comparison."
    mean_pseudo = (
        float(np.mean([row["pseudo_total"] for row in pseudo_rows]))
        if pseudo_rows
        else float("nan")
    )
    report = f"""# Submission 6 Nested-Equivalent Reconstruction

**Status:** {status}

{metric_text}

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
- Completed outer folds: {completed or 'none'}.
- Mean pseudo-label count across completed folds: {mean_pseudo:.1f}.

## Interpretation boundary

This is a nested-equivalent validation reference, not a byte-for-byte replay of
the submitted full-data model.  Each fold necessarily trains on 80% of labelled
rows, and its encoder/target encodings exclude the outer validation fold.  See
`recipe_equivalence_audit.csv` for the component-by-component audit.

The vector is approved as the canonical future comparison only after all five
folds finish, the predeclared absolute weighted-F1 gap to the public LB is at
most {SANITY_ABS_GAP_TOLERANCE:.3f}, and an opt-in full-data production replay
verifies the archived probability artifacts.  Until then,
`canonical_reference_approved` is false in `global_metrics.csv`.

No test submission and no final candidate test probability vector were created.
Model-level checkpoints under `checkpoints/` are resumable and are bound to the
input/recipe hash in `run_manifest.json`.
"""
    atomic_write_text(OUTPUT_DIR / "submission6_nested_report.md", report)


def main() -> None:
    args = parse_args()
    requested_folds = parse_outer_folds(args.outer_folds)
    np.random.seed(SEED)
    recipe = load_frozen_recipe()
    run_signature = write_provenance(recipe)
    write_recipe_equivalence_audit(run_signature)

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    y = (train["vital_status"] == "Dead").astype(np.int8).to_numpy()
    if len(train) != 24_000 or len(test) != 36_000:
        raise ValueError(
            f"Unexpected competition data sizes: train={len(train)}, test={len(test)}"
        )
    outer = StratifiedKFold(N_OUTER_FOLDS, shuffle=True, random_state=SEED)
    outer_splits = list(outer.split(train, y))
    write_structural_audit(outer_splits, y, recipe)

    if args.production_replay:
        if args.plan_only:
            raise ValueError("--production-replay and --plan-only are mutually exclusive")
        run_production_replay(train, test, y, recipe, run_signature)
        collect_outputs(y, outer_splits, run_signature)
        print("Production replay comparison complete; no submission was generated.")
        return

    if args.plan_only:
        fit_idx, validation_idx = outer_splits[0]
        _, _, _, details = build_outer_base_features(
            train.iloc[fit_idx].reset_index(drop=True),
            train.iloc[validation_idx].reset_index(drop=True),
            test.reset_index(drop=True),
            recipe,
        )
        atomic_write_text(
            OUTPUT_DIR / "plan_only_validation.json",
            json.dumps(
                {
                    "status": "plan_only_ok",
                    "run_signature": run_signature,
                    "first_fold_preprocessing": details,
                    "requested_outer_folds": [fold + 1 for fold in requested_folds],
                    "estimated_model_fits_full_run": 600,
                },
                indent=2,
            )
            + "\n",
        )
        collect_outputs(y, outer_splits, run_signature)
        print("Plan-only validation completed; no models were fitted.")
        return

    for outer_fold in requested_folds:
        final_path = FOLD_DIR / f"outer_fold_{outer_fold + 1:02d}_final.npz"
        if final_path.exists():
            artifact = np.load(final_path, allow_pickle=False)
            validate_checkpoint_signature(artifact, run_signature, final_path)
            print(f"OUTER FOLD {outer_fold + 1}: already complete", flush=True)
            continue
        fit_idx, validation_idx = outer_splits[outer_fold]
        fit_raw = train.iloc[fit_idx].reset_index(drop=True)
        validation_raw = train.iloc[validation_idx].reset_index(drop=True)
        test_raw = test.reset_index(drop=True)
        y_fit = y[fit_idx]
        print(
            f"\nOUTER FOLD {outer_fold + 1}/{N_OUTER_FOLDS}: "
            f"fit={len(fit_idx):,}, validation={len(validation_idx):,}",
            flush=True,
        )
        fit_base, validation_base, test_base, feature_details = build_outer_base_features(
            fit_raw, validation_raw, test_raw, recipe
        )
        teacher_validation, teacher_test, teacher_iterations = train_teacher(
            outer_fold,
            fit_base,
            fit_raw,
            y_fit,
            validation_base,
            validation_raw,
            test_base,
            test_raw,
            recipe,
            run_signature,
        )
        student_validation, pseudo_counts, student_iterations = train_student(
            outer_fold,
            fit_base,
            fit_raw,
            y_fit,
            validation_base,
            validation_raw,
            test_base,
            test_raw,
            teacher_test,
            recipe,
            run_signature,
        )
        final_validation = (
            FINAL_TEACHER_WEIGHT * teacher_validation
            + FINAL_STUDENT_WEIGHT * student_validation
        )
        atomic_save_npz(
            final_path,
            run_signature=np.array(run_signature),
            fit_idx=fit_idx,
            validation_idx=validation_idx,
            y_validation=y[validation_idx],
            teacher_validation=teacher_validation,
            student_validation=student_validation,
            final_validation=final_validation,
            pseudo_counts=pseudo_counts,
            teacher_best_iterations=teacher_iterations,
            student_best_iterations=student_iterations,
            feature_details_json=np.array(json.dumps(feature_details, sort_keys=True)),
        )
        fold_row = metric_row(
            "submission6_nested", final_validation, y[validation_idx], 1
        )
        print(
            f"OUTER FOLD {outer_fold + 1} COMPLETE: weighted F1="
            f"{fold_row['official_support_weighted_f1']:.6f}, "
            f"pseudo={int(pseudo_counts[0]):,}",
            flush=True,
        )
        collect_outputs(y, outer_splits, run_signature)

    _, _, final_oof, _ = collect_outputs(y, outer_splits, run_signature)
    complete = int(np.isfinite(final_oof).sum())
    print(
        f"\nNested reconstruction checkpointed: {complete:,}/{len(y):,} OOF rows complete."
    )
    print(f"Outputs: {OUTPUT_DIR}")
    print("No test submission was generated.")


if __name__ == "__main__":
    main()
