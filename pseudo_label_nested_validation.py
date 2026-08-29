"""Nested audit of 90% versus 95% pseudo-label confidence.

Each outer fold is never used to create pseudo-labels, target encodings, or
models.  A six-model teacher fitted on the other four folds predicts the
unlabelled competition test set.  Separate six-model students are then fitted
with 90%/10% or 95%/5% pseudo-labels and evaluated on the untouched outer
fold.  Hyperparameters and blend weights are extracted directly from
``pipeline_pseudo90.py``.  The nested audit fits one model per family without
the candidate pipeline's inner five-fold early-stopped averaging, so it is a
controlled paired proxy rather than an exact recipe-equivalent replay.

The historical ``oof_step1.npy`` vector is retained as a useful frozen tree
proxy, but it is not described as recipe-equivalent pseudo-label OOF.  The
paired 95% control generated here is the valid like-for-like baseline.

The competition metric is support-weighted F1.  Earlier versions of this
audit used sklearn's default binary F1 (Dead-class F1).  Outputs now make both
metrics explicit and use support-weighted F1 for the submission gate.  The
old ambiguous field names are retained only as documented legacy aliases.

Outer-fold labels are isolated, but the ordinal encoder is fitted once on all
training features before the outer split.  That operation is target-free; it
still means this is not a fully end-to-end nested preprocessing replay.
"""

from __future__ import annotations

import ast
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder


warnings.filterwarnings("ignore")

SEED = 42
N_OUTER_FOLDS = 5
FIXED_POSITIVE_RATE = 0.845
BOOTSTRAP_REPEATS = 1_000
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "diagnostic_outputs" / "pseudo90_nested"
CHECKPOINT_PATH = OUTPUT_DIR / "nested_oof_checkpoint.npz"
SOURCE_PATH = ROOT / "pipeline_pseudo90.py"
OFFICIAL_F1_AVERAGE = "weighted"
METRIC_SCHEMA_VERSION = "2_weighted_f1_primary"


def official_weighted_f1(y_true: np.ndarray, predictions: np.ndarray) -> float:
    """Competition metric: support-weighted F1 across Alive and Dead."""

    return float(
        f1_score(
            y_true,
            predictions,
            average=OFFICIAL_F1_AVERAGE,
            zero_division=0,
        )
    )


def legacy_dead_class_f1(y_true: np.ndarray, predictions: np.ndarray) -> float:
    """Historical local metric: binary F1 with Dead encoded as class 1."""

    return float(f1_score(y_true, predictions, average="binary", zero_division=0))


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


def load_frozen_recipe() -> dict[str, object]:
    """Extract definitions without executing the training pipeline."""

    wanted_functions = {"build_features", "smoothed_te"}
    wanted_assignments = {
        "cat_cols",
        "te_cols",
        "SMOOTHING_M",
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
        "best_alpha",
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
    namespace: dict[str, object] = {"np": np, "pd": pd}
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, SOURCE_PATH.name, "exec"), namespace)
    return namespace


def as_float_matrix(frame: pd.DataFrame | np.ndarray) -> np.ndarray:
    values = frame.values if isinstance(frame, pd.DataFrame) else frame
    return np.nan_to_num(values.astype(np.float32), nan=-999.0)


def cross_fitted_target_encoding(
    raw_train: pd.DataFrame,
    y_train: np.ndarray,
    raw_evaluations: dict[str, pd.DataFrame],
    te_cols: list[str],
    smoothed_te,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    train_encoded = pd.DataFrame(index=np.arange(len(raw_train)))
    eval_encoded = {
        name: pd.DataFrame(index=np.arange(len(frame)))
        for name, frame in raw_evaluations.items()
    }
    inner = StratifiedKFold(5, shuffle=True, random_state=seed)
    for column in te_cols:
        train_series = raw_train[column].reset_index(drop=True)
        encoded = np.zeros(len(raw_train), dtype=np.float64)
        for fit_idx, held_idx in inner.split(train_series, y_train):
            _, held_values, _ = smoothed_te(
                train_series.iloc[fit_idx], y_train[fit_idx], train_series.iloc[held_idx]
            )
            encoded[held_idx] = held_values
        train_encoded[f"{column}_te"] = encoded
        for name, frame in raw_evaluations.items():
            _, values, _ = smoothed_te(
                train_series, y_train, frame[column].reset_index(drop=True)
            )
            eval_encoded[name][f"{column}_te"] = values
    return train_encoded, eval_encoded


def fit_six_model_blend(
    base_train: pd.DataFrame,
    raw_train: pd.DataFrame,
    y_train: np.ndarray,
    base_evaluations: dict[str, pd.DataFrame],
    raw_evaluations: dict[str, pd.DataFrame],
    recipe: dict[str, object],
    seed: int,
) -> dict[str, np.ndarray]:
    """Fit the frozen six-model blend without using evaluation labels."""

    train_te, evaluation_te = cross_fitted_target_encoding(
        raw_train,
        y_train,
        raw_evaluations,
        recipe["te_cols"],
        recipe["smoothed_te"],
        seed,
    )
    x_train_v5 = as_float_matrix(base_train)
    x_train_v4 = as_float_matrix(
        np.hstack([base_train.to_numpy(), train_te.to_numpy()])
    )
    x_eval_v5 = {name: as_float_matrix(frame) for name, frame in base_evaluations.items()}
    x_eval_v4 = {
        name: as_float_matrix(
            np.hstack([base_evaluations[name].to_numpy(), evaluation_te[name].to_numpy()])
        )
        for name in base_evaluations
    }

    predictions = {name: np.zeros(len(frame), dtype=np.float64) for name, frame in base_evaluations.items()}

    model_specs = [
        (lgb.LGBMClassifier, dict(recipe["v5_lgb"]), x_train_v5, x_eval_v5, float(recipe["W5_LGB"]), "v5_lgb"),
        (xgb.XGBClassifier, dict(recipe["v5_xgb"]), x_train_v5, x_eval_v5, float(recipe["W5_XGB"]), "v5_xgb"),
        (CatBoostClassifier, dict(recipe["v5_cb"]), x_train_v5, x_eval_v5, float(recipe["W5_CB"]), "v5_cb"),
        (lgb.LGBMClassifier, dict(recipe["v4_lgb"]), x_train_v4, x_eval_v4, float(recipe["W4_LGB"]), "v4_lgb"),
        (xgb.XGBClassifier, dict(recipe["v4_xgb"]), x_train_v4, x_eval_v4, float(recipe["W4_XGB"]), "v4_xgb"),
        (CatBoostClassifier, dict(recipe["v4_cb"]), x_train_v4, x_eval_v4, float(recipe["W4_CB"]), "v4_cb"),
    ]
    v5_predictions = {name: np.zeros(len(frame)) for name, frame in base_evaluations.items()}
    v4_predictions = {name: np.zeros(len(frame)) for name, frame in base_evaluations.items()}
    for constructor, parameters, x_train, evaluation_arrays, weight, label in model_specs:
        parameters["random_state" if "random_state" in parameters else "random_seed"] = seed
        model = constructor(**parameters)
        model.fit(x_train, y_train)
        destination = v5_predictions if label.startswith("v5") else v4_predictions
        for name, x_eval in evaluation_arrays.items():
            destination[name] += weight * model.predict_proba(x_eval)[:, 1]
        print(f"      fitted {label}", flush=True)

    alpha = float(recipe["best_alpha"])
    for name in predictions:
        predictions[name] = alpha * v5_predictions[name] + (1.0 - alpha) * v4_predictions[name]
    return predictions


def top_rate_predictions(scores: np.ndarray, rate: float) -> np.ndarray:
    count = int(round(len(scores) * rate))
    # Stable descending order makes earlier row order the deterministic
    # tie-breaker, matching the current validation and submission audits.
    order = np.argsort(-scores, kind="mergesort")
    predictions = np.zeros(len(scores), dtype=np.int8)
    predictions[order[:count]] = 1
    return predictions


def bootstrap_f1_delta(
    y: np.ndarray,
    candidate_scores: np.ndarray,
    control_scores: np.ndarray,
    repeats: int,
    seed: int,
    *,
    metric,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    deltas = np.empty(repeats)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    for iteration in range(repeats):
        sampled = np.concatenate(
            [rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)]
        )
        candidate = top_rate_predictions(candidate_scores[sampled], FIXED_POSITIVE_RATE)
        control = top_rate_predictions(control_scores[sampled], FIXED_POSITIVE_RATE)
        deltas[iteration] = metric(y[sampled], candidate) - metric(
            y[sampled], control
        )
    return float(deltas.mean()), float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def main() -> None:
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    recipe = load_frozen_recipe()
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    y = (train["vital_status"] == "Dead").astype(np.int8).to_numpy()

    build_features = recipe["build_features"]
    base_train = build_features(train).reset_index(drop=True)
    base_test = build_features(test).reset_index(drop=True)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_categorical = train[recipe["cat_cols"]].fillna("__NA__").astype(str)
    test_categorical = test[recipe["cat_cols"]].fillna("__NA__").astype(str)
    encoder.fit(train_categorical)
    encoded_train = pd.DataFrame(
        encoder.transform(train_categorical),
        columns=[f"{column}_enc" for column in recipe["cat_cols"]],
    )
    encoded_test = pd.DataFrame(
        encoder.transform(test_categorical),
        columns=[f"{column}_enc" for column in recipe["cat_cols"]],
    )
    base_train = pd.concat([base_train, encoded_train], axis=1)
    base_test = pd.concat([base_test, encoded_test], axis=1)

    teacher_oof = np.full(len(train), np.nan)
    pseudo95_oof = np.full(len(train), np.nan)
    pseudo90_oof = np.full(len(train), np.nan)
    completed = np.zeros(N_OUTER_FOLDS, dtype=bool)
    # Axis 1 is [95%, 90%]; axis 2 is [total, Dead, Alive].
    pseudo_counts = np.full((N_OUTER_FOLDS, 2, 3), -1, dtype=np.int64)
    if CHECKPOINT_PATH.exists():
        checkpoint = np.load(CHECKPOINT_PATH, allow_pickle=False)
        teacher_oof = checkpoint["teacher_oof"]
        pseudo95_oof = checkpoint["pseudo95_oof"]
        pseudo90_oof = checkpoint["pseudo90_oof"]
        completed = checkpoint["completed"].astype(bool)
        if "pseudo_counts" in checkpoint.files:
            pseudo_counts = checkpoint["pseudo_counts"].astype(np.int64)
        elif (OUTPUT_DIR / "pseudo_counts.csv").exists():
            previous_counts = pd.read_csv(OUTPUT_DIR / "pseudo_counts.csv")
            for row in previous_counts.itertuples(index=False):
                confidence_index = 0 if int(row.confidence) == 95 else 1
                pseudo_counts[int(row.outer_fold) - 1, confidence_index] = [
                    int(row.pseudo_total), int(row.pseudo_dead), int(row.pseudo_alive)
                ]
        print(f"Resuming checkpoint: {int(completed.sum())}/{N_OUTER_FOLDS} folds complete")

    def write_count_artifacts() -> None:
        rows = []
        for fold_index in range(N_OUTER_FOLDS):
            for confidence_index, confidence in enumerate((95, 90)):
                total, dead, alive = pseudo_counts[fold_index, confidence_index]
                if total >= 0:
                    rows.append(
                        {
                            "outer_fold": fold_index + 1,
                            "confidence": confidence,
                            "pseudo_total": int(total),
                            "pseudo_dead": int(dead),
                            "pseudo_alive": int(alive),
                        }
                    )
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "pseudo_counts.csv", index=False)

    outer = StratifiedKFold(N_OUTER_FOLDS, shuffle=True, random_state=SEED)
    for fold, (fit_idx, validation_idx) in enumerate(outer.split(base_train, y)):
        if completed[fold] and (pseudo_counts[fold] >= 0).all():
            continue
        predictions_already_complete = bool(completed[fold])
        print(f"\nOUTER FOLD {fold + 1}/{N_OUTER_FOLDS}", flush=True)
        teacher = fit_six_model_blend(
            base_train.iloc[fit_idx].reset_index(drop=True),
            train.iloc[fit_idx].reset_index(drop=True),
            y[fit_idx],
            {
                "validation": base_train.iloc[validation_idx].reset_index(drop=True),
                "test": base_test,
            },
            {
                "validation": train.iloc[validation_idx].reset_index(drop=True),
                "test": test,
            },
            recipe,
            SEED + fold * 101,
        )
        if not predictions_already_complete:
            teacher_oof[validation_idx] = teacher["validation"]

        for confidence_index, (confidence, destination) in enumerate(
            ((0.95, pseudo95_oof), (0.90, pseudo90_oof))
        ):
            mask = (teacher["test"] >= confidence) | (teacher["test"] <= 1.0 - confidence)
            pseudo_y = (teacher["test"][mask] >= confidence).astype(np.int8)
            pseudo_counts[fold, confidence_index] = [
                int(mask.sum()), int(pseudo_y.sum()), int((pseudo_y == 0).sum())
            ]
            if predictions_already_complete:
                continue
            combined_base = pd.concat(
                [base_train.iloc[fit_idx].reset_index(drop=True), base_test.loc[mask].reset_index(drop=True)],
                ignore_index=True,
            )
            combined_raw = pd.concat(
                [train.iloc[fit_idx].reset_index(drop=True), test.loc[mask].reset_index(drop=True)],
                ignore_index=True,
            )
            combined_y = np.concatenate([y[fit_idx], pseudo_y])
            print(
                f"    {confidence:.0%}: {int(mask.sum()):,} pseudo labels "
                f"({int(pseudo_y.sum()):,} Dead, {int((pseudo_y == 0).sum()):,} Alive)",
                flush=True,
            )
            student = fit_six_model_blend(
                combined_base,
                combined_raw,
                combined_y,
                {"validation": base_train.iloc[validation_idx].reset_index(drop=True)},
                {"validation": train.iloc[validation_idx].reset_index(drop=True)},
                recipe,
                SEED + fold * 101 + int(confidence * 1000),
            )["validation"]
            destination[validation_idx] = 0.5 * teacher["validation"] + 0.5 * student

        if not predictions_already_complete:
            completed[fold] = True
        np.savez_compressed(
            CHECKPOINT_PATH,
            teacher_oof=teacher_oof,
            pseudo95_oof=pseudo95_oof,
            pseudo90_oof=pseudo90_oof,
            completed=completed,
            pseudo_counts=pseudo_counts,
        )
        write_count_artifacts()

    if not (np.isfinite(teacher_oof).all() and np.isfinite(pseudo95_oof).all() and np.isfinite(pseudo90_oof).all()):
        raise RuntimeError("Nested OOF vectors are incomplete")

    frozen_tree = np.load(ROOT / "oof_step1.npy", allow_pickle=False)
    frozen_y = np.load(ROOT / "y_step1.npy", allow_pickle=False)
    if not np.array_equal(y, frozen_y):
        raise ValueError("Frozen labels do not match train.csv")
    candidates = {
        "frozen_tree_proxy": frozen_tree,
        "nested_teacher": teacher_oof,
        "nested_pseudo95_control": pseudo95_oof,
        "nested_pseudo90_candidate": pseudo90_oof,
    }
    rows = []
    for name, scores in candidates.items():
        predictions = top_rate_predictions(scores, FIXED_POSITIVE_RATE)
        rows.append(
            {
                "candidate": name,
                "metric_schema_version": METRIC_SCHEMA_VERSION,
                "rows": len(y),
                "roc_auc": roc_auc_score(y, scores),
                "fixed_positive_rate": float(predictions.mean()),
                "official_weighted_f1_at_fixed_rate": official_weighted_f1(
                    y, predictions
                ),
                "official_weighted_f1_at_probability_0_5": official_weighted_f1(
                    y, scores >= 0.5
                ),
                "legacy_dead_class_f1_at_fixed_rate": legacy_dead_class_f1(
                    y, predictions
                ),
                "legacy_dead_class_f1_at_probability_0_5": legacy_dead_class_f1(
                    y, scores >= 0.5
                ),
                # Backward-compatible aliases.  They intentionally retain the
                # old binary-Dead semantics; new code must use the explicit
                # official_weighted_* columns above.
                "fixed_rate_f1": legacy_dead_class_f1(y, predictions),
                "f1_at_probability_0_5": legacy_dead_class_f1(
                    y, scores >= 0.5
                ),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT_DIR / "nested_oof_metrics.csv", index=False)
    np.savez_compressed(
        OUTPUT_DIR / "nested_oof_predictions.npz",
        y=y,
        teacher_oof=teacher_oof,
        pseudo95_oof=pseudo95_oof,
        pseudo90_oof=pseudo90_oof,
        frozen_tree_oof=frozen_tree,
    )

    mean_delta, lower_delta, upper_delta = bootstrap_f1_delta(
        y,
        pseudo90_oof,
        pseudo95_oof,
        BOOTSTRAP_REPEATS,
        SEED + 9_000,
        metric=official_weighted_f1,
    )
    legacy_mean_delta, legacy_lower_delta, legacy_upper_delta = bootstrap_f1_delta(
        y,
        pseudo90_oof,
        pseudo95_oof,
        BOOTSTRAP_REPEATS,
        SEED + 9_000,
        metric=legacy_dead_class_f1,
    )
    official_metric_column = "official_weighted_f1_at_fixed_rate"
    legacy_metric_column = "legacy_dead_class_f1_at_fixed_rate"
    f1_90 = float(
        metrics.loc[
            metrics.candidate == "nested_pseudo90_candidate",
            official_metric_column,
        ].iloc[0]
    )
    f1_95 = float(
        metrics.loc[
            metrics.candidate == "nested_pseudo95_control",
            official_metric_column,
        ].iloc[0]
    )
    legacy_f1_90 = float(
        metrics.loc[
            metrics.candidate == "nested_pseudo90_candidate", legacy_metric_column
        ].iloc[0]
    )
    legacy_f1_95 = float(
        metrics.loc[
            metrics.candidate == "nested_pseudo95_control", legacy_metric_column
        ].iloc[0]
    )
    gate_pass = lower_delta >= -0.001 and f1_90 >= f1_95 - 0.001
    summary = {
        "validation_design": "5-fold outer holdout; test pseudo-labels and models rebuilt without outer-fold labels",
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "official_metric": "support-weighted F1 across Alive and Dead",
        "legacy_metric": "binary Dead-class F1; retained for historical comparison only",
        "preprocessing_limitation": "ordinal categorical encoder was fitted target-free on all training feature rows before outer splitting",
        "recipe_equivalent_to_submission11": False,
        "recipe_equivalence_limitation": "nested students fit one full model per family; Submission 11 used five-fold early-stopped student averaging",
        "positive_rate_provenance_limitation": "the fixed 84.5% rate is a historically chosen policy and must not be treated as independent validation evidence",
        "rules_review_note": "pseudo-labels are automated model predictions from provided test features, not hidden/manual labels; confirm transductive pseudo-labeling is permitted by the organizer",
        "fixed_positive_rate": FIXED_POSITIVE_RATE,
        "pseudo90_official_weighted_f1_at_fixed_rate": f1_90,
        "pseudo95_official_weighted_f1_at_fixed_rate": f1_95,
        "pseudo90_minus_pseudo95_official_weighted_f1": f1_90 - f1_95,
        "official_weighted_f1_paired_stratified_bootstrap_mean_delta": mean_delta,
        "official_weighted_f1_paired_stratified_bootstrap_95pct_interval": [lower_delta, upper_delta],
        "pseudo90_legacy_dead_class_f1_at_fixed_rate": legacy_f1_90,
        "pseudo95_legacy_dead_class_f1_at_fixed_rate": legacy_f1_95,
        "pseudo90_minus_pseudo95_legacy_dead_class_f1": legacy_f1_90
        - legacy_f1_95,
        "legacy_dead_class_f1_paired_stratified_bootstrap_mean_delta": legacy_mean_delta,
        "legacy_dead_class_f1_paired_stratified_bootstrap_95pct_interval": [
            legacy_lower_delta,
            legacy_upper_delta,
        ],
        # Backward-compatible aliases preserve the old binary-Dead meaning.
        "pseudo90_fixed_rate_f1": legacy_f1_90,
        "pseudo95_fixed_rate_f1": legacy_f1_95,
        "pseudo90_minus_pseudo95_f1": legacy_f1_90 - legacy_f1_95,
        "paired_stratified_bootstrap_mean_delta": legacy_mean_delta,
        "paired_stratified_bootstrap_95pct_interval": [
            legacy_lower_delta,
            legacy_upper_delta,
        ],
        "legacy_alias_note": "unqualified *_f1 fields are deprecated aliases for binary Dead-class F1",
        "gate_tolerance": -0.001,
        "gate_metric": "official_weighted_f1",
        "submission_a_gate_pass": bool(gate_pass),
        "frozen_tree_is_recipe_equivalent": False,
    }
    (OUTPUT_DIR / "validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("\n", metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
