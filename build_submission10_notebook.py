"""Build the exact historical Submission 10 prediction-replay notebook.

The historical neural-network fold checkpoints were not preserved. This
notebook therefore reconstructs the scored prediction file from the team's
preserved V6 and neural-network probability artifacts. It validates those
artifacts by SHA-256 before blending and does not describe the replay as a
fresh model retrain.
"""

from __future__ import annotations

import base64
from pathlib import Path
import zlib

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "insight_2_0_submission10.ipynb"
V6_SOURCE_PATH = ROOT / "archive" / "probs_v6_final.npy"
NN_SOURCE_PATH = ROOT / "archive" / "probs_nn.npy"

ASCII_TRANSLATION = str.maketrans(
    {
        "\u00d7": "x",
        "\u2013": "-",
        "\u2014": "-",
        "\u2192": "->",
        "\u2264": "<=",
        "\u2265": ">=",
        "\u2500": "-",
    }
)


def compressed_base64(path: Path) -> str:
    """Return a portable compressed representation of an immutable artifact."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing preserved probability artifact: {path}")
    return base64.b64encode(zlib.compress(path.read_bytes(), level=9)).decode("ascii")


V6_PAYLOAD = compressed_base64(V6_SOURCE_PATH)
NN_PAYLOAD = compressed_base64(NN_SOURCE_PATH)


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.translate(ASCII_TRANSLATION).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.translate(ASCII_TRANSLATION).strip() + "\n")


def replace_once(source: str, old: str, new: str, description: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"Could not uniquely locate {description}")
    return source.replace(old, new)


def embedded_tree_pipeline() -> str:
    source = (ROOT / "pipeline_v6.py").read_text(encoding="utf-8")
    source = replace_once(
        source,
        'OUTPUT_DIR = Path("artifacts/v6_rerun")',
        'OUTPUT_DIR = ROOT / "artifacts" / "submission10_notebook" / "tree"',
        "the Submission 6 output directory",
    )
    source = replace_once(
        source,
        'train_df = pd.read_csv("train.csv")',
        "train_df = pd.read_csv(TRAIN_PATH)",
        "the Submission 6 training-data path",
    )
    source = replace_once(
        source,
        'test_df = pd.read_csv("test.csv")',
        "test_df = pd.read_csv(TEST_PATH)",
        "the Submission 6 test-data path",
    )
    return (
        "# Complete Submission 6 tree/pseudo-label training implementation.\n"
        "# Outputs remain isolated under artifacts/submission10_notebook/tree/.\n"
        + source
    )


def embedded_nn_pipeline() -> str:
    source = (ROOT / "pipeline_nn.py").read_text(encoding="utf-8")
    source = replace_once(
        source,
        'train_df = pd.read_csv("train.csv")',
        "train_df = pd.read_csv(TRAIN_PATH)",
        "the NN training-data path",
    )
    source = replace_once(
        source,
        'test_df = pd.read_csv("test.csv")',
        "test_df = pd.read_csv(TEST_PATH)",
        "the NN test-data path",
    )
    source = replace_once(
        source,
        "if torch.cuda.is_available():\n    torch.cuda.manual_seed(SEED)\n",
        """if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# The original run selected the available accelerator automatically. This
# reproducibility audit uses deterministic CPU execution and saves each fold.
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)
NN_OUTPUT_DIR = ROOT / "artifacts" / "submission10_notebook" / "nn"
NN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NN_CHECKPOINTS = []
""",
        "the NN seed block",
    )
    source = replace_once(
        source,
        'device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")',
        'device = torch.device("cpu")',
        "the historical NN device selection",
    )
    source = replace_once(
        source,
        "    model.load_state_dict(best_model)\n    model.eval()\n",
        """    model.load_state_dict(best_model)
    checkpoint_path = NN_OUTPUT_DIR / f"fold_{fold + 1:02d}_best_state.pt"
    torch.save(
        {key: value.detach().cpu() for key, value in best_model.items()},
        checkpoint_path,
    )
    NN_CHECKPOINTS.append(checkpoint_path)
    model.eval()
""",
        "the NN best-model restore block",
    )
    source = replace_once(
        source,
        'np.save("probs_nn.npy", test_preds)\nnp.save("oof_nn.npy", oof_preds)',
        'np.save(NN_OUTPUT_DIR / "probs_nn_fresh.npy", test_preds)\nnp.save(NN_OUTPUT_DIR / "oof_nn_fresh.npy", oof_preds)',
        "the NN probability outputs",
    )
    source = replace_once(
        source,
        'sub.to_csv("submission_nn.csv", index=False)',
        'sub.to_csv(NN_OUTPUT_DIR / "nn_only_submission.csv", index=False, lineterminator="\\n")',
        "the NN-only submission output",
    )
    source = replace_once(
        source,
        'print(f"  submission_nn.csv generated: {dead_count:,} Dead ({dead_count/len(test_preds):.2%}) at th={target_th:.3f}")',
        'print(f"  nn_only_submission.csv generated: {dead_count:,} Dead ({dead_count/len(test_preds):.2%}) at th={target_th:.3f}")',
        "the NN completion message",
    )
    source = replace_once(
        source,
        'print(f"  Wait until tomorrow to submit this!")',
        'print(f"  Fresh NN artifacts and checkpoints: {NN_OUTPUT_DIR}")',
        "the obsolete NN submission reminder",
    )
    return (
        "# Complete historical Submission 10 NN recipe, retrained from raw data.\n"
        "# Prediction-affecting features and hyperparameters are unchanged.\n"
        + source
    )


nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.12"},
}

nb.cells = [
    markdown(
        """
# Insight 2.0 - Submission 10 Modelling and Exact Prediction Reproduction

This notebook documents and executes the complete Submission 10 modelling
recipe: the Submission 6 tree/pseudo-label ensemble, an entity-embedding neural
network, an 80%/20% probability blend, and a deterministic top-30,411 decision
rule. It includes EDA, preprocessing, feature engineering, fixed tuned
hyperparameters, model fitting, validation, and output checks.

Fresh training outputs are retained as a reproducibility audit. The exact
historical scored CSV is reconstructed separately from the immutable
probability artifacts saved at the time of the original run. That reconstruction
is byte-identical to the file that scored `0.876616` on the Public Leaderboard.

**Scope:** this is exact prediction-artifact reproduction, not an exact fresh
retrain of the historical neural network. The original five NN fold
checkpoints and complete runtime state were not preserved. The distinction is
kept explicit throughout the notebook.
"""
    ),
    markdown(
        """
## 1. Environment and data contract

The full workflow requires the organizer-provided `train.csv` and `test.csv`.
It uses manually specified NumPy/pandas/scikit-learn preprocessing, LightGBM,
XGBoost, CatBoost, PyTorch, and Matplotlib. The preserved historical
probability vectors are embedded as compressed, hash-verified notebook data so
no external model artifact is needed for the exact final-file check.
"""
    ),
    code(
        """
from pathlib import Path
import base64
import hashlib
import io
import platform
import zlib

import numpy as np
import pandas as pd
import sklearn
import lightgbm
import xgboost
import catboost
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score

RUN_DIR = Path.cwd().resolve()
ROOT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else RUN_DIR

def locate_data_files():
    candidate_dirs = [RUN_DIR, ROOT]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidate_dirs.extend(
            sorted({path.parent for path in kaggle_input.rglob("train.csv")})
        )

    seen = set()
    for directory in candidate_dirs:
        directory = directory.resolve()
        if directory in seen:
            continue
        seen.add(directory)
        train_path = directory / "train.csv"
        test_path = directory / "test.csv"
        if not (train_path.is_file() and test_path.is_file()):
            continue
        try:
            train_columns = set(pd.read_csv(train_path, nrows=2).columns)
            test_columns = set(pd.read_csv(test_path, nrows=2).columns)
        except Exception:
            continue
        if (
            "vital_status" in train_columns
            and "vital_status" not in test_columns
            and "patient_id" in train_columns
            and "patient_id" in test_columns
            and test_columns == train_columns - {"vital_status"}
        ):
            return train_path, test_path

    raise FileNotFoundError(
        "Could not locate the organizer train.csv and test.csv together. "
        "Attach the competition data to the notebook or place both files "
        "beside the notebook."
    )

TRAIN_PATH, TEST_PATH = locate_data_files()

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
assert "vital_status" in train.columns
assert "patient_id" in test.columns
assert "vital_status" not in test.columns
assert set(test.columns) == set(train.columns) - {"vital_status"}
assert len(train) == 24_000
assert len(test) == 36_000
assert train["patient_id"].is_unique
assert test["patient_id"].is_unique
assert not train["patient_id"].isna().any()
assert not test["patient_id"].isna().any()
assert set(train["vital_status"].unique()) == {"Alive", "Dead"}
y_binary = train["vital_status"].eq("Dead").to_numpy(dtype=np.int8)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sha256_bytes(values):
    return hashlib.sha256(values).hexdigest()

print("Python:", platform.python_version())
print("Training data:", TRAIN_PATH)
print("Test data:", TEST_PATH)
runtime_versions = {
    "NumPy": np.__version__,
    "pandas": pd.__version__,
    "scikit-learn": sklearn.__version__,
    "LightGBM": lightgbm.__version__,
    "XGBoost": xgboost.__version__,
    "CatBoost": catboost.__version__,
    "PyTorch": torch.__version__,
}
display(pd.Series(runtime_versions, name="runtime version").to_frame())
print("Train/test shapes:", train.shape, test.shape)
print("Dead prevalence:", f"{y_binary.mean():.3%}")
"""
    ),
    markdown(
        """
## 2. Exploratory data analysis

The compact EDA checks target imbalance, missingness, stage-associated outcome
rates, age bands, and train/test categorical coverage. These checks informed
the missingness indicators, ordinal stage features, and robust unknown-category
handling used by the models. Patient identifiers are never used as features.
"""
    ),
    code(
        """
eda_summary = pd.DataFrame({
    "dtype": train.dtypes.astype(str),
    "missing_count": train.isna().sum(),
    "missing_rate": train.isna().mean(),
    "unique_values": train.nunique(dropna=False),
}).sort_values("missing_rate", ascending=False)
display(eda_summary.head(15))

fig, axes = plt.subplots(1, 3, figsize=(17, 4))
train["vital_status"].value_counts().reindex(["Alive", "Dead"]).plot.bar(
    ax=axes[0], color=["#4C78A8", "#E45756"]
)
axes[0].set_title("Target distribution")
axes[0].set_ylabel("Patients")

eda_summary.head(12).sort_values("missing_rate")["missing_rate"].plot.barh(
    ax=axes[1], color="#72B7B2"
)
axes[1].set_title("Highest missingness rates")
axes[1].set_xlabel("Missing fraction")

stage_rate = train.groupby("summary_stage", dropna=False)["vital_status"].apply(
    lambda values: values.eq("Dead").mean()
).sort_values()
stage_rate.plot.barh(ax=axes[2], color="#F58518")
axes[2].set_title("Dead rate by summary stage")
axes[2].set_xlabel("Dead fraction")
plt.tight_layout()
plt.show()

age_audit = train.groupby("age_recode", dropna=False).agg(
    patients=("patient_id", "size"),
    dead_rate=("vital_status", lambda values: values.eq("Dead").mean()),
)
display(age_audit)

categorical_columns = train.drop(
    columns=["patient_id", "vital_status"]
).select_dtypes(include=["object", "string"]).columns
coverage = []
for column in categorical_columns:
    train_values = set(train[column].fillna("__NA__").astype(str))
    test_values = set(test[column].fillna("__NA__").astype(str))
    coverage.append({
        "feature": column,
        "test_only_categories": len(test_values - train_values),
    })
display(pd.DataFrame(coverage).sort_values(
    "test_only_categories", ascending=False
).head(15))
"""
    ),
    markdown(
        """
## 3. Preprocessing, feature engineering, and fixed models

The tree branch uses two manually designed feature views. Both include age
midpoint, diagnosis year, TNM order, positive-node ratio, metastasis count,
tumour-size reconciliation, treatment indicators, missingness indicators, and
target-free category frequencies. One view additionally uses fold-safe target
encoding. LightGBM, XGBoost, and CatBoost are averaged within each view; the two
views are blended and retrained once with high-confidence model-generated
pseudo labels.

The diversity branch standardizes clinical numeric features and learns entity
embeddings for 29 categorical variables. Its MLP uses hidden widths 256/128/64,
batch normalization, ReLU, dropout 0.3, BCE loss, AdamW, five stratified folds,
batch size 1,024, 15 epochs, and lowest-validation-loss checkpoint selection.

All parameters are frozen from development. No hyperparameter search or AutoML
pipeline generation is performed during the final reproducibility run.
"""
    ),
    markdown(
        """
## 4. Submission 6 tree/pseudo-label training

The following cell embeds the complete maintained Submission 6 implementation.
It fits both feature views and all six model components from `train.csv`, creates
the pseudo-label set from model confidence, retrains the student ensemble, and
writes fresh outputs under `artifacts/submission10_notebook/tree/`.
"""
    ),
    code(embedded_tree_pipeline()),
    markdown(
        """
## 5. Neural-network training

This cell embeds the historical Submission 10 NN architecture and training
recipe. For cross-platform auditability, the fresh run uses deterministic CPU
execution and saves the best state from each of its five folds. This fresh run
tests the recipe; it is not represented as the unavailable historical fold
state.
"""
    ),
    code(embedded_nn_pipeline()),
    markdown(
        """
## 6. Preserved historical model artifacts

Submission 6 combined LightGBM, XGBoost, and CatBoost across target-encoded and
non-target-encoded feature views, followed by one pass of pseudo-label
training. The neural component used entity embeddings for categorical fields
and an MLP for the combined categorical and continuous representation.

The embedded arrays below are the saved test-set inference outputs from those
two model families. Their original `.npy` byte hashes are checked after
decompression so a modified payload cannot silently generate a different
prediction file.
"""
    ),
    code(
        """
EXPECTED_V6_SHA256 = "aca54c31462449df432e1edda5da81a6d04e242c8985cfde0e5983c6d0d92ab6"
EXPECTED_NN_SHA256 = "7ec4721ae7d4eccb35ebc5821014e581ad2da4e872775d8a4845f37423b1ce46"
"""
        + f"\nV6_NPY_ZLIB_BASE64 = {V6_PAYLOAD!r}\n"
        + f"NN_NPY_ZLIB_BASE64 = {NN_PAYLOAD!r}\n"
        + """
v6_npy_bytes = zlib.decompress(base64.b64decode(V6_NPY_ZLIB_BASE64))
nn_npy_bytes = zlib.decompress(base64.b64decode(NN_NPY_ZLIB_BASE64))

actual_v6_sha256 = sha256_bytes(v6_npy_bytes)
actual_nn_sha256 = sha256_bytes(nn_npy_bytes)
assert actual_v6_sha256 == EXPECTED_V6_SHA256, "Unexpected V6 probability artifact"
assert actual_nn_sha256 == EXPECTED_NN_SHA256, "Unexpected NN probability artifact"

historical_v6_probability = np.load(
    io.BytesIO(v6_npy_bytes), allow_pickle=False
).astype(np.float64)
historical_nn_probability = np.load(
    io.BytesIO(nn_npy_bytes), allow_pickle=False
).astype(np.float64)

for name, values in {
    "Submission 6": historical_v6_probability,
    "Neural network": historical_nn_probability,
}.items():
    assert values.shape == (len(test),), f"{name} probability shape mismatch"
    assert np.isfinite(values).all(), f"{name} contains non-finite values"
    assert np.logical_and(values >= 0.0, values <= 1.0).all(), f"{name} is outside [0, 1]"

print("Submission 6 probability SHA-256:", actual_v6_sha256)
print("Neural-network probability SHA-256:", actual_nn_sha256)
"""
    ),
    markdown(
        """
## 7. Exact Submission 10 reconstruction

The blend weights and class count were fixed before this replay. Stable
mergesort ordering provides deterministic tie-breaking at the decision
boundary.
"""
    ),
    code(
        """
TREE_WEIGHT = 0.80
NN_WEIGHT = 0.20
DEAD_COUNT = 30_411

submission10_probability = (
    TREE_WEIGHT * historical_v6_probability
    + NN_WEIGHT * historical_nn_probability
)

order = np.argsort(-submission10_probability, kind="mergesort")
is_dead = np.zeros(len(test), dtype=bool)
is_dead[order[:DEAD_COUNT]] = True

submission = pd.DataFrame(
    {
        "patient_id": test["patient_id"].to_numpy(),
        "vital_status": np.where(is_dead, "Dead", "Alive"),
    }
)

OUTPUT_PATH = ROOT / "submission.csv"
PROBABILITY_OUTPUT_PATH = ROOT / "probs_submission10.npy"
submission.to_csv(OUTPUT_PATH, index=False, lineterminator="\\n")
np.save(PROBABILITY_OUTPUT_PATH, submission10_probability)

print("Wrote:", OUTPUT_PATH)
print("Rows:", len(submission))
print("Dead:", int(is_dead.sum()))
print("Alive:", int((~is_dead).sum()))
"""
    ),
    markdown(
        """
## 8. Exact output verification

The structural checks enforce the Kaggle submission contract. The final hash
is a strict gate here because this notebook intentionally replays immutable
probability artifacts rather than retraining hardware-sensitive models.
"""
    ),
    code(
        """
EXPECTED_SUBMISSION10_SHA256 = "333af97cfbc16ffdcc2d9f910000664c443694239c60e67d6504af18687e86f1"

assert list(submission.columns) == ["patient_id", "vital_status"]
assert len(submission) == 36_000
assert submission["patient_id"].equals(test["patient_id"])
assert submission["patient_id"].is_unique
assert not submission.isna().any().any()
assert set(submission["vital_status"].unique()) == {"Alive", "Dead"}
assert int(submission["vital_status"].eq("Dead").sum()) == DEAD_COUNT

actual_submission_sha256 = sha256(OUTPUT_PATH)
assert actual_submission_sha256 == EXPECTED_SUBMISSION10_SHA256

archived_submission_path = ROOT / "archive" / "submission10.csv"
if archived_submission_path.is_file():
    assert OUTPUT_PATH.read_bytes() == archived_submission_path.read_bytes()

print("Submission 10 SHA-256:", actual_submission_sha256)
print("Exact historical prediction replay: PASS")
print("Verified historical Public Leaderboard score: 0.876616")
"""
    ),
    markdown(
        """
## 9. Fresh-training audit and interpretation

The fresh models fitted in Sections 4–5 are evaluated separately from the
historical replay. Their output shows that the complete recipe remains
executable while making the historical NN reproducibility boundary measurable.
The OOF blend below is a screening comparison because the tree and NN branches
do not share a fully nested outer-fold design.
"""
    ),
    code(
        """
def stable_top_k(probabilities, count):
    ranking = np.argsort(-np.asarray(probabilities), kind="mergesort")
    labels = np.zeros(len(ranking), dtype=np.int8)
    labels[ranking[:count]] = 1
    return labels

AUDIT_DIR = ROOT / "artifacts" / "submission10_notebook"
FRESH_TREE_PATH = AUDIT_DIR / "tree" / "probs_v6_final.npy"
FRESH_NN_PATH = AUDIT_DIR / "nn" / "probs_nn_fresh.npy"
FRESH_NN_OOF_PATH = AUDIT_DIR / "nn" / "oof_nn_fresh.npy"
for path in (FRESH_TREE_PATH, FRESH_NN_PATH, FRESH_NN_OOF_PATH):
    if not path.is_file():
        raise FileNotFoundError(f"Fresh training output missing: {path}")

fresh_tree_probability = np.load(FRESH_TREE_PATH, allow_pickle=False)
fresh_nn_probability = np.load(FRESH_NN_PATH, allow_pickle=False)
fresh_nn_oof = np.load(FRESH_NN_OOF_PATH, allow_pickle=False)
assert fresh_tree_probability.shape == fresh_nn_probability.shape == (len(test),)
assert fresh_nn_oof.shape == (len(train),)

fresh_blend_probability = 0.80 * fresh_tree_probability + 0.20 * fresh_nn_probability
fresh_is_dead = stable_top_k(fresh_blend_probability, DEAD_COUNT).astype(bool)
fresh_submission = pd.DataFrame({
    "patient_id": test["patient_id"].to_numpy(),
    "vital_status": np.where(fresh_is_dead, "Dead", "Alive"),
})
FRESH_SUBMISSION_PATH = AUDIT_DIR / "submission_fresh_retrain.csv"
fresh_submission.to_csv(FRESH_SUBMISSION_PATH, index=False, lineterminator="\\n")

oof_count = round(0.845 * len(train))
fresh_oof_blend = 0.80 * oof_blend + 0.20 * fresh_nn_oof
fresh_oof_labels = stable_top_k(fresh_oof_blend, oof_count)
fresh_weighted_f1 = f1_score(
    y_binary, fresh_oof_labels, average="weighted", zero_division=0
)
fresh_dead_class_f1 = f1_score(
    y_binary, fresh_oof_labels, average="binary", zero_division=0
)
fresh_auc = roc_auc_score(y_binary, fresh_oof_blend)

fresh_vs_historical_changes = int(np.sum(fresh_is_dead != is_dead))
fresh_audit = pd.Series({
    "fresh_tree_probability_sha256": sha256(FRESH_TREE_PATH),
    "fresh_nn_probability_sha256": sha256(FRESH_NN_PATH),
    "fresh_submission_sha256": sha256(FRESH_SUBMISSION_PATH),
    "fresh_vs_historical_label_changes": fresh_vs_historical_changes,
    "fresh_OOF_support_weighted_F1": fresh_weighted_f1,
    "fresh_OOF_Dead_class_F1": fresh_dead_class_f1,
    "fresh_OOF_ROC_AUC": fresh_auc,
    "saved_fresh_NN_checkpoints": len(NN_CHECKPOINTS),
})
display(fresh_audit.to_frame("value"))

assert len(NN_CHECKPOINTS) == 5
fresh_tree_exact = sha256(FRESH_TREE_PATH) == EXPECTED_V6_SHA256
print("Fresh tree byte match:", "PASS" if fresh_tree_exact else "ENVIRONMENT DRIFT")
if not fresh_tree_exact:
    print(
        "The fresh tree probabilities differ at byte level, which can occur "
        "across library, operating-system, and hardware versions. The exact "
        "historical Submission 10 replay above is unaffected."
    )

# Representative importance from the retained pseudo-trained v4 CatBoost
# component. This is a component diagnostic, not a causal interpretation.
representative_names = list(X_train_base.columns) + [
    f"{column}_te" for column in te_cols
]
if hasattr(m, "get_feature_importance"):
    representative_values = np.asarray(m.get_feature_importance(), dtype=float)
    if len(representative_values) == len(representative_names):
        importance = (
            pd.DataFrame({"feature": representative_names, "importance": representative_values})
            .sort_values("importance", ascending=False)
            .head(15)
            .sort_values("importance")
        )
        ax = importance.plot.barh(
            x="feature", y="importance", legend=False, figsize=(9, 6), color="#4C78A8"
        )
        ax.set_title("Representative pseudo-trained CatBoost importance")
        ax.set_xlabel("CatBoost importance")
        ax.set_ylabel("")
        plt.tight_layout()
        plt.show()

age_rows = []
age_values = train["age_recode"].fillna("Missing").astype(str)
for age_band, indices in age_values.groupby(age_values).groups.items():
    indices = np.asarray(list(indices))
    if len(indices) < 200:
        continue
    age_rows.append({
        "age_band": age_band,
        "patients": len(indices),
        "support_weighted_F1": f1_score(
            y_binary[indices], fresh_oof_labels[indices],
            average="weighted", zero_division=0,
        ),
    })
display(pd.DataFrame(age_rows).sort_values("support_weighted_F1"))
"""
    ),
    markdown(
        """
## 10. Results and limitations

Running this notebook regenerates the exact historical Submission 10 CSV. An
identical prediction file receives the same score when evaluated against the
same Kaggle labels.

The notebook also performs EDA, feature engineering, full Submission 6
training, five-fold NN training with newly saved checkpoints, OOF screening,
feature-importance inspection, and an age-band audit. The fresh model output is
kept under `artifacts/submission10_notebook/` and never replaces the exact
historical replay written to `submission.csv`.

The replay establishes the provenance of the scored predictions, blend, and
decision rule. It does not establish that the fresh neural-network run is the
unavailable historical fold state. Exact historical NN-training identity would
require the original checkpoints and runtime state; the notebook reports the
fresh-versus-historical label difference rather than hiding it.

No AutoML pipeline generator, external data, manual test labels, or row-level
leaderboard probing is used.
"""
    ),
]

nbf.write(nb, NOTEBOOK_PATH)
print(f"Wrote {NOTEBOOK_PATH} with {len(nb.cells)} cells")
