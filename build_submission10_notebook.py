"""Build the separate, self-contained Submission 10 recipe notebook.

The generated notebook performs a clean tree and neural-network retrain before
forming the historical 80/20 blend.  Frozen historical probabilities are used
only in a clearly separated, optional post-training provenance audit; they are
never substituted for the fresh predictions.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "insight_2_0_submission10.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def replace_once(source: str, old: str, new: str, description: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"Could not uniquely locate {description}")
    return source.replace(old, new)


def embedded_tree_pipeline() -> str:
    source = (ROOT / "pipeline_v6.py").read_text(encoding="utf-8")
    source = replace_once(
        source,
        'OUTPUT_DIR = Path("artifacts/v6_rerun")',
        'OUTPUT_DIR = Path("artifacts/submission10_notebook/tree")',
        "the Submission 6 output declaration",
    )
    return (
        "# Submission 6 tree/pseudo-label implementation embedded for portability.\n"
        "# It reads train.csv/test.csv and writes only below the Submission 10 audit directory.\n"
        + source
    )


def embedded_nn_pipeline() -> str:
    source = (ROOT / "pipeline_nn.py").read_text(encoding="utf-8")
    source = replace_once(
        source,
        "if torch.cuda.is_available():\n    torch.cuda.manual_seed(SEED)\n",
        """if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# The historical code selected CUDA/MPS/CPU automatically.  This audit forces
# CPU and deterministic algorithms to make the fresh rerun more portable.
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
        "the historical automatic device selection",
    )
    source = replace_once(
        source,
        "    model.load_state_dict(best_model)\n    model.eval()\n",
        """    model.load_state_dict(best_model)
    checkpoint_path = NN_OUTPUT_DIR / f"fold_{fold + 1:02d}_best_state.pt"
    torch.save({key: value.detach().cpu() for key, value in best_model.items()}, checkpoint_path)
    NN_CHECKPOINTS.append(checkpoint_path)
    model.eval()
""",
        "the best-model restore block",
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
        "the obsolete submission reminder",
    )
    return (
        "# Historical Submission 10 NN recipe embedded for portability.\n"
        "# Prediction-affecting hyperparameters and preprocessing remain unchanged.\n"
        + source
    )


nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.14"},
}

nb.cells = [
    markdown(
        """
# Insight 2.0 — Submission 10 Recipe Reproduction Audit

This is a separate, executable notebook for the historical Submission 10
method: 80% of the Submission 6 tree/pseudo-label probability and 20% of an
entity-embedding neural-network probability, followed by a stable top-30,411
decision rule. The official objective is support-weighted F1, and output labels
are `Dead` and `Alive`.

**Scope and status:** this notebook retrains both model families from
`train.csv` and `test.csv`; its fresh prediction does not load historical test
probabilities. The original NN fold checkpoints were not preserved. A controlled
preflight rerun on the audited CPU environment was deterministic across two
runs, but differed from the archived NN vector and changed 100 hard labels in
the final blend. Consequently, this is an honest reproduction of the
Submission 10 **recipe**, not a claim of byte-identical regeneration of the
historical scored CSV.

The optional historical appendix runs only after fresh training and, when the
archived arrays are present, verifies the exact historical artifact separately.
It never replaces the fresh prediction. Organizer confirmation is still needed
before presenting artifact replay as satisfying an exact-reproducibility rule.

The verified Submission 6 notebook remains
`insight_2_0_consolidated.ipynb` and is not changed by this notebook.
"""
    ),
    markdown(
        """
## 1. Environment and data contract

The workflow uses only the organizer-provided training and test tables for
fresh model fitting. It validates schema and identifiers and records the exact
runtime. The audited environment uses NumPy 2.4.3, pandas 3.0.3,
scikit-learn 1.8.0, LightGBM 4.7.0, XGBoost 3.2.0, CatBoost 1.2.10, and
PyTorch 2.12.0. The fresh NN audit deliberately uses CPU with deterministic
algorithms; the historical script automatically selected the available device.
"""
    ),
    code(
        """
from pathlib import Path
import hashlib
import json
import platform
import warnings

import numpy as np
import pandas as pd
import sklearn
import lightgbm
import xgboost
import catboost
import torch
from sklearn.metrics import f1_score

ROOT = Path.cwd().resolve()
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test.csv"
for required in (TRAIN_PATH, TEST_PATH):
    if not required.is_file():
        raise FileNotFoundError(f"Missing required competition file: {required}")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
assert "vital_status" not in test.columns
assert set(test.columns) == set(train.columns) - {"vital_status"}
assert train["patient_id"].is_unique and test["patient_id"].is_unique
assert not train["patient_id"].isna().any() and not test["patient_id"].isna().any()
assert set(train["vital_status"].unique()) == {"Dead", "Alive"}

y_binary = train["vital_status"].eq("Dead").astype("int8")
AUDITED_VERSIONS = {
    "numpy": "2.4.3",
    "pandas": "3.0.3",
    "scikit-learn": "1.8.0",
    "lightgbm": "4.7.0",
    "xgboost": "3.2.0",
    "catboost": "1.2.10",
    "torch": "2.12.0",
}
RUNTIME_VERSIONS = {
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit-learn": sklearn.__version__,
    "lightgbm": lightgbm.__version__,
    "xgboost": xgboost.__version__,
    "catboost": catboost.__version__,
    "torch": torch.__version__,
}
VERSION_MATCHES = {
    name: RUNTIME_VERSIONS[name] == expected
    for name, expected in AUDITED_VERSIONS.items()
}
print("Python:", platform.python_version())
display(pd.DataFrame({
    "audited": AUDITED_VERSIONS,
    "runtime": RUNTIME_VERSIONS,
    "exact_match": VERSION_MATCHES,
}))
print("train/test shapes:", train.shape, test.shape)
print("Dead prevalence:", f"{y_binary.mean():.3%}")
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())
"""
    ),
    markdown(
        """
## 2. Exploratory data analysis

The compact EDA checks class imbalance, missingness, stage, age, and train/test
category coverage. These are useful both for model design and for detecting
schema drift before training. Patient identifiers are never model features.
"""
    ),
    code(
        """
import matplotlib.pyplot as plt

summary = pd.DataFrame({
    "dtype": train.dtypes.astype(str),
    "missing_count": train.isna().sum(),
    "missing_rate": train.isna().mean(),
    "unique_values": train.nunique(dropna=False),
}).sort_values("missing_rate", ascending=False)
display(summary.head(15))

fig, axes = plt.subplots(1, 3, figsize=(17, 4))
train["vital_status"].value_counts().reindex(["Alive", "Dead"]).plot.bar(
    ax=axes[0], color=["#4C78A8", "#E45756"]
)
axes[0].set_title("Target distribution")
axes[0].set_ylabel("Patients")

summary.head(12).sort_values("missing_rate")["missing_rate"].plot.barh(
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
).sort_index()
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
## 3. Feature engineering and model rationale

The tree component uses two manually designed views: a non-target-encoded view
and a fold-safe target-encoded view. Features include age midpoint, diagnosis
year, TNM order, positive-node ratio, metastasis count, tumour-size
reconciliation, treatment indicators, missingness indicators, and target-free
category frequencies. Six LightGBM/XGBoost/CatBoost components are blended,
then retrained once with high-confidence model-generated pseudo labels.

The diversity component uses standardized clinical numeric features plus
entity embeddings for 29 categorical variables. Its MLP has hidden widths
256/128/64, batch normalization, ReLU, dropout 0.3, BCE loss, AdamW, five
stratified folds, batch size 1,024, 15 epochs, and lowest-validation-loss model
selection. All hyperparameters are frozen; no AutoML pipeline generator is
used.

The production combination is a deterministic 80% tree / 20% NN probability
blend. A stable rank rule selects exactly 30,411 `Dead` predictions so the
model comparison is not confounded by a class-count change.
"""
    ),
    markdown(
        """
## 4. Fresh Submission 6 tree training

This cell embeds the complete historical Submission 6 implementation. It reads
only the two competition tables and writes fresh artifacts under
`artifacts/submission10_notebook/tree/`. Cached tree probabilities are not
prediction inputs.
"""
    ),
    code(embedded_tree_pipeline()),
    markdown(
        """
## 5. Fresh neural-network training

This cell embeds the historical Submission 10 neural-network recipe. The model
architecture, features, folds, optimizer, and training schedule are unchanged.
For future auditability, this rerun forces CPU deterministic execution and now
saves the selected state for every fold under
`artifacts/submission10_notebook/nn/`. Historical fold states were not saved.
"""
    ),
    code(embedded_nn_pipeline()),
    markdown(
        """
## 6. Fresh 80/20 blend, validation, and historical provenance

The fresh submission below uses only probabilities produced by the two
preceding training cells. It is structurally validated and written to
`artifacts/submission10_notebook/submission.csv`.

Only after that output exists does the optional historical audit load archived
arrays. It reports correlation, errors, hashes, and label disagreement. It
also writes a separately named historical replay for provenance when—and only
when—the archived arrays pass their expected hashes. That replay never replaces
the fresh output.
"""
    ),
    code(
        """
warnings.filterwarnings("default")

def stable_top_k(probabilities, count):
    order = np.argsort(-np.asarray(probabilities), kind="mergesort")
    labels = np.zeros(len(order), dtype=np.int8)
    labels[order[:count]] = 1
    return labels

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

OUTPUT_DIR = ROOT / "artifacts" / "submission10_notebook"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TREE_PROBABILITY_PATH = OUTPUT_DIR / "tree" / "probs_v6_final.npy"
NN_PROBABILITY_PATH = OUTPUT_DIR / "nn" / "probs_nn_fresh.npy"
NN_OOF_PATH = OUTPUT_DIR / "nn" / "oof_nn_fresh.npy"
FRESH_PROBABILITY_PATH = OUTPUT_DIR / "probs_submission10_fresh.npy"
FRESH_SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"

tree_probabilities = np.load(TREE_PROBABILITY_PATH, allow_pickle=False)
nn_probabilities = np.load(NN_PROBABILITY_PATH, allow_pickle=False)
nn_oof_probabilities = np.load(NN_OOF_PATH, allow_pickle=False)
for name, values, expected_length in (
    ("tree", tree_probabilities, len(test)),
    ("NN", nn_probabilities, len(test)),
    ("NN OOF", nn_oof_probabilities, len(train)),
):
    assert values.shape == (expected_length,), f"{name} shape mismatch"
    assert np.isfinite(values).all(), f"{name} contains non-finite values"
    assert ((values >= 0.0) & (values <= 1.0)).all(), f"{name} outside [0,1]"

target_dead_count = 30_411
fresh_probabilities = 0.80 * tree_probabilities + 0.20 * nn_probabilities
fresh_labels = stable_top_k(fresh_probabilities, target_dead_count)
fresh_submission = pd.DataFrame({
    "patient_id": test["patient_id"].copy(),
    "vital_status": np.where(fresh_labels == 1, "Dead", "Alive"),
})
np.save(FRESH_PROBABILITY_PATH, fresh_probabilities)
fresh_submission.to_csv(FRESH_SUBMISSION_PATH, index=False, lineterminator="\\n")

assert fresh_submission.columns.tolist() == ["patient_id", "vital_status"]
assert len(fresh_submission) == len(test) == 36_000
assert fresh_submission["patient_id"].equals(test["patient_id"])
assert fresh_submission["patient_id"].is_unique
assert fresh_submission["vital_status"].isin(["Dead", "Alive"]).all()
assert not fresh_submission.isna().any().any()
assert int(fresh_submission["vital_status"].eq("Dead").sum()) == target_dead_count

# This is a screening proxy only: the tree vector is the pre-pseudo teacher OOF,
# while the final test-side tree probability includes pseudo-label retraining.
proxy_rate_count = round(0.845 * len(train))
tree_proxy_labels = stable_top_k(oof_blend, proxy_rate_count)
nn_oof_labels = stable_top_k(nn_oof_probabilities, proxy_rate_count)
blend_proxy_probabilities = 0.80 * oof_blend + 0.20 * nn_oof_probabilities
blend_proxy_labels = stable_top_k(blend_proxy_probabilities, proxy_rate_count)
fresh_oof_metrics = {
    "tree_pre_pseudo_weighted_f1": float(f1_score(
        y, tree_proxy_labels, average="weighted", zero_division=0
    )),
    "nn_weighted_f1": float(f1_score(
        y, nn_oof_labels, average="weighted", zero_division=0
    )),
    "blend_pre_pseudo_proxy_weighted_f1": float(f1_score(
        y, blend_proxy_labels, average="weighted", zero_division=0
    )),
}

EXPECTED_HISTORICAL = {
    "tree_probability_sha256": "aca54c31462449df432e1edda5da81a6d04e242c8985cfde0e5983c6d0d92ab6",
    "nn_probability_sha256": "7ec4721ae7d4eccb35ebc5821014e581ad2da4e872775d8a4845f37423b1ce46",
    "submission10_sha256": "333af97cfbc16ffdcc2d9f910000664c443694239c60e67d6504af18687e86f1",
}
report = {
    "status": "fresh_recipe_retrain",
    "fresh_output": {
        "submission_path": str(FRESH_SUBMISSION_PATH.relative_to(ROOT)),
        "submission_sha256": sha256(FRESH_SUBMISSION_PATH),
        "tree_probability_sha256": sha256(TREE_PROBABILITY_PATH),
        "nn_probability_sha256": sha256(NN_PROBABILITY_PATH),
        "rows": len(fresh_submission),
        "dead": int(fresh_labels.sum()),
        "alive": int((fresh_labels == 0).sum()),
    },
    "fresh_oof_screening_metrics": fresh_oof_metrics,
    "historical_expected_hashes": EXPECTED_HISTORICAL,
    "historical_audit": {"available": False},
}

historical_tree_path = ROOT / "archive" / "probs_v6_final.npy"
historical_nn_path = ROOT / "archive" / "probs_nn.npy"
historical_submission_path = ROOT / "archive" / "submission10.csv"
if all(path.is_file() for path in (
    historical_tree_path, historical_nn_path, historical_submission_path
)):
    historical_tree_hash = sha256(historical_tree_path)
    historical_nn_hash = sha256(historical_nn_path)
    historical_submission_hash = sha256(historical_submission_path)
    trusted_historical_inputs = (
        historical_tree_hash == EXPECTED_HISTORICAL["tree_probability_sha256"]
        and historical_nn_hash == EXPECTED_HISTORICAL["nn_probability_sha256"]
        and historical_submission_hash == EXPECTED_HISTORICAL["submission10_sha256"]
    )

    historical_nn = np.load(historical_nn_path, allow_pickle=False)
    historical_submission = pd.read_csv(historical_submission_path)
    assert historical_submission["patient_id"].equals(test["patient_id"])
    historical_labels = historical_submission["vital_status"].eq("Dead").to_numpy()
    label_disagreements = int(np.sum(fresh_labels.astype(bool) != historical_labels))
    nn_mae = float(np.mean(np.abs(nn_probabilities - historical_nn)))
    nn_max_error = float(np.max(np.abs(nn_probabilities - historical_nn)))
    nn_pearson = float(np.corrcoef(nn_probabilities, historical_nn)[0, 1])
    nn_spearman = float(pd.Series(nn_probabilities).corr(
        pd.Series(historical_nn), method="spearman"
    ))

    replay_path = None
    replay_hash = None
    replay_exact = None
    if trusted_historical_inputs:
        historical_tree = np.load(historical_tree_path, allow_pickle=False)
        replay_probabilities = 0.80 * historical_tree + 0.20 * historical_nn
        replay_labels = stable_top_k(replay_probabilities, target_dead_count)
        replay = pd.DataFrame({
            "patient_id": test["patient_id"].copy(),
            "vital_status": np.where(replay_labels == 1, "Dead", "Alive"),
        })
        replay_path = OUTPUT_DIR / "historical_replay_submission10.csv"
        replay.to_csv(replay_path, index=False, lineterminator="\\n")
        replay_hash = sha256(replay_path)
        replay_exact = replay_hash == EXPECTED_HISTORICAL["submission10_sha256"]
        if not replay_exact:
            warnings.warn("Historical replay did not reproduce the audited CSV hash.")

    report["historical_audit"] = {
        "available": True,
        "trusted_expected_hashes": trusted_historical_inputs,
        "fresh_nn_matches_historical_bytes": sha256(NN_PROBABILITY_PATH) == historical_nn_hash,
        "fresh_submission_matches_historical_bytes": sha256(FRESH_SUBMISSION_PATH) == historical_submission_hash,
        "fresh_vs_historical_label_disagreements": label_disagreements,
        "fresh_vs_historical_nn_mae": nn_mae,
        "fresh_vs_historical_nn_max_error": nn_max_error,
        "fresh_vs_historical_nn_pearson": nn_pearson,
        "fresh_vs_historical_nn_spearman": nn_spearman,
        "historical_replay_path": (
            str(replay_path.relative_to(ROOT)) if replay_path is not None else None
        ),
        "historical_replay_sha256": replay_hash,
        "historical_replay_exact": replay_exact,
    }
    if label_disagreements:
        warnings.warn(
            f"Fresh recipe retrain differs from historical Submission 10 on "
            f"{label_disagreements:,} rows; no frozen prediction was substituted."
        )

REPORT_PATH = OUTPUT_DIR / "reproduction_report.json"
REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n", encoding="utf-8")

print("Fresh structural validation: PASS")
print("Fresh output:", FRESH_SUBMISSION_PATH)
print("Rows / Dead / Alive:", len(fresh_submission), int(fresh_labels.sum()), int((fresh_labels == 0).sum()))
print("Fresh SHA-256:", report["fresh_output"]["submission_sha256"])
print("OOF screening metrics:", fresh_oof_metrics)
print("Historical audit:", report["historical_audit"])
print("Machine-readable report:", REPORT_PATH)
"""
    ),
    markdown(
        """
## 7. Interpretation, evidence, and limitations

### Representative tree-component importance

The chart below reports feature importance from the retained final-fold,
pseudo-trained v4 CatBoost component. It is one component of a larger ensemble,
not a causal or ensemble-wide importance measure.
"""
    ),
    code(
        """
representative_feature_names = list(X_train_base.columns) + [
    f"{column}_te" for column in te_cols
]

if hasattr(m, "get_feature_importance"):
    representative_importance = np.asarray(m.get_feature_importance(), dtype=float)
    if len(representative_importance) == len(representative_feature_names):
        importance_table = (
            pd.DataFrame({
                "feature": representative_feature_names,
                "importance": representative_importance,
            })
            .sort_values("importance", ascending=False)
            .head(15)
            .sort_values("importance")
        )
        ax = importance_table.plot.barh(
            x="feature",
            y="importance",
            figsize=(9, 6),
            legend=False,
            color="#4C78A8",
        )
        ax.set_title("Representative final-fold v4 CatBoost feature importance")
        ax.set_xlabel("CatBoost importance")
        ax.set_ylabel("")
        plt.tight_layout()
        plt.show()
        display(importance_table.sort_values(
            "importance", ascending=False
        ).reset_index(drop=True))
    else:
        warnings.warn("Representative importance length did not match feature names.")
else:
    warnings.warn("The retained component did not expose CatBoost importance.")
"""
    ),
    markdown(
        """
### Validation record and subgroup context

Historical saved-OOF screening favored the 80/20 blend over the pure tree
candidate, but those tree and NN vectors used different cross-validation
schemes and are not an end-to-end nested estimate. Submission 10 scored
`0.876616` on the Public Leaderboard, below Submission 6 (`0.877258`) and
Submission 12 (`0.877460`). The planned Submission 10 selection is therefore a
private-split diversity decision, not a claim that Public score proved it best.

The leakage-safe Submission 6 reconstruction found an age-55–59 weighted-F1
trough (`0.8381`) and a weak localized-stage slice. Dedicated follow-ups found
no sufficiently stable targeted fix; the NN blend is global and is not claimed
to resolve either subgroup uniformly.

### Results and limitations

The fresh output is the result of complete model fitting from the organizer
tables. The historical replay, when available, is a provenance demonstration
from preserved probability artifacts and must not be confused with retraining.
The original NN checkpoints and a contemporaneous environment manifest were
not preserved, so an exact historical neural-network retrain cannot be claimed.

The audited clean-kernel run completed in `516.84` seconds (about 8 minutes
37 seconds). Its fresh tree probability matched the historical tree bytes, but
the deterministic fresh NN did not: the final fresh blend differed from
historical Submission 10 on 108 rows and had SHA-256
`d624009d5e12e58c157bee34b664ebaf23eba05f9613f2c5e7ccd0c65a98cf34`.
The separately named historical replay matched the scored Submission 10 hash
`333af97cfbc16ffdcc2d9f910000664c443694239c60e67d6504af18687e86f1`.
This proves artifact provenance but does not close the historical NN-retrain
gap.

CPU deterministic controls and newly saved checkpoints make this fresh rerun
repeatable going forward, but numerical identity can still depend on the
supported platform and package builds. Pseudo-labeling can reinforce confident
teacher errors, and the fixed class-count rule may transfer imperfectly if
hidden prevalence differs. Neither Public LB nor saved-OOF screening reveals
the hidden Private ordering.

The workflow uses manually specified preprocessing, features, models, folds,
and ensemble weights. It uses no AutoML pipeline generator, external data,
manual test-set labels, or row-level leaderboard probing. Written organizer
clarification remains advisable for pseudo-labeling and for using historical
artifact replay as evidence of the exact scored file.
"""
    ),
]

nbf.write(nb, NOTEBOOK_PATH)
print(f"Wrote {NOTEBOOK_PATH} with {len(nb.cells)} cells")
