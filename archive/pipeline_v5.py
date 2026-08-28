"""
================================================================================
Insight 2.0 Datathon 2026 — Pipeline v5: Clean Rebuild
================================================================================
Target:  vital_status (Dead=1, Alive=0)
Metric:  F1-score (Dead is positive class)
Baseline: All-Dead ≈ 0.907 F1 — model MUST beat this.

Architecture:
  1. Load & Audit Data
  2. Clean Placeholder Missing Values
  3. Feature Engineering (numeric + missingness indicators + interactions)
  4. Modeling: LightGBM + XGBoost + CatBoost with native categoricals
     - Sanity check: CV F1 > 0.907 (all-Dead baseline)
     - scale_pos_weight experiment
  5. Threshold Selection via OOF F1 sweep
  6. Ensembling (probability averaging → single threshold)
  7. Full-Train Final Fit → Submission
  8. Validation & Sanity Checks
================================================================================
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
N_SPLITS = 5
np.random.seed(SEED)

SEP = "=" * 80

# ============================================================================
# STEP 1: LOAD & AUDIT DATA
# ============================================================================
print(SEP)
print("  STEP 1: Load & Audit Data")
print(SEP)

train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")
test_ids = test_df["patient_id"].values

y = (train_df["vital_status"] == "Dead").astype(int).values
DEAD_RATE = y.mean()

print(f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
print(f"  Dead: {y.sum():,} ({DEAD_RATE:.2%}) | Alive: {(1-y).sum():,} ({1-DEAD_RATE:.2%})")

# Leakage check
assert len(np.intersect1d(train_df["patient_id"], test_ids)) == 0, "LEAK!"

# All-Dead baseline
all_dead_f1 = f1_score(y, np.ones_like(y), pos_label=1)
print(f"\n  *** ALL-DEAD BASELINE F1: {all_dead_f1:.6f} ***")
print(f"  Any model MUST beat {all_dead_f1:.4f} or it's broken.\n")

# Missing value audit
print("  Missing rates (>1%):")
for c in train_df.columns:
    if c in ("patient_id", "vital_status"): continue
    tr_m = train_df[c].isna().mean()
    te_m = test_df[c].isna().mean()
    if max(tr_m, te_m) > 0.01:
        print(f"    {c:<45} train={tr_m:.3f} test={te_m:.3f}")

# Numeric vs categorical columns
NUMERIC_COLS = ["year_of_diagnosis", "regional_nodes_examined",
                "regional_nodes_positive", "histologic_type_icdo3"]
CATEGORICAL_COLS = [c for c in train_df.columns
                    if c not in NUMERIC_COLS + ["patient_id", "vital_status"]]
print(f"\n  Numeric columns: {len(NUMERIC_COLS)}")
print(f"  Categorical columns: {len(CATEGORICAL_COLS)}")

# ============================================================================
# STEP 2: CLEAN PLACEHOLDER MISSING VALUES
# ============================================================================
print(f"\n{SEP}")
print("  STEP 2: Clean Placeholder Missing Values")
print(SEP)

# SEER uses "Blank(s)", "Unknown", "999", etc. as missing indicators.
# For tree models, keeping these as categories is often BETTER than NaN.
# We'll convert to NaN only for numeric-adjacent fields where the literal
# string blocks numeric interpretation.

def clean_seer(df):
    """Normalize SEER placeholder values. Modifies df in-place."""
    df = df.copy()

    # rx_summ_scope_reglnsur2003: 66% missing — add indicator
    df["scope_reglnsur_missing"] = df["rx_summ_scope_reglnsur2003"].isna().astype(int)

    # For tree models, "Blank(s)" and "Unknown" are kept as categories
    # because they carry signal (e.g., "Unknown" stage often → worse prognosis).
    # We only convert truly numeric columns that have string placeholders.

    return df

train_clean = clean_seer(train_df)
test_clean  = clean_seer(test_df)
print("  Cleaned. Added missingness indicators.")

# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================
print(f"\n{SEP}")
print("  STEP 3: Feature Engineering")
print(SEP)

def build_features(df):
    """Extract features. No target information used."""
    F = pd.DataFrame(index=df.index)

    # ── Age: convert bins to midpoint ──
    def _age(s):
        if pd.isna(s): return np.nan
        s = str(s).strip()
        if "90+" in s: return 92.5
        parts = s.replace(" years", "").split("-")
        try: return (float(parts[0]) + float(parts[1])) / 2
        except: return np.nan
    F["age_midpoint"] = df["age_recode"].apply(_age)
    F["age_missing"] = df["age_recode"].isna().astype(int)

    # ── Year ──
    F["year_of_diagnosis"] = df["year_of_diagnosis"].astype(float)
    F["years_since_dx"] = 2024.0 - df["year_of_diagnosis"]

    # ── Sex ──
    F["is_male"] = (df["sex"] == "Male").astype(int)

    # ── Nodes ──
    F["nodes_examined"] = df["regional_nodes_examined"].astype(float)
    F["nodes_examined_zero"] = (df["regional_nodes_examined"] == 0).astype(int)
    npos = df["regional_nodes_positive"].astype(float)
    F["nodes_positive"] = np.where(npos < 98, npos, np.nan)
    F["nodes_positive_unknown"] = (npos >= 98).astype(int)
    # node_positive_ratio (handle divide-by-zero)
    F["node_positive_ratio"] = np.where(
        (F["nodes_examined"] > 0) & (~np.isnan(F["nodes_positive"])),
        F["nodes_positive"] / np.maximum(F["nodes_examined"], 1),
        np.nan
    )

    # ── Tumor Size ──
    def _parse_size(v, blanks=("Blank(s)", "999")):
        if pd.isna(v): return np.nan
        s = str(v).strip()
        if s.startswith("Unknown") or s in blanks: return np.nan
        try: return float(s)
        except: return np.nan
    F["tumor_size_ot"]  = df["tumor_size_overtime"].apply(_parse_size)
    F["tumor_size_sm"]  = df["tumor_size_summary"].apply(_parse_size)
    F["tumor_size_best"] = F["tumor_size_sm"].fillna(F["tumor_size_ot"])
    F["tumor_size_missing"] = F["tumor_size_best"].isna().astype(int)

    def _cs_num(v):
        if pd.isna(v): return np.nan
        s = str(v).strip()
        if s in ("Blank(s)",): return np.nan
        try:
            val = float(s)
            return np.nan if val == 999 else val
        except: return np.nan
    F["cs_tumor_size"]  = df["cs_tumor_size20042015"].apply(_cs_num)
    F["cs_extension"]   = df["cs_extension20042015"].apply(_cs_num)

    # ── Summary Stage (ordinal) ──
    stage_map = {"Blank(s)": np.nan, "Unknown/unstaged": np.nan,
                 "Localized": 1, "Regional": 2, "Distant": 3}
    F["stage_ord"] = df["summary_stage"].map(stage_map).astype(float)
    F["stage_missing"] = F["stage_ord"].isna().astype(int)
    F["is_distant"]  = (df["summary_stage"] == "Distant").astype(int)
    F["is_localized"] = (df["summary_stage"] == "Localized").astype(int)

    # ── TNM (ordinal encoding) ──
    t_map = {"Blank(s)": np.nan, "88": np.nan, "TX": np.nan,
             "T0": 0, "Ta": 1, "Tis": 1, "Tis(DCIS)": 1, "Tis(LCIS)": 1, "Tis(Paget)": 1,
             "T1": 2, "T1mi": 2, "T1a": 3, "T1a1": 3, "T1a2": 3,
             "T1b": 4, "T1b1": 4, "T1b2": 4, "T1c": 5,
             "T2": 6, "T2a": 7, "T2b": 8,
             "T3": 9, "T4": 10, "T4a": 11, "T4b": 12, "T4c": 12, "T4d": 13, "T4e": 13}
    n_map = {"Blank(s)": np.nan, "88": np.nan, "NX": np.nan,
             "N0": 0, "N1": 1, "N1a": 1, "N1b": 2, "N1c": 2,
             "N2": 3, "N2a": 3, "N2b": 4, "N2c": 4,
             "N3": 5, "N3a": 5, "N3b": 6, "N3c": 6}
    m_map = {"Blank(s)": np.nan, "88": np.nan, "MX": np.nan,
             "M0": 0, "M1": 1, "M1a": 2, "M1b": 3, "M1c": 4}
    F["t_stage"] = df["derived_eod2018t_recode2018"].map(t_map).astype(float)
    F["n_stage"] = df["derived_eod2018n_recode2018"].map(n_map).astype(float)
    F["m_stage"] = df["derived_eod2018m_recode2018"].map(m_map).astype(float)
    F["is_m1"] = df["derived_eod2018m_recode2018"].str.startswith("M1", na=False).astype(int)

    # ── Composite AJCC-style stage ──
    # Simple numeric composite: T + N + M*3 (gives M heaviest weight)
    F["tnm_composite"] = F["t_stage"].fillna(0) + F["n_stage"].fillna(0) + F["m_stage"].fillna(0) * 3

    # ── Metastasis sites ──
    met_cols = ["seer_combined_metsatdxbone2010", "seer_combined_metsatdxbrain2010",
                "seer_combined_metsatdxliver2010", "seer_combined_metsatdxlung2010"]
    for mc in met_cols:
        short = mc.replace("seer_combined_metsatdx", "").replace("2010", "")
        F[f"met_{short}_yes"] = (df[mc] == "Yes").astype(int)
    F["met_count"] = sum(F[f"met_{mc.replace('seer_combined_metsatdx','').replace('2010','')}_yes"] for mc in met_cols)
    F["any_mets"] = (F["met_count"] >= 1).astype(int)

    # ── Surgery / Treatment ──
    F["surgery_done"] = (df["reason_nocancer_directed_surgery"] == "Surgery performed").astype(int)
    F["surgery_not_rec"] = df["reason_nocancer_directed_surgery"].str.contains("Not recommended", na=False).astype(int)
    F["died_before_surg"] = df["reason_nocancer_directed_surgery"].str.contains("died prior", na=False).astype(int)
    F["autopsy_or_dc"] = df["reason_nocancer_directed_surgery"].str.contains("death certificate|autopsy", na=False).astype(int)

    F["radiation_given"] = (~df["radiation_recode"].isin([
        "None/Unknown",
        "No radiation and/or no surgery; unknown if surgery and/or radiation given",
        "Refused (1988+)", "Recommended, unknown if administered"
    ]) & df["radiation_recode"].notna()).astype(int)

    # ── Grade ──
    grade_map = {"Blank(s)": np.nan, "Unknown": np.nan,
                 "Well differentiated; Grade I": 1,
                 "Moderately differentiated; Grade II": 2,
                 "Poorly differentiated; Grade III": 3,
                 "Undifferentiated; anaplastic; Grade IV": 4}
    F["grade_ord"] = df["grade_recode_thru2017"].map(grade_map).astype(float)
    F["grade_missing"] = F["grade_ord"].isna().astype(int)

    # ── Histology groups ──
    def _histo_group(c):
        try: c = int(c)
        except: return 0
        if c in (8140,8141,8143,8144,8145,8147,8148,8211,8230,8250,8251,8252,
                 8253,8254,8255,8256,8257,8260,8263,8310,8323,8333,8480,8481,
                 8490,8550,8551,8570,8574): return 1  # Adenocarcinoma
        elif c in (8070,8071,8072,8073,8074,8075,8076,8078,8083,8084): return 2  # Squamous
        elif c in (8041,8042,8043,8044,8045): return 3  # Small cell
        elif c in (8012,8013,8014): return 4  # Large cell
        elif c in (8000,8001,8002,8003,8004,8005,8010,8011,8020,8021,8022): return 5  # NOS
        elif c in (8240,8241,8242,8243,8244,8245,8246,8249): return 6  # Carcinoid
        else: return 7  # Other
    F["histo_group"] = df["histologic_type_icdo3"].apply(_histo_group)
    F["is_small_cell"] = (F["histo_group"] == 3).astype(int)
    F["histologic_raw"] = df["histologic_type_icdo3"].astype(float)

    # ── Missingness indicators ──
    F["scope_reglnsur_missing"] = df["rx_summ_scope_reglnsur2003"].isna().astype(int)
    F["eod_m_blank"] = (df["derived_eod2018m_recode2018"] == "Blank(s)").astype(int)
    F["grade_blank"] = (df["grade_recode_thru2017"] == "Blank(s)").astype(int)

    # ── Interactions ──
    F["age_x_years"] = F["age_midpoint"] * F["years_since_dx"]
    F["age_x_stage"] = F["age_midpoint"] * F["stage_ord"]
    F["stage_x_surgery"] = F["stage_ord"] * F["surgery_done"]
    F["mets_x_surgery"] = F["any_mets"] * F["surgery_done"]
    F["age_x_mets"] = F["age_midpoint"] * F["any_mets"]
    F["years_x_stage"] = F["years_since_dx"] * F["stage_ord"]

    return F

X_train_num = build_features(train_clean)
X_test_num  = build_features(test_clean)

# ── Categorical columns: ordinal encode for LGB/XGB, keep raw for CatBoost ──
cat_for_encoding = [
    "age_recode", "race", "sex", "origin", "primary_site",
    "marital_status_at_diagnosis", "sequence_number", "site_recode_icdo3_who2008",
    "grade_recode_thru2017", "laterality", "diagnostic_confirmation",
    "summary_stage", "derived_eod2018t_recode2018", "derived_eod2018n_recode2018",
    "derived_eod2018m_recode2018",
    "seer_combined_metsatdxbone2010", "seer_combined_metsatdxbrain2010",
    "seer_combined_metsatdxliver2010", "seer_combined_metsatdxlung2010",
    "rx_summ_surgprim_site19982022", "rx_summ_surgprim_site20232023",
    "rx_summ_scope_reglnsur2003", "rx_summ_surgothregdis2003",
    "rx_summ_surgradseq", "reason_nocancer_directed_surgery", "radiation_recode",
    "tumor_size_overtime", "tumor_size_summary",
    "cs_tumor_size20042015", "cs_extension20042015",
]

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
enc.fit(train_clean[cat_for_encoding].fillna("__NA__").astype(str))

tr_cat_enc = pd.DataFrame(
    enc.transform(train_clean[cat_for_encoding].fillna("__NA__").astype(str)),
    columns=[f"{c}_enc" for c in cat_for_encoding], index=train_df.index
)
te_cat_enc = pd.DataFrame(
    enc.transform(test_clean[cat_for_encoding].fillna("__NA__").astype(str)),
    columns=[f"{c}_enc" for c in cat_for_encoding], index=test_df.index
)

X_train = pd.concat([X_train_num, tr_cat_enc], axis=1)
X_test  = pd.concat([X_test_num, te_cat_enc], axis=1)

# Fill NaN for tree models
X_train_np = np.nan_to_num(X_train.values.astype(np.float32), nan=-999.0)
X_test_np  = np.nan_to_num(X_test.values.astype(np.float32), nan=-999.0)

feat_names = list(X_train.columns)
print(f"  Feature matrix: {X_train.shape[1]} features")
print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

# ============================================================================
# STEP 4: MODELING — SANITY CHECK + BASELINE + OPTUNA TUNING
# ============================================================================
print(f"\n{SEP}")
print("  STEP 4: Modeling — Sanity Check & Hyperparameter Tuning")
print(SEP)

# ── 4a: Sanity check — plain LightGBM, 5-fold CV ──
print("\n  [4a] Sanity Check: Plain LightGBM, default params, 5-fold CV")
skf_sanity = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
sanity_f1s = []
for fold, (tr_i, val_i) in enumerate(skf_sanity.split(X_train_np, y)):
    m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=SEED,
                            verbose=-1, n_jobs=-1)
    m.fit(X_train_np[tr_i], y[tr_i],
          eval_set=[(X_train_np[val_i], y[val_i])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    pred = m.predict(X_train_np[val_i])
    f = f1_score(y[val_i], pred, pos_label=1)
    sanity_f1s.append(f)
    print(f"    Fold {fold+1}: F1={f:.5f}")

sanity_mean = np.mean(sanity_f1s)
print(f"  → Mean CV F1: {sanity_mean:.5f}")
if sanity_mean < all_dead_f1:
    print(f"  *** WARNING: CV F1 ({sanity_mean:.4f}) < All-Dead ({all_dead_f1:.4f})! ***")
    print(f"  *** Model is broken — would be better to predict all Dead. ***")
else:
    print(f"  ✓ CV F1 ({sanity_mean:.4f}) > All-Dead ({all_dead_f1:.4f}). Pipeline is correct.")

# ── 4b: scale_pos_weight experiment ──
print("\n  [4b] scale_pos_weight experiment")
alive_count = (y == 0).sum()
dead_count  = (y == 1).sum()
# For Dead=positive with 83% prevalence, scale_pos_weight < 1 would
# downweight the majority class. But with F1 on the majority class,
# we actually don't want to penalize Dead predictions.
# Try weight = alive_count / dead_count (standard) and 1.0 (no weighting)
for spw_label, spw_val in [("none (1.0)", 1.0), ("alive/dead", alive_count/dead_count)]:
    spw_f1s = []
    for tr_i, val_i in skf_sanity.split(X_train_np, y):
        m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                scale_pos_weight=spw_val,
                                random_state=SEED, verbose=-1, n_jobs=-1)
        m.fit(X_train_np[tr_i], y[tr_i],
              eval_set=[(X_train_np[val_i], y[val_i])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        pred = m.predict(X_train_np[val_i])
        spw_f1s.append(f1_score(y[val_i], pred, pos_label=1))
    print(f"    scale_pos_weight={spw_label}: Mean F1={np.mean(spw_f1s):.5f}")

# ── 4c: Optuna tuning ──
OPTUNA_TRIALS = 20
print(f"\n  [4c] Optuna Tuning ({OPTUNA_TRIALS} trials/model, 3-fold objective)")

def optuna_lgb(trial):
    p = {
        "n_estimators": 1200,
        "learning_rate": trial.suggest_float("lr", 0.01, 0.08, log=True),
        "max_depth": trial.suggest_int("md", 4, 8),
        "num_leaves": trial.suggest_int("nl", 20, 80),
        "subsample": trial.suggest_float("ss", 0.65, 0.95),
        "colsample_bytree": trial.suggest_float("cs", 0.45, 0.90),
        "min_child_samples": trial.suggest_int("mcs", 10, 60),
        "reg_alpha": trial.suggest_float("ra", 0.001, 3.0, log=True),
        "reg_lambda": trial.suggest_float("rl", 0.01, 5.0, log=True),
        "random_state": SEED, "n_jobs": -1, "verbose": -1,
    }
    scores = []
    for tr, val in StratifiedKFold(3, shuffle=True, random_state=SEED).split(X_train_np, y):
        m = lgb.LGBMClassifier(**p)
        m.fit(X_train_np[tr], y[tr],
              eval_set=[(X_train_np[val], y[val])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        prob = m.predict_proba(X_train_np[val])[:, 1]
        scores.append(f1_score(y[val], (prob >= 0.5).astype(int), pos_label=1))
    return np.mean(scores)

def optuna_xgb(trial):
    p = {
        "n_estimators": 1200,
        "learning_rate": trial.suggest_float("lr", 0.01, 0.08, log=True),
        "max_depth": trial.suggest_int("md", 4, 8),
        "subsample": trial.suggest_float("ss", 0.65, 0.95),
        "colsample_bytree": trial.suggest_float("cs", 0.45, 0.90),
        "min_child_weight": trial.suggest_int("mcw", 3, 25),
        "reg_alpha": trial.suggest_float("ra", 0.001, 3.0, log=True),
        "reg_lambda": trial.suggest_float("rl", 0.01, 5.0, log=True),
        "gamma": trial.suggest_float("g", 0.0, 2.0),
        "eval_metric": "logloss", "tree_method": "hist",
        "random_state": SEED, "n_jobs": -1,
    }
    scores = []
    for tr, val in StratifiedKFold(3, shuffle=True, random_state=SEED).split(X_train_np, y):
        m = xgb.XGBClassifier(**p, early_stopping_rounds=50)
        m.fit(X_train_np[tr], y[tr], eval_set=[(X_train_np[val], y[val])], verbose=False)
        prob = m.predict_proba(X_train_np[val])[:, 1]
        scores.append(f1_score(y[val], (prob >= 0.5).astype(int), pos_label=1))
    return np.mean(scores)

def optuna_cb(trial):
    p = {
        "iterations": 1200,
        "learning_rate": trial.suggest_float("lr", 0.01, 0.08, log=True),
        "depth": trial.suggest_int("d", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2", 0.3, 8.0, log=True),
        "bagging_temperature": trial.suggest_float("bt", 0.0, 3.0),
        "random_strength": trial.suggest_float("rs", 0.0, 3.0),
        "eval_metric": "Logloss", "random_seed": SEED, "verbose": False,
    }
    scores = []
    for tr, val in StratifiedKFold(3, shuffle=True, random_state=SEED).split(X_train_np, y):
        m = CatBoostClassifier(**p, early_stopping_rounds=50)
        m.fit(X_train_np[tr], y[tr], eval_set=(X_train_np[val], y[val]), verbose=False)
        prob = m.predict_proba(X_train_np[val])[:, 1]
        scores.append(f1_score(y[val], (prob >= 0.5).astype(int), pos_label=1))
    return np.mean(scores)

print("  Tuning LightGBM...")
study_lgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study_lgb.optimize(optuna_lgb, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
print(f"  → LGB best F1: {study_lgb.best_value:.5f}")

print("  Tuning XGBoost...")
study_xgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study_xgb.optimize(optuna_xgb, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
print(f"  → XGB best F1: {study_xgb.best_value:.5f}")

print("  Tuning CatBoost...")
study_cb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study_cb.optimize(optuna_cb, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
print(f"  → CB  best F1: {study_cb.best_value:.5f}")

# Build final params
bp = study_lgb.best_params
lgb_params = {
    "n_estimators": 1500, "learning_rate": bp["lr"], "max_depth": bp["md"],
    "num_leaves": bp["nl"], "subsample": bp["ss"], "colsample_bytree": bp["cs"],
    "min_child_samples": bp["mcs"], "reg_alpha": bp["ra"], "reg_lambda": bp["rl"],
    "random_state": SEED, "n_jobs": -1, "verbose": -1,
}
bp = study_xgb.best_params
xgb_params = {
    "n_estimators": 1500, "learning_rate": bp["lr"], "max_depth": bp["md"],
    "subsample": bp["ss"], "colsample_bytree": bp["cs"],
    "min_child_weight": bp["mcw"], "reg_alpha": bp["ra"], "reg_lambda": bp["rl"],
    "gamma": bp["g"], "eval_metric": "logloss", "tree_method": "hist",
    "random_state": SEED, "n_jobs": -1,
}
bp = study_cb.best_params
cb_params = {
    "iterations": 1500, "learning_rate": bp["lr"], "depth": bp["d"],
    "l2_leaf_reg": bp["l2"], "bagging_temperature": bp["bt"],
    "random_strength": bp["rs"], "eval_metric": "Logloss",
    "random_seed": SEED, "verbose": False,
}

print(f"\n  Final LGB: {lgb_params}")
print(f"  Final XGB: {xgb_params}")
print(f"  Final CB:  {cb_params}")

# ============================================================================
# STEP 5 & 6: 5-FOLD CV — OOF PREDICTIONS → THRESHOLD SWEEP → ENSEMBLE
# ============================================================================
print(f"\n{SEP}")
print("  STEP 5 & 6: 5-Fold CV → OOF → Threshold → Ensemble")
print(SEP)

oof_lgb = np.zeros(len(y), dtype=np.float64)
oof_xgb = np.zeros(len(y), dtype=np.float64)
oof_cb  = np.zeros(len(y), dtype=np.float64)

skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)

for fold, (tr_i, val_i) in enumerate(skf.split(X_train_np, y)):
    Xtr, ytr = X_train_np[tr_i], y[tr_i]
    Xval, yval = X_train_np[val_i], y[val_i]

    # LightGBM
    m_lgb = lgb.LGBMClassifier(**lgb_params)
    m_lgb.fit(Xtr, ytr, eval_set=[(Xval, yval)],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
    oof_lgb[val_i] = m_lgb.predict_proba(Xval)[:, 1]

    # XGBoost
    m_xgb = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=80)
    m_xgb.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    oof_xgb[val_i] = m_xgb.predict_proba(Xval)[:, 1]

    # CatBoost
    m_cb = CatBoostClassifier(**cb_params, early_stopping_rounds=80)
    m_cb.fit(Xtr, ytr, eval_set=(Xval, yval), verbose=False)
    oof_cb[val_i] = m_cb.predict_proba(Xval)[:, 1]

    # Per-fold stats
    for name, oof in [("LGB", oof_lgb), ("XGB", oof_xgb), ("CB", oof_cb)]:
        f = f1_score(yval, (oof[val_i] >= 0.5).astype(int), pos_label=1)
        if name == "CB":
            print(f"  Fold {fold+1}: LGB={f1_score(yval, (oof_lgb[val_i]>=0.5).astype(int), pos_label=1):.5f}  "
                  f"XGB={f1_score(yval, (oof_xgb[val_i]>=0.5).astype(int), pos_label=1):.5f}  "
                  f"CB={f:.5f}")

# ── Ensemble weight search ──
print("\n  Searching ensemble weights...")
best_w, best_wf1 = (1/3, 1/3, 1/3), 0.0
for w1 in np.arange(0.15, 0.55, 0.05):
    for w2 in np.arange(0.15, 0.55, 0.05):
        w3 = round(1 - w1 - w2, 2)
        if w3 < 0.10 or w3 > 0.55: continue
        ens = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cb
        f = f1_score(y, (ens >= 0.5).astype(int), pos_label=1)
        if f > best_wf1:
            best_wf1 = f
            best_w = (round(w1, 2), round(w2, 2), w3)

W_LGB, W_XGB, W_CB = best_w
oof_ens = W_LGB * oof_lgb + W_XGB * oof_xgb + W_CB * oof_cb
print(f"  Best weights: LGB={W_LGB}, XGB={W_XGB}, CB={W_CB}")
print(f"  OOF Ensemble F1 (@ th=0.5): {best_wf1:.5f}")

# ── Threshold sweep (the critical step) ──
print(f"\n  Threshold Sweep (OOF Ensemble, Dead=1 positive):")
print(f"  {'Thresh':>7} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Dead%':>8}")

sweep = []
for th in np.arange(0.05, 0.96, 0.01):
    pred = (oof_ens >= th).astype(int)
    f1 = f1_score(y, pred, pos_label=1)
    prec = precision_score(y, pred, pos_label=1, zero_division=0)
    rec = recall_score(y, pred, pos_label=1)
    dpct = pred.mean()
    sweep.append((th, f1, prec, rec, dpct))

best_th_entry = max(sweep, key=lambda x: x[1])
BEST_TH = best_th_entry[0]

# Print a focused range around the optimum
for th, f1, prec, rec, dpct in sweep:
    tag = ""
    if abs(th - BEST_TH) < 0.001: tag = " ← F1 OPTIMAL"
    if abs(th - 0.50) < 0.001: tag += " ← DEFAULT"
    # Show range around optimum and key thresholds
    if abs(th - BEST_TH) < 0.10 or abs(th - 0.50) < 0.02 or tag:
        print(f"  {th:7.2f} {f1:8.5f} {prec:8.4f} {rec:8.4f} {dpct:8.3f}{tag}")

print(f"\n  *** CHOSEN THRESHOLD: {BEST_TH:.2f} ***")
print(f"  *** OOF F1 at chosen threshold: {best_th_entry[1]:.5f} ***")
print(f"  *** Dead%: {best_th_entry[4]:.1%}, Precision: {best_th_entry[2]:.4f}, Recall: {best_th_entry[3]:.4f} ***")

# Sanity check vs all-Dead
if best_th_entry[1] < all_dead_f1:
    print(f"\n  *** CRITICAL: OOF F1 ({best_th_entry[1]:.4f}) < All-Dead ({all_dead_f1:.4f})! ***")
else:
    print(f"\n  ✓ OOF F1 ({best_th_entry[1]:.4f}) > All-Dead ({all_dead_f1:.4f}). Model adds value.")

# Show classification report at chosen threshold
oof_labels = (oof_ens >= BEST_TH).astype(int)
print(f"\n  OOF Classification Report (threshold={BEST_TH:.2f}):")
print(classification_report(y, oof_labels, target_names=["Alive", "Dead"], digits=4))

# Per-fold F1 variance at chosen threshold
fold_f1s = []
for fold, (_, val_i) in enumerate(skf.split(X_train_np, y)):
    f = f1_score(y[val_i], (oof_ens[val_i] >= BEST_TH).astype(int), pos_label=1)
    fold_f1s.append(f)
    print(f"  Fold {fold+1} F1 (@ th={BEST_TH:.2f}): {f:.5f}")
print(f"  → Mean: {np.mean(fold_f1s):.5f} ± {np.std(fold_f1s):.5f}")

# ============================================================================
# STEP 7: FULL-TRAIN FINAL FIT → SUBMISSION
# ============================================================================
print(f"\n{SEP}")
print("  STEP 7: Full-Train Final Fit → Submission")
print(SEP)

# Fit each model on ALL training data, predict test
print("  Training LightGBM on full train...")
final_lgb = lgb.LGBMClassifier(**lgb_params)
final_lgb.fit(X_train_np, y)

print("  Training XGBoost on full train...")
final_xgb = xgb.XGBClassifier(**xgb_params)
final_xgb.fit(X_train_np, y, verbose=False)

print("  Training CatBoost on full train...")
final_cb = CatBoostClassifier(**cb_params)
final_cb.fit(X_train_np, y, verbose=False)

# Predict test
test_prob_lgb = final_lgb.predict_proba(X_test_np)[:, 1]
test_prob_xgb = final_xgb.predict_proba(X_test_np)[:, 1]
test_prob_cb  = final_cb.predict_proba(X_test_np)[:, 1]

test_ens = W_LGB * test_prob_lgb + W_XGB * test_prob_xgb + W_CB * test_prob_cb

print(f"\n  Test ensemble prob stats: mean={test_ens.mean():.4f}, median={np.median(test_ens):.4f}")

# Apply threshold
test_preds = (test_ens >= BEST_TH).astype(int)
test_status = np.where(test_preds == 1, "Dead", "Alive")
dead_n = test_preds.sum()
print(f"  Threshold: {BEST_TH:.2f}")
print(f"  Test predictions: Dead={dead_n:,} ({dead_n/len(test_preds):.1%}), Alive={len(test_preds)-dead_n:,}")

# Write submission
sub = pd.DataFrame({"patient_id": test_ids, "vital_status": test_status})
sub.to_csv("submission.csv", index=False)
print(f"  → Written: submission.csv")

# Also generate an all-Dead submission for LB baseline comparison
sub_all_dead = pd.DataFrame({"patient_id": test_ids, "vital_status": "Dead"})
sub_all_dead.to_csv("submission_all_dead.csv", index=False)
print(f"  → Written: submission_all_dead.csv (baseline)")

# ============================================================================
# STEP 8: VALIDATION & SANITY CHECKS
# ============================================================================
print(f"\n{SEP}")
print("  STEP 8: Validation & Sanity Checks")
print(SEP)

# Re-read and verify
sub_check = pd.read_csv("submission.csv")
print(f"  Rows: {len(sub_check)} (expected 36,000)")
print(f"  Columns: {list(sub_check.columns)}")
print(f"  Unique patient_ids: {sub_check['patient_id'].nunique()}")
print(f"  Labels: {sub_check['vital_status'].unique()}")
print(f"  No missing: {sub_check.isna().sum().sum() == 0}")
print(f"  IDs match test.csv: {(sub_check['patient_id'].values == test_ids).all()}")
print(f"  No duplicates: {sub_check.duplicated('patient_id').sum() == 0}")

test_dead_pct = (sub_check["vital_status"] == "Dead").mean()
print(f"\n  Test Dead%: {test_dead_pct:.1%} (train: {DEAD_RATE:.1%})")
if abs(test_dead_pct - DEAD_RATE) < 0.10:
    print(f"  ✓ Test class balance is within 10% of train — plausible.")
else:
    print(f"  ⚠ Test class balance differs from train by {abs(test_dead_pct-DEAD_RATE):.1%}")

# ── Final Summary ──
print(f"\n{SEP}")
print("  FINAL SUMMARY")
print(SEP)
print(f"  Models: LightGBM + XGBoost + CatBoost")
print(f"  Ensemble Weights: LGB={W_LGB}, XGB={W_XGB}, CB={W_CB}")
print(f"  Chosen Threshold: {BEST_TH:.2f}")
print(f"  OOF F1 (5-fold CV): {best_th_entry[1]:.5f}")
print(f"  All-Dead Baseline F1: {all_dead_f1:.5f}")
print(f"  Test Dead%: {test_dead_pct:.1%}")
print(f"  Submission: submission.csv ({len(sub_check):,} rows)")
print(SEP)
