"""
================================================================================
Insight 2.0 Datathon 2026 — Pipeline v4: Maximum Generalization
================================================================================
Target: vital_status (Dead = 1, Alive = 0)
Evaluation Metric: F1-score (Dead as positive class)

Architecture:
1. Data Quality & Leakage Audit
2. Train/Test Distribution Shift Diagnostics (Adversarial Validation + Chi-Square)
3. SEER-Aware Feature Engineering with Explicit Missingness Indicators
4. Bayesian m-Estimate Smoothed Target Encoding (Leak-Free Inner CV)
5. Moderate Optuna Hyperparameter Tuning (15 trials/model, 3-fold objective)
6. 3×5 Repeated Stratified CV with Nested Threshold Selection
7. Multi-Threshold Submission Generation for LB Experimentation
8. Subgroup & Edge-Case Diagnostics
================================================================================
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             classification_report, roc_auc_score)
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Configuration ──
SEED = 42
N_SPLITS = 5
N_REPEATS = 3          # 3 seeds × 5 folds = 15 evaluations
SMOOTHING_M = 20.0     # Bayesian m-estimate smoothing
OPTUNA_TRIALS = 15     # Moderate tuning per model
OPTUNA_CV_FOLDS = 3    # Fast 3-fold for tuning objective

np.random.seed(SEED)

print("=" * 80)
print("  INSIGHT 2.0 — PIPELINE v4: MAXIMUM GENERALIZATION")
print("=" * 80)

# ============================================================================
# SECTION 1: DATA LOADING & LEAKAGE AUDIT
# ============================================================================
print("\n>>> [1/9] Data Loading & Leakage Audit")

train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

train_ids = train_df["patient_id"].values
test_ids  = test_df["patient_id"].values

# Leakage checks
assert len(np.intersect1d(train_ids, test_ids)) == 0, "LEAK: Overlapping patient_ids!"
assert len(test_ids) == 36000

y_train = (train_df["vital_status"] == "Dead").astype(int).values
GLOBAL_DEAD_RATE = float(y_train.mean())

print(f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
print(f"  Dead={y_train.sum():,} ({GLOBAL_DEAD_RATE:.2%}) | Alive={len(y_train)-y_train.sum():,} ({1-GLOBAL_DEAD_RATE:.2%})")

# Duplicate check
train_feats = train_df.drop(columns=["patient_id", "vital_status"])
test_feats  = test_df.drop(columns=["patient_id"])
print(f"  Train duplicate rows: {train_feats.duplicated().sum()}")
print(f"  Test  duplicate rows: {test_feats.duplicated().sum()}")

# Death rate by year (key survival data characteristic)
print("\n  Death Rate by Year of Diagnosis:")
for yr in sorted(train_df["year_of_diagnosis"].unique()):
    mask = train_df["year_of_diagnosis"] == yr
    dr = y_train[mask].mean()
    print(f"    {yr}: Dead={dr:.1%} (n={mask.sum():,})")

# ============================================================================
# SECTION 2: DISTRIBUTION SHIFT DIAGNOSTICS
# ============================================================================
print("\n>>> [2/9] Distribution Shift Diagnostics")

# Adversarial Validation
adv_X = pd.concat([train_feats, test_feats], ignore_index=True)
adv_y = np.array([0]*len(train_df) + [1]*len(test_df))
for col in adv_X.columns:
    adv_X[col] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1
                                ).fit_transform(adv_X[[col]].astype(str)).astype(np.float32)

adv_clf = RandomForestClassifier(n_estimators=60, max_depth=6, random_state=SEED, n_jobs=-1)
adv_aucs = []
for tr_i, val_i in StratifiedKFold(3, shuffle=True, random_state=SEED).split(adv_X, adv_y):
    adv_clf.fit(adv_X.iloc[tr_i], adv_y[tr_i])
    adv_aucs.append(roc_auc_score(adv_y[val_i], adv_clf.predict_proba(adv_X.iloc[val_i])[:,1]))

mean_adv = np.mean(adv_aucs)
print(f"  Adversarial Validation AUC: {mean_adv:.4f}")
if mean_adv < 0.53:
    print("  → Train/Test are i.i.d. Stratified K-Fold is optimal.")
else:
    print("  → WARNING: Detectable covariate shift.")

# Chi-square tests
print("\n  Chi-Square Homogeneity Tests:")
for feat in ["age_recode", "summary_stage", "primary_site", "derived_eod2018m_recode2018",
             "reason_nocancer_directed_surgery", "grade_recode_thru2017", "sex", "race",
             "seer_combined_metsatdxbone2010", "radiation_recode"]:
    tr_c = train_df[feat].fillna("NA").value_counts()
    te_c = test_df[feat].fillna("NA").value_counts()
    cats = list(set(tr_c.index) | set(te_c.index))
    cont = np.vstack([[tr_c.get(c,0) for c in cats], [te_c.get(c,0) for c in cats]])
    _, pval, _, _ = chi2_contingency(cont)
    flag = "" if pval > 0.01 else " ⚠"
    print(f"    {feat:<40}: p={pval:.4f}{flag}")

# Year distribution comparison
print("\n  Year Distribution Comparison:")
for yr in sorted(train_df["year_of_diagnosis"].unique()):
    tr_pct = (train_df["year_of_diagnosis"]==yr).mean()*100
    te_pct = (test_df["year_of_diagnosis"]==yr).mean()*100
    print(f"    {yr}: Train={tr_pct:.1f}% | Test={te_pct:.1f}% | Δ={tr_pct-te_pct:+.1f}%")

# ============================================================================
# SECTION 3: FEATURE ENGINEERING
# ============================================================================
print("\n>>> [3/9] Feature Engineering")

def build_features(df):
    """Target-agnostic feature extraction. No label information used."""
    F = pd.DataFrame(index=df.index)

    # ── Temporal ──
    F["year_of_diagnosis"] = df["year_of_diagnosis"].astype(float)
    F["years_since_dx"] = 2024.0 - df["year_of_diagnosis"]

    # ── Age ──
    def _age(s):
        if pd.isna(s): return np.nan
        s = str(s).strip()
        if "90+" in s: return 92.5
        parts = s.replace(" years","").split("-")
        try: return (float(parts[0])+float(parts[1]))/2
        except: return np.nan
    F["age"] = df["age_recode"].apply(_age)
    F["age_missing"] = df["age_recode"].isna().astype(float)
    F["is_elderly"] = (F["age"] >= 75).astype(float)

    # ── Sex ──
    F["is_male"] = (df["sex"]=="Male").astype(float)

    # ── Missingness indicators (SEER-aware) ──
    F["stage_blank"]   = (df["summary_stage"]=="Blank(s)").astype(float)
    F["stage_unknown"] = df["summary_stage"].isin(["Unknown/unstaged","Unknown"]).astype(float)
    F["eod_m_blank"]   = (df["derived_eod2018m_recode2018"]=="Blank(s)").astype(float)
    F["grade_blank"]   = (df["grade_recode_thru2017"]=="Blank(s)").astype(float)
    F["nodes_scope_na"]= df["rx_summ_scope_reglnsur2003"].isna().astype(float)

    # ── Tumor Size ──
    def _size_ot(v):
        if pd.isna(v): return np.nan
        s = str(v).strip()
        if s.startswith("Unknown") or s=="Blank(s)": return -1
        try: return float(s)
        except: return -1

    def _size_sm(v):
        if pd.isna(v): return np.nan
        s = str(v).strip()
        if s in ("Blank(s)","999"): return -1
        try: return float(s)
        except: return -1

    F["tsize_ot"] = df["tumor_size_overtime"].apply(_size_ot)
    F["tsize_sm"] = df["tumor_size_summary"].apply(_size_sm)
    F["tsize_missing"] = ((F["tsize_ot"]<0)&(F["tsize_sm"]<0)).astype(float)
    best = np.where(F["tsize_sm"]>0, F["tsize_sm"], F["tsize_ot"])
    F["tsize_clean"] = np.where(best>0, best, np.nan)

    # CS tumor size and extension (numeric)
    def _cs_num(v):
        if pd.isna(v): return np.nan
        s = str(v).strip()
        if s=="Blank(s)": return -1
        try:
            val = float(s)
            return -1 if val==999 else val
        except: return -1
    F["cs_tsize"] = df["cs_tumor_size20042015"].apply(_cs_num)
    F["cs_ext"]   = df["cs_extension20042015"].apply(_cs_num)

    # ── Nodes ──
    F["nodes_exam"] = df["regional_nodes_examined"].astype(float)
    F["nodes_exam_none"] = (df["regional_nodes_examined"]==0).astype(float)
    npos = df["regional_nodes_positive"].astype(float)
    F["nodes_pos_unk"] = (npos>=98).astype(float)
    F["nodes_pos"] = np.where(npos<98, npos, -1)
    F["nodes_ratio"] = np.where(
        (F["nodes_exam"]>0)&(F["nodes_pos"]>=0),
        F["nodes_pos"]/np.maximum(F["nodes_exam"],1), -1)

    # ── Metastasis ──
    mets = ["seer_combined_metsatdxbone2010","seer_combined_metsatdxbrain2010",
            "seer_combined_metsatdxliver2010","seer_combined_metsatdxlung2010"]
    for m in mets:
        F[f"{m}_yes"] = (df[m]=="Yes").astype(float)
        F[f"{m}_unk"] = (df[m].isin(["Unknown","Blank(s)"])|df[m].isna()).astype(float)
    F["mets_count"] = sum(F[f"{m}_yes"] for m in mets)
    F["any_mets"]   = (F["mets_count"]>=1).astype(float)
    F["mets_unk_count"] = sum(F[f"{m}_unk"] for m in mets)

    # ── Summary Stage ──
    smap = {"Blank(s)":-1,"Unknown/unstaged":0,"Localized":1,"Regional":2,"Distant":3}
    F["stage_ord"] = df["summary_stage"].map(smap).fillna(0).astype(float)
    F["is_distant"] = (df["summary_stage"]=="Distant").astype(float)
    F["is_local"]   = (df["summary_stage"]=="Localized").astype(float)

    # ── TNM ──
    t_map = {"Blank(s)":-1,"88":0,"TX":0,"Ta":2,"T0":1,"Tis":2,"Tis(DCIS)":2,"Tis(LCIS)":2,"Tis(Paget)":2,
             "T1":3,"T1mi":3,"T1a":4,"T1a1":4,"T1a2":4,"T1b":5,"T1b1":5,"T1b2":5,"T1c":6,
             "T2":7,"T2a":8,"T2b":9,"T3":10,"T4":11,"T4a":12,"T4b":13,"T4c":13,"T4d":14,"T4e":14}
    n_map = {"Blank(s)":-1,"88":0,"NX":0,"N0":1,"N1":2,"N1a":2,"N1b":3,"N1c":3,
             "N2":4,"N2a":4,"N2b":5,"N2c":5,"N3":6,"N3a":6,"N3b":7,"N3c":7}
    m_map = {"Blank(s)":-1,"88":0,"MX":0,"M0":1,"M1":2,"M1a":3,"M1b":4,"M1c":5}
    F["t_ord"] = df["derived_eod2018t_recode2018"].map(t_map).fillna(0).astype(float)
    F["n_ord"] = df["derived_eod2018n_recode2018"].map(n_map).fillna(0).astype(float)
    F["m_ord"] = df["derived_eod2018m_recode2018"].map(m_map).fillna(0).astype(float)
    F["is_m1"] = df["derived_eod2018m_recode2018"].str.startswith("M1",na=False).astype(float)

    # ── Surgery & Treatment ──
    F["surg_done"] = (df["reason_nocancer_directed_surgery"]=="Surgery performed").astype(float)
    F["surg_not_rec"] = df["reason_nocancer_directed_surgery"].str.contains("Not recommended",na=False).astype(float)
    F["died_pre_surg"] = df["reason_nocancer_directed_surgery"].str.contains("died prior",na=False).astype(float)
    F["autopsy_dc"] = df["reason_nocancer_directed_surgery"].str.contains("death certificate|autopsy",na=False).astype(float)

    F["rad_given"] = (~df["radiation_recode"].isin([
        "None/Unknown","No radiation and/or no surgery; unknown if surgery and/or radiation given",
        "Refused (1988+)","Recommended, unknown if administered"
    ]) & df["radiation_recode"].notna()).astype(float)

    # ── Grade ──
    grade_map = {"Blank(s)":-1,"Unknown":0,
                 "Well differentiated; Grade I":1,"Moderately differentiated; Grade II":2,
                 "Poorly differentiated; Grade III":3,"Undifferentiated; anaplastic; Grade IV":4}
    F["grade_ord"] = df["grade_recode_thru2017"].map(grade_map).fillna(0).astype(float)
    F["grade_missing"] = (F["grade_ord"]<=0).astype(float)

    # ── Histology ──
    def _histo(c):
        try: c=int(c)
        except: return 0
        if c in (8140,8141,8143,8144,8145,8147,8148,8211,8230,8250,8251,8252,8253,8254,8255,8256,8257,8260,8263,8310,8323,8333,8480,8481,8490,8550,8551,8570,8574): return 1
        elif c in (8070,8071,8072,8073,8074,8075,8076,8078,8083,8084): return 2
        elif c in (8041,8042,8043,8044,8045): return 3
        elif c in (8012,8013,8014): return 4
        elif c in (8000,8001,8002,8003,8004,8005,8010,8011,8020,8021,8022): return 5
        elif c in (8240,8241,8242,8243,8244,8245,8246,8249): return 6
        else: return 7
    F["histo_grp"] = df["histologic_type_icdo3"].apply(_histo).astype(float)
    F["is_small_cell"] = (F["histo_grp"]==3).astype(float)
    F["histologic_type_icdo3_raw"] = df["histologic_type_icdo3"].astype(float)

    # ── Interactions ──
    F["age_x_yrs"] = F["age"] * F["years_since_dx"]
    F["age_x_stage"] = F["age"] * np.maximum(F["stage_ord"],0)
    F["stage_x_surg"] = F["stage_ord"] * F["surg_done"]
    F["mets_x_surg"] = F["any_mets"] * F["surg_done"]
    F["age_x_mets"] = F["age"] * F["any_mets"]
    F["yrs_x_stage"] = F["years_since_dx"] * np.maximum(F["stage_ord"],0)

    # ── Frequency encoding (target-free) ──
    for col_name, col_data in [("primary_site", df["primary_site"]),
                                ("histologic_type_icdo3", df["histologic_type_icdo3"])]:
        freq = col_data.astype(str).map(col_data.astype(str).value_counts(normalize=True))
        F[f"{col_name}_freq"] = freq.astype(float)

    return F

X_tr_eng = build_features(train_df)
X_te_eng = build_features(test_df)

# ── Ordinal encode raw categoricals ──
raw_cat_cols = [
    "age_recode","race","sex","origin","primary_site",
    "marital_status_at_diagnosis","sequence_number","site_recode_icdo3_who2008",
    "grade_recode_thru2017","laterality","diagnostic_confirmation",
    "summary_stage","derived_eod2018t_recode2018","derived_eod2018n_recode2018",
    "derived_eod2018m_recode2018",
    "seer_combined_metsatdxbone2010","seer_combined_metsatdxbrain2010",
    "seer_combined_metsatdxliver2010","seer_combined_metsatdxlung2010",
    "rx_summ_surgprim_site19982022","rx_summ_surgprim_site20232023",
    "rx_summ_scope_reglnsur2003","rx_summ_surgothregdis2003",
    "rx_summ_surgradseq","reason_nocancer_directed_surgery","radiation_recode",
    "tumor_size_overtime","tumor_size_summary","cs_tumor_size20042015","cs_extension20042015",
]

enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
enc.fit(train_df[raw_cat_cols].fillna("__NA__").astype(str))

tr_cat = pd.DataFrame(enc.transform(train_df[raw_cat_cols].fillna("__NA__").astype(str)),
                       columns=[f"{c}_enc" for c in raw_cat_cols], index=train_df.index)
te_cat = pd.DataFrame(enc.transform(test_df[raw_cat_cols].fillna("__NA__").astype(str)),
                       columns=[f"{c}_enc" for c in raw_cat_cols], index=test_df.index)

X_train_base = pd.concat([X_tr_eng, tr_cat], axis=1)
X_test_base  = pd.concat([X_te_eng, te_cat], axis=1)
print(f"  Feature matrix: Train={X_train_base.shape}, Test={X_test_base.shape}")

# ── Target Encoding Columns ──
te_cols = ["histologic_type_icdo3","primary_site","diagnostic_confirmation",
           "reason_nocancer_directed_surgery","rx_summ_surgprim_site19982022"]

# ============================================================================
# SECTION 4: TARGET ENCODING HELPER
# ============================================================================
def smoothed_te(tr_series, target, te_series, m=SMOOTHING_M):
    """Bayesian m-estimate: μ_c = (n*mean + m*prior) / (n+m)"""
    prior = target.mean()
    df = pd.DataFrame({"c": tr_series.astype(str), "y": target})
    stats = df.groupby("c")["y"].agg(["count","mean"])
    encoded = (stats["count"]*stats["mean"] + m*prior) / (stats["count"]+m)
    return encoded, te_series.astype(str).map(encoded).fillna(prior).values, prior

# Full-train TE for final test predictions
te_test = pd.DataFrame(index=test_df.index)
for col in te_cols:
    _, vals, _ = smoothed_te(train_df[col], y_train, test_df[col])
    te_test[f"{col}_te"] = vals

X_test_final = pd.concat([X_test_base, te_test], axis=1).values.astype(np.float32)
X_test_final = np.nan_to_num(X_test_final, nan=-999.0)

# ============================================================================
# SECTION 5: OPTUNA HYPERPARAMETER TUNING (Moderate: 15 trials per model)
# ============================================================================
print("\n>>> [4/9] Optuna Hyperparameter Tuning (15 trials/model, 3-fold CV)")

# Prepare a quick feature matrix for tuning (no TE to avoid complexity)
X_tune = X_train_base.values.astype(np.float32)
X_tune = np.nan_to_num(X_tune, nan=-999.0)

def optuna_objective_lgb(trial):
    p = {
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("lr", 0.02, 0.08, log=True),
        "max_depth": trial.suggest_int("md", 5, 8),
        "num_leaves": trial.suggest_int("nl", 31, 80),
        "subsample": trial.suggest_float("ss", 0.7, 0.95),
        "colsample_bytree": trial.suggest_float("cs", 0.5, 0.9),
        "min_child_samples": trial.suggest_int("mcs", 15, 50),
        "reg_alpha": trial.suggest_float("ra", 0.01, 2.0, log=True),
        "reg_lambda": trial.suggest_float("rl", 0.1, 3.0, log=True),
        "random_state": SEED, "n_jobs": -1, "verbose": -1,
    }
    scores = []
    for tr, val in StratifiedKFold(OPTUNA_CV_FOLDS, shuffle=True, random_state=SEED).split(X_tune, y_train):
        m = lgb.LGBMClassifier(**p)
        m.fit(X_tune[tr], y_train[tr], eval_set=[(X_tune[val], y_train[val])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        pred = m.predict_proba(X_tune[val])[:,1]
        scores.append(f1_score(y_train[val], (pred>=0.5).astype(int), pos_label=1))
    return np.mean(scores)

def optuna_objective_xgb(trial):
    p = {
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("lr", 0.02, 0.08, log=True),
        "max_depth": trial.suggest_int("md", 5, 8),
        "subsample": trial.suggest_float("ss", 0.7, 0.95),
        "colsample_bytree": trial.suggest_float("cs", 0.5, 0.9),
        "min_child_weight": trial.suggest_int("mcw", 3, 20),
        "reg_alpha": trial.suggest_float("ra", 0.01, 2.0, log=True),
        "reg_lambda": trial.suggest_float("rl", 0.1, 3.0, log=True),
        "gamma": trial.suggest_float("g", 0.0, 2.0),
        "eval_metric": "logloss", "tree_method": "hist",
        "random_state": SEED, "n_jobs": -1,
    }
    scores = []
    for tr, val in StratifiedKFold(OPTUNA_CV_FOLDS, shuffle=True, random_state=SEED).split(X_tune, y_train):
        m = xgb.XGBClassifier(**p, early_stopping_rounds=50)
        m.fit(X_tune[tr], y_train[tr], eval_set=[(X_tune[val], y_train[val])], verbose=False)
        pred = m.predict_proba(X_tune[val])[:,1]
        scores.append(f1_score(y_train[val], (pred>=0.5).astype(int), pos_label=1))
    return np.mean(scores)

def optuna_objective_cb(trial):
    p = {
        "iterations": 1000,
        "learning_rate": trial.suggest_float("lr", 0.02, 0.08, log=True),
        "depth": trial.suggest_int("d", 5, 8),
        "l2_leaf_reg": trial.suggest_float("l2", 0.5, 5.0, log=True),
        "bagging_temperature": trial.suggest_float("bt", 0.0, 3.0),
        "random_strength": trial.suggest_float("rs", 0.0, 3.0),
        "eval_metric": "Logloss", "random_seed": SEED, "verbose": False,
    }
    scores = []
    for tr, val in StratifiedKFold(OPTUNA_CV_FOLDS, shuffle=True, random_state=SEED).split(X_tune, y_train):
        m = CatBoostClassifier(**p, early_stopping_rounds=50)
        m.fit(X_tune[tr], y_train[tr], eval_set=(X_tune[val], y_train[val]), verbose=False)
        pred = m.predict_proba(X_tune[val])[:,1]
        scores.append(f1_score(y_train[val], (pred>=0.5).astype(int), pos_label=1))
    return np.mean(scores)

# Run Optuna
print("  Tuning LightGBM...")
study_lgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study_lgb.optimize(optuna_objective_lgb, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
print(f"  → LGB best 3-fold F1: {study_lgb.best_value:.5f}")

print("  Tuning XGBoost...")
study_xgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study_xgb.optimize(optuna_objective_xgb, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
print(f"  → XGB best 3-fold F1: {study_xgb.best_value:.5f}")

print("  Tuning CatBoost...")
study_cb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study_cb.optimize(optuna_objective_cb, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
print(f"  → CB  best 3-fold F1: {study_cb.best_value:.5f}")

# Build final params from Optuna results
bp_lgb = study_lgb.best_params
lgb_params = {
    "n_estimators": 1500,
    "learning_rate": bp_lgb["lr"], "max_depth": bp_lgb["md"],
    "num_leaves": bp_lgb["nl"], "subsample": bp_lgb["ss"],
    "colsample_bytree": bp_lgb["cs"], "min_child_samples": bp_lgb["mcs"],
    "reg_alpha": bp_lgb["ra"], "reg_lambda": bp_lgb["rl"],
    "random_state": SEED, "n_jobs": -1, "verbose": -1,
}

bp_xgb = study_xgb.best_params
xgb_params = {
    "n_estimators": 1500,
    "learning_rate": bp_xgb["lr"], "max_depth": bp_xgb["md"],
    "subsample": bp_xgb["ss"], "colsample_bytree": bp_xgb["cs"],
    "min_child_weight": bp_xgb["mcw"],
    "reg_alpha": bp_xgb["ra"], "reg_lambda": bp_xgb["rl"],
    "gamma": bp_xgb["g"],
    "eval_metric": "logloss", "tree_method": "hist",
    "random_state": SEED, "n_jobs": -1,
}

bp_cb = study_cb.best_params
cb_params = {
    "iterations": 1500,
    "learning_rate": bp_cb["lr"], "depth": bp_cb["d"],
    "l2_leaf_reg": bp_cb["l2"],
    "bagging_temperature": bp_cb["bt"], "random_strength": bp_cb["rs"],
    "eval_metric": "Logloss", "random_seed": SEED, "verbose": False,
}

print(f"\n  Final LGB params: {lgb_params}")
print(f"  Final XGB params: {xgb_params}")
print(f"  Final CB  params: {cb_params}")

# ============================================================================
# SECTION 6: REPEATED 5-FOLD CV WITH LEAK-FREE TARGET ENCODING
# ============================================================================
print(f"\n>>> [5/9] Repeated {N_REPEATS}×{N_SPLITS}-Fold CV Evaluation")

total_folds = N_REPEATS * N_SPLITS
oof_lgb = np.zeros((N_REPEATS, len(y_train)), dtype=np.float32)
oof_xgb = np.zeros((N_REPEATS, len(y_train)), dtype=np.float32)
oof_cb  = np.zeros((N_REPEATS, len(y_train)), dtype=np.float32)

test_lgb = np.zeros(len(test_df), dtype=np.float32)
test_xgb = np.zeros(len(test_df), dtype=np.float32)
test_cb  = np.zeros(len(test_df), dtype=np.float32)

fold_metrics = []  # (repeat, fold, lgb_f1, xgb_f1, cb_f1, ens_f1, nested_th)

for rep in range(N_REPEATS):
    seed_r = SEED + rep * 111
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed_r)
    print(f"\n  ── Repeat {rep+1}/{N_REPEATS} (seed={seed_r}) ──")

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_base, y_train)):
        # Target encoding (leak-free: inner CV for train, outer for val)
        tr_te = pd.DataFrame(index=range(len(tr_idx)))
        val_te = pd.DataFrame(index=range(len(val_idx)))

        for col in te_cols:
            tr_s = train_df[col].iloc[tr_idx].reset_index(drop=True)
            val_s = train_df[col].iloc[val_idx].reset_index(drop=True)
            _, val_enc, _ = smoothed_te(tr_s, y_train[tr_idx], val_s)

            inner_enc = np.zeros(len(tr_idx))
            for itr, ival in StratifiedKFold(5, shuffle=True, random_state=seed_r).split(tr_s, y_train[tr_idx]):
                _, ie, _ = smoothed_te(tr_s.iloc[itr], y_train[tr_idx][itr], tr_s.iloc[ival])
                inner_enc[ival] = ie

            tr_te[f"{col}_te"] = inner_enc
            val_te[f"{col}_te"] = val_enc

        X_tr = np.hstack([X_train_base.iloc[tr_idx].values, tr_te.values]).astype(np.float32)
        X_val = np.hstack([X_train_base.iloc[val_idx].values, val_te.values]).astype(np.float32)
        X_tr  = np.nan_to_num(X_tr, nan=-999.0)
        X_val = np.nan_to_num(X_val, nan=-999.0)
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        # LightGBM
        m_lgb = lgb.LGBMClassifier(**lgb_params)
        m_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        vp_lgb = m_lgb.predict_proba(X_val)[:,1]
        oof_lgb[rep, val_idx] = vp_lgb
        test_lgb += m_lgb.predict_proba(X_test_final)[:,1] / total_folds

        # XGBoost
        m_xgb = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=80)
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        vp_xgb = m_xgb.predict_proba(X_val)[:,1]
        oof_xgb[rep, val_idx] = vp_xgb
        test_xgb += m_xgb.predict_proba(X_test_final)[:,1] / total_folds

        # CatBoost
        m_cb = CatBoostClassifier(**cb_params, early_stopping_rounds=80)
        m_cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        vp_cb = m_cb.predict_proba(X_val)[:,1]
        oof_cb[rep, val_idx] = vp_cb
        test_cb += m_cb.predict_proba(X_test_final)[:,1] / total_folds

        # Ensemble and nested threshold (on train split only)
        vp_ens = 0.40*vp_lgb + 0.35*vp_xgb + 0.25*vp_cb
        tr_ens = 0.40*m_lgb.predict_proba(X_tr)[:,1] + 0.35*m_xgb.predict_proba(X_tr)[:,1] + 0.25*m_cb.predict_proba(X_tr)[:,1]

        best_th, best_f1_th = 0.50, 0.0
        for th in np.arange(0.40, 0.60, 0.01):
            f = f1_score(y_tr, (tr_ens>=th).astype(int), pos_label=1)
            if f > best_f1_th: best_th, best_f1_th = round(th,2), f

        f_lgb = f1_score(y_val, (vp_lgb>=0.5).astype(int), pos_label=1)
        f_xgb = f1_score(y_val, (vp_xgb>=0.5).astype(int), pos_label=1)
        f_cb  = f1_score(y_val, (vp_cb>=0.5).astype(int), pos_label=1)
        f_ens = f1_score(y_val, (vp_ens>=best_th).astype(int), pos_label=1)
        fold_metrics.append((rep, fold, f_lgb, f_xgb, f_cb, f_ens, best_th))

        print(f"    F{fold+1} LGB={f_lgb:.5f} XGB={f_xgb:.5f} CB={f_cb:.5f} Ens(th={best_th})={f_ens:.5f}")

# Average OOF across repeats
oof_lgb_avg = oof_lgb.mean(axis=0)
oof_xgb_avg = oof_xgb.mean(axis=0)
oof_cb_avg  = oof_cb.mean(axis=0)

# ============================================================================
# SECTION 7: ENSEMBLE WEIGHT SELECTION & THRESHOLD ANALYSIS
# ============================================================================
print("\n>>> [6/9] Ensemble Weight Selection & Threshold Analysis")

# Find best simple weights on OOF
best_w, best_wf1 = (0.33, 0.34, 0.33), 0.0
for w1 in np.arange(0.20, 0.55, 0.05):
    for w2 in np.arange(0.20, 0.55, 0.05):
        w3 = round(1-w1-w2, 2)
        if w3 < 0.15 or w3 > 0.50: continue
        ens = w1*oof_lgb_avg + w2*oof_xgb_avg + w3*oof_cb_avg
        f = f1_score(y_train, (ens>=0.5).astype(int), pos_label=1)
        if f > best_wf1:
            best_wf1 = f
            best_w = (round(w1,2), round(w2,2), w3)

W_LGB, W_XGB, W_CB = best_w
print(f"  Best weights (@ th=0.5): LGB={W_LGB}, XGB={W_XGB}, CB={W_CB} → OOF F1={best_wf1:.5f}")

oof_ens = W_LGB*oof_lgb_avg + W_XGB*oof_xgb_avg + W_CB*oof_cb_avg
test_ens = W_LGB*test_lgb + W_XGB*test_xgb + W_CB*test_cb

# Comprehensive threshold sweep
print("\n  Threshold Sweep (OOF Ensemble):")
print(f"  {'Thresh':>7} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Dead%':>8}")
sweep_results = []
for th in np.arange(0.40, 0.65, 0.005):
    pred = (oof_ens >= th).astype(int)
    f1 = f1_score(y_train, pred, pos_label=1)
    prec = precision_score(y_train, pred, pos_label=1)
    rec = recall_score(y_train, pred, pos_label=1)
    dead_pct = pred.mean()
    sweep_results.append((th, f1, prec, rec, dead_pct))

# Find key thresholds
f1_optimal = max(sweep_results, key=lambda x: x[1])
calibrated = min(sweep_results, key=lambda x: abs(x[4] - GLOBAL_DEAD_RATE))

for th, f1, prec, rec, dpct in sweep_results:
    tag = ""
    if abs(th - f1_optimal[0]) < 0.001: tag = " ← F1 OPTIMAL"
    if abs(th - calibrated[0]) < 0.001: tag = " ← CALIBRATED (≈83% Dead)"
    if abs(th - 0.50) < 0.001: tag = " ← DEFAULT"
    if dpct > 0.82 and dpct < 0.86:
        print(f"  {th:7.3f} {f1:8.5f} {prec:8.4f} {rec:8.4f} {dpct:8.3f}{tag}")
    elif tag:
        print(f"  {th:7.3f} {f1:8.5f} {prec:8.4f} {rec:8.4f} {dpct:8.3f}{tag}")

print(f"\n  F1-Optimal:  th={f1_optimal[0]:.3f} → F1={f1_optimal[1]:.5f}, Dead%={f1_optimal[4]:.1%}")
print(f"  Calibrated:  th={calibrated[0]:.3f} → F1={calibrated[1]:.5f}, Dead%={calibrated[4]:.1%}")

# Nested threshold stability
nested_ths = [m[6] for m in fold_metrics]
print(f"\n  Nested Threshold Stability:")
print(f"    Per-fold thresholds: {nested_ths}")
print(f"    Mean={np.mean(nested_ths):.3f} ± {np.std(nested_ths):.3f}, Median={np.median(nested_ths):.3f}")

# ============================================================================
# SECTION 8: VALIDATION REPORT & SUBGROUP DIAGNOSTICS
# ============================================================================
print("\n" + "=" * 80)
print(">>> [7/9] Validation Report")
print("=" * 80)

fm = np.array(fold_metrics)
print(f"\n  Model Performance (Mean ± Std across {total_folds} folds @ th=0.50):")
print(f"    LGB: {fm[:,2].mean():.5f} ± {fm[:,2].std():.5f} [{fm[:,2].min():.5f}, {fm[:,2].max():.5f}]")
print(f"    XGB: {fm[:,3].mean():.5f} ± {fm[:,3].std():.5f} [{fm[:,3].min():.5f}, {fm[:,3].max():.5f}]")
print(f"    CB:  {fm[:,4].mean():.5f} ± {fm[:,4].std():.5f} [{fm[:,4].min():.5f}, {fm[:,4].max():.5f}]")
print(f"    Ens: {fm[:,5].mean():.5f} ± {fm[:,5].std():.5f} [{fm[:,5].min():.5f}, {fm[:,5].max():.5f}]")

corr_df = pd.DataFrame({"LGB": oof_lgb_avg, "XGB": oof_xgb_avg, "CB": oof_cb_avg})
print(f"\n  Prediction Correlation:")
print(corr_df.corr().round(4).to_string())

# Choose primary submission threshold: calibrated (matches train Dead%)
PRIMARY_TH = calibrated[0]
print(f"\n  PRIMARY SUBMISSION THRESHOLD: {PRIMARY_TH:.3f} (Dead% ≈ {GLOBAL_DEAD_RATE:.0%})")

oof_labels = (oof_ens >= PRIMARY_TH).astype(int)
print(f"\n  OOF at PRIMARY threshold ({PRIMARY_TH:.3f}):")
print(classification_report(y_train, oof_labels, target_names=["Alive","Dead"], digits=4))

print("\n>>> [8/9] Subgroup Diagnostics")
def subgrp(name, mask):
    if mask.sum() < 20: return
    sy, sp = y_train[mask], oof_labels[mask]
    print(f"  {name:<42} N={mask.sum():<5} Dead%={sy.mean():.1%} F1={f1_score(sy, sp, pos_label=1, zero_division=0):.4f}")

subgrp("Old Diagnoses (2013-2015)", train_df["year_of_diagnosis"]<=2015)
subgrp("Recent Diagnoses (2021-2023)", train_df["year_of_diagnosis"]>=2021)
subgrp("Distant/Metastatic", train_df["summary_stage"]=="Distant")
subgrp("Localized", train_df["summary_stage"]=="Localized")
subgrp("Surgery Performed", train_df["reason_nocancer_directed_surgery"]=="Surgery performed")
subgrp("Small Cell", X_tr_eng["is_small_cell"]==1)
subgrp("Missing Tumor Size", X_tr_eng["tsize_missing"]==1)
subgrp("Elderly (≥80)", X_tr_eng["age"]>=80)
subgrp("Young (≤50)", X_tr_eng["age"]<=50)
subgrp("Borderline [0.40-0.60]", (oof_ens>=0.40)&(oof_ens<=0.60))

# ============================================================================
# SECTION 9: MULTI-THRESHOLD SUBMISSION GENERATION
# ============================================================================
print("\n" + "=" * 80)
print(">>> [9/9] Generating Submissions")
print("=" * 80)

# Generate submissions at multiple thresholds for LB experimentation
thresholds_to_generate = {
    "submission.csv": PRIMARY_TH,           # Primary: calibrated to 83% Dead
    "submission_f1opt.csv": f1_optimal[0],   # F1-optimal on OOF
    "submission_050.csv": 0.50,              # Default Bayesian boundary
}

for fname, th in thresholds_to_generate.items():
    preds = (test_ens >= th).astype(int)
    status = np.where(preds==1, "Dead", "Alive")
    sub = pd.DataFrame({"patient_id": test_ids, "vital_status": status})

    assert len(sub) == 36000
    assert sub["vital_status"].isin(["Dead","Alive"]).all()
    assert sub["patient_id"].nunique() == 36000

    sub.to_csv(fname, index=False)
    dead_n = (preds==1).sum()
    print(f"  {fname:<25} th={th:.3f} → Dead={dead_n:,} ({preds.mean():.1%}), Alive={36000-dead_n:,}")

# Also show which Dead% the best LB submission had
print(f"\n  Reference: Your best LB score (0.875506) had Dead%=84.5% (30,431/36,000)")
print(f"  Train ground truth Dead%: {GLOBAL_DEAD_RATE:.1%}")

print("\n" + "=" * 80)
print(">>> [10/10] Pseudo-Labeling & Final Retrain")
print("=" * 80)

# Identify confident test predictions
CONFIDENT_DEAD_TH = 0.95
CONFIDENT_ALIVE_TH = 0.05

pseudo_dead_mask = test_ens >= CONFIDENT_DEAD_TH
pseudo_alive_mask = test_ens <= CONFIDENT_ALIVE_TH
pseudo_mask = pseudo_dead_mask | pseudo_alive_mask

n_pseudo_dead = pseudo_dead_mask.sum()
n_pseudo_alive = pseudo_alive_mask.sum()
n_pseudo = n_pseudo_dead + n_pseudo_alive

print(f"  Confident Dead  (P >= {CONFIDENT_DEAD_TH}): {n_pseudo_dead:,}")
print(f"  Confident Alive (P <= {CONFIDENT_ALIVE_TH}): {n_pseudo_alive:,}")
print(f"  Total Pseudo-Labeled: {n_pseudo:,} ({(n_pseudo/len(test_df)):.1%} of test set)")

if n_pseudo > 1000:
    # ── Augment Training Data ──
    X_pseudo = X_test_base[pseudo_mask]
    y_pseudo = pseudo_dead_mask[pseudo_mask].astype(int)

    # We need to re-do Target Encoding on the combined dataset
    X_train_comb_base = pd.concat([X_train_base, X_pseudo], ignore_index=True)
    y_train_comb = np.concatenate([y_train, y_pseudo])

    # Re-encode Target Encoding columns for combined dataset
    te_comb = pd.DataFrame(index=range(len(X_train_comb_base)))
    comb_raw_cols = {col: pd.concat([train_df[col], test_df[col][pseudo_mask]], ignore_index=True) for col in te_cols}
    
    for col in te_cols:
        tr_s = comb_raw_cols[col]
        # We need leak-free inner CV for the combined dataset for retraining
        inner_enc = np.zeros(len(tr_s))
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        for itr, ival in skf.split(tr_s, y_train_comb):
            _, ie, _ = smoothed_te(tr_s.iloc[itr], y_train_comb[itr], tr_s.iloc[ival])
            inner_enc[ival] = ie
        te_comb[f"{col}_te"] = inner_enc
    
    X_train_comb = np.hstack([X_train_comb_base.values, te_comb.values]).astype(np.float32)
    X_train_comb = np.nan_to_num(X_train_comb, nan=-999.0)

    # ── Retrain Models (Single 5-fold pass to save time) ──
    print("\n  Retraining on Augmented Dataset (Train + Pseudo-Test)...")
    test_lgb_pseudo = np.zeros(len(test_df), dtype=np.float32)
    test_xgb_pseudo = np.zeros(len(test_df), dtype=np.float32)
    test_cb_pseudo  = np.zeros(len(test_df), dtype=np.float32)

    skf = StratifiedKFold(5, shuffle=True, random_state=SEED+999)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_comb, y_train_comb)):
        X_tr, y_tr = X_train_comb[tr_idx], y_train_comb[tr_idx]
        X_val, y_val = X_train_comb[val_idx], y_train_comb[val_idx]

        m_lgb = lgb.LGBMClassifier(**lgb_params)
        m_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        test_lgb_pseudo += m_lgb.predict_proba(X_test_final)[:,1] / 5

        m_xgb = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=80)
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        test_xgb_pseudo += m_xgb.predict_proba(X_test_final)[:,1] / 5

        m_cb = CatBoostClassifier(**cb_params, early_stopping_rounds=80)
        m_cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        test_cb_pseudo += m_cb.predict_proba(X_test_final)[:,1] / 5

        print(f"    Pseudo-F{fold+1} done")

    test_ens_pseudo = W_LGB*test_lgb_pseudo + W_XGB*test_xgb_pseudo + W_CB*test_cb_pseudo
    
    # ── Generate Pseudo Submission ──
    # Let's match the 84.5% Dead rate that worked best
    th_pseudo = 0.50
    for th in np.arange(0.40, 0.70, 0.005):
        dpct = (test_ens_pseudo >= th).mean()
        if dpct <= 0.845:
            th_pseudo = th
            break
            
    preds_pseudo = (test_ens_pseudo >= th_pseudo).astype(int)
    status_pseudo = np.where(preds_pseudo==1, "Dead", "Alive")
    sub_pseudo = pd.DataFrame({"patient_id": test_ids, "vital_status": status_pseudo})
    sub_pseudo.to_csv("submission_pseudo.csv", index=False)
    
    dead_n = preds_pseudo.sum()
    print(f"\n  submission_pseudo.csv th={th_pseudo:.3f} → Dead={dead_n:,} ({preds_pseudo.mean():.1%})")
else:
    print("  Not enough confident predictions for effective pseudo-labeling.")


print("\n" + "=" * 80)
print("  PIPELINE v4 (w/ Pseudo-Labeling) COMPLETE.")
print("  Recommended: Submit submission_pseudo.csv to test the augmentation.")
print("=" * 80)

