"""
Pipeline v6: Maximum Private-Test Generalization
=================================================
Three techniques to improve beyond v5's 0.8767:

1. ADD TARGET ENCODING BACK — v4 had it, v5 dropped it. Adding it back
   creates a richer feature set that captures category-level mortality rates.

2. BLEND v4 + v5 MODELS — v4 (with TE + a different stored parameter set) and v5
   (without TE + different params) capture complementary patterns.
   Blending their probabilities reduces variance.

3. PSEUDO-LABELING — Use confident test predictions (>95% Dead or <5% Dead)
   to augment training data, then retrain.

Final test predictions = blend of all approaches at 84.5% Dead.
"""

import warnings; warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier

SEED = 42
np.random.seed(SEED)
SMOOTHING_M = 20.0
LEAKAGE_CLEAN = False
OUTPUT_DIR = Path("artifacts/v6_rerun")

print("=" * 80)
print("  PIPELINE v6: Maximum Private-Test Generalization")
print("=" * 80)

# ── Load Data ──
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
test_ids = test_df["patient_id"].values
y = (train_df["vital_status"] == "Dead").astype(int).values
print(f"  Train: {len(train_df):,} | Test: {len(test_df):,} | Dead%: {y.mean():.2%}")

# ============================================================================
# FEATURE ENGINEERING (same as v5 + target encoding from v4)
# ============================================================================
print("\n>>> Building Features...")

def build_features(df):
    F = pd.DataFrame(index=df.index)
    def _age(s):
        if pd.isna(s): return np.nan
        s = str(s).strip()
        if "90+" in s: return 92.5
        parts = s.replace(" years","").split("-")
        try: return (float(parts[0])+float(parts[1]))/2
        except: return np.nan
    F["age_midpoint"] = df["age_recode"].apply(_age)
    F["age_missing"] = df["age_recode"].isna().astype(int)
    F["year_of_diagnosis"] = df["year_of_diagnosis"].astype(float)
    F["years_since_dx"] = 2024.0 - df["year_of_diagnosis"]
    F["is_male"] = (df["sex"] == "Male").astype(int)
    F["nodes_examined"] = df["regional_nodes_examined"].astype(float)
    F["nodes_examined_zero"] = (df["regional_nodes_examined"] == 0).astype(int)
    npos = df["regional_nodes_positive"].astype(float)
    F["nodes_positive"] = np.where(npos < 98, npos, np.nan)
    F["nodes_positive_unknown"] = (npos >= 98).astype(int)
    F["node_positive_ratio"] = np.where(
        (F["nodes_examined"]>0)&(~np.isnan(F["nodes_positive"])),
        F["nodes_positive"]/np.maximum(F["nodes_examined"],1), np.nan)
    def _ps(v):
        if pd.isna(v): return np.nan
        s = str(v).strip()
        if s.startswith("Unknown") or s in ("Blank(s)","999"): return np.nan
        try: return float(s)
        except: return np.nan
    F["tumor_size_ot"] = df["tumor_size_overtime"].apply(_ps)
    F["tumor_size_sm"] = df["tumor_size_summary"].apply(_ps)
    F["tumor_size_best"] = F["tumor_size_sm"].fillna(F["tumor_size_ot"])
    F["tumor_size_missing"] = F["tumor_size_best"].isna().astype(int)
    def _cs(v):
        if pd.isna(v): return np.nan
        s = str(v).strip()
        if s == "Blank(s)": return np.nan
        try:
            val = float(s)
            return np.nan if val == 999 else val
        except: return np.nan
    F["cs_tumor_size"] = df["cs_tumor_size20042015"].apply(_cs)
    F["cs_extension"] = df["cs_extension20042015"].apply(_cs)
    smap = {"Blank(s)":np.nan,"Unknown/unstaged":np.nan,"Localized":1,"Regional":2,"Distant":3}
    F["stage_ord"] = df["summary_stage"].map(smap).astype(float)
    F["stage_missing"] = F["stage_ord"].isna().astype(int)
    F["is_distant"] = (df["summary_stage"]=="Distant").astype(int)
    F["is_localized"] = (df["summary_stage"]=="Localized").astype(int)
    t_map={"Blank(s)":np.nan,"88":np.nan,"TX":np.nan,"T0":0,"Ta":1,"Tis":1,"Tis(DCIS)":1,"Tis(LCIS)":1,"Tis(Paget)":1,"T1":2,"T1mi":2,"T1a":3,"T1a1":3,"T1a2":3,"T1b":4,"T1b1":4,"T1b2":4,"T1c":5,"T2":6,"T2a":7,"T2b":8,"T3":9,"T4":10,"T4a":11,"T4b":12,"T4c":12,"T4d":13,"T4e":13}
    n_map={"Blank(s)":np.nan,"88":np.nan,"NX":np.nan,"N0":0,"N1":1,"N1a":1,"N1b":2,"N1c":2,"N2":3,"N2a":3,"N2b":4,"N2c":4,"N3":5,"N3a":5,"N3b":6,"N3c":6}
    m_map={"Blank(s)":np.nan,"88":np.nan,"MX":np.nan,"M0":0,"M1":1,"M1a":2,"M1b":3,"M1c":4}
    F["t_stage"]=df["derived_eod2018t_recode2018"].map(t_map).astype(float)
    F["n_stage"]=df["derived_eod2018n_recode2018"].map(n_map).astype(float)
    F["m_stage"]=df["derived_eod2018m_recode2018"].map(m_map).astype(float)
    F["is_m1"]=df["derived_eod2018m_recode2018"].str.startswith("M1",na=False).astype(int)
    F["tnm_composite"]=F["t_stage"].fillna(0)+F["n_stage"].fillna(0)+F["m_stage"].fillna(0)*3
    mc=["seer_combined_metsatdxbone2010","seer_combined_metsatdxbrain2010",
        "seer_combined_metsatdxliver2010","seer_combined_metsatdxlung2010"]
    for m in mc:
        short=m.replace("seer_combined_metsatdx","").replace("2010","")
        F[f"met_{short}_yes"]=(df[m]=="Yes").astype(int)
    F["met_count"]=sum((df[m]=="Yes").astype(int) for m in mc)
    F["any_mets"]=(F["met_count"]>=1).astype(int)
    F["surgery_done"]=(df["reason_nocancer_directed_surgery"]=="Surgery performed").astype(int)
    F["surgery_not_rec"]=df["reason_nocancer_directed_surgery"].str.contains("Not recommended",na=False).astype(int)
    F["died_before_surg"]=df["reason_nocancer_directed_surgery"].str.contains("died prior",na=False).astype(int)
    F["autopsy_or_dc"]=df["reason_nocancer_directed_surgery"].str.contains("death certificate|autopsy",na=False).astype(int)
    F["radiation_given"]=(~df["radiation_recode"].isin([
        "None/Unknown","No radiation and/or no surgery; unknown if surgery and/or radiation given",
        "Refused (1988+)","Recommended, unknown if administered"
    ])&df["radiation_recode"].notna()).astype(int)
    gmap={"Blank(s)":np.nan,"Unknown":np.nan,"Well differentiated; Grade I":1,
          "Moderately differentiated; Grade II":2,"Poorly differentiated; Grade III":3,
          "Undifferentiated; anaplastic; Grade IV":4}
    F["grade_ord"]=df["grade_recode_thru2017"].map(gmap).astype(float)
    F["grade_missing"]=F["grade_ord"].isna().astype(int)
    def _hg(c):
        try: c=int(c)
        except: return 0
        if c in (8140,8141,8143,8144,8145,8147,8148,8211,8230,8250,8251,8252,8253,8254,8255,8256,8257,8260,8263,8310,8323,8333,8480,8481,8490,8550,8551,8570,8574): return 1
        elif c in (8070,8071,8072,8073,8074,8075,8076,8078,8083,8084): return 2
        elif c in (8041,8042,8043,8044,8045): return 3
        elif c in (8012,8013,8014): return 4
        elif c in (8000,8001,8002,8003,8004,8005,8010,8011,8020,8021,8022): return 5
        elif c in (8240,8241,8242,8243,8244,8245,8246,8249): return 6
        else: return 7
    F["histo_group"]=df["histologic_type_icdo3"].apply(_hg)
    F["is_small_cell"]=(F["histo_group"]==3).astype(int)
    F["histologic_raw"]=df["histologic_type_icdo3"].astype(float)
    F["scope_reglnsur_missing"]=df["rx_summ_scope_reglnsur2003"].isna().astype(int)
    F["eod_m_blank"]=(df["derived_eod2018m_recode2018"]=="Blank(s)").astype(int)
    F["grade_blank"]=(df["grade_recode_thru2017"]=="Blank(s)").astype(int)
    F["age_x_years"]=F["age_midpoint"]*F["years_since_dx"]
    F["age_x_stage"]=F["age_midpoint"]*F["stage_ord"]
    F["stage_x_surgery"]=F["stage_ord"]*F["surgery_done"]
    F["mets_x_surgery"]=F["any_mets"]*F["surgery_done"]
    F["age_x_mets"]=F["age_midpoint"]*F["any_mets"]
    F["years_x_stage"]=F["years_since_dx"]*F["stage_ord"]
    # Frequency encoding (target-free)
    for col_name, col_data in [("primary_site", df["primary_site"]),
                                ("histologic_type_icdo3", df["histologic_type_icdo3"])]:
        freq = col_data.astype(str).map(col_data.astype(str).value_counts(normalize=True))
        F[f"{col_name}_freq"] = freq.astype(float)
    if LEAKAGE_CLEAN:
        F = F.drop(columns=["died_before_surg", "autopsy_or_dc"])
    return F

X_tr_num = build_features(train_df)
X_te_num = build_features(test_df)

# Ordinal encode categoricals
cat_cols=["age_recode","race","sex","origin","primary_site","marital_status_at_diagnosis",
          "sequence_number","site_recode_icdo3_who2008","grade_recode_thru2017","laterality",
          "diagnostic_confirmation","summary_stage","derived_eod2018t_recode2018",
          "derived_eod2018n_recode2018","derived_eod2018m_recode2018",
          "seer_combined_metsatdxbone2010","seer_combined_metsatdxbrain2010",
          "seer_combined_metsatdxliver2010","seer_combined_metsatdxlung2010",
          "rx_summ_surgprim_site19982022","rx_summ_surgprim_site20232023",
          "rx_summ_scope_reglnsur2003","rx_summ_surgothregdis2003",
          "rx_summ_surgradseq","reason_nocancer_directed_surgery","radiation_recode",
          "tumor_size_overtime","tumor_size_summary","cs_tumor_size20042015","cs_extension20042015"]
if LEAKAGE_CLEAN:
    cat_cols.remove("reason_nocancer_directed_surgery")

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
enc.fit(train_df[cat_cols].fillna("__NA__").astype(str))
tr_cat = pd.DataFrame(enc.transform(train_df[cat_cols].fillna("__NA__").astype(str)),
                       columns=[f"{c}_enc" for c in cat_cols], index=train_df.index)
te_cat = pd.DataFrame(enc.transform(test_df[cat_cols].fillna("__NA__").astype(str)),
                       columns=[f"{c}_enc" for c in cat_cols], index=test_df.index)

X_train_base = pd.concat([X_tr_num, tr_cat], axis=1)
X_test_base = pd.concat([X_te_num, te_cat], axis=1)

# Target Encoding columns
te_cols = ["histologic_type_icdo3","primary_site","diagnostic_confirmation",
           "reason_nocancer_directed_surgery","rx_summ_surgprim_site19982022"]
if LEAKAGE_CLEAN:
    te_cols.remove("reason_nocancer_directed_surgery")

def smoothed_te(tr_s, target, te_s, m=SMOOTHING_M):
    prior = target.mean()
    df = pd.DataFrame({"c": tr_s.astype(str), "y": target})
    stats = df.groupby("c")["y"].agg(["count","mean"])
    encoded = (stats["count"]*stats["mean"] + m*prior) / (stats["count"]+m)
    return encoded, te_s.astype(str).map(encoded).fillna(prior).values, prior

# Full-train TE for test
te_test = pd.DataFrame(index=test_df.index)
for col in te_cols:
    _, vals, _ = smoothed_te(train_df[col], y, test_df[col])
    te_test[f"{col}_te"] = vals

X_test_with_te = np.nan_to_num(
    pd.concat([X_test_base, te_test], axis=1).values.astype(np.float32), nan=-999.0)
X_test_no_te = np.nan_to_num(X_test_base.values.astype(np.float32), nan=-999.0)

print(f"  Features (no TE): {X_test_no_te.shape[1]}")
print(f"  Features (with TE): {X_test_with_te.shape[1]}")

# ============================================================================
# MODEL CONFIGS — Two diverse param sets
# ============================================================================
# v5 stored parameters (no TE)
v5_lgb = {'n_estimators':1500,'learning_rate':0.04158,'max_depth':4,'num_leaves':53,
           'subsample':0.6706,'colsample_bytree':0.6861,'min_child_samples':37,
           'reg_alpha':1.5411,'reg_lambda':0.3901,'random_state':42,'n_jobs':-1,'verbose':-1}
v5_xgb = {'n_estimators':1500,'learning_rate':0.05443,'max_depth':5,
           'subsample':0.7332,'colsample_bytree':0.6476,'min_child_weight':7,
           'reg_alpha':0.09336,'reg_lambda':0.03991,'gamma':1.1909,
           'eval_metric':'logloss','tree_method':'hist','random_state':42,'n_jobs':-1}
v5_cb  = {'iterations':1500,'learning_rate':0.07444,'depth':7,
           'l2_leaf_reg':0.7346,'bagging_temperature':0.01295,'random_strength':1.9376,
           'eval_metric':'Logloss','random_seed':42,'verbose':False,
           'allow_writing_files':False}

# v4 stored parameters (with TE)
v4_lgb = {'n_estimators':1500,'learning_rate':0.02261,'max_depth':5,'num_leaves':33,
           'subsample':0.7813,'colsample_bytree':0.6555,'min_child_samples':24,
           'reg_alpha':0.8071,'reg_lambda':0.3365,'random_state':42,'n_jobs':-1,'verbose':-1}
v4_xgb = {'n_estimators':1500,'learning_rate':0.02261,'max_depth':5,
           'subsample':0.7113,'colsample_bytree':0.6301,'min_child_weight':9,
           'reg_alpha':0.04211,'reg_lambda':1.6755,'gamma':0.7135,
           'eval_metric':'logloss','tree_method':'hist','random_state':42,'n_jobs':-1}
v4_cb  = {'iterations':1500,'learning_rate':0.02483,'depth':5,
           'l2_leaf_reg':3.674,'bagging_temperature':1.803,'random_strength':2.124,
           'eval_metric':'Logloss','random_seed':42,'verbose':False,
           'allow_writing_files':False}

# v5 weights
W5_LGB, W5_XGB, W5_CB = 0.30, 0.15, 0.55
# v4 weights
W4_LGB, W4_XGB, W4_CB = 0.35, 0.20, 0.45

# ============================================================================
# PHASE 1: Train BOTH model sets via 3×5 fold-averaged predictions
# ============================================================================
print("\n>>> Phase 1: Dual Model Training (3×5 CV, fold-averaged)")

N_REPS = 3
N_FOLDS = 5
total = N_REPS * N_FOLDS

# v5 models (no TE)
v5_test_lgb = np.zeros(len(test_df))
v5_test_xgb = np.zeros(len(test_df))
v5_test_cb  = np.zeros(len(test_df))

# v4 models (with TE)
v4_test_lgb = np.zeros(len(test_df))
v4_test_xgb = np.zeros(len(test_df))
v4_test_cb  = np.zeros(len(test_df))

# OOF predictions for stacking
oof_v5 = np.zeros(len(y))
oof_v4 = np.zeros(len(y))
oof_counts = np.zeros(len(y))

for rep in range(N_REPS):
    seed_r = SEED + rep * 111
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed_r)
    print(f"\n  ── Repeat {rep+1}/{N_REPS} (seed={seed_r}) ──")

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_base, y)):

        # ── v5: No Target Encoding ──
        X_tr_v5 = np.nan_to_num(X_train_base.iloc[tr_idx].values.astype(np.float32), nan=-999.0)
        X_val_v5 = np.nan_to_num(X_train_base.iloc[val_idx].values.astype(np.float32), nan=-999.0)

        m = lgb.LGBMClassifier(**v5_lgb)
        m.fit(X_tr_v5, y[tr_idx], eval_set=[(X_val_v5, y[val_idx])],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        v5_test_lgb += m.predict_proba(X_test_no_te)[:, 1] / total
        vp5_lgb = m.predict_proba(X_val_v5)[:, 1]

        m = xgb.XGBClassifier(**v5_xgb, early_stopping_rounds=80)
        m.fit(X_tr_v5, y[tr_idx], eval_set=[(X_val_v5, y[val_idx])], verbose=False)
        v5_test_xgb += m.predict_proba(X_test_no_te)[:, 1] / total
        vp5_xgb = m.predict_proba(X_val_v5)[:, 1]

        m = CatBoostClassifier(**v5_cb, early_stopping_rounds=80)
        m.fit(X_tr_v5, y[tr_idx], eval_set=(X_val_v5, y[val_idx]), verbose=False)
        v5_test_cb += m.predict_proba(X_test_no_te)[:, 1] / total
        vp5_cb = m.predict_proba(X_val_v5)[:, 1]

        v5_val = W5_LGB*vp5_lgb + W5_XGB*vp5_xgb + W5_CB*vp5_cb

        # ── v4: With Target Encoding (leak-free inner CV) ──
        tr_te = pd.DataFrame(index=range(len(tr_idx)))
        val_te = pd.DataFrame(index=range(len(val_idx)))
        for col in te_cols:
            tr_s = train_df[col].iloc[tr_idx].reset_index(drop=True)
            val_s = train_df[col].iloc[val_idx].reset_index(drop=True)
            _, val_enc, _ = smoothed_te(tr_s, y[tr_idx], val_s)
            inner_enc = np.zeros(len(tr_idx))
            for itr, ival in StratifiedKFold(5, shuffle=True, random_state=seed_r).split(tr_s, y[tr_idx]):
                _, ie, _ = smoothed_te(tr_s.iloc[itr], y[tr_idx][itr], tr_s.iloc[ival])
                inner_enc[ival] = ie
            tr_te[f"{col}_te"] = inner_enc
            val_te[f"{col}_te"] = val_enc

        X_tr_v4 = np.nan_to_num(np.hstack([X_train_base.iloc[tr_idx].values, tr_te.values]).astype(np.float32), nan=-999.0)
        X_val_v4 = np.nan_to_num(np.hstack([X_train_base.iloc[val_idx].values, val_te.values]).astype(np.float32), nan=-999.0)

        m = lgb.LGBMClassifier(**v4_lgb)
        m.fit(X_tr_v4, y[tr_idx], eval_set=[(X_val_v4, y[val_idx])],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        v4_test_lgb += m.predict_proba(X_test_with_te)[:, 1] / total
        vp4_lgb = m.predict_proba(X_val_v4)[:, 1]

        m = xgb.XGBClassifier(**v4_xgb, early_stopping_rounds=80)
        m.fit(X_tr_v4, y[tr_idx], eval_set=[(X_val_v4, y[val_idx])], verbose=False)
        v4_test_xgb += m.predict_proba(X_test_with_te)[:, 1] / total
        vp4_xgb = m.predict_proba(X_val_v4)[:, 1]

        m = CatBoostClassifier(**v4_cb, early_stopping_rounds=80)
        m.fit(X_tr_v4, y[tr_idx], eval_set=(X_val_v4, y[val_idx]), verbose=False)
        v4_test_cb += m.predict_proba(X_test_with_te)[:, 1] / total
        vp4_cb = m.predict_proba(X_val_v4)[:, 1]

        v4_val = W4_LGB*vp4_lgb + W4_XGB*vp4_xgb + W4_CB*vp4_cb

        # Store OOF
        oof_v5[val_idx] += v5_val
        oof_v4[val_idx] += v4_val
        oof_counts[val_idx] += 1

        f5 = f1_score(y[val_idx], (v5_val >= 0.5).astype(int), pos_label=1)
        f4 = f1_score(y[val_idx], (v4_val >= 0.5).astype(int), pos_label=1)
        print(f"    F{fold+1} Dead-class F1: v5={f5:.5f}  v4={f4:.5f}")

# Average OOF across repeats
oof_v5 /= oof_counts
oof_v4 /= oof_counts

# Ensemble test predictions
v5_ens = W5_LGB*v5_test_lgb + W5_XGB*v5_test_xgb + W5_CB*v5_test_cb
v4_ens = W4_LGB*v4_test_lgb + W4_XGB*v4_test_xgb + W4_CB*v4_test_cb

# ============================================================================
# PHASE 2: Find optimal blend ratio (v5 vs v4)
# ============================================================================
print("\n>>> Phase 2: Optimal Blend Ratio")

best_alpha, best_blend_f1 = 0.5, 0.0
for alpha in np.arange(0.0, 1.01, 0.05):
    blend = alpha * oof_v5 + (1 - alpha) * oof_v4
    f = f1_score(y, (blend >= 0.5).astype(int), pos_label=1)
    if f > best_blend_f1:
        best_blend_f1 = f
        best_alpha = round(alpha, 2)

print(f"  Best blend: {best_alpha:.0%} v5 + {1-best_alpha:.0%} v4 → OOF Dead-class F1={best_blend_f1:.5f}")

# Also check individual OOF scores
f5_oof = f1_score(y, (oof_v5 >= 0.5).astype(int), pos_label=1)
f4_oof = f1_score(y, (oof_v4 >= 0.5).astype(int), pos_label=1)
print(f"  v5 alone OOF Dead-class F1: {f5_oof:.5f}")
print(f"  v4 alone OOF Dead-class F1: {f4_oof:.5f}")
print(f"  Blend   OOF Dead-class F1:  {best_blend_f1:.5f}")

# Blend test predictions
test_blend = best_alpha * v5_ens + (1 - best_alpha) * v4_ens
oof_blend = best_alpha * oof_v5 + (1 - best_alpha) * oof_v4

# ============================================================================
# PHASE 3: Pseudo-labeling
# ============================================================================
print("\n>>> Phase 3: Pseudo-Labeling")

CONF_DEAD = 0.95
CONF_ALIVE = 0.05
pseudo_dead = test_blend >= CONF_DEAD
pseudo_alive = test_blend <= CONF_ALIVE
pseudo_mask = pseudo_dead | pseudo_alive

n_pd, n_pa = pseudo_dead.sum(), pseudo_alive.sum()
print(f"  Confident Dead  (≥{CONF_DEAD}): {n_pd:,}")
print(f"  Confident Alive (≤{CONF_ALIVE}): {n_pa:,}")
print(f"  Total Pseudo: {n_pd+n_pa:,} ({(n_pd+n_pa)/len(test_df):.1%} of test)")

if (n_pd + n_pa) > 500:
    # Augment training data
    X_pseudo = X_test_base[pseudo_mask]
    y_pseudo = pseudo_dead[pseudo_mask].astype(int)

    X_comb_base = pd.concat([X_train_base, X_pseudo], ignore_index=True)
    y_comb = np.concatenate([y, y_pseudo])

    # Full-train TE for combined
    comb_raw = {col: pd.concat([train_df[col], test_df[col][pseudo_mask]], ignore_index=True) for col in te_cols}
    te_comb = pd.DataFrame(index=range(len(X_comb_base)))
    for col in te_cols:
        _, vals, _ = smoothed_te(comb_raw[col], y_comb, test_df[col])
        te_test[f"{col}_te"] = vals  # update test TE
        # Inner CV TE for combined train
        inner_enc = np.zeros(len(y_comb))
        for itr, ival in StratifiedKFold(5, shuffle=True, random_state=SEED).split(comb_raw[col], y_comb):
            _, ie, _ = smoothed_te(comb_raw[col].iloc[itr], y_comb[itr], comb_raw[col].iloc[ival])
            inner_enc[ival] = ie
        te_comb[f"{col}_te"] = inner_enc

    X_comb_v4 = np.nan_to_num(np.hstack([X_comb_base.values, te_comb.values]).astype(np.float32), nan=-999.0)
    X_comb_v5 = np.nan_to_num(X_comb_base.values.astype(np.float32), nan=-999.0)
    X_test_te_updated = np.nan_to_num(pd.concat([X_test_base, te_test], axis=1).values.astype(np.float32), nan=-999.0)

    # Retrain with pseudo labels (5-fold, single pass)
    print("  Retraining with pseudo-labeled data...")
    ps_v5_lgb = np.zeros(len(test_df))
    ps_v5_xgb = np.zeros(len(test_df))
    ps_v5_cb  = np.zeros(len(test_df))
    ps_v4_lgb = np.zeros(len(test_df))
    ps_v4_xgb = np.zeros(len(test_df))
    ps_v4_cb  = np.zeros(len(test_df))

    skf = StratifiedKFold(5, shuffle=True, random_state=SEED+999)
    for fold, (tr_i, val_i) in enumerate(skf.split(X_comb_v5, y_comb)):
        # v5 (no TE)
        m = lgb.LGBMClassifier(**v5_lgb)
        m.fit(X_comb_v5[tr_i], y_comb[tr_i], eval_set=[(X_comb_v5[val_i], y_comb[val_i])],
              callbacks=[lgb.early_stopping(80,verbose=False),lgb.log_evaluation(0)])
        ps_v5_lgb += m.predict_proba(X_test_no_te)[:,1] / 5

        m = xgb.XGBClassifier(**v5_xgb, early_stopping_rounds=80)
        m.fit(X_comb_v5[tr_i], y_comb[tr_i], eval_set=[(X_comb_v5[val_i], y_comb[val_i])], verbose=False)
        ps_v5_xgb += m.predict_proba(X_test_no_te)[:,1] / 5

        m = CatBoostClassifier(**v5_cb, early_stopping_rounds=80)
        m.fit(X_comb_v5[tr_i], y_comb[tr_i], eval_set=(X_comb_v5[val_i], y_comb[val_i]), verbose=False)
        ps_v5_cb += m.predict_proba(X_test_no_te)[:,1] / 5

        # v4 (with TE)
        m = lgb.LGBMClassifier(**v4_lgb)
        m.fit(X_comb_v4[tr_i], y_comb[tr_i], eval_set=[(X_comb_v4[val_i], y_comb[val_i])],
              callbacks=[lgb.early_stopping(80,verbose=False),lgb.log_evaluation(0)])
        ps_v4_lgb += m.predict_proba(X_test_te_updated)[:,1] / 5

        m = xgb.XGBClassifier(**v4_xgb, early_stopping_rounds=80)
        m.fit(X_comb_v4[tr_i], y_comb[tr_i], eval_set=[(X_comb_v4[val_i], y_comb[val_i])], verbose=False)
        ps_v4_xgb += m.predict_proba(X_test_te_updated)[:,1] / 5

        m = CatBoostClassifier(**v4_cb, early_stopping_rounds=80)
        m.fit(X_comb_v4[tr_i], y_comb[tr_i], eval_set=(X_comb_v4[val_i], y_comb[val_i]), verbose=False)
        ps_v4_cb += m.predict_proba(X_test_te_updated)[:,1] / 5

        print(f"    Pseudo-F{fold+1} done")

    ps_v5_ens = W5_LGB*ps_v5_lgb + W5_XGB*ps_v5_xgb + W5_CB*ps_v5_cb
    ps_v4_ens = W4_LGB*ps_v4_lgb + W4_XGB*ps_v4_xgb + W4_CB*ps_v4_cb
    ps_blend = best_alpha * ps_v5_ens + (1 - best_alpha) * ps_v4_ens

    # Final ensemble: blend original + pseudo (50/50)
    final_ens = 0.5 * test_blend + 0.5 * ps_blend
    print(f"  Final = 50% original blend + 50% pseudo-labeled blend")
else:
    final_ens = test_blend
    print("  Not enough confident predictions. Using original blend only.")

# ============================================================================
# PHASE 4: Generate Submissions
# ============================================================================
print("\n>>> Phase 4: Generate Submissions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save all probabilities
np.save(OUTPUT_DIR / "probs_v6_final.npy", final_ens)
np.save(OUTPUT_DIR / "probs_v6_blend.npy", test_blend)

# Find threshold for 84.5% Dead
for th in np.arange(0.40, 0.75, 0.001):
    dpct = (final_ens >= th).mean()
    if dpct <= 0.845:
        break

# Generate main submission
preds = (final_ens >= th).astype(int)
sub = pd.DataFrame({"patient_id": test_ids, "vital_status": np.where(preds==1,"Dead","Alive")})
sub.to_csv(OUTPUT_DIR / "submission.csv", index=False)
dead_n = preds.sum()
print(f"  {OUTPUT_DIR / 'submission.csv'}: th={th:.3f} → Dead={dead_n:,} ({preds.mean():.1%})")

# Also generate blend-only (no pseudo) at 84.5%
for th2 in np.arange(0.40, 0.75, 0.001):
    if (test_blend >= th2).mean() <= 0.845:
        break
preds2 = (test_blend >= th2).astype(int)
sub2 = pd.DataFrame({"patient_id": test_ids, "vital_status": np.where(preds2==1,"Dead","Alive")})
sub2.to_csv(OUTPUT_DIR / "submission_blend.csv", index=False)
print(f"  {OUTPUT_DIR / 'submission_blend.csv'}: th={th2:.3f} → Dead={preds2.sum():,} ({preds2.mean():.1%})")

# Comparison with previous best
print(f"\n  Historical pre-v6 LB benchmark: 0.876730 at Dead%=84.5%")
print(f"  Historical v5 probabilities are retained in archive/probs_foldavg.npy")

print(f"\n{'='*80}")
print(f"  PIPELINE v6 COMPLETE")
print(f"  Rerun artifacts were isolated under {OUTPUT_DIR}; canonical submission.csv was not touched")
print(f"{'='*80}")
