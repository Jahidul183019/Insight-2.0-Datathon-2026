"""
Pipeline v8: Iterative Pseudo-Labeling (Final Push)
===================================================
1. Uses the highly accurate probabilities from Submission 6 (`archive/probs_v6_final.npy`).
2. Applies an ultra-strict threshold (Dead > 98%, Alive < 5%) to extract ~14,000 completely pure test samples.
3. Retrains the 6 top-tier models (3 with Target Encoding, 3 without) on Train + Pure Pseudo.
4. Generates the final submission, calibrated strictly to 84.48% Dead.
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier

SEED = 42
np.random.seed(SEED)
SMOOTHING_M = 20.0

print("=" * 80)
print("  PIPELINE v8: Iterative Pseudo-Labeling")
print("=" * 80)

# ── Load Data ──
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
test_ids = test_df["patient_id"].values
y = (train_df["vital_status"] == "Dead").astype(int).values
print(f"  Train: {len(train_df):,} | Test: {len(test_df):,} | Dead%: {y.mean():.2%}")

# Load Sub 6 Probabilities
print("\n>>> Loading Submission 6 Probabilities...")
try:
    test_blend = np.load("archive/probs_v6_final.npy")
except:
    test_blend = np.load("probs_v6_final.npy")

# ============================================================================
# FEATURE ENGINEERING 
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
    for col_name, col_data in [("primary_site", df["primary_site"]),
                                ("histologic_type_icdo3", df["histologic_type_icdo3"])]:
        freq = col_data.astype(str).map(col_data.astype(str).value_counts(normalize=True))
        F[f"{col_name}_freq"] = freq.astype(float)
    return F

X_tr_num = build_features(train_df)
X_te_num = build_features(test_df)

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

X_test_no_te = np.nan_to_num(X_test_base.values.astype(np.float32), nan=-999.0)

# ============================================================================
# MODEL CONFIGS
# ============================================================================
# v5 Optuna params (no TE)
v5_lgb = {'n_estimators':1500,'learning_rate':0.04158,'max_depth':4,'num_leaves':53,
           'subsample':0.6706,'colsample_bytree':0.6861,'min_child_samples':37,
           'reg_alpha':1.5411,'reg_lambda':0.3901,'random_state':42,'n_jobs':-1,'verbose':-1}
v5_xgb = {'n_estimators':1500,'learning_rate':0.05443,'max_depth':5,
           'subsample':0.7332,'colsample_bytree':0.6476,'min_child_weight':7,
           'reg_alpha':0.09336,'reg_lambda':0.03991,'gamma':1.1909,
           'eval_metric':'logloss','tree_method':'hist','random_state':42,'n_jobs':-1}
v5_cb  = {'iterations':1500,'learning_rate':0.07444,'depth':7,
           'l2_leaf_reg':0.7346,'bagging_temperature':0.01295,'random_strength':1.9376,
           'eval_metric':'Logloss','random_seed':42,'verbose':False}

# v4 Optuna params (with TE)
v4_lgb = {'n_estimators':1500,'learning_rate':0.02261,'max_depth':5,'num_leaves':33,
           'subsample':0.7813,'colsample_bytree':0.6555,'min_child_samples':24,
           'reg_alpha':0.8071,'reg_lambda':0.3365,'random_state':42,'n_jobs':-1,'verbose':-1}
v4_xgb = {'n_estimators':1500,'learning_rate':0.02261,'max_depth':5,
           'subsample':0.7113,'colsample_bytree':0.6301,'min_child_weight':9,
           'reg_alpha':0.04211,'reg_lambda':1.6755,'gamma':0.7135,
           'eval_metric':'logloss','tree_method':'hist','random_state':42,'n_jobs':-1}
v4_cb  = {'iterations':1500,'learning_rate':0.02483,'depth':5,
           'l2_leaf_reg':3.674,'bagging_temperature':1.803,'random_strength':2.124,
           'eval_metric':'Logloss','random_seed':42,'verbose':False}

# weights
W5_LGB, W5_XGB, W5_CB = 0.30, 0.15, 0.55
W4_LGB, W4_XGB, W4_CB = 0.35, 0.20, 0.45
best_alpha = 0.5

# ============================================================================
# PHASE 1: Iterative Pseudo-labeling (Ultra Strict Threshold)
# ============================================================================
print("\n>>> Pseudo-Labeling with Ultra-Strict Threshold")

CONF_DEAD = 0.98   # Stricter than Sub 6 (which was 0.95)
CONF_ALIVE = 0.05  # Keep Alive generous because of class imbalance
pseudo_dead = test_blend >= CONF_DEAD
pseudo_alive = test_blend <= CONF_ALIVE
pseudo_mask = pseudo_dead | pseudo_alive

n_pd, n_pa = pseudo_dead.sum(), pseudo_alive.sum()
print(f"  Confident Dead  (≥{CONF_DEAD}): {n_pd:,}")
print(f"  Confident Alive (≤{CONF_ALIVE}): {n_pa:,}")
print(f"  Total Pseudo: {n_pd+n_pa:,} ({(n_pd+n_pa)/len(test_df):.1%} of test)")

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

# Retrain with pseudo labels (5-fold)
print("\n  Retraining 6 models on augmented dataset...")
ps_v5_lgb = np.zeros(len(test_df))
ps_v5_xgb = np.zeros(len(test_df))
ps_v5_cb  = np.zeros(len(test_df))
ps_v4_lgb = np.zeros(len(test_df))
ps_v4_xgb = np.zeros(len(test_df))
ps_v4_cb  = np.zeros(len(test_df))

N_FOLDS_FINAL = 5
skf = StratifiedKFold(N_FOLDS_FINAL, shuffle=True, random_state=SEED+999)

for fold, (tr_i, val_i) in enumerate(skf.split(X_comb_v5, y_comb)):
    print(f"\n  -- Fold {fold+1}/{N_FOLDS_FINAL} --")
    
    # v5 (no TE)
    print("    Training v5 LightGBM...")
    m = lgb.LGBMClassifier(**v5_lgb)
    m.fit(X_comb_v5[tr_i], y_comb[tr_i], eval_set=[(X_comb_v5[val_i], y_comb[val_i])],
          callbacks=[lgb.early_stopping(80,verbose=False),lgb.log_evaluation(0)])
    ps_v5_lgb += m.predict_proba(X_test_no_te)[:,1] / N_FOLDS_FINAL

    print("    Training v5 XGBoost...")
    m = xgb.XGBClassifier(**v5_xgb, early_stopping_rounds=80)
    m.fit(X_comb_v5[tr_i], y_comb[tr_i], eval_set=[(X_comb_v5[val_i], y_comb[val_i])], verbose=False)
    ps_v5_xgb += m.predict_proba(X_test_no_te)[:,1] / N_FOLDS_FINAL

    print("    Training v5 CatBoost...")
    m = CatBoostClassifier(**v5_cb, early_stopping_rounds=80)
    m.fit(X_comb_v5[tr_i], y_comb[tr_i], eval_set=(X_comb_v5[val_i], y_comb[val_i]), verbose=False)
    ps_v5_cb += m.predict_proba(X_test_no_te)[:,1] / N_FOLDS_FINAL

    # v4 (with TE)
    print("    Training v4 LightGBM...")
    m = lgb.LGBMClassifier(**v4_lgb)
    m.fit(X_comb_v4[tr_i], y_comb[tr_i], eval_set=[(X_comb_v4[val_i], y_comb[val_i])],
          callbacks=[lgb.early_stopping(80,verbose=False),lgb.log_evaluation(0)])
    ps_v4_lgb += m.predict_proba(X_test_te_updated)[:,1] / N_FOLDS_FINAL

    print("    Training v4 XGBoost...")
    m = xgb.XGBClassifier(**v4_xgb, early_stopping_rounds=80)
    m.fit(X_comb_v4[tr_i], y_comb[tr_i], eval_set=[(X_comb_v4[val_i], y_comb[val_i])], verbose=False)
    ps_v4_xgb += m.predict_proba(X_test_te_updated)[:,1] / N_FOLDS_FINAL

    print("    Training v4 CatBoost...")
    m = CatBoostClassifier(**v4_cb, early_stopping_rounds=80)
    m.fit(X_comb_v4[tr_i], y_comb[tr_i], eval_set=(X_comb_v4[val_i], y_comb[val_i]), verbose=False)
    ps_v4_cb += m.predict_proba(X_test_te_updated)[:,1] / N_FOLDS_FINAL

ps_v5_ens = W5_LGB*ps_v5_lgb + W5_XGB*ps_v5_xgb + W5_CB*ps_v5_cb
ps_v4_ens = W4_LGB*ps_v4_lgb + W4_XGB*ps_v4_xgb + W4_CB*ps_v4_cb
ps_blend = best_alpha * ps_v5_ens + (1 - best_alpha) * ps_v4_ens

# Final ensemble: blend original Sub 6 probabilities + new ultra-pure pseudo probabilities
# We blend 50/50 because test_blend is already extremely good (0.877), we just want the regularization of v8
final_ens = 0.5 * test_blend + 0.5 * ps_blend

# ============================================================================
# PHASE 2: Generate Final Submissions
# ============================================================================
print("\n>>> Phase 2: Generating Final Submission (84.48% Rule)")

# Find threshold for 84.48% Dead (the golden ratio)
best_th = 0.5
for th in np.arange(0.40, 0.75, 0.001):
    dpct = (final_ens >= th).mean()
    if dpct <= 0.8449: # Target 84.48%
        best_th = th
        break

preds = (final_ens >= best_th).astype(int)
sub = pd.DataFrame({"patient_id": test_ids, "vital_status": np.where(preds==1,"Dead","Alive")})
sub.to_csv("submission.csv", index=False)
dead_n = preds.sum()
print(f"  submission.csv: th={best_th:.3f} → Dead={dead_n:,} ({preds.mean():.2%})")

print(f"\n{'='*80}")
print(f"  PIPELINE v8 COMPLETE")
print(f"  Final submission generated as submission.csv")
print(f"{'='*80}")
