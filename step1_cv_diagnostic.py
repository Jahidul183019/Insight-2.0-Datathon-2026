import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier

SEED = 42
np.random.seed(SEED)
SMOOTHING_M = 20.0

print("=" * 80)
print("  STEP 1: CV-Threshold Diagnostic")
print("=" * 80)

# ── Load Data ──
train_df = pd.read_csv("train.csv")
y = (train_df["vital_status"] == "Dead").astype(int).values

# ============================================================================
# FEATURE ENGINEERING (From v6)
# ============================================================================
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

print("  Building features...")
X_tr_num = build_features(train_df)

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

X_train_base = pd.concat([X_tr_num, tr_cat], axis=1)

te_cols = ["histologic_type_icdo3","primary_site","diagnostic_confirmation",
           "reason_nocancer_directed_surgery","rx_summ_surgprim_site19982022"]

def smoothed_te(tr_s, target, te_s, m=SMOOTHING_M):
    prior = target.mean()
    df = pd.DataFrame({"c": tr_s.astype(str), "y": target})
    stats = df.groupby("c")["y"].agg(["count","mean"])
    encoded = (stats["count"]*stats["mean"] + m*prior) / (stats["count"]+m)
    return encoded, te_s.astype(str).map(encoded).fillna(prior).values, prior

# Configs
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

W5_LGB, W5_XGB, W5_CB = 0.30, 0.15, 0.55
W4_LGB, W4_XGB, W4_CB = 0.35, 0.20, 0.45

N_REPS = 3
N_FOLDS = 5

oof_v5 = np.zeros(len(y))
oof_v4 = np.zeros(len(y))
oof_counts = np.zeros(len(y))

# To store fold data for threshold diagnostic
fold_data = []

print("  Training 15 folds (3x5)...")
for rep in range(N_REPS):
    seed_r = SEED + rep * 111
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed_r)
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_base, y)):
        
        # v5 (No TE)
        X_tr_v5 = np.nan_to_num(X_train_base.iloc[tr_idx].values.astype(np.float32), nan=-999.0)
        X_val_v5 = np.nan_to_num(X_train_base.iloc[val_idx].values.astype(np.float32), nan=-999.0)

        m = lgb.LGBMClassifier(**v5_lgb)
        m.fit(X_tr_v5, y[tr_idx], eval_set=[(X_val_v5, y[val_idx])], callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        vp5_lgb = m.predict_proba(X_val_v5)[:, 1]

        m = xgb.XGBClassifier(**v5_xgb, early_stopping_rounds=80)
        m.fit(X_tr_v5, y[tr_idx], eval_set=[(X_val_v5, y[val_idx])], verbose=False)
        vp5_xgb = m.predict_proba(X_val_v5)[:, 1]

        m = CatBoostClassifier(**v5_cb, early_stopping_rounds=80)
        m.fit(X_tr_v5, y[tr_idx], eval_set=(X_val_v5, y[val_idx]), verbose=False)
        vp5_cb = m.predict_proba(X_val_v5)[:, 1]

        v5_val = W5_LGB*vp5_lgb + W5_XGB*vp5_xgb + W5_CB*vp5_cb
        
        # v4 (With TE)
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
        m.fit(X_tr_v4, y[tr_idx], eval_set=[(X_val_v4, y[val_idx])], callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        vp4_lgb = m.predict_proba(X_val_v4)[:, 1]

        m = xgb.XGBClassifier(**v4_xgb, early_stopping_rounds=80)
        m.fit(X_tr_v4, y[tr_idx], eval_set=[(X_val_v4, y[val_idx])], verbose=False)
        vp4_xgb = m.predict_proba(X_val_v4)[:, 1]

        m = CatBoostClassifier(**v4_cb, early_stopping_rounds=80)
        m.fit(X_tr_v4, y[tr_idx], eval_set=(X_val_v4, y[val_idx]), verbose=False)
        vp4_cb = m.predict_proba(X_val_v4)[:, 1]

        v4_val = W4_LGB*vp4_lgb + W4_XGB*vp4_xgb + W4_CB*vp4_cb

        # Save fold info
        fold_data.append({
            'rep': rep,
            'fold': fold,
            'val_idx': val_idx.copy(),
            'y_val': y[val_idx].copy(),
            'v5_val': v5_val.copy(),
            'v4_val': v4_val.copy()
        })
        np.savez_compressed(
            "cv_fold_checkpoint.npz",
            rep=np.array([d["rep"] for d in fold_data]),
            fold=np.array([d["fold"] for d in fold_data]),
            val_idx=np.array([d["val_idx"] for d in fold_data], dtype=object),
            y_val=np.array([d["y_val"] for d in fold_data], dtype=object),
            v5_val=np.array([d["v5_val"] for d in fold_data], dtype=object),
            v4_val=np.array([d["v4_val"] for d in fold_data], dtype=object),
            allow_pickle=True,
        )
        
        oof_v5[val_idx] += v5_val
        oof_v4[val_idx] += v4_val
        oof_counts[val_idx] += 1
        print(f"    R{rep+1}-F{fold+1} done")

oof_v5 /= oof_counts
oof_v4 /= oof_counts

# Find best global alpha
best_alpha, best_blend_f1 = 0.5, 0.0
for alpha in np.arange(0.0, 1.01, 0.05):
    blend = alpha * oof_v5 + (1 - alpha) * oof_v4
    f = f1_score(y, (blend >= 0.5).astype(int), pos_label=1)
    if f > best_blend_f1:
        best_blend_f1 = f
        best_alpha = round(alpha, 2)

print(f"\n  Global Best blend: {best_alpha:.0%} v5 + {1-best_alpha:.0%} v4")

# Calculate optimal threshold PER FOLD
print("\n>>> Per-Fold Optimal Threshold Analysis")
optimal_thresholds = []

for idx, data in enumerate(fold_data):
    val_y = data['y_val']
    val_blend = best_alpha * data['v5_val'] + (1 - best_alpha) * data['v4_val']
    
    best_f1 = 0
    best_th = 0.5
    for th in np.arange(0.3, 0.7, 0.005):
        f1 = f1_score(val_y, (val_blend >= th).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            
    optimal_thresholds.append(best_th)
    print(f"  R{data['rep']+1}-F{data['fold']+1}: Opt Th = {best_th:.3f} | F1 = {best_f1:.5f}")

t = np.array(optimal_thresholds)
print("\n=== THRESHOLD DIAGNOSTIC SUMMARY ===")
print(f"Min:  {t.min():.3f}")
print(f"Max:  {t.max():.3f}")
print(f"Mean: {t.mean():.3f}")
print(f"Std:  {t.std():.4f}")

if t.std() > 0.05:
    print("CONCLUSION: HIGH VARIANCE (Unstable calibration). The 84.5% rule is safer.")
else:
    print("CONCLUSION: LOW VARIANCE (Stable calibration). The model is reliable.")

# Save for Step 2
oof_blend = best_alpha * oof_v5 + (1 - best_alpha) * oof_v4
np.save("oof_step1.npy", oof_blend)
np.save("y_step1.npy", y)
print("Saved OOF arrays for Step 2.")
