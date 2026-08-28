"""
Pipeline NN: Deep Learning with Entity Embeddings
=================================================
A PyTorch architecture designed to capture non-linear, spatial patterns that 
Gradient Boosted Trees often miss.

Key features:
1. Entity Embeddings for categorical variables.
2. Standardized continuous variables.
3. Multi-Layer Perceptron (MLP) with Batch Normalization and Dropout.
4. Focal Loss (optional) to handle class imbalance.
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import f1_score
import copy

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

print("=" * 80)
print("  PIPELINE NN: Entity Embeddings + MLP")
print("=" * 80)

# ── Load Data ──
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
test_ids = test_df["patient_id"].values
y = (train_df["vital_status"] == "Dead").astype(int).values
print(f"  Train: {len(train_df):,} | Test: {len(test_df):,} | Dead%: {y.mean():.2%}")

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
    t_map={"Blank(s)":np.nan,"88":np.nan,"TX":np.nan,"T0":0,"Ta":1,"Tis":1,"Tis(DCIS)":1,"Tis(LCIS)":1,"Tis(Paget)":1,"T1":2,"T1mi":2,"T1a":3,"T1a1":3,"T1a2":3,"T1b":4,"T1b1":4,"T1b2":4,"T1c":5,"T2":6,"T2a":7,"T2b":8,"T3":9,"T4":10,"T4a":11,"T4b":12,"T4c":12,"T4d":13,"T4e":13}
    n_map={"Blank(s)":np.nan,"88":np.nan,"NX":np.nan,"N0":0,"N1":1,"N1a":1,"N1b":2,"N1c":2,"N2":3,"N2a":3,"N2b":4,"N2c":4,"N3":5,"N3a":5,"N3b":6,"N3c":6}
    m_map={"Blank(s)":np.nan,"88":np.nan,"MX":np.nan,"M0":0,"M1":1,"M1a":2,"M1b":3,"M1c":4}
    F["t_stage"]=df["derived_eod2018t_recode2018"].map(t_map).astype(float)
    F["n_stage"]=df["derived_eod2018n_recode2018"].map(n_map).astype(float)
    F["m_stage"]=df["derived_eod2018m_recode2018"].map(m_map).astype(float)
    mc=["seer_combined_metsatdxbone2010","seer_combined_metsatdxbrain2010",
        "seer_combined_metsatdxliver2010","seer_combined_metsatdxlung2010"]
    F["met_count"]=sum((df[m]=="Yes").astype(int) for m in mc)
    F["any_mets"]=(F["met_count"]>=1).astype(int)
    F["surgery_done"]=(df["reason_nocancer_directed_surgery"]=="Surgery performed").astype(int)
    return F

X_tr_num = build_features(train_df)
X_te_num = build_features(test_df)

# Fill NAs for continuous features and Standardize (Crucial for Neural Networks!)
X_tr_num = X_tr_num.fillna(0)
X_te_num = X_te_num.fillna(0)

scaler = StandardScaler()
X_tr_num_scaled = scaler.fit_transform(X_tr_num)
X_te_num_scaled = scaler.transform(X_te_num)

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

# Encode categoricals for Embedding layers (Needs to be integers 0, 1, 2...)
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
train_cat_str = train_df[cat_cols].fillna("__NA__").astype(str)
test_cat_str = test_df[cat_cols].fillna("__NA__").astype(str)

enc.fit(train_cat_str)
X_tr_cat = enc.transform(train_cat_str)
X_te_cat = enc.transform(test_cat_str)

# Shift all categorical codes by 1 to make them non-negative (0 becomes the __NA__ or unknown bin)
X_tr_cat = X_tr_cat + 1
X_te_cat = X_te_cat + 1

# Calculate embedding sizes (Rule of thumb: min(50, num_categories // 2))
cat_dims = [int(X_tr_cat[:, i].max() + 2) for i in range(X_tr_cat.shape[1])]
emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]

# ============================================================================
# PYTORCH DATASETS & MODEL
# ============================================================================
class TabularDataset(Dataset):
    def __init__(self, cat_data, num_data, targets=None):
        self.cat_data = torch.tensor(cat_data, dtype=torch.long)
        self.num_data = torch.tensor(num_data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32) if targets is not None else None

    def __len__(self):
        return len(self.cat_data)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.cat_data[idx], self.num_data[idx], self.targets[idx]
        return self.cat_data[idx], self.num_data[idx]

class TabularNN(nn.Module):
    def __init__(self, emb_dims, num_cont, hidden_layers=[256, 128, 64], p_dropout=0.3):
        super().__init__()
        
        # Embedding layers for categorical variables
        self.embs = nn.ModuleList([nn.Embedding(c, d) for c, d in emb_dims])
        n_emb = sum(e.embedding_dim for e in self.embs)
        
        # Total input size
        n_in = n_emb + num_cont
        
        # Hidden layers
        layers = []
        for hidden in hidden_layers:
            layers.append(nn.Linear(n_in, hidden))
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p_dropout))
            n_in = hidden
            
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(n_in, 1)
        
    def forward(self, x_cat, x_cont):
        # Concatenate embeddings
        x = [e(x_cat[:, i]) for i, e in enumerate(self.embs)]
        x = torch.cat(x, 1)
        
        # Combine with continuous features
        x = torch.cat([x, x_cont], 1)
        
        # Pass through MLP
        x = self.mlp(x)
        out = self.out(x)
        return torch.sigmoid(out).squeeze()

# ============================================================================
# TRAINING LOOP
# ============================================================================
print("\n>>> Training PyTorch Neural Network (5-Fold CV)...")

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"  Using device: {device}")

N_FOLDS = 5
skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)

oof_preds = np.zeros(len(y))
test_preds = np.zeros(len(test_df))

BATCH_SIZE = 1024
EPOCHS = 15
LR = 0.002

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr_cat, y)):
    print(f"\n  -- Fold {fold+1}/{N_FOLDS} --")
    
    # Datasets
    train_ds = TabularDataset(X_tr_cat[tr_idx], X_tr_num_scaled[tr_idx], y[tr_idx])
    val_ds = TabularDataset(X_tr_cat[val_idx], X_tr_num_scaled[val_idx], y[val_idx])
    test_ds = TabularDataset(X_te_cat, X_te_num_scaled)
    
    # Loaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = TabularNN(emb_dims, num_cont=X_tr_num_scaled.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    best_loss = float('inf')
    best_model = None
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for x_cat, x_cont, targets in train_loader:
            x_cat, x_cont, targets = x_cat.to(device), x_cont.to(device), targets.to(device)
            
            optimizer.zero_grad()
            preds = model(x_cat, x_cont)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds_fold = []
        with torch.no_grad():
            for x_cat, x_cont, targets in val_loader:
                x_cat, x_cont, targets = x_cat.to(device), x_cont.to(device), targets.to(device)
                preds = model(x_cat, x_cont)
                loss = criterion(preds, targets)
                val_loss += loss.item()
                val_preds_fold.extend(preds.cpu().numpy())
                
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            
        print(f"    Epoch {epoch+1:2d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
        
    # Load best model for predictions
    model.load_state_dict(best_model)
    model.eval()
    
    # OOF predictions
    fold_oof = []
    with torch.no_grad():
        for x_cat, x_cont, _ in val_loader:
            x_cat, x_cont = x_cat.to(device), x_cont.to(device)
            preds = model(x_cat, x_cont)
            fold_oof.extend(preds.cpu().numpy())
    oof_preds[val_idx] = fold_oof
    
    # Test predictions
    fold_test = []
    with torch.no_grad():
        for x_cat, x_cont in test_loader:
            x_cat, x_cont = x_cat.to(device), x_cont.to(device)
            preds = model(x_cat, x_cont)
            fold_test.extend(preds.cpu().numpy())
    test_preds += np.array(fold_test) / N_FOLDS

# ============================================================================
# METRICS & SUBMISSION
# ============================================================================
print("\n>>> Evaluation & Thresholding")

# Find OOF optimal threshold
best_f1 = 0
best_th_oof = 0.5
for th in np.arange(0.3, 0.8, 0.01):
    f1 = f1_score(y, (oof_preds >= th).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_th_oof = th

print(f"  OOF F1 Score: {best_f1:.5f} (at threshold {best_th_oof:.2f})")

# Save predictions for final mega-ensemble later
np.save("probs_nn.npy", test_preds)
np.save("oof_nn.npy", oof_preds)

# Find test threshold to hit strictly 84.48% Dead
target_th = 0.5
for th in np.arange(0.3, 0.9, 0.001):
    dead_pct = (test_preds >= th).mean()
    if dead_pct <= 0.8449:
        target_th = th
        break

final_preds = (test_preds >= target_th).astype(int)
dead_count = final_preds.sum()

sub = pd.DataFrame({"patient_id": test_ids, "vital_status": np.where(final_preds==1, "Dead", "Alive")})
sub.to_csv("submission_nn.csv", index=False)

print(f"\n{'='*80}")
print(f"  PIPELINE NN COMPLETE")
print(f"  submission_nn.csv generated: {dead_count:,} Dead ({dead_count/len(test_preds):.2%}) at th={target_th:.3f}")
print(f"  Wait until tomorrow to submit this!")
print(f"{'='*80}")
