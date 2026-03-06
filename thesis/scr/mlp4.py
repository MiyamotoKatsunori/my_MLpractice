import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================
# 設定"Isg","mf", "pw", "CH7", "CH8","CH14","CH15","T_0", "T_90", "T_180", "T_270","T_0_dTdt", "T_90_dTdt", "T_180_dTdt", "T_270_dTdt",
# ============================
csv_path = r"C:\Users\katsu\OneDrive\my_practice\thesis\data\Pinput.csv"
inrange_path  = r"C:\Users\katsu\OneDrive\my_practice\thesis\data\inrange.csv"
outrange_path = r"C:\Users\katsu\OneDrive\my_practice\thesis\data\peva.csv"
out_dir = r"C:\Users\katsu\OneDrive\my_practice\thesis\output\ad_noch"
os.makedirs(out_dir, exist_ok=True)

input_cols = [
    "Isg","mf", "pw",
    "T_0", "T_90", "T_180", "T_270","T_0_dTdt", "T_90_dTdt", "T_180_dTdt", "T_270_dTdt",
    "I_aps"
]

target_col = "I_sg"
use_robust = False

epochs = 60
eval_every = 1
batch_size = 64
lr = 5e-4
weight_decay = 1e-5
num_runs = 5
base_seed = 2010
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true.reshape(-1,1), y_pred.reshape(-1,1)))

def to_numeric_df(df, cols):
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

# ====== MLPモデル ======
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

def prepare_main_dataset():
    df = pd.read_csv(csv_path)
    df = to_numeric_df(df, input_cols + [target_col])
    df = df.dropna(subset=input_cols + [target_col]).reset_index(drop=True)
    X = df[input_cols].values.astype(np.float32)
    y_true = df[target_col].values.reshape(-1,1).astype(np.float32)
    scaler_X = RobustScaler() if use_robust else StandardScaler()
    scaler_y = RobustScaler() if use_robust else StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y_true)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    y_t = torch.tensor(y_scaled, dtype=torch.float32)
    ds = TensorDataset(X_t, y_t)
    n = len(ds); n_val = int(n * 0.2)
    return ds, n, n_val, scaler_X, scaler_y, input_cols, y_true

def prepare_external_data(path, used_cols, scaler_X, scaler_y):
    df2 = pd.read_csv(path)
    df2 = to_numeric_df(df2, used_cols + [target_col])
    df2 = df2.dropna().reset_index(drop=True)
    X = df2[used_cols].values.astype(np.float32)
    y_true = df2[target_col].values.reshape(-1,1).astype(np.float32)
    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y_true)
    return X_scaled, y_true, y_scaled

# ============================
# グローバル最良追跡
# ============================
best_global_rmse = float("inf")
best_state, best_scalers, best_cols, best_tag = None, None, None, ""

# ============================
# 繰り返し実験ループ
# ============================
all_train_hist, all_val_hist = [], []
all_in_hist, all_out_hist = [], []

for run in range(1, num_runs+1):
    seed = base_seed + run
    set_seed(seed)
    print(f"\n==== Run {run}/{num_runs} ====")

    ds, n, n_val, scaler_X, scaler_y, used_cols, y_true = prepare_main_dataset()
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n-n_val, n_val], generator=gen)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ld   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    X_in_scaled, y_in_true, _  = prepare_external_data(inrange_path, used_cols, scaler_X, scaler_y)
    X_out_scaled, y_out_true, _ = prepare_external_data(outrange_path, used_cols, scaler_X, scaler_y)

    model = MLP(len(used_cols))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    tr_hist, va_hist = [], []
    in_hist, out_hist = {}, {}

    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_ld:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(train_ld)

        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_ld:
                va_loss += criterion(model(xb), yb).item()
        va_loss /= len(val_ld)
        print(f"Run {run} Epoch[{ep}/{epochs}] Train:{tr_loss:.6f} Val:{va_loss:.6f}")

        with torch.no_grad():
            train_idx, val_idx = train_ds.indices, val_ds.indices
            y_tr_scaled = model(ds.tensors[0][train_idx]).numpy()
            y_va_scaled = model(ds.tensors[0][val_idx]).numpy()
        y_tr = scaler_y.inverse_transform(y_tr_scaled)
        y_va = scaler_y.inverse_transform(y_va_scaled)
        y_tr_true = scaler_y.inverse_transform(ds.tensors[1][train_idx].numpy())
        y_va_true = scaler_y.inverse_transform(ds.tensors[1][val_idx].numpy())
        tr_hist.append(rmse(y_tr_true, y_tr))
        va_hist.append(rmse(y_va_true, y_va))

        if ep % eval_every == 0 or ep == 1:
            with torch.no_grad():
                y_in_scaled  = model(torch.tensor(X_in_scaled, dtype=torch.float32)).numpy()
                y_out_scaled = model(torch.tensor(X_out_scaled, dtype=torch.float32)).numpy()
            y_in  = scaler_y.inverse_transform(y_in_scaled)
            y_out = scaler_y.inverse_transform(y_out_scaled)
            rmse_in  = rmse(y_in_true,  y_in)
            rmse_out = rmse(y_out_true, y_out)
            in_hist[ep]  = rmse_in
            out_hist[ep] = rmse_out
            if rmse_out < best_global_rmse:
                best_global_rmse = rmse_out
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                best_scalers = {"scaler_X": scaler_X, "scaler_y": scaler_y}
                best_cols = used_cols
                best_tag = f"run{run}_epoch{ep}"
                print(f"** New best outrange RMSE {rmse_out:.6f} at {best_tag}")

    all_train_hist.append(tr_hist)
    all_val_hist.append(va_hist)
    all_in_hist.append(in_hist)
    all_out_hist.append(out_hist)

# ============================
# 各履歴をCSV出力
# ============================
pd.DataFrame(all_train_hist).T.to_csv(os.path.join(out_dir, "train_rmse_history.csv"), index_label="epoch")
pd.DataFrame(all_val_hist).T.to_csv(os.path.join(out_dir, "val_rmse_history.csv"), index_label="epoch")

def dict_list_to_df(hist_list):
    df = pd.DataFrame()
    for i, h in enumerate(hist_list, 1):
        temp = pd.DataFrame(list(h.items()), columns=["epoch", f"Run{i}"]).set_index("epoch")
        df = pd.concat([df, temp], axis=1)
    return df

dict_list_to_df(all_in_hist).to_csv(os.path.join(out_dir, "inrange_rmse_history.csv"))
dict_list_to_df(all_out_hist).to_csv(os.path.join(out_dir, "outrange_rmse_history.csv"))

# ============================
# 集約（All Runs）グラフ出力
# ============================
def plot_all(series_list, label, filename):
    plt.figure(figsize=(7,4))
    for i, s in enumerate(series_list,1):
        plt.plot(np.arange(1, len(s)+1), s, label=f"Run{i}", linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("RMSE [mA]")
    plt.title(label)
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()

plot_all(all_train_hist, "Train RMSE (All Runs)", "aggregate_train.png")
plot_all(all_val_hist,   "Val RMSE (All Runs)",   "aggregate_val.png")

plt.figure(figsize=(7,4))
for i,d in enumerate(all_in_hist,1):
    xs = sorted(d.keys()); ys = [d[k] for k in xs]
    plt.plot(xs, ys,  label=f"Run{i}")
plt.xlabel("Epoch"); plt.ylabel("Inrange RMSE [mA]")
plt.title("Inrange RMSE (All Runs)")
plt.grid(True); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "aggregate_inrange.png"))
plt.close()

plt.figure(figsize=(7,4))
for i,d in enumerate(all_out_hist,1):
    xs = sorted(d.keys()); ys = [d[k] for k in xs]
    plt.plot(xs, ys, label=f"Run{i}")
plt.xlabel("Epoch"); plt.ylabel("Outrange RMSE [mA]")
plt.title("Outrange RMSE (All Runs)")
plt.grid(True); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "aggregate_outrange.png"))
plt.close()

# ============================
# ベストモデル保存
# ============================
if best_state:
    best_model_path = os.path.join(out_dir, "best_outrange_model.pth")
    best_scaler_path = os.path.join(out_dir, "best_outrange_scalers.joblib")
    dummy = MLP(len(best_cols))
    dummy.load_state_dict(best_state)
    torch.save(dummy.state_dict(), best_model_path)
    joblib.dump({"scaler_X": best_scalers["scaler_X"], "scaler_y": best_scalers["scaler_y"], "input_cols": best_cols}, best_scaler_path)
    print(f"\n✅ Saved best outrange model ({best_tag}, RMSE={best_global_rmse:.6f})")
else:
    print("\n⚠️ No best model found")

print(f"\nAll results saved to: {out_dir}")