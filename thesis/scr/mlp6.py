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

# 設定
csv_path = r"C:\Users\katsu\OneDrive\my_practice\thesis\data\Pinput.csv"
inrange_path  = r"C:\Users\katsu\OneDrive\my_practice\thesis\data\inrange.csv"
outrange_path = r"C:\Users\katsu\OneDrive\my_practice\thesis\data\peva.csv"
out_dir = r"C:\Users\katsu\OneDrive\my_practice\thesis\output\ad_ratio"
os.makedirs(out_dir, exist_ok=True)

input_cols = [
    "T_0", "T_90", "T_180", "T_270",
    "T_0_dTdt", "T_90_dTdt", "T_180_dTdt", "T_270_dTdt",
    "mf", "pw", "I_aps"
]
phys_col = "Isg"
target_col = "I_sg"
use_robust = False

epochs = 60
eval_every = 1
batch_size = 64
lr = 5e-4
weight_decay = 1e-5
num_runs = 5
base_seed = 2025

# 関数群
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

# ===== MLP（残差ではなく比率を出力する）
class RatioMLP(nn.Module):
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

# ===== メインデータ準備（比率モデル対応）
def prepare_main_dataset():
    df = pd.read_csv(csv_path)
    df = to_numeric_df(df, input_cols + [phys_col, target_col])
    df = df.dropna().reset_index(drop=True)

    # 比率学習なので Isg=0 は除外
    df = df[df[phys_col] != 0].reset_index(drop=True)

    X = df[input_cols].values.astype(np.float32)
    I_phys = df[phys_col].values.reshape(-1,1).astype(np.float32)
    y_true = df[target_col].values.reshape(-1,1).astype(np.float32)

    # ------ 学習ターゲット：比率 ------
    y_ratio = y_true / I_phys

    scaler_X = RobustScaler() if use_robust else StandardScaler()
    scaler_y = RobustScaler() if use_robust else StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y_ratio)

    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    y_t = torch.tensor(y_scaled, dtype=torch.float32)
    I_phys_t = torch.tensor(I_phys, dtype=torch.float32)

    ds = TensorDataset(X_t, y_t, I_phys_t)
    n = len(ds); n_val = int(n * 0.2)
    return ds, n, n_val, scaler_X, scaler_y, input_cols, y_true, I_phys

# ===== 外部データも比率を再構成
def prepare_external_data(path, used_cols, scaler_X):
    df2 = pd.read_csv(path)
    df2 = to_numeric_df(df2, used_cols + [phys_col, target_col])
    df2 = df2.dropna().reset_index(drop=True)

    df2 = df2[df2[phys_col] != 0].reset_index(drop=True)

    X = df2[used_cols].values.astype(np.float32)
    I_phys = df2[phys_col].values.reshape(-1,1).astype(np.float32)
    y_true = df2[target_col].values.reshape(-1,1).astype(np.float32)
    X_scaled = scaler_X.transform(X)
    return X_scaled, I_phys, y_true

# ============================
# グローバル最良追跡
# ============================
best_global_rmse = float("inf")
best_state, best_scalers, best_cols, best_tag = None, None, None, ""

# ============================
# 全 run のログ格納
# ============================
all_train_hist, all_val_hist = [], []
all_in_hist, all_out_hist = [], []

# ============================
# 繰り返し実験ループ
# ============================
for run in range(1, num_runs+1):
    seed = base_seed + run
    set_seed(seed)
    print(f"\n==== Run {run}/{num_runs} ====")

    ds, n, n_val, scaler_X, scaler_y, used_cols, y_true, I_phys = prepare_main_dataset()
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n-n_val, n_val], generator=gen)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ld   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    X_in_scaled, I_in_phys, y_in_true = prepare_external_data(inrange_path, used_cols, scaler_X)
    X_out_scaled, I_out_phys, y_out_true = prepare_external_data(outrange_path, used_cols, scaler_X)

    model = RatioMLP(len(used_cols))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    tr_hist, va_hist = [], []
    in_hist, out_hist = {}, {}

    # ===== 学習ループ =====
    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for xb, yb, _ in train_ld:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(train_ld)

        # ===== Val loss =====
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb, _ in val_ld:
                va_loss += criterion(model(xb), yb).item()
        va_loss /= len(val_ld)
        print(f"Run {run} Epoch[{ep}/{epochs}] Train:{tr_loss:.6f} Val:{va_loss:.6f}")

        # ===== 実スケール RMSE（比率 → Isg_pred） =====
        with torch.no_grad():
            train_idx, val_idx = train_ds.indices, val_ds.indices
            ratio_tr_scaled = model(ds.tensors[0][train_idx]).numpy()
            ratio_va_scaled = model(ds.tensors[0][val_idx]).numpy()

        ratio_tr = scaler_y.inverse_transform(ratio_tr_scaled)
        ratio_va = scaler_y.inverse_transform(ratio_va_scaled)

        I_pred_tr = I_phys[train_idx] * ratio_tr
        I_pred_va = I_phys[val_idx]  * ratio_va

        tr_rmse = rmse(y_true[train_idx], I_pred_tr)
        va_rmse = rmse(y_true[val_idx], I_pred_va)

        tr_hist.append(tr_rmse)
        va_hist.append(va_rmse)

        # ===== inrange/out_range 評価 =====
        if ep % eval_every == 0 or ep == 1:
            with torch.no_grad():
                ratio_in_scaled  = model(torch.tensor(X_in_scaled, dtype=torch.float32)).numpy()
                ratio_out_scaled = model(torch.tensor(X_out_scaled, dtype=torch.float32)).numpy()

            ratio_in  = scaler_y.inverse_transform(ratio_in_scaled)
            ratio_out = scaler_y.inverse_transform(ratio_out_scaled)

            y_in_pred  = I_in_phys  * ratio_in
            y_out_pred = I_out_phys * ratio_out

            rmse_in  = rmse(y_in_true,  y_in_pred)
            rmse_out = rmse(y_out_true, y_out_pred)

            in_hist[ep]  = rmse_in
            out_hist[ep] = rmse_out

            if rmse_out < best_global_rmse:
                best_global_rmse = rmse_out
                best_state  = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
                best_scalers = {"scaler_X": scaler_X, "scaler_y": scaler_y}
                best_cols = used_cols
                best_tag = f"run{run}_epoch{ep}"
                print(f"** new best outrange RMSE {rmse_out:.6f} at {best_tag}")

    # ===== 各 run のグラフ保存 =====
    x_ep = np.arange(1, epochs+1)

    plt.figure()
    plt.plot(x_ep, tr_hist, label="Train RMSE")
    plt.plot(x_ep, va_hist, label="Val RMSE")
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"run{run}_train_val.png"))
    plt.close()

    # in/out
    x_in  = sorted(in_hist.keys())
    y_in  = [in_hist[e] for e in x_in]
    x_out = sorted(out_hist.keys())
    y_out = [out_hist[e] for e in x_out]

    plt.figure()
    plt.plot(x_in,  y_in ,label="inrange")
    plt.plot(x_out, y_out, label="outrange")
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"run{run}_in_out.png"))
    plt.close()

    # 全 run 保存
    all_train_hist.append(tr_hist)
    all_val_hist.append(va_hist)
    all_in_hist.append(in_hist)
    all_out_hist.append(out_hist)

# ============================
# 集約グラフ（全 run ）
# ============================
def plot_all(series_list, title, filename):
    plt.figure(figsize=(7,4))
    for i, s in enumerate(series_list,1):
        xs = np.arange(1, len(s)+1)
        plt.plot(xs, s, label=f"Run{i}")
    plt.grid(True); plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("RMSE [mA]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()

plot_all(all_train_hist, "Train RMSE (All Runs)", "aggregate_train.png")
plot_all(all_val_hist,   "Val RMSE (All Runs)",   "aggregate_val.png")

# inrange
plt.figure(figsize=(7,4))
for i, d in enumerate(all_in_hist, 1):
    xs = sorted(d.keys())
    ys = [d[k] for k in xs]
    plt.plot(xs, ys, label=f"Run{i}")
plt.legend(); plt.grid(True)
plt.xlabel("Epoch"); plt.ylabel("RMSE[mA]")
plt.title("Inrange RMSE (All Runs)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "aggregate_inrange.png"))
plt.close()

# outrange
plt.figure(figsize=(7,4))
for i, d in enumerate(all_out_hist, 1):
    xs = sorted(d.keys())
    ys = [d[k] for k in xs]
    plt.plot(xs, ys, label=f"Run{i}")
plt.legend(); plt.grid(True)
plt.xlabel("Epoch"); plt.ylabel("RMSE[mA]")
plt.title("Outrange RMSE (All Runs)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "aggregate_outrange.png"))
plt.close()

# ============================
# CSV 保存
# ============================
pd.DataFrame(all_train_hist).T.to_csv(os.path.join(out_dir, "train_rmse_history.csv"))
pd.DataFrame(all_val_hist).T.to_csv(os.path.join(out_dir, "val_rmse_history.csv"))

def dict_list_to_df(hist_list):
    df = pd.DataFrame()
    for i, h in enumerate(hist_list,1):
        temp = pd.DataFrame(list(h.items()), columns=["epoch", f"Run{i}"])
        temp = temp.set_index("epoch")
        df = pd.concat([df, temp], axis=1)
    return df

dict_list_to_df(all_in_hist ).to_csv(os.path.join(out_dir, "inrange_rmse_history.csv"))
dict_list_to_df(all_out_hist).to_csv(os.path.join(out_dir, "outrange_rmse_history.csv"))

# ============================
# ベストモデル保存
# ============================
if best_state:
    model_path = os.path.join(out_dir, "best_outrange_model.pth")
    scaler_path = os.path.join(out_dir, "best_outrange_scalers.joblib")
    dummy = RatioMLP(len(best_cols))
    dummy.load_state_dict(best_state)
    torch.save(dummy.state_dict(), model_path)
    joblib.dump({"scaler_X":best_scalers["scaler_X"],"scaler_y":best_scalers["scaler_y"],"input_cols":best_cols}, scaler_path)
    print(f"\nSaved BEST model at {model_path} ({best_tag})  RMSE={best_global_rmse:.6f}")
else:
    print("\nNo best model found.")

print(f"\nAll results saved to: {out_dir}")
