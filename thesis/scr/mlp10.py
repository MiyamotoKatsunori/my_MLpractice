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
import csv

# ============================
# パス設定
# ============================
csv_path = r"C:\Users\katsu\OneDrive\my_practice\thesis\data\input.csv"
inrange_path  = r"C:\Users\katsu\OneDrive\my_practice\thesis\data\inrange.csv"
outrange_path = r"C:\Users\katsu\OneDrive\my_practice\thesis\data\outrange.csv"
out_dir = r"C:\Users\katsu\OneDrive\my_practice\thesis\best\addl5"
os.makedirs(out_dir, exist_ok=True)

# ============================
# データ列名
# ============================
input_cols = [
    "T_0", "T_90", "T_180", "T_270",
    "T_0_dTdt", "T_90_dTdt", "T_180_dTdt", "T_270_dTdt",
    "mf", "pw", "I_aps"
]
phys_col = "Isg"
target_col = "I_sg"

# ============================================
# 設定
# ============================================
config = {
    "hidden": 64,
    "num_layers": 5,
    "activation": "relu",
    "initializer": "default",
    "use_dropout": True,
    "dropout_p": 0.1,
    "use_layernorm": False,

    "opt_name": "adam",
    "lr": 5e-4,
    "weight_decay": 1e-5,

    "epochs": 60,
    "batch_size": 64,
    "num_runs": 10,
    "base_seed": 2024,
    "eval_every": 1,

    "use_robust": False,
}

epochs     = config["epochs"]
batch_size = config["batch_size"]
num_runs   = config["num_runs"]
base_seed  = config["base_seed"]
eval_every = config["eval_every"]
use_robust = config["use_robust"]

smooth_k = 2
epoch_window = 3


# ============================================
# 関数群
# ============================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true.reshape(-1,1), y_pred.reshape(-1,1)))

def smooth(series, k=2):
    s = []
    L = len(series)
    for i in range(L):
        left = max(0, i-k)
        right = min(L, i+k+1)
        s.append(np.mean(series[left:right]))
    return s

def to_numeric_df(df, cols):
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


# ============================================
# ResidualMLP
# ============================================
class ResidualMLP(nn.Module):
    def __init__(self, d, cfg):
        super().__init__()
        hidden        = cfg["hidden"]
        num_layers    = cfg["num_layers"]
        act_name      = cfg["activation"]
        use_dropout   = cfg["use_dropout"]
        dropout_p     = cfg["dropout_p"]
        use_layernorm = cfg["use_layernorm"]

        if act_name == "relu":
            act = nn.ReLU()
            kaiming_nonlin = "relu"
        elif act_name == "tanh":
            act = nn.Tanh()
            kaiming_nonlin = "tanh"
        elif act_name == "gelu":
            act = nn.GELU()
            kaiming_nonlin = "relu"
        else:
            raise ValueError(f"Unknown activation: {act_name}")

        layers = []
        layers.append(nn.Linear(d, hidden))
        if use_layernorm: layers.append(nn.LayerNorm(hidden))
        layers.append(act)

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden, hidden))
            if use_layernorm: layers.append(nn.LayerNorm(hidden))
            layers.append(act)

        if use_dropout:
            layers.append(nn.Dropout(dropout_p))

        layers.append(nn.Linear(hidden, 1))

        self.net = nn.Sequential(*layers)
        self.kaiming_nonlin = kaiming_nonlin
        self.apply(lambda m: self.init_weights(m, cfg["initializer"]))

    def init_weights(self, m, initializer):
        if isinstance(m, nn.Linear):
            if initializer == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight, nonlinearity=self.kaiming_nonlin)
            elif initializer == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight, nonlinearity=self.kaiming_nonlin)
            elif initializer == "xavier_normal":
                nn.init.xavier_normal_(m.weight)
            elif initializer == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ============================================
# Optimizer factory
# ============================================
def get_optimizer(opt_name, params, lr, weight_decay):
    opt_name = opt_name.lower()
    if opt_name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "adagrad":
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif opt_name == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


# ============================================
# データ準備
# ============================================
def prepare_main_dataset():
    df = pd.read_csv(csv_path)
    df = to_numeric_df(df, input_cols + [phys_col, target_col])
    df = df.dropna(subset=input_cols + [phys_col, target_col]).reset_index(drop=True)

    X = df[input_cols].values.astype(np.float32)
    I_phys = df[phys_col].values.reshape(-1,1).astype(np.float32)
    y_true = df[target_col].values.reshape(-1,1).astype(np.float32)
    y_resid = y_true - I_phys

    scaler_X = RobustScaler() if use_robust else StandardScaler()
    scaler_y = RobustScaler() if use_robust else StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y_resid)

    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    y_t = torch.tensor(y_scaled, dtype=torch.float32)
    I_phys_t = torch.tensor(I_phys, dtype=torch.float32)

    ds = TensorDataset(X_t, y_t, I_phys_t)
    n = len(ds)
    n_val = int(n * 0.2)

    return ds, n, n_val, scaler_X, scaler_y, input_cols, y_true, I_phys


def prepare_external_data(path, used_cols, scaler_X):
    df2 = pd.read_csv(path)
    df2 = to_numeric_df(df2, used_cols + [phys_col, target_col])
    df2 = df2.dropna().reset_index(drop=True)
    X = df2[used_cols].values.astype(np.float32)
    I_phys = df2[phys_col].values.reshape(-1,1).astype(np.float32)
    y_true = df2[target_col].values.reshape(-1,1).astype(np.float32)
    X_scaled = scaler_X.transform(X)
    return X_scaled, I_phys, y_true


# ============================================
# グローバル最良追跡
# ============================================
best_global_rmse = float("inf")
best_state, best_scalers, best_cols, best_tag = None, None, None, ""

# ============================================
# 繰り返し実験ループ
# ============================================
all_train_hist, all_val_hist = [], []
all_in_hist, all_out_hist = [], []

best_train_rmse_list = []
best_out_rmse_list = []
avg_train_rmse_list = []

for run in range(1, num_runs+1):

    seed = base_seed + run
    set_seed(seed)
    print(f"\n==== Run {run}/{num_runs} ====")

    best_rmse_out_run = float("inf")
    best_train_rmse_run = None
    best_epoch_run = None

    ds, n, n_val, scaler_X, scaler_y, used_cols, y_true, I_phys = prepare_main_dataset()
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n-n_val, n_val], generator=gen)

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ld   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    X_in_scaled, I_in_phys, y_in_true = prepare_external_data(inrange_path, used_cols, scaler_X)
    X_out_scaled, I_out_phys, y_out_true = prepare_external_data(outrange_path, used_cols, scaler_X)

    model = ResidualMLP(len(used_cols), config)
    optimizer = get_optimizer(config["opt_name"], model.parameters(), config["lr"], config["weight_decay"])
    criterion = nn.MSELoss()

    tr_hist, va_hist = [], []
    in_hist, out_hist = {}, {}

    for ep in range(1, epochs+1):

        # ---------- Train ----------
        model.train()
        tr_loss = 0.0
        for xb, yb, _ in train_ld:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(train_ld)

        # ---------- Validation ----------
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb, _ in val_ld:
                va_loss += criterion(model(xb), yb).item()
        va_loss /= len(val_ld)

        print(f"Run {run} Epoch[{ep}/{epochs}] Train:{tr_loss:.6f} Val:{va_loss:.6f}")

        # ---------- RMSE ----------
        with torch.no_grad():
            train_idx, val_idx = train_ds.indices, val_ds.indices
            ΔI_tr_scaled = model(ds.tensors[0][train_idx]).numpy()
            ΔI_va_scaled = model(ds.tensors[0][val_idx]).numpy()

        ΔI_tr = scaler_y.inverse_transform(ΔI_tr_scaled)
        ΔI_va = scaler_y.inverse_transform(ΔI_va_scaled)

        I_pred_tr = I_phys[train_idx] + ΔI_tr
        I_pred_va = I_phys[val_idx] + ΔI_va

        tr_rmse = rmse(y_true[train_idx], I_pred_tr)
        va_rmse = rmse(y_true[val_idx], I_pred_va)

        tr_hist.append(tr_rmse)
        va_hist.append(va_rmse)

        # ---------- In / Out ----------
        if ep % eval_every == 0 or ep == 1:
            with torch.no_grad():
                ΔI_in_scaled  = model(torch.tensor(X_in_scaled, dtype=torch.float32)).numpy()
                ΔI_out_scaled = model(torch.tensor(X_out_scaled, dtype=torch.float32)).numpy()

            ΔI_in  = scaler_y.inverse_transform(ΔI_in_scaled)
            ΔI_out = scaler_y.inverse_transform(ΔI_out_scaled)

            y_in_pred  = I_in_phys  + ΔI_in
            y_out_pred = I_out_phys + ΔI_out

            rmse_in  = rmse(y_in_true,  y_in_pred)
            rmse_out = rmse(y_out_true, y_out_pred)

            in_hist[ep] = rmse_in
            out_hist[ep] = rmse_out

            # --- run 内 best 更新 ---
            if rmse_out < best_rmse_out_run:
                best_rmse_out_run  = rmse_out
                best_train_rmse_run = tr_rmse
                best_epoch_run = ep
                print(f"Run {run}: New best OUTRMSE {rmse_out:.6f} at epoch {ep}")

            # --- global best 更新 ---
            if rmse_out < best_global_rmse:
                best_global_rmse = rmse_out
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_scalers = {"scaler_X": scaler_X, "scaler_y": scaler_y}
                best_cols = used_cols
                best_tag = f"run{run}_epoch{ep}"
                print(f"** Global best OUTRMSE {rmse_out:.6f} at {best_tag}")

    # best epoch ± window 平均
    if best_epoch_run is None:
        avg_train_rmse = None
    else:
        start = max(1, best_epoch_run - epoch_window)
        end   = min(epochs, best_epoch_run + epoch_window)
        vals = tr_hist[start-1 : end]
        avg_train_rmse = float(np.mean(vals))

    best_train_rmse_list.append(best_train_rmse_run)
    best_out_rmse_list.append(best_rmse_out_run)
    avg_train_rmse_list.append(avg_train_rmse)

    all_train_hist.append(tr_hist)
    all_val_hist.append(va_hist)
    all_in_hist.append(in_hist)
    all_out_hist.append(out_hist)


# ============================================
# 出力 CSV
# ============================================
df_best = pd.DataFrame({
    "run": np.arange(1, num_runs+1),
    "best_out_rmse": best_out_rmse_list,
    "best_train_rmse_at_best_out": best_train_rmse_list,
    "avg_train_rmse_around_best_out": avg_train_rmse_list
})
df_best.to_csv(os.path.join(out_dir, "best_rmse_per_run.csv"), index=False)


# ============================================
# RMSE履歴 CSV 保存 (★追加部分★)
# ============================================

def convert_dict_history(all_hist, epochs):
    arr = []
    for ep in range(1, epochs+1):
        row = []
        for run_dict in all_hist:
            row.append(run_dict.get(ep, np.nan))
        arr.append(row)
    return arr

# ---- Train / Val ----
train_arr = list(zip(*all_train_hist))
val_arr   = list(zip(*all_val_hist))

train_csv = os.path.join(out_dir, "train_rmse_history.csv")
with open(train_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch"] + list(range(num_runs)))
    for ep in range(1, epochs+1):
        writer.writerow([ep] + list(train_arr[ep-1]))

val_csv = os.path.join(out_dir, "val_rmse_history.csv")
with open(val_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch"] + list(range(num_runs)))
    for ep in range(1, epochs+1):
        writer.writerow([ep] + list(val_arr[ep-1]))

# ---- Inrange / Outrange ----
in_arr  = convert_dict_history(all_in_hist, epochs)
out_arr = convert_dict_history(all_out_hist, epochs)

in_csv = os.path.join(out_dir, "inrange_rmse_history.csv")
with open(in_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch"] + list(range(num_runs)))
    for ep in range(1, epochs+1):
        writer.writerow([ep] + list(in_arr[ep-1]))

out_csv = os.path.join(out_dir, "outrange_rmse_history.csv")
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch"] + list(range(num_runs)))
    for ep in range(1, epochs+1):
        writer.writerow([ep] + list(out_arr[ep-1]))

print("\n📁 Saved RMSE history CSVs:")
print(" -", train_csv)
print(" -", val_csv)
print(" -", in_csv)
print(" -", out_csv)


# ============================================
# グラフ出力（全 run）
# ============================================
def plot_all(series_list, label, filename):
    plt.figure(figsize=(7,4))
    for i, s in enumerate(series_list, 1):
        xs = np.arange(1, len(s)+1)
        plt.plot(xs, s, linewidth=2, label=f"Run{i}")
    plt.xlabel("Epoch"); plt.ylabel("RMSE [mA]")
    plt.title(label)
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()

plot_all(all_train_hist, "Train RMSE (All Runs)", "aggregate_train.png")
plot_all(all_val_hist,   "Val RMSE (All Runs)",   "aggregate_val.png")

plt.figure(figsize=(7,4))
for i, d in enumerate(all_in_hist, 1):
    xs = sorted(d.keys())
    ys = [d[k] for k in xs]
    plt.plot(xs, ys, linewidth=2, label=f"Run{i}")
plt.xlabel("Epoch"); plt.ylabel("RMSE [mA]")
plt.title("Inrange RMSE (All Runs)")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "aggregate_inrange.png"))
plt.close()

plt.figure(figsize=(7,4))
for i, d in enumerate(all_out_hist, 1):
    xs = sorted(d.keys())
    ys = [d[k] for k in xs]
    plt.plot(xs, ys, linewidth=2, label=f"Run{i}")
plt.xlabel("Epoch"); plt.ylabel("RMSE [mA]")
plt.title("Outrange RMSE (All Runs)")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "aggregate_outrange.png"))
plt.close()


# ============================================
# ベストモデル保存
# ============================================
if best_state:
    best_model_path  = os.path.join(out_dir, "best_outrange_model.pth")
    best_scaler_path = os.path.join(out_dir, "best_outrange_scalers.joblib")

    dummy = ResidualMLP(len(best_cols), config)
    dummy.load_state_dict(best_state)

    torch.save(dummy.state_dict(), best_model_path)
    joblib.dump(
        {
            "scaler_X": best_scalers["scaler_X"],
            "scaler_y": best_scalers["scaler_y"],
            "input_cols": best_cols,
            "phys_col": phys_col
        },
        best_scaler_path
    )

    print(f"\n✅ Saved best outrange model ({best_tag}, RMSE={best_global_rmse:.6f})")
else:
    print("\n⚠️ No best model found")

print(f"\nAll results saved to: {out_dir}")
