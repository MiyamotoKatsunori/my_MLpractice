import pandas as pd, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib, os

# ============================
# 1. データ読み込み"CH14", "CH15", "CH7", "CH8","T_0_dTdt", "T_90_dTdt", "T_180_dTdt", "T_270_dTdt","T_0", "T_90", "T_180", "T_270",
# ============================
csv_path = r"C:\Users\katsu\OneDrive\my_practice\thesis\data\input.csv"

# 入力：温度群＋流量・電力など（Isg は除外する）
input_cols = [
    "T_0", "T_90", "T_180", "T_270",
    "T_0_dTdt", "T_90_dTdt", "T_180_dTdt", "T_270_dTdt",
     "mf", "pw"
]
phys_col = "Isg"     # 理論電流値
target_col = "I_sg"  # 実測電流
use_robust = False

df = pd.read_csv(csv_path)
df[input_cols + [phys_col, target_col]] = df[input_cols + [phys_col, target_col]].apply(pd.to_numeric, errors="coerce")
df = df.replace([np.inf, -np.inf], np.nan)
before = len(df)
df = df.dropna(subset=input_cols + [phys_col, target_col]).reset_index(drop=True)
print(f"DropNA: {before-len(df)} rows removed")

# 定数列を除外
stds = df[input_cols].std()
const_cols = stds[stds == 0].index.tolist()
if const_cols:
    print("定数列を除外:", const_cols)
    input_cols = [c for c in input_cols if c not in const_cols]

# 入出力準備
X = df[input_cols].values.astype(np.float32)
I_phys = df[phys_col].values.reshape(-1, 1).astype(np.float32)
y = df[target_col].values.reshape(-1, 1).astype(np.float32)

# 学習目標は ΔI = I_true - I_phys
y_residual = y - I_phys

# ========= スケーリング =========
if use_robust:
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
else:
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_res_scaled = scaler_y.fit_transform(y_residual)

# Tensor変換
X_t = torch.tensor(X_scaled, dtype=torch.float32)
y_t = torch.tensor(y_res_scaled, dtype=torch.float32)
I_phys_t = torch.tensor(I_phys, dtype=torch.float32)  # 評価用に保持

# ========= データ分割 =========
ds = TensorDataset(X_t, y_t, I_phys_t)
n = len(ds); n_val = int(n * 0.2)
gen = torch.Generator().manual_seed(42)
train_ds, val_ds = random_split(ds, [n - n_val, n_val], generator=gen)

train_ld = DataLoader(train_ds, batch_size=64, shuffle=True)
val_ld   = DataLoader(val_ds, batch_size=64, shuffle=False)

# ============================
# 2. MLP定義（ΔI予測モデル）
# ============================
class ResidualMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)  # 出力は ΔI（スケーリング済み）

model = ResidualMLP(X_t.shape[1])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
epochs = 50
best = float('inf'); best_state = None

# ============================
# 3. 学習ループ
# ============================
for ep in range(1, epochs + 1):
    model.train()
    tr = 0.0
    for xb, yb, _ in train_ld:
        optimizer.zero_grad(set_to_none=True)
        pred_res = model(xb)
        loss = criterion(pred_res, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr += loss.item()
    tr /= len(train_ld)

    model.eval(); va = 0.0
    with torch.no_grad():
        for xb, yb, _ in val_ld:
            pred_res = model(xb)
            va += criterion(pred_res, yb).item()
    va /= len(val_ld)

    print(f"Epoch [{ep}/{epochs}]  Train: {tr:.6f}  Val: {va:.6f}")
    if va < best:
        best = va
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)

torch.save(model.state_dict(), r"C:\Users\katsu\OneDrive\my_practice\thesis\output\hybrid_model.pth")
joblib.dump({
    "scaler_X": scaler_X, "scaler_y": scaler_y,
    "input_cols": input_cols, "phys_col": phys_col
}, r"C:\Users\katsu\OneDrive\my_practice\thesis\output\hybrid_scaler.joblib")
print("保存完了")

# ============================
# 4. 評価（スケール戻し）
# ============================
def eval_metrics(y_true, y_pred, name=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    print(f"{name} RMSE: {rmse:.6f}, MAE: {mae:.6f}")

model.eval()
with torch.no_grad():
    # 訓練・検証データでΔIを予測
    train_idx, val_idx = train_ds.indices, val_ds.indices
    ΔI_train_scaled = model(X_t[train_idx]).numpy()
    ΔI_val_scaled   = model(X_t[val_idx]).numpy()

    ΔI_train = scaler_y.inverse_transform(ΔI_train_scaled)
    ΔI_val   = scaler_y.inverse_transform(ΔI_val_scaled)

    # I_pred = I_phys + ΔI_pred
    I_pred_train = I_phys[train_idx] + ΔI_train
    I_pred_val   = I_phys[val_idx] + ΔI_val
    I_true_train = y[train_idx]
    I_true_val   = y[val_idx]

print("\n スケールを戻した評価（ハイブリッドモデル）")
eval_metrics(I_true_train, I_pred_train, "Train")
eval_metrics(I_true_val, I_pred_val, "Val")

# ============================
# 5. 外部データ評価
# ============================
def evaluate_external_data(csv_path, model, scaler_pack):
    df2 = pd.read_csv(csv_path)
    df2[scaler_pack["input_cols"] + [scaler_pack["phys_col"], target_col]] = \
        df2[scaler_pack["input_cols"] + [scaler_pack["phys_col"], target_col]].apply(pd.to_numeric, errors="coerce")
    df2 = df2.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    X2 = df2[scaler_pack["input_cols"]].values.astype(np.float32)
    I2_phys = df2[scaler_pack["phys_col"]].values.reshape(-1,1).astype(np.float32)
    y2_true = df2[target_col].values.reshape(-1,1).astype(np.float32)

    X2_scaled = scaler_pack["scaler_X"].transform(X2)
    with torch.no_grad():
        ΔI2_scaled = model(torch.tensor(X2_scaled, dtype=torch.float32)).numpy()
    ΔI2 = scaler_pack["scaler_y"].inverse_transform(ΔI2_scaled)

    y2_pred = I2_phys + ΔI2

    print(f"\n=== {os.path.basename(csv_path)} の評価結果 ===")
    eval_metrics(y2_true, y2_pred, "  Test")

scaler_pack = {"scaler_X": scaler_X, "scaler_y": scaler_y, "input_cols": input_cols, "phys_col": phys_col}
evaluate_external_data(r"C:\Users\katsu\OneDrive\my_practice\thesis\data\inrange.csv", model, scaler_pack)
evaluate_external_data(r"C:\Users\katsu\OneDrive\my_practice\thesis\data\outrange.csv", model, scaler_pack)
