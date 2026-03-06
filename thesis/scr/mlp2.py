import pandas as pd, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os

# ============================
# 1. データ読み込み"CH14", "CH15", "CH7", "CH8", 
# ============================
csv_path=r"C:\Users\katsu\OneDrive\my_practice\thesis\data\input.csv"

input_cols = [
     "T_0", "T_90", "T_180", "T_270", "T_0_dTdt", "T_90_dTdt", "T_180_dTdt", "T_270_dTdt","Isg","mf","pw","I_aps"
]
target_col = "I_sg"
use_robust = False

df = pd.read_csv(csv_path)
df[input_cols + [target_col]] = df[input_cols + [target_col]].apply(pd.to_numeric, errors="coerce")
df = df.replace([np.inf, -np.inf], np.nan)
before = len(df)
df = df.dropna(subset=input_cols + [target_col]).reset_index(drop=True)
print(f"DropNA: {before-len(df)} rows removed")

stds = df[input_cols].std()
const_cols = stds[stds == 0].index.tolist()
if const_cols:
    print("定数列を除外:", const_cols)
    input_cols = [c for c in input_cols if c not in const_cols]

X = df[input_cols].values.astype(np.float32)
y = df[target_col].values.reshape(-1,1).astype(np.float32)

if use_robust:
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
else:
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)

if np.isclose(y.std(), 0.0):
    print("y が定数です。出力のスケーリングをスキップします。")
    y_scaled = y.copy()
    scaler_y = None
else:
    y_scaled = scaler_y.fit_transform(y)

def _chk(a, name):
    nan = np.isnan(a).sum(); inf = np.isinf(a).sum()
    if nan or inf:
        raise ValueError(f"{name} に NaN/inf が含まれています: nan={nan}, inf={inf}")
_chk(X_scaled, "X_scaled")
_chk(y_scaled, "y_scaled")

X_t = torch.tensor(X_scaled, dtype=torch.float32)
y_t = torch.tensor(y_scaled, dtype=torch.float32)

ds = TensorDataset(X_t, y_t)
n = len(ds); n_val = int(n*0.2)
gen = torch.Generator().manual_seed(42)
train_ds, val_ds = random_split(ds, [n-n_val, n_val], generator=gen)

train_ld = DataLoader(train_ds, batch_size=64, shuffle=True)
val_ld   = DataLoader(val_ds, batch_size=64, shuffle=False)

# ============================
# 2. MLPモデル定義
# ============================
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

model = MLP(X_t.shape[1])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
epochs = 50
best = float('inf'); best_state = None

def clip_grads(model, max_norm=1.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

# ============================
# 3. 学習ループ
# ============================
for ep in range(1, epochs+1):
    model.train()
    tr = 0.0
    for xb, yb in train_ld:
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        clip_grads(model, 1.0)
        optimizer.step()
        tr += loss.item()
    tr /= len(train_ld)

    model.eval(); va = 0.0
    with torch.no_grad():
        for xb, yb in val_ld:
            pred = model(xb)
            va += criterion(pred, yb).item()
    va /= len(val_ld)

    print(f"Epoch [{ep}/{epochs}]  Train: {tr:.6f}  Val: {va:.6f}")
    if va < best:
        best = va
        best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

model.load_state_dict(best_state)
torch.save(model.state_dict(), r"C:\Users\katsu\OneDrive\my_practice\thesis\output\model1.pth")
joblib.dump({"scaler_X": scaler_X, "scaler_y": scaler_y, "input_cols": input_cols},
            r"C:\Users\katsu\OneDrive\my_practice\thesis\output\model1.joblib")
print("保存完了")

# ============================
# 4. 🔍 追加部分①：スケールを戻した誤差比較
# ============================
model.eval()
with torch.no_grad():
    # train, val それぞれの予測値を取得
    y_train_pred_scaled = model(X_t[train_ds.indices]).numpy()
    y_val_pred_scaled   = model(X_t[val_ds.indices]).numpy()

# スケールを戻す
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
y_val_pred   = scaler_y.inverse_transform(y_val_pred_scaled)
y_train_true = scaler_y.inverse_transform(y_t[train_ds.indices].numpy())
y_val_true   = scaler_y.inverse_transform(y_t[val_ds.indices].numpy())

# RMSE・MAEを計算
def eval_metrics(y_true, y_pred, name=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    print(f"{name} RMSE: {rmse:.6f}, MAE: {mae:.6f}")

print("\n スケールを戻した評価 ")
eval_metrics(y_train_true, y_train_pred, "Train")
eval_metrics(y_val_true, y_val_pred, "Val")

# ============================
# 5. 🔍 追加部分②：inrange/outRange評価
# ============================

def evaluate_external_data(csv_path, model, scaler_pack):
    df2 = pd.read_csv(csv_path)
    df2[scaler_pack["input_cols"]] = df2[scaler_pack["input_cols"]].apply(pd.to_numeric, errors="coerce")
    df2 = df2.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    X2 = df2[scaler_pack["input_cols"]].values.astype(np.float32)
    y2 = df2[target_col].values.reshape(-1,1).astype(np.float32)

    X2_scaled = scaler_pack["scaler_X"].transform(X2)
    y2_scaled = scaler_pack["scaler_y"].transform(y2)

    with torch.no_grad():
        y2_pred_scaled = model(torch.tensor(X2_scaled, dtype=torch.float32)).numpy()

    y2_pred = scaler_pack["scaler_y"].inverse_transform(y2_pred_scaled)
    y2_true = scaler_pack["scaler_y"].inverse_transform(y2_scaled)

    print(f"\n=== {os.path.basename(csv_path)} の評価結果 ===")
    eval_metrics(y2_true, y2_pred, "  Test")

# 評価実行
scaler_pack = {"scaler_X": scaler_X, "scaler_y": scaler_y, "input_cols": input_cols}
evaluate_external_data(r"C:\Users\katsu\OneDrive\my_practice\thesis\data\inrange.csv", model, scaler_pack)
evaluate_external_data(r"C:\Users\katsu\OneDrive\my_practice\thesis\data\outrange.csv", model, scaler_pack)
