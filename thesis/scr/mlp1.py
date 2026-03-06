import pandas as pd, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib

# ============================
# 1. データ読み込み
# ============================
# CSVファイルには以下の列を想定
#   - 条件を区別する列: condition_id（任意）
#   - 入力特徴量（例: T1, T2, T3, T4, flow, power, voltage, current, ...）
#   - 物理モデル出力: y_phys
#   - 実測出力: y
csv_path=r"C:\Users\katsu\OneDrive\my_practice\thesis\data\input.csv"

# 入力列を定義（物理モデル出力も含める）
input_cols = [
    "T_0", "T_90", "T_180", "T_270", "mf", "pw", "T_0_dTdt",
    "T_90_dTdt", "T_180_dTdt", "T_270_dTdt", "Isg", "I_aps", "CH14", "CH15", "CH7", "CH8"
]
target_col = "I_sg"
use_robust = False  # 外れ値が強いなら True に

# ========= 1) 読み込み & クレンジング =========
df = pd.read_csv(csv_path)

# 数値変換（非数値は NaN）
df[input_cols + [target_col]] = df[input_cols + [target_col]].apply(pd.to_numeric, errors="coerce")
# ±inf を NaN に
df = df.replace([np.inf, -np.inf], np.nan)

# 欠損を削除（もしくは前方補間など適宜）
before = len(df)
df = df.dropna(subset=input_cols + [target_col]).reset_index(drop=True)
print(f"DropNA: {before-len(df)} rows removed")

# 定数列を検出 → 除外（あるいは後でスケール=1に補正）
stds = df[input_cols].std()
const_cols = stds[stds == 0].index.tolist()
if const_cols:
    print("定数列を除外:", const_cols)
    input_cols = [c for c in input_cols if c not in const_cols]

X = df[input_cols].values.astype(np.float32)
y = df[target_col].values.reshape(-1,1).astype(np.float32)

# ========= 2) スケーリング（ゼロ分散対策） =========
if use_robust:
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()  # yのスケーリングが不要ならコメントアウト
else:
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)

# yの分散がゼロならスケーリングはやめる
if np.isclose(y.std(), 0.0):
    print("y が定数です。出力のスケーリングをスキップします。")
    y_scaled = y.copy()
    scaler_y = None
else:
    y_scaled = scaler_y.fit_transform(y)

# スケーリング後の NaN/inf をチェック
def _chk(a, name):
    nan = np.isnan(a).sum(); inf = np.isinf(a).sum()
    if nan or inf:
        raise ValueError(f"{name} に NaN/inf が含まれています: nan={nan}, inf={inf}")
_chk(X_scaled, "X_scaled")
_chk(y_scaled, "y_scaled")

# ========= 3) Tensor & Split =========
X_t = torch.tensor(X_scaled, dtype=torch.float32)
y_t = torch.tensor(y_scaled, dtype=torch.float32)

ds = TensorDataset(X_t, y_t)
n = len(ds); n_val = int(n*0.2)
gen = torch.Generator().manual_seed(42)  # 再現性
train_ds, val_ds = random_split(ds, [n-n_val, n_val], generator=gen)

train_ld = DataLoader(train_ds, batch_size=64, shuffle=True)
val_ld   = DataLoader(val_ds, batch_size=64, shuffle=False)

# ========= 4) MLP =========
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

# ========= 5) 学習設定（安定化） =========
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)  # lr下げ＋L2
epochs = 100
best = float('inf'); best_state = None

# 勾配クリップ
def clip_grads(model, max_norm=1.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

# ========= 6) ループ =========
for ep in range(1, epochs+1):
    model.train()
    tr = 0.0
    for xb, yb in train_ld:
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = criterion(pred, yb)
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError("Train loss became NaN/Inf")
        loss.backward()
        clip_grads(model, 1.0)
        optimizer.step()
        tr += loss.item()
    tr /= len(train_ld)

    model.eval(); va = 0.0
    with torch.no_grad():
        for xb, yb in val_ld:
            pred = model(xb)
            loss = criterion(pred, yb)
            va += loss.item()
    va /= len(val_ld)

    print(f"Epoch [{ep}/{epochs}]  Train: {tr:.6f}  Val: {va:.6f}")
    if va < best:
        best = va
        best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

# 復元 & 保存
model.load_state_dict(best_state)
torch.save(model.state_dict(), r"C:\Users\katsu\OneDrive\my_practice\thesis\output\model1.pth")
joblib.dump({"scaler_X": scaler_X, "scaler_y": scaler_y, "input_cols": input_cols},
            r"C:\Users\katsu\OneDrive\my_practice\thesis\output\model1.joblib")
print("保存完了")