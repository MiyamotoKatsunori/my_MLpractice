import numpy as np
import torch
import torch.nn as nn
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

# =====================================================
# ★ ユーザー環境に合わせて変更してください
# =====================================================
outrange_path = r"C:\Users\katsu\OneDrive\my_practice\thesis\data\outrange.csv"
model_path    = r"C:\Users\katsu\OneDrive\my_practice\thesis\output\hybrid_runs_line\best_outrange_model.pth"
scaler_path   = r"C:\Users\katsu\OneDrive\my_practice\thesis\output\hybrid_runs_line\best_outrange_scalers.joblib"
save_dir      = r"C:\Users\katsu\OneDrive\my_practice\thesis\pdp"

os.makedirs(save_dir, exist_ok=True)

# =====================================================
# ★ 変数のスイープ範囲（必要なものだけ指定）
# =====================================================
custom_ranges = {
     "mf": (0, 100),
     "pw": (0, 10),
     "T_0": (0, 100),
     "T_90": (0, 100),
     "T_180": (0, 100),
     "T_270": (0, 100),
     "T_0_dTdt": (-0.2, 0.2),
     "T_90_dTdt": (-0.2, 0.2),
     "T_180_dTdt": (-0.2, 0.2),
     "T_270_dTdt": (-0.2, 0.2),

     # ★ 新しく追加する範囲設定（平均用）
     "T_avg": (0, 100),
     "dTdt_avg": (-0.2, 0.2),
}

# =====================================================
# ResidualMLP（学習時と同じ構造）
# =====================================================
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
        return self.net(x)

# =====================================================
# モデル・スケーラー読み込み
# =====================================================
scalers = joblib.load(scaler_path)
scaler_X = scalers["scaler_X"]
scaler_y = scalers["scaler_y"]
input_cols = scalers["input_cols"]

model = ResidualMLP(len(input_cols))
state = torch.load(model_path, map_location="cpu")
model.load_state_dict(state)
model.eval()

print("Loaded model and scalers.")
print("Input columns:", input_cols)

# =====================================================
# データ読み込み（範囲推定に使用）
# =====================================================
df = pd.read_csv(outrange_path)
df = df.dropna(subset=input_cols).reset_index(drop=True)
X = df[input_cols].values.astype(np.float32)

# =====================================================
# ΔI 予測関数
# =====================================================
def predict_deltaI(model, X_scaled):
    with torch.no_grad():
        ΔI_scaled = model(torch.tensor(X_scaled, dtype=torch.float32))
    ΔI = scaler_y.inverse_transform(ΔI_scaled.numpy())
    return ΔI[:, 0]

# 平均ベース点（PDP用に他の変数を固定する値）
X_mean = X.mean(axis=0).copy()

# =====================================================
# ★ 通常の入力変数の PDP（個別）
# =====================================================
for i, col in enumerate(input_cols):
    print(f"\n=== Wide-range PDP (NN residual only): {col} ===")

    # スイープ範囲決定
    if col in custom_ranges:
        x_min, x_max = custom_ranges[col]
    else:
        raw_min, raw_max = np.min(X[:, i]), np.max(X[:, i])
        width = raw_max - raw_min
        x_min = raw_min - 0.3 * width
        x_max = raw_max + 0.3 * width

    x_grid = np.linspace(x_min, x_max, 150)

    preds = []
    for x_val in x_grid:
        X_base = X_mean.copy()
        X_base[i] = x_val
        X_scaled_tmp = scaler_X.transform([X_base])
        delta = predict_deltaI(model, X_scaled_tmp)
        preds.append(delta[0])

    # 保存
    plt.figure(figsize=(6,4))
    plt.plot(x_grid, preds, linewidth=2)
    plt.xlabel(col)
    plt.ylabel("NN Residual ΔI [mA]")
    plt.title(f"Wide-Range NN Residual Dependence on {col}")
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"pdp_deltaI_{col}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")

# =====================================================
# ★ 温度平均 T_avg
# =====================================================
print("\n=== PDP: Temperature Average T_avg ===")

temp_cols = ["T_0", "T_90", "T_180", "T_270"]
idx_temp = [input_cols.index(c) for c in temp_cols]

# 範囲設定（custom があれば優先）
if "T_avg" in custom_ranges:
    t_min, t_max = custom_ranges["T_avg"]
else:
    raw_min = np.min(X[:, idx_temp])
    raw_max = np.max(X[:, idx_temp])
    width = raw_max - raw_min
    t_min = raw_min - 0.3 * width
    t_max = raw_max + 0.3 * width

T_avg_grid = np.linspace(t_min, t_max, 150)
preds_Tavg = []

for T_avg in T_avg_grid:
    X_base = X_mean.copy()
    for c in temp_cols:
        X_base[input_cols.index(c)] = T_avg

    X_scaled_tmp = scaler_X.transform([X_base])
    delta = predict_deltaI(model, X_scaled_tmp)
    preds_Tavg.append(delta[0])

plt.figure(figsize=(6,4))
plt.plot(T_avg_grid, preds_Tavg, linewidth=2)
plt.xlabel("T_avg [°C]")
plt.ylabel("NN Residual ΔI [mA]")
plt.title("Dependence on Temperature Average (T_avg)")
plt.grid(True)
plt.tight_layout()

save_path = os.path.join(save_dir, "pdp_deltaI_T_avg.png")
plt.savefig(save_path, dpi=300)
plt.close()
print(f"Saved: {save_path}")

# =====================================================
# ★ 温度微分平均 dTdt_avg
# =====================================================
print("\n=== PDP: Temperature Derivative Average dTdt_avg ===")

dtemp_cols = ["T_0_dTdt", "T_90_dTdt", "T_180_dTdt", "T_270_dTdt"]
idx_dtemp = [input_cols.index(c) for c in dtemp_cols]

# custom_ranges があればそちらを使う
if "dTdt_avg" in custom_ranges:
    t_min, t_max = custom_ranges["dTdt_avg"]
else:
    raw_min = np.min(X[:, idx_dtemp])
    raw_max = np.max(X[:, idx_dtemp])
    width = raw_max - raw_min
    t_min = raw_min - 0.3 * width
    t_max = raw_max + 0.3 * width

dTdt_avg_grid = np.linspace(t_min, t_max, 150)
preds_dTdtavg = []

for dTdt_avg in dTdt_avg_grid:
    X_base = X_mean.copy()
    for c in dtemp_cols:
        X_base[input_cols.index(c)] = dTdt_avg

    X_scaled_tmp = scaler_X.transform([X_base])
    delta = predict_deltaI(model, X_scaled_tmp)
    preds_dTdtavg.append(delta[0])

plt.figure(figsize=(6,4))
plt.plot(dTdt_avg_grid, preds_dTdtavg, linewidth=2)
plt.xlabel("dTdt_avg [K/s]")
plt.ylabel("NN Residual ΔI [mA]")
plt.title("Dependence on Temperature Derivative Average (dTdt_avg)")
plt.grid(True)
plt.tight_layout()

save_path = os.path.join(save_dir, "pdp_deltaI_dTdt_avg.png")
plt.savefig(save_path, dpi=300)
plt.close()
print(f"Saved: {save_path}")

# =====================================================
# ★ ここから追加：T_0, T_90, T_180, T_270 を1枚にまとめたPDP
# =====================================================
print("\n=== Combined PDP: T_0, T_90, T_180, T_270 ===")

# 共通スイープ範囲を決定（custom_ranges があればそれに基づく）
temp_mins = []
temp_maxs = []
for c in temp_cols:
    if c in custom_ranges:
        temp_mins.append(custom_ranges[c][0])
        temp_maxs.append(custom_ranges[c][1])
    else:
        idx = input_cols.index(c)
        raw_min, raw_max = X[:, idx].min(), X[:, idx].max()
        width = raw_max - raw_min
        temp_mins.append(raw_min - 0.3 * width)
        temp_maxs.append(raw_max + 0.3 * width)

t_min_all = min(temp_mins)
t_max_all = max(temp_maxs)
T_grid = np.linspace(t_min_all, t_max_all, 150)

plt.figure(figsize=(7,5))

for c in temp_cols:
    preds_c = []
    for val in T_grid:
        X_base = X_mean.copy()
        X_base[input_cols.index(c)] = val
        X_scaled_tmp = scaler_X.transform([X_base])
        delta = predict_deltaI(model, X_scaled_tmp)
        preds_c.append(delta[0])
    plt.plot(T_grid, preds_c, linewidth=2, label=c)

plt.xlabel("Temperature [°C]")
plt.ylabel("NN Residual ΔI [mA]")
plt.title("PDP of Temperatures (T_0, T_90, T_180, T_270)")
plt.legend()
plt.grid(True)
plt.tight_layout()

save_path = os.path.join(save_dir, "pdp_deltaI_T_all.png")
plt.savefig(save_path, dpi=300)
plt.close()
print(f"Saved combined temperature PDP: {save_path}")

# =====================================================
# ★ ここから追加：T_0_dTdt, ... を1枚にまとめたPDP
# =====================================================
print("\n=== Combined PDP: T_*_dTdt ===")

dtemp_mins = []
dtemp_maxs = []
for c in dtemp_cols:
    if c in custom_ranges:
        dtemp_mins.append(custom_ranges[c][0])
        dtemp_maxs.append(custom_ranges[c][1])
    else:
        idx = input_cols.index(c)
        raw_min, raw_max = X[:, idx].min(), X[:, idx].max()
        width = raw_max - raw_min
        dtemp_mins.append(raw_min - 0.3 * width)
        dtemp_maxs.append(raw_max + 0.3 * width)

dt_min_all = min(dtemp_mins)
dt_max_all = max(dtemp_maxs)
dTdt_grid = np.linspace(dt_min_all, dt_max_all, 150)

plt.figure(figsize=(7,5))

for c in dtemp_cols:
    preds_c = []
    for val in dTdt_grid:
        X_base = X_mean.copy()
        X_base[input_cols.index(c)] = val
        X_scaled_tmp = scaler_X.transform([X_base])
        delta = predict_deltaI(model, X_scaled_tmp)
        preds_c.append(delta[0])
    plt.plot(dTdt_grid, preds_c, linewidth=2, label=c)

plt.xlabel("dT/dt [K/s]")
plt.ylabel("NN Residual ΔI [mA]")
plt.title("PDP of Temperature Derivatives (T_*_dTdt)")
plt.legend()
plt.grid(True)
plt.tight_layout()

save_path = os.path.join(save_dir, "pdp_deltaI_dTdt_all.png")
plt.savefig(save_path, dpi=150)
plt.close()
print(f"Saved combined dTdt PDP: {save_path}")

print("\n=== All PDP graphs saved to:", save_dir, "===")
