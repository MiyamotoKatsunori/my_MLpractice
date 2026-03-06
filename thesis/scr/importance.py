import numpy as np
import torch
import torch.nn as nn
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

# =====================================================
# パス設定
# =====================================================
outrange_path = r"C:\Users\katsu\OneDrive\my_practice\thesis\data\outrange.csv"
model_path    = r"C:\Users\katsu\OneDrive\my_practice\thesis\output\hybrid_runs_line\best_outrange_model.pth"
scaler_path   = r"C:\Users\katsu\OneDrive\my_practice\thesis\output\hybrid_runs_line\best_outrange_scalers.joblib"
save_dir      = r"C:\Users\katsu\OneDrive\my_practice\thesis\importance"

os.makedirs(save_dir, exist_ok=True)

# =====================================================
# ResidualMLP（学習時と同じ）
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

df = pd.read_csv(outrange_path)
df = df.dropna(subset=input_cols + ["Isg", "I_sg"]).reset_index(drop=True)

X = df[input_cols].values.astype(np.float32)
I_phys = df["Isg"].values.reshape(-1, 1).astype(np.float32)
y_true = df["I_sg"].values.reshape(-1, 1).astype(np.float32)

# 残差 ΔI
y_resid = y_true - I_phys
X_scaled = scaler_X.transform(X)

# =====================================================
# 予測関数（ΔIのみ）
# =====================================================
def predict_deltaI(model, X_scaled):
    with torch.no_grad():
        out_scaled = model(torch.tensor(X_scaled, dtype=torch.float32))
    return scaler_y.inverse_transform(out_scaled.numpy())[:, 0]


# =====================================================
# 元のRMSE
# =====================================================
delta_pred_orig = predict_deltaI(model, X_scaled)
rmse_orig = np.sqrt(np.mean((y_resid[:,0] - delta_pred_orig)**2))

print("Original RMSE (ΔI):", rmse_orig)

# =====================================================
# 入力特徴量の重要度
# =====================================================
names = []
importances = []

for j, col in enumerate(input_cols):

    X_perm = X_scaled.copy()
    idx = np.random.permutation(len(X_perm))
    X_perm[:, j] = X_perm[idx, j]

    delta_pred_perm = predict_deltaI(model, X_perm)
    rmse_perm = np.sqrt(np.mean((y_resid[:,0] - delta_pred_perm)**2))

    imp = rmse_perm - rmse_orig
    names.append(col)
    importances.append(imp)

    print(f"{col}: {imp:.6f}")


# =====================================================
# 温度平均 T_avg の重要度
# =====================================================
temp_cols = ["T_0", "T_90", "T_180", "T_270"]
temp_idx = [input_cols.index(c) for c in temp_cols]

X_perm = X_scaled.copy()
idx = np.random.permutation(len(X_perm))

# 4つを揃えてシャッフル
for k in temp_idx:
    X_perm[:, k] = X_scaled[idx, k]

delta_pred_perm = predict_deltaI(model, X_perm)
rmse_perm = np.sqrt(np.mean((y_resid[:,0] - delta_pred_perm)**2))
imp_Tavg = rmse_perm - rmse_orig

names.append("T_avg")
importances.append(imp_Tavg)

print(f"T_avg: {imp_Tavg:.6f}")


# =====================================================
# 温度微分平均 dTdt_avg の重要度
# =====================================================
dtemp_cols = ["T_0_dTdt", "T_90_dTdt", "T_180_dTdt", "T_270_dTdt"]
dtemp_idx = [input_cols.index(c) for c in dtemp_cols]

X_perm = X_scaled.copy()
idx = np.random.permutation(len(X_perm))

# 揃えてシャッフル
for k in dtemp_idx:
    X_perm[:, k] = X_scaled[idx, k]

delta_pred_perm = predict_deltaI(model, X_perm)
rmse_perm = np.sqrt(np.mean((y_resid[:,0] - delta_pred_perm)**2))
imp_dTdt = rmse_perm - rmse_orig

names.append("dTdt_avg")
importances.append(imp_dTdt)

print(f"dTdt_avg: {imp_dTdt:.6f}")


# =====================================================
# 可視化
# =====================================================
plt.figure(figsize=(12,5))
plt.bar(names, importances)
plt.xticks(rotation=45)
plt.ylabel("Permutation Importance (ΔRMSE of ΔI)")
plt.title("Feature Importance (including T_avg & dTdt_avg)")
plt.tight_layout()

save_path = os.path.join(save_dir, "feature_importance_outrange.png")
plt.savefig(save_path, dpi=300)
plt.close()

print("\nSaved:", save_path)
