import numpy as np
import torch
import torch.nn as nn
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# ★ ユーザー環境に合わせて適宜修正
# =====================================================
outrange_path = r"C:\Users\katsu\OneDrive\my_practice\thesis\data\outrange.csv"
model_path = r"C:\Users\katsu\OneDrive\my_practice\thesis\output\hybrid_runs_line\best_outrange_model.pth"
scaler_path = r"C:\Users\katsu\OneDrive\my_practice\thesis\output\hybrid_runs_line\best_outrange_scalers.joblib"

# =====================================================
# ResidualMLP (元スクリプトと同じ構造)
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
# モデル + スケーラー読み込み
# =====================================================
scalers = joblib.load(scaler_path)
scaler_X = scalers["scaler_X"]
scaler_y = scalers["scaler_y"]
input_cols = scalers["input_cols"]
phys_col   = scalers["phys_col"]

# モデル構築 & 読み込み
model = ResidualMLP(len(input_cols))
state = torch.load(model_path, map_location="cpu")
model.load_state_dict(state)
model.eval()

print("Loaded model and scalers.")
print("Input columns:", input_cols)
print("Phys col:", phys_col)

# =====================================================
# データ読み込み（outrange）
# =====================================================
df = pd.read_csv(outrange_path)
df = df.dropna(subset=input_cols + [phys_col]).reset_index(drop=True)

X = df[input_cols].values.astype(np.float32)
I_phys = df[phys_col].values.astype(np.float32).reshape(-1, 1)

# スケール後のデータ
X_scaled = scaler_X.transform(X)


# =====================================================
# 予測関数（物理モデル + 残差）
# =====================================================
def hybrid_predict(model, X_scaled, I_phys):
    """物理モデル + NN残差 で総合予測を返す"""
    with torch.no_grad():
        ΔI_scaled = model(torch.tensor(X_scaled, dtype=torch.float32))
    ΔI = scaler_y.inverse_transform(ΔI_scaled.numpy())
    return I_phys + ΔI


# =====================================================
# 部分依存プロット（PDP）
# =====================================================
for i, col in enumerate(input_cols):
    print(f"\n=== PDP: {col} ===")

    # 観測データの 1〜99% 範囲でスイープ
    x_min, x_max = np.percentile(X[:, i], 1), np.percentile(X[:, i], 99)
    x_grid = np.linspace(x_min, x_max, 100)

    preds = []

    for x_val in x_grid:
        # 平均のサンプルを作り、この入力だけ変化させる
        X_base = X.mean(axis=0).copy()
        X_base[i] = x_val

        # スケール変換
        X_scaled_tmp = scaler_X.transform([X_base])

        # 予測
        y_pred = hybrid_predict(model, X_scaled_tmp, I_phys.mean())
        preds.append(y_pred[0, 0])

    # 図の描画
    plt.figure(figsize=(6,4))
    plt.plot(x_grid, preds, linewidth=2)
    plt.xlabel(col)
    plt.ylabel("Predicted I_sg [mA]")
    plt.title(f"Dependence on {col}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
