import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from mlp2 import MLP  # mlp2.py と同じディレクトリにある場合

# ==============================
# 1️⃣ モデルとスケーラーの読み込み
# ==============================
save_dir = r"C:\Users\katsu\OneDrive\my_practice\thesis\output"
model_path = f"{save_dir}\\model1.pth"
scaler_pack = joblib.load(f"{save_dir}\\model1.joblib")

scaler_X = scaler_pack["scaler_X"]
scaler_y = scaler_pack["scaler_y"]
input_cols = scaler_pack["input_cols"]

# モデル構築とパラメータ読み込み
model = MLP(len(input_cols))
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# ==============================
# 2️⃣ 平均入力ベクトルを作成
# ==============================
# 平均値ベクトルを基準点とする
mean_vec = scaler_X.mean_

# Isg のインデックスを特定
isg_idx = input_cols.index("Isg")
print(f"Isg index = {isg_idx}")

# ==============================
# 3️⃣ Isg を変化させて推論
# ==============================
# Isg を変化させる範囲（例: 平均±3σ）
isg_mean = mean_vec[isg_idx]
isg_std  = scaler_X.scale_[isg_idx]
isg_values = np.linspace(isg_mean - 3*isg_std, isg_mean + 3*isg_std, 100)

# 入力ベクトル群を生成
X_varied = np.tile(mean_vec, (len(isg_values), 1))
X_varied[:, isg_idx] = isg_values  # Isg だけを変化させる

# スケーリング
X_varied_t = torch.tensor(X_varied, dtype=torch.float32)

# 推論
with torch.no_grad():
    y_scaled = model(X_varied_t).numpy()

# 出力のスケールを戻す
y_pred = scaler_y.inverse_transform(y_scaled)

# ==============================
# 4️⃣ グラフ描画
# ==============================
plt.figure(figsize=(6, 4))
plt.plot(isg_values, y_pred, lw=2)
plt.xlabel("Theoretical Current Isg (input)")
plt.ylabel("Predicted Current I_sg (output)")
plt.title("Model Dependence on Theoretical Current (Isg)")
plt.grid(True)
plt.tight_layout()
plt.show()
