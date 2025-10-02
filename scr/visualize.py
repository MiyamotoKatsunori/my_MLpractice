# 可視化スクリプト
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

# データ読み込み 
df = pd.read_csv(r"C:\Users\katsu\OneDrive\my_practice\scr\data\train_data.csv")
X = df[["x1", "x2"]].values
y = df["y"].values

X_tensor = torch.tensor(X, dtype=torch.float32)

# MLPモデル定義
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MLP()
model.load_state_dict(torch.load(r"C:\Users\katsu\OneDrive\my_practice\scr\data\mlp_model.pth"))
model.eval()
print("学習済みモデルを読み込み")

# 予測 
with torch.no_grad():
    y_pred = model(X_tensor).numpy().flatten()

# 真関数
x1 = X[:, 0]
x2 = X[:, 1]
y_true = np.sin(x1) + 0.5*np.cos(x2) + 0.1*x1**2 - 0.2*x2**2

# 3Dプロット 
fig = plt.figure(figsize=(12,6))

# 真関数 
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_trisurf(x1, x2, y_true, cmap="viridis", alpha=0.8)
ax1.set_title("True Function")
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
ax1.set_zlabel("y")

# 予測値
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_trisurf(x1, x2, y_pred, cmap="plasma", alpha=0.8)
ax2.set_title("MLP Prediction")
ax2.set_xlabel("x1")
ax2.set_ylabel("x2")
ax2.set_zlabel("y")

plt.tight_layout()
plt.savefig(r"C:\Users\katsu\OneDrive\my_practice\scr\data\graph.png")
plt.show()
