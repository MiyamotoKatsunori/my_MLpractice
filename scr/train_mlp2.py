import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

#データ読み込み
df = pd.read_csv(r"C:\Users\katsu\OneDrive\my_practice\scr\data\train_data.csv")  # 修正: dataフォルダ
X = df[["x1", "x2"]].values
y = df["y"].values

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# MLPモデル
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

# 損失関数 & 最適化
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 学習ループ
epochs = 50
for epoch in range(epochs):
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 学習後の予測
with torch.no_grad():
    y_pred = model(X_tensor).numpy()

# 真関数 
x1 = X[:, 0]
x2 = X[:, 1]
y_true = np.sin(x1) + 0.5*np.cos(x2) + 0.1*x1**2 - 0.2*x2  # train_data.pyで定義した式

# グラフ描画
plt.figure(figsize=(8,6))
plt.scatter(range(len(y_true)), y_true, color="blue", label="True Function", alpha=0.6)
plt.scatter(range(len(y_pred)), y_pred, color="red", label="MLP Prediction", alpha=0.6)
plt.xlabel("Sample index")
plt.ylabel("y")
plt.title("True Function vs MLP Prediction")
plt.legend()
plt.savefig(r"C:\Users\katsu\OneDrive\my_practice\scr\data\graph.png")
plt.show()
