import os
import numpy as np
import pandas as pd

# データ件数
N = 10000  

# 入力データ（ランダム生成）
x1 = np.random.uniform(-5, 5, N)
x2 = np.random.uniform(-5, 5, N)

# 出力（例：三角関数 + 多項式 + ノイズ）
y = np.sin(x1) + 0.5 * np.cos(x2) + 0.1 * x1**2 - 0.2 * x2 **2 + np.random.normal(0, 0.1, N)

# DataFrameにまとめる
df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

# 保存先ディレクトリ
save_dir = r"C:\Users\katsu\OneDrive\my_practice\scr\data"
os.makedirs(save_dir, exist_ok=True)  # 無ければ自動で作成

# ファイルパス
file_path = os.path.join(save_dir, "train_data.csv")

# CSVに保存
df.to_csv(file_path, index=False)

print(f"CSVを作成: {file_path}")
