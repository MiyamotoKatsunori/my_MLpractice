import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==== 比較したいフォルダ ====
models = {
    "Black box" : r"C:\Users\katsu\OneDrive\my_practice\thesis\output\ad_black",

    "GB1-4"  : r"C:\Users\katsu\OneDrive\my_practice\thesis\output\ad_noch",
    "GB2"    : r"C:\Users\katsu\OneDrive\my_practice\thesis\output\ad_ratio",
    "GB3"  : r"C:\Users\katsu\OneDrive\my_practice\thesis\output\ad_hybrid",
}

# ==== 可視化するメトリック ====
metric = "outrange"

# ==== カスタマイズ設定 ====

# 軸フォントサイズ
AXIS_LABEL_SIZE = 20
TICK_SIZE = 18
TITLE_SIZE = 18
LEGEND_SIZE = 15

# 線の太さ
LINE_WIDTH = 2.5

# 凡例位置（"upper right", "lower left", "best", etc）
LEGEND_LOC = "best"

# y軸の範囲（None のままなら自動）
YMIN = None
YMAX = 8.0

# x軸の表示間隔
XTICK_INTERVAL = 10   # 5 エポック間隔で tick を打つ


# ==== 描画開始 ====
plt.figure(figsize=(10,6))

for label, path in models.items():
    csv_path = os.path.join(path, f"{metric}_rmse_history.csv")

    if not os.path.exists(csv_path):
        print(f"⚠️ Missing: {csv_path}")
        continue

    df = pd.read_csv(csv_path, index_col=0)

    mean = df.mean(axis=1)
    std  = df.std(axis=1)

    epochs = mean.index.astype(int)

    # 平均線
    plt.plot(epochs, mean, linewidth=LINE_WIDTH, label=label)

    # 標準偏差帯
    plt.fill_between(epochs, mean - std, mean + std, alpha=0.15)
# ==== White box（一定値のライン）====
#WHITE_BOX_RMSE = RMSE = 8.441447992327006
#plt.plot(epochs, [WHITE_BOX_RMSE]*len(epochs),
        #linestyle="--", linewidth=LINE_WIDTH,
         #label="White box")

# ==== 軸とタイトル ====
plt.xlabel("Epoch", fontsize=AXIS_LABEL_SIZE)
plt.ylabel(r"RMSE [mA]", fontsize=AXIS_LABEL_SIZE)
plt.title(f"Comparison of mean (2W0.4kPa) RMSE across models",
          fontsize=TITLE_SIZE)

# y軸の範囲を自由に設定
if (YMIN is not None) or (YMAX is not None):
    plt.ylim(YMIN, YMAX)

# x軸の目盛りを指定 (例: 0,5,10,15,...)
plt.xticks(np.arange(min(epochs), max(epochs)+1, XTICK_INTERVAL),
           fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)

# グリッド
plt.grid(True, linestyle="--", alpha=0.6)

# 凡例
plt.legend(fontsize=LEGEND_SIZE, loc=LEGEND_LOC)

plt.tight_layout()

out_path = rf"C:\Users\katsu\OneDrive\my_practice\thesis\output\adthesis_{metric}.png"
plt.savefig(out_path, dpi=300)
plt.show()

print(f"\nSaved: {out_path}")



