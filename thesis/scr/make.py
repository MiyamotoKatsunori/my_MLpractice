import pandas as pd
import matplotlib.pyplot as plt

# ================================
# ★ 文字サイズ設定（自由に変更）
# ================================
title_size = 18      # タイトル
label_size = 18      # 軸ラベル
tick_size = 18       # 軸目盛（数字）
legend_size = 18     # 凡例
# ================================

# ======== CSV読み込み ========
csv_path = r"C:\Users\katsu\OneDrive\my_practice\thesis\output\blockbox\outrange_rmse_history.csv"   # ← CSVファイルパス
df = pd.read_csv(csv_path)

# ======== 列指定 ========
# 例：1列目 x、2〜6列目が y1〜y5 とする
x = df.iloc[:, 0]
y_cols = df.columns[1:6]   # ← y列が5本あると仮定（自動で取得）

# ======== グラフ描画 ========
plt.figure(figsize=(8, 6))

for col in y_cols:
    plt.plot(x, df[col], linewidth=2, label=col)   # 線のみ（点なし）

# ======== タイトル・軸ラベル ========
#plt.title("Outrange for all run", fontsize=title_size)
plt.xlabel(df.columns[0], fontsize=label_size)
plt.ylabel("RSME[mA]", fontsize=label_size)

# ======== 目盛りサイズ ========
plt.tick_params(axis='both', labelsize=tick_size)

# ======== 凡例 ========
plt.legend(fontsize=legend_size)

# ======== 装飾 ========
plt.grid(True)
plt.tight_layout()
plt.show()
