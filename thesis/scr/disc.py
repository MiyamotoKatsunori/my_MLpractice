import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# smoothing（平滑化）関数
# ---------------------------------------------------------
def smooth(series, k=3):
    s = []
    L = len(series)
    for i in range(L):
        left = max(0, i-k)
        right = min(L, i+k+1)
        s.append(np.mean(series[left:right]))
    return np.array(s)


folder_map = {
    r"C:\Users\katsu\OneDrive\my_practice\thesis\best\l364": "default",
    r"C:\Users\katsu\OneDrive\my_practice\thesis\best\l3100": "100nodes",
    r"C:\Users\katsu\OneDrive\my_practice\thesis\best\addl5": "5 liner",
    r"C:\Users\katsu\OneDrive\my_practice\thesis\best\adm": "Ir=3e-4,decay=3e-5",
    r"C:\Users\katsu\OneDrive\my_practice\thesis\best\drop01": "dropout=0.1",
    r"C:\Users\katsu\OneDrive\my_practice\thesis\best\weight9-4": "decay=9e-4",
}
folders = list(folder_map.keys())
folder_labels = list(folder_map.values())


# ---------------------------------------------------------
# ★ 複数 run 除外設定（自由に編集可能）
# ---------------------------------------------------------
exclude = {
    "100nodes": [1],    # run2 を除外
    "default": [4, 2],
    "5 liner": [2,3,0],
    "Ir=3e-4,decay=3e-5":[1,8,4,6],
    "dropout=0.1": [6,8,3,1]
}


# ---------------------------------------------------------
# データ読み込み（run 除外に対応）
# ---------------------------------------------------------
all_train = []
all_train_smooth = []
all_out = []
folder_ids = []

for fid, folder in enumerate(folders):

    label = folder_labels[fid]  # 例 "100nodes"

    train_path = os.path.join(folder, "train_rmse_history.csv")
    out_path   = os.path.join(folder, "outrange_rmse_history.csv")

    if not os.path.exists(train_path) or not os.path.exists(out_path):
        print(f"⚠ CSV not found in: {folder}")
        continue

    df_train = pd.read_csv(train_path)
    df_out   = pd.read_csv(out_path)

    train_mat = df_train.iloc[:, 1:].values  # (epochs, runs)
    out_mat   = df_out.iloc[:, 1:].values    # (epochs, runs)

    num_runs = train_mat.shape[1]

    # -----------------------------
    # ★ 各 run を処理（複数 run 除外に対応）
    # -----------------------------
    for r in range(num_runs):

        # 除外 run のチェック
        if label in exclude and r in exclude[label]:
            print(f"→ EXCLUDED: {label} run{r}")
            continue

        # Train（生）
        tr = train_mat[:, r]

        # Train（平滑化）
        tr_s = smooth(tr, k=2)

        # Outrange
        out = out_mat[:, r]

        # flatten 保存
        all_train.extend(tr)
        all_train_smooth.extend(tr_s)
        all_out.extend(out)
        folder_ids.extend([fid] * len(tr))


# numpy に変換
all_train = np.array(all_train)
all_train_smooth = np.array(all_train_smooth)
all_out = np.array(all_out)
folder_ids = np.array(folder_ids)



# ---------------------------------------------------------
# ① 元の散布図（train vs outrange）
# ---------------------------------------------------------
plt.figure(figsize=(14, 8))

scatter = plt.scatter(
    all_train,
    all_out,
    c=folder_ids,
    cmap="tab10",
    alpha=0.65,
    s=30
)

plt.xlabel("Train RMSE")
plt.ylabel("Outrange RMSE")
plt.title("Train vs Outrange RMSE (all epochs, filtered runs)")
plt.grid(True)

handles = [
    plt.Line2D([], [], marker="o", linestyle="-", 
               color=scatter.cmap(scatter.norm(i)))
    for i in range(len(folders))
]
plt.legend(handles, folder_labels, title="Folders", loc="best")

plt.tight_layout()

save_path = r"C:\Users\katsu\OneDrive\my_practice\thesis\best\train_outrange_allpoints_filtered.png"
plt.savefig(save_path, dpi=150)
plt.show()

print("Saved:", save_path)



# ---------------------------------------------------------
# ② ★ 平滑化版（train_smooth vs outrange）
# ---------------------------------------------------------
plt.figure(figsize=(12, 8))

scatter2 = plt.scatter(
    all_train_smooth,
    all_out,
    c=folder_ids,
    cmap="tab10",
    alpha=0.65,
    s=30
)

plt.xlabel("Smoothed Train RMSE")
plt.ylabel("Outrange RMSE")
plt.title("Train vs Outrange RMSE (all epochs)")
plt.grid(True)

handles2 = [
    plt.Line2D([], [], marker="o", linestyle="-", 
               color=scatter2.cmap(scatter2.norm(i)))
    for i in range(len(folders))
]
plt.legend(handles2, folder_labels, title="Change Para", loc="best")

plt.tight_layout()

save_path2 = r"C:\Users\katsu\OneDrive\my_practice\thesis\best\train_outrange_smoothed_filtered.png"
plt.savefig(save_path2, dpi=150)
plt.show()

print("Saved:", save_path2)


