"""
data/preprocess.py
AI4I 2020 Predictive Maintenance Dataset 预处理脚本
=====================================================
数据来源: UCI Machine Learning Repository (ID=601)
         S. Matzka, "Explainable Artificial Intelligence for Predictive
         Maintenance Applications," AI4I 2020, doi:10.1109/AI4I49448.2020.00023
原始规模: 10,000 条 × 14 列
特征维度: 8 = 5 数值特征 + 3 列 Type one-hot (L/M/H)
标签:     二分类 (Machine failure) + 多分类 (5 种故障模式)
依赖:     numpy, pandas, scikit-learn（无需 imbalanced-learn）
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors

# ============================================================
# 0. 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROC_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(PROC_DIR, exist_ok=True)

RAW_CSV = os.path.join(RAW_DIR, "ai4i2020.csv")
SEQ_LEN = 10          # LSTM 滑动窗口长度
RANDOM_STATE = 42

# ============================================================
# 手动实现 SMOTE（无需 imbalanced-learn）
# ============================================================
def manual_smote(X, y, random_state=42, k_neighbors=5):
    """
    对二分类/多分类数据做 SMOTE 过采样，使各类别数量与多数类一致。

    参数:
        X: 2D array, shape (n_samples, n_features) — 展平后的特征矩阵
        y: 1D array, shape (n_samples,)            — 标签
        random_state: int                           — 随机种子
        k_neighbors: int                            — K近邻数

    返回:
        X_res: 2D array — 过采样后的特征矩阵
        y_res: 1D array — 过采样后的标签
    """
    rng = np.random.RandomState(random_state)
    classes, counts = np.unique(y, return_counts=True)
    majority_count = counts.max()

    X_resampled = [X.copy()]
    y_resampled = [y.copy()]

    for cls in classes:
        n_current = np.sum(y == cls)
        if n_current >= majority_count:
            continue

        X_minority = X[y == cls]
        n_synthetic = majority_count - n_current

        # K 近邻（排除自身，所以 fit 时 n_neighbors = k+1，查询时取第 2~k+1 个）
        k = min(k_neighbors, n_current - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X_minority)
        _, indices = nn.kneighbors(X_minority)
        # indices[:, 0] 是自身，indices[:, 1:] 是邻居
        neighbor_indices = indices[:, 1:]

        # 生成合成样本
        synthetic = np.empty((n_synthetic, X.shape[1]), dtype=X.dtype)
        for i in range(n_synthetic):
            # 随机选一个少数类样本
            ref_idx = rng.randint(0, n_current)
            # 随机选一个邻居
            nn_idx = rng.randint(0, k)
            neighbor = X_minority[neighbor_indices[ref_idx, nn_idx]]
            # 线性插值
            lam = rng.uniform(0, 1)
            synthetic[i] = X_minority[ref_idx] + lam * (neighbor - X_minority[ref_idx])

        X_resampled.append(synthetic)
        y_resampled.append(np.full(n_synthetic, cls, dtype=y.dtype))

    X_res = np.vstack(X_resampled)
    y_res = np.concatenate(y_resampled)

    # 打乱顺序
    shuffle_idx = rng.permutation(len(y_res))
    return X_res[shuffle_idx], y_res[shuffle_idx]


# ============================================================
# 1. 加载 & 清洗
# ============================================================
print("=" * 60)
print("[1/7] 加载原始数据")
print("=" * 60)

# 兼容多种常见文件名
for fname in ["ai4i2020.csv", "ai4i_2020.csv", "predictive_maintenance.csv"]:
    candidate = os.path.join(RAW_DIR, fname)
    if os.path.exists(candidate):
        RAW_CSV = candidate
        break

df = pd.read_csv(RAW_CSV)
print(f"  原始数据规模: {df.shape[0]} 行 × {df.shape[1]} 列")
print(f"  列名: {list(df.columns)}")

# ---------- 缺失值检查 ----------
missing = df.isnull().sum()
if missing.sum() > 0:
    print(f"  ⚠ 发现缺失值:\n{missing[missing > 0]}")
    df.dropna(inplace=True)
    print(f"  已删除含缺失值的行，剩余: {df.shape[0]} 行")
else:
    print("  ✓ 无缺失值")

# ---------- 去除无业务意义的 ID 列 ----------
drop_cols = []
for c in df.columns:
    cl = c.lower().replace(" ", "")
    if cl in ("udi", "uid", "productid", "product_id", "type.1"):
        drop_cols.append(c)
if drop_cols:
    df.drop(columns=drop_cols, inplace=True)
    print(f"  已删除 ID 列: {drop_cols}")

# ---------- 统一列名（去空格、统一小写） ----------
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
print(f"  清洗后列名: {list(df.columns)}")

# ============================================================
# 2. Type 编码 — One-Hot (L / M / H)
# ============================================================
print("\n" + "=" * 60)
print("[2/7] Type 字段 One-Hot 编码")
print("=" * 60)

type_col = "type"
assert type_col in df.columns, f"找不到 '{type_col}' 列，请检查原始数据"

type_dummies = pd.get_dummies(df[type_col], prefix="type", dtype=np.float32)
# 确保 L / M / H 三列齐全（即使某类缺失也不影响维度）
for suffix in ["L", "M", "H"]:
    col_name = f"type_{suffix}"
    if col_name not in type_dummies.columns:
        type_dummies[col_name] = 0.0
# 固定列顺序
type_dummies = type_dummies[["type_L", "type_M", "type_H"]]
df = pd.concat([df.drop(columns=[type_col]), type_dummies], axis=1)
print(f"  One-Hot 列: {list(type_dummies.columns)}")
print(f"  分布: L={type_dummies['type_L'].sum():.0f}, "
      f"M={type_dummies['type_M'].sum():.0f}, "
      f"H={type_dummies['type_H'].sum():.0f}")

# ============================================================
# 3. 标签编码
# ============================================================
print("\n" + "=" * 60)
print("[3/7] 标签编码")
print("=" * 60)

# ---- 二分类标签: Machine failure (0/1) ----
binary_col = "machine_failure"
assert binary_col in df.columns, f"找不到 '{binary_col}' 列"
y_binary = df[binary_col].astype(np.int32).values
print(f"  二分类标签 '{binary_col}': "
      f"正常={np.sum(y_binary == 0)}, 故障={np.sum(y_binary == 1)}")

# ---- 多分类标签: 5 种故障模式 → 0~5 ----
failure_modes = ["twf", "hdf", "pwf", "osf", "rnf"]
# 动态匹配列名（兼容大小写差异）
actual_fm_cols = []
for fm in failure_modes:
    matched = [c for c in df.columns if c == fm or c.startswith(fm)]
    if matched:
        actual_fm_cols.append(matched[0])
    else:
        raise KeyError(f"找不到故障模式列 '{fm}'，现有列: {list(df.columns)}")

# 编码规则: 0=无故障, 1=TWF, 2=HDF, 3=PWF, 4=OSF, 5=RNF
y_multi = np.zeros(len(df), dtype=np.int32)
for idx, col in enumerate(actual_fm_cols):
    mask = df[col].astype(int).values == 1
    y_multi[mask] = idx + 1

le = LabelEncoder()
le.classes_ = np.array(["No_Failure"] + [c.upper() for c in failure_modes])
print(f"  多分类标签映射: {dict(enumerate(le.classes_))}")
for cls_id, cls_name in enumerate(le.classes_):
    cnt = np.sum(y_multi == cls_id)
    print(f"    {cls_id} ({cls_name}): {cnt} 条 ({cnt / len(y_multi) * 100:.2f}%)")

# ============================================================
# 4. 特征提取 & 标准化
# ============================================================
print("\n" + "=" * 60)
print("[4/7] 特征提取与标准化")
print("=" * 60)

# 5 个数值特征
numeric_features = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]
# 动态匹配（兼容列名中的单位标注如 [K]）
actual_num_cols = []
for nf in numeric_features:
    matched = [c for c in df.columns if nf in c.replace("[", "").replace("]", "")]
    if matched:
        actual_num_cols.append(matched[0])
    else:
        raise KeyError(f"找不到数值特征列 '{nf}'，现有列: {list(df.columns)}")

# 3 个 one-hot 列
onehot_cols = ["type_L", "type_M", "type_H"]

# 合并: 5 数值 + 3 one-hot = 8 维
feature_cols = actual_num_cols + onehot_cols
assert len(feature_cols) == 8, f"特征维度不为 8，实际: {len(feature_cols)}"
print(f"  特征列 ({len(feature_cols)} 维): {feature_cols}")

X_raw = df[feature_cols].values.astype(np.float32)

# StandardScaler（先占位，划分后再 fit）
scaler = StandardScaler()

# ============================================================
# 5. 数据集划分 7 : 1.5 : 1.5
# ============================================================
print("\n" + "=" * 60)
print("[5/7] 数据集划分 (7 : 1.5 : 1.5)")
print("=" * 60)

# 第一步: 70% 训练, 30% 临时
X_train, X_temp, y_bin_train, y_bin_temp, y_mul_train, y_mul_temp = \
    train_test_split(X_raw, y_binary, y_multi,
                     test_size=0.30, random_state=RANDOM_STATE,
                     stratify=y_binary)

# 第二步: 临时 30% → 50% 验证 + 50% 测试 (即总体的 15% + 15%)
X_val, X_test, y_bin_val, y_bin_test, y_mul_val, y_mul_test = \
    train_test_split(X_temp, y_bin_temp, y_mul_temp,
                     test_size=0.50, random_state=RANDOM_STATE,
                     stratify=y_bin_temp)

print(f"  训练集: {X_train.shape[0]} 条 ({X_train.shape[0] / len(X_raw) * 100:.1f}%)")
print(f"  验证集: {X_val.shape[0]} 条 ({X_val.shape[0] / len(X_raw) * 100:.1f}%)")
print(f"  测试集: {X_test.shape[0]} 条 ({X_test.shape[0] / len(X_raw) * 100:.1f}%)")

# ---- 仅用训练集 fit scaler（只对 5 个数值特征 fit） ----
scaler.fit(X_train[:, :5])
print(f"  Scaler 已 fit（仅训练集 5 个数值特征）")


def apply_scaler(X, scaler):
    """对前 5 列数值特征做标准化，后 3 列 one-hot 保持原值"""
    X_out = X.copy()
    X_out[:, :5] = scaler.transform(X[:, :5])
    return X_out


X_train_s = apply_scaler(X_train, scaler)
X_val_s = apply_scaler(X_val, scaler)
X_test_s = apply_scaler(X_test, scaler)

# ============================================================
# 6. 构建 3D 序列 & SMOTE 过采样（仅训练集）
# ============================================================
print("\n" + "=" * 60)
print("[6/7] 构建 LSTM 3D 序列 & SMOTE 过采样")
print("=" * 60)


def build_sequences(X, y_bin, y_mul, seq_len):
    """
    滑动窗口构建 3D 序列。
    X: (N, 8) → X_seq: (N - seq_len + 1, seq_len, 8)
    标签取窗口最后一个时间步的值。
    """
    n_samples = X.shape[0]
    n_seqs = n_samples - seq_len + 1
    X_seq = np.zeros((n_seqs, seq_len, X.shape[1]), dtype=np.float32)
    y_bin_seq = np.zeros(n_seqs, dtype=np.int32)
    y_mul_seq = np.zeros(n_seqs, dtype=np.int32)

    for i in range(n_seqs):
        X_seq[i] = X[i: i + seq_len]
        y_bin_seq[i] = y_bin[i + seq_len - 1]   # 取窗口末尾标签
        y_mul_seq[i] = y_mul[i + seq_len - 1]

    return X_seq, y_bin_seq, y_mul_seq


# ---- 6a. 训练集构建 3D 序列 ----
X_train_3d, y_bin_train_3d, y_mul_train_3d = build_sequences(
    X_train_s, y_bin_train, y_mul_train, SEQ_LEN
)
print(f"  训练集 3D 序列: {X_train_3d.shape}")
print(f"    二分类分布: 正常={np.sum(y_bin_train_3d == 0)}, "
      f"故障={np.sum(y_bin_train_3d == 1)}")

# ---- 6b. SMOTE 过采样（仅训练集） ----
# ⚠ 风险点③: SMOTE 原生只接受 2D 表格数据 (n_samples, n_features)
#    对 LSTM 输入的 3D 张量 (n_samples, seq_len, n_features)
#    必须: 展平 → SMOTE → reshape 回 3D
print("\n  --- SMOTE 过采样流程（手动实现，无需 imbalanced-learn） ---")
print(f"  [步骤1] 原始 3D 张量 shape: {X_train_3d.shape}")

n_samples, seq_len, n_features = X_train_3d.shape

# 展平: (n_samples, seq_len, n_features) → (n_samples, seq_len * n_features)
X_train_flat = X_train_3d.reshape(n_samples, -1)
print(f"  [步骤2] 展平后 shape: {X_train_flat.shape}  "
      f"(= {n_samples} × {seq_len}*{n_features})")

# 手动 SMOTE 过采样 — 二分类标签
X_train_flat_res, y_bin_train_res = manual_smote(
    X_train_flat, y_bin_train_3d, random_state=RANDOM_STATE
)
print(f"  [步骤3] SMOTE 后 shape: {X_train_flat_res.shape}, "
      f"标签分布: 正常={np.sum(y_bin_train_res == 0)}, "
      f"故障={np.sum(y_bin_train_res == 1)}")

# reshape 回 3D: (n_resampled, seq_len * n_features) → (n_resampled, seq_len, n_features)
X_train_3d_res = X_train_flat_res.reshape(-1, seq_len, n_features)
print(f"  [步骤4] reshape 回 3D: {X_train_3d_res.shape}")

# 多分类标签同步过采样
X_train_flat_res_mul, y_mul_train_res = manual_smote(
    X_train_flat, y_mul_train_3d, random_state=RANDOM_STATE
)
X_train_3d_res_mul = X_train_flat_res_mul.reshape(-1, seq_len, n_features)
print(f"  多分类 SMOTE 后 shape: {X_train_3d_res_mul.shape}")

# ---- 6c. 验证集 / 测试集构建 3D 序列（不做 SMOTE） ----
X_val_3d, y_bin_val_3d, y_mul_val_3d = build_sequences(
    X_val_s, y_bin_val, y_mul_val, SEQ_LEN
)
X_test_3d, y_bin_test_3d, y_mul_test_3d = build_sequences(
    X_test_s, y_bin_test, y_mul_test, SEQ_LEN
)
print(f"\n  验证集 3D: {X_val_3d.shape}")
print(f"  测试集 3D: {X_test_3d.shape}")

# ============================================================
# 7. 保存所有产出
# ============================================================
print("\n" + "=" * 60)
print("[7/7] 保存产出文件")
print("=" * 60)

# ---- 保存 scaler ----
scaler_path = os.path.join(PROC_DIR, "scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print(f"  ✓ Scaler       → {scaler_path}")

# ---- 保存 LabelEncoder ----
le_path = os.path.join(PROC_DIR, "label_encoder.pkl")
with open(le_path, "wb") as f:
    pickle.dump(le, f)
print(f"  ✓ LabelEncoder → {le_path}")

# ---- 保存 2D 数据（用于传统 ML 基线） ----
np.savez_compressed(
    os.path.join(PROC_DIR, "splits_2d.npz"),
    X_train=X_train_s, y_bin_train=y_bin_train, y_mul_train=y_mul_train,
    X_val=X_val_s, y_bin_val=y_bin_val, y_mul_val=y_mul_val,
    X_test=X_test_s, y_bin_test=y_bin_test, y_mul_test=y_mul_test,
)
print(f"  ✓ 2D 数据      → {os.path.join(PROC_DIR, 'splits_2d.npz')}")

# ---- 保存 3D 序列数据（用于 LSTM） ----
np.savez_compressed(
    os.path.join(PROC_DIR, "splits_3d.npz"),
    X_train=X_train_3d_res, y_bin_train=y_bin_train_res,
    X_train_mul=X_train_3d_res_mul, y_mul_train=y_mul_train_res,
    X_val=X_val_3d, y_bin_val=y_bin_val_3d, y_mul_val=y_mul_val_3d,
    X_test=X_test_3d, y_bin_test=y_bin_test_3d, y_mul_test=y_mul_test_3d,
)
print(f"  ✓ 3D 序列      → {os.path.join(PROC_DIR, 'splits_3d.npz')}")

# ============================================================
# 8. 统计信息汇总（供报告"数据资源构建"章节）
# ============================================================
print("\n" + "=" * 60)
print("统计信息汇总")
print("=" * 60)

stats = {
    "数据来源": {
        "名称": "AI4I 2020 Predictive Maintenance Dataset",
        "来源": "UCI Machine Learning Repository (ID=601)",
        "论文": 'S. Matzka, "Explainable AI for Predictive Maintenance",'
                " AI4I 2020, doi:10.1109/AI4I49448.2020.00023",
        "原始规模": f"{df.shape[0]} 行 × {df.shape[1]} 列",
        "缺失值": "无",
    },
    "特征工程": {
        "输入维度": 8,
        "数值特征(5)": actual_num_cols,
        "One-Hot特征(3)": onehot_cols,
        "标准化方法": "StandardScaler (仅 fit 训练集数值特征)",
    },
    "标签信息": {
        "二分类": {
            "含义": "Machine failure (0=正常, 1=故障)",
            "总体分布": {
                "正常": int(np.sum(y_binary == 0)),
                "故障": int(np.sum(y_binary == 1)),
                "故障率": f"{np.sum(y_binary == 1) / len(y_binary) * 100:.2f}%",
            },
        },
        "多分类": {
            "含义": "5 种故障模式 + 无故障",
            "类别映射": {int(k): v for k, v in enumerate(le.classes_)},
            "总体分布": {
                le.classes_[i]: int(np.sum(y_multi == i))
                for i in range(len(le.classes_))
            },
        },
    },
    "数据集划分": {
        "比例": "7 : 1.5 : 1.5",
        "训练集": X_train.shape[0],
        "验证集": X_val.shape[0],
        "测试集": X_test.shape[0],
    },
    "SMOTE过采样": {
        "实现方式": "手动 numpy 实现（无需 imbalanced-learn）",
        "应用范围": "仅训练集",
        "3D处理策略": "展平(n, seq*n_feat) → SMOTE → reshape(n, seq, n_feat)",
        "序列长度": SEQ_LEN,
        "过采样后训练集(二分类)": int(X_train_3d_res.shape[0]),
        "过采样后训练集(多分类)": int(X_train_3d_res_mul.shape[0]),
        "过采样后二分类分布": {
            "正常": int(np.sum(y_bin_train_res == 0)),
            "故障": int(np.sum(y_bin_train_res == 1)),
        },
    },
    "产出文件": {
        "scaler": "data/processed/scaler.pkl",
        "label_encoder": "data/processed/label_encoder.pkl",
        "2D数据": "data/processed/splits_2d.npz",
        "3D序列": "data/processed/splits_3d.npz",
    },
}

# 打印统计摘要
print(json.dumps(stats, indent=2, ensure_ascii=False))

# 保存统计信息 JSON
stats_path = os.path.join(PROC_DIR, "dataset_stats.json")
with open(stats_path, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)
print(f"\n  ✓ 统计信息 → {stats_path}")

# ============================================================
# 验证产出完整性
# ============================================================
print("\n" + "=" * 60)
print("产出验证")
print("=" * 60)

data_3d = np.load(os.path.join(PROC_DIR, "splits_3d.npz"))
print(f"  3D 训练集: X={data_3d['X_train'].shape}, "
      f"y_bin={data_3d['y_bin_train'].shape}, "
      f"y_mul={data_3d['y_mul_train'].shape}")
print(f"  3D 验证集: X={data_3d['X_val'].shape}, "
      f"y_bin={data_3d['y_bin_val'].shape}, "
      f"y_mul={data_3d['y_mul_val'].shape}")
print(f"  3D 测试集: X={data_3d['X_test'].shape}, "
      f"y_bin={data_3d['y_bin_test'].shape}, "
      f"y_mul={data_3d['y_mul_test'].shape}")

assert data_3d["X_train"].shape[1] == SEQ_LEN, "seq_len 不匹配"
assert data_3d["X_train"].shape[2] == 8, "特征维度不为 8"
print("\n  ✓ 所有维度校验通过！")

# ============================================================
# 9. 数值特征描述性统计（均值 / 标准差 / 最值 / 均方根 RMS）
# ============================================================
print("\n" + "=" * 60)
print("数值特征描述性统计（原始物理单位）")
print("=" * 60)

feat_stat_rows = []
for col in actual_num_cols:
    x = df[col].astype(float).values
    feat_stat_rows.append({
        "特征": col,
        "样本数": int(len(x)),
        "均值": float(np.mean(x)),
        "标准差": float(np.std(x)),
        "最小值": float(np.min(x)),
        "最大值": float(np.max(x)),
        "均方根(RMS)": float(np.sqrt(np.mean(x ** 2))),
    })

feat_stats_df = pd.DataFrame(feat_stat_rows)
feat_stats_path = os.path.join(PROC_DIR, "feature_stats.csv")
feat_stats_df.to_csv(feat_stats_path, index=False, encoding="utf-8-sig")
print(f"\n{feat_stats_df.round(4).to_string(index=False)}")
print(f"\n  ✓ 特征统计 → {feat_stats_path}")

print("\n预处理完成。")
