# -*- coding: utf-8 -*-
"""
训练随机森林分类器（算法模块 1），含决策阈值调优。

加载 data/processed/splits_2d.npz（2D 特征，用 class_weight 平衡不平衡）。

用法：
    python train_classifier.py                              # 使用项目内默认数据
    python train_classifier.py --data <splits_2d.npz 路径>   # 指定其他数据

产出：
    checkpoints/rf_binary.pkl       二分类模型
    checkpoints/rf_multi.pkl        多分类模型
    checkpoints/classifier_config.txt  (最优阈值 + 类别数)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from models.rf_classifier import save_model, train_binary, train_multi


def find_best_threshold(probs: np.ndarray, y_true: np.ndarray):
    """在 [0.05, 0.95] 扫描阈值，返回使 F1 最大的阈值及其 F1。"""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        pred = (probs >= t).astype(int)
        f = f1_score(y_true, pred)
        if f > best_f1:
            best_f1, best_t = f, float(round(t, 2))
    return best_t, best_f1


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(script_dir / "data" / "processed" / "splits_2d.npz"),
        help="splits_2d.npz 路径（默认项目内 data/processed/splits_2d.npz）",
    )
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--out", type=str, default="checkpoints")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = script_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载 2D 数据
    print("[1/4] 加载 2D 数据...")
    d = np.load(args.data)
    X_train = d["X_train"]; y_bin_train = d["y_bin_train"]; y_mul_train = d["y_mul_train"]
    X_val = d["X_val"]; y_bin_val = d["y_bin_val"]; y_mul_val = d["y_mul_val"]
    X_test = d["X_test"]; y_bin_test = d["y_bin_test"]; y_mul_test = d["y_mul_test"]
    print(f"  训练集: {X_train.shape}  验证集: {X_val.shape}  测试集: {X_test.shape}")

    # 2. 训练二分类随机森林 + 阈值调优
    print("[2/4] 训练二分类随机森林...")
    rf_bin = train_binary(X_train, y_bin_train, n_estimators=args.n_estimators)
    probs_val = rf_bin.predict_proba(X_val)[:, 1]
    best_t, val_f1 = find_best_threshold(probs_val, y_bin_val)
    print(f"  最优阈值 = {best_t:.2f}  验证集 F1 = {val_f1:.4f}")

    # 3. 训练多分类随机森林
    print("[3/4] 训练多分类随机森林...")
    rf_mul = train_multi(X_train, y_mul_train, n_estimators=args.n_estimators)

    # 4. 测试集评估
    print("[4/4] 测试集评估...")
    probs_test = rf_bin.predict_proba(X_test)[:, 1]
    pred_05 = (probs_test >= 0.5).astype(int)
    pred_t = (probs_test >= best_t).astype(int)
    tp = int(((pred_t == 1) & (y_bin_test == 1)).sum())
    fp = int(((pred_t == 1) & (y_bin_test == 0)).sum())
    fn = int(((pred_t == 0) & (y_bin_test == 1)).sum())
    tn = int(((pred_t == 0) & (y_bin_test == 0)).sum())
    pred_mul = rf_mul.predict(X_test)

    print(f"  二分类 @0.5     : acc={accuracy_score(y_bin_test, pred_05):.4f}  F1={f1_score(y_bin_test, pred_05):.4f}")
    print(f"  二分类 @{best_t:.2f}   : acc={accuracy_score(y_bin_test, pred_t):.4f}  F1={f1_score(y_bin_test, pred_t):.4f}")
    print(f"  混淆矩阵 (阈值{best_t:.2f}): TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  多分类(6类)    : acc={accuracy_score(y_mul_test, pred_mul):.4f}  F1(macro)={f1_score(y_mul_test, pred_mul, average='macro'):.4f}")

    # 保存
    save_model(rf_bin, str(out_dir / "rf_binary.pkl"))
    save_model(rf_mul, str(out_dir / "rf_multi.pkl"))
    (out_dir / "classifier_config.txt").write_text(f"threshold={best_t:.4f}\nnum_classes=6\n")
    print(f"  已保存: rf_binary.pkl, rf_multi.pkl, classifier_config.txt")

    # ------------------------------------------------------------------
    # 可视化：保存图到 figures/
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = script_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 特征名（与 preprocess.py 的 8 维顺序一致）
    feat_names = ["空气温度", "工艺温度", "转速", "扭矩", "刀具磨损", "类型L", "类型M", "类型H"]

    # (1) 阈值-F1 扫描曲线
    threshs = np.arange(0.05, 0.96, 0.01)
    f1s = [f1_score(y_bin_val, (probs_val >= t).astype(int)) for t in threshs]
    plt.figure(figsize=(7, 4))
    plt.plot(threshs, f1s, marker="o", markersize=3)
    plt.axvline(best_t, linestyle="--", label=f"最优阈值 = {best_t:.2f}")
    plt.xlabel("阈值"); plt.ylabel("F1")
    plt.title("阈值扫描与 F1（验证集）")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "rf_threshold_f1.png", dpi=150)
    plt.close()

    # (2) 混淆矩阵热力图
    cm = confusion_matrix(y_bin_test, pred_t)
    thresh_cm = cm.max() / 2
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                     color="white" if cm[i, j] > thresh_cm else "black")
    plt.xticks([0, 1], ["正常", "故障"]); plt.yticks([0, 1], ["正常", "故障"])
    plt.xlabel("预测"); plt.ylabel("真实")
    plt.title(f"混淆矩阵（阈值 {best_t:.2f}）")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(fig_dir / "rf_confusion_matrix.png", dpi=150)
    plt.close()

    # (3) 特征重要性
    importances = rf_bin.feature_importances_
    order = np.argsort(importances)
    plt.figure(figsize=(8, 5))
    plt.barh(range(len(importances)), importances[order])
    plt.yticks(range(len(importances)), [feat_names[i] for i in order])
    plt.xlabel("特征重要性")
    plt.title("随机森林特征重要性（二分类）")
    plt.tight_layout()
    plt.savefig(fig_dir / "rf_feature_importance.png", dpi=150)
    plt.close()

    print(f"  已保存图: figures/rf_threshold_f1.png, rf_confusion_matrix.png, rf_feature_importance.png")


if __name__ == "__main__":
    main()
