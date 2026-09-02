# -*- coding: utf-8 -*-
"""
训练 LSTM 自编码器（算法模块 2，异常检测）。

只用"正常"设备样本训练重构。加载 splits_3d.npz，取二分类标签为 0 的
正常序列训练，以验证集正常样本重构误差的 mean + 3*std 作为阈值 τ。

用法：
    python train_ae.py                              # 使用项目内默认数据
    python train_ae.py --data <splits_3d.npz 路径>   # 指定其他数据

产出：
    checkpoints/ae_best.pth
    checkpoints/ae_config.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.lstm_ae import LSTMAutoencoder, compute_threshold, reconstruction_error


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(script_dir / "data" / "processed" / "splits_3d.npz"),
        help="splits_3d.npz 路径（默认项目内 data/processed/splits_3d.npz）",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--out", type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = script_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据，筛选正常序列
    print("[1/4] 加载数据并筛选正常样本...")
    d = np.load(args.data)
    X_train = d["X_train"]
    y_bin_train = d["y_bin_train"]
    X_test = d["X_test"]
    y_bin_test = d["y_bin_test"]

    normal_mask = y_bin_train == 0
    X_normal = X_train[normal_mask]  # 正常序列（SMOTE 不改变多数类）
    print(f"  正常序列数: {len(X_normal)}")

    n = len(X_normal)
    n_train = int(n * 0.8)
    X_train_ae = X_normal[:n_train]
    X_val_ae = X_normal[n_train:]

    input_size = X_normal.shape[2]

    train_ds = TensorDataset(torch.tensor(X_train_ae, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val_ae, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # 2. 模型
    model = LSTMAutoencoder(input_size=input_size).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 3. 训练 + 早停
    print("[2/4] 训练自编码器...")
    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    train_loss_hist = []
    val_loss_hist = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for (xb,) in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                recon = model(xb)
                val_loss += criterion(recon, xb).item() * xb.size(0)
        train_avg = total_loss / len(train_ds)
        avg_val = val_loss / len(val_ds)
        train_loss_hist.append(train_avg)
        val_loss_hist.append(avg_val)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  train={train_avg:.6f}  val={avg_val:.6f}")

        if avg_val < best_loss:
            best_loss = avg_val
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  早停于 epoch {epoch}")
                break

    # 4. 计算阈值 τ = mean + 3*std
    print("[3/4] 计算阈值 τ...")
    model.load_state_dict(best_state)
    model.eval()
    errors = []
    with torch.no_grad():
        for (xb,) in val_loader:
            xb = xb.to(device)
            recon = model(xb)
            errors.append(reconstruction_error(recon, xb).cpu())
    errors = torch.cat(errors)
    threshold = compute_threshold(errors, k=3.0)
    print(f"  τ = {threshold:.6f}  (mean={errors.mean():.6f}, std={errors.std():.6f})")

    torch.save(best_state, out_dir / "ae_best.pth")
    (out_dir / "ae_config.txt").write_text(f"threshold={threshold:.8f}\n")
    print(f"[4/4] 已保存: {out_dir / 'ae_best.pth'}")

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

    def calc_errors(model, X, device, batch_size=64):
        """计算一组序列的重构误差（MSE）。"""
        model.eval()
        errs = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
                errs.append(reconstruction_error(model(xb), xb).cpu())
        return torch.cat(errs)

    # (1) loss 收敛曲线
    plt.figure(figsize=(7, 4))
    epochs = range(1, len(train_loss_hist) + 1)
    plt.plot(epochs, train_loss_hist, label="训练 loss")
    plt.plot(epochs, val_loss_hist, label="验证 loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title("自编码器训练收敛曲线")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "ae_loss_curve.png", dpi=150)
    plt.close()

    # (2) 重构误差分布：正常 vs 异常
    normal_errs = calc_errors(model, X_normal, device)
    fault_mask = y_bin_test == 1
    fault_errs = calc_errors(model, X_test[fault_mask], device) if fault_mask.sum() > 0 else None

    plt.figure(figsize=(7, 4))
    plt.hist(normal_errs.numpy(), bins=40, alpha=0.6, label="正常样本")
    if fault_errs is not None:
        plt.hist(fault_errs.numpy(), bins=40, alpha=0.6, label="异常样本")
    plt.axvline(threshold, linestyle="--", label=f"阈值 τ = {threshold:.4f}")
    plt.xlabel("重构误差"); plt.ylabel("样本数")
    plt.title("重构误差分布（正常 vs 异常）")
    plt.legend(); plt.tight_layout()
    plt.savefig(fig_dir / "ae_error_distribution.png", dpi=150)
    plt.close()

    # (3) 原始 vs 重构序列（刀具磨损，索引 4）
    sample = X_normal[0]
    with torch.no_grad():
        recon_sample = model(torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
    feat_idx = 4
    steps = range(1, sample.shape[0] + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(steps, sample[:, feat_idx], "o-", label="原始")
    plt.plot(steps, recon_sample[:, feat_idx], "s--", label="重构")
    plt.xlabel("时间步"); plt.ylabel("刀具磨损（标准化后）")
    plt.title("原始序列 vs 重构序列（刀具磨损）")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "ae_reconstruction.png", dpi=150)
    plt.close()

    print(f"  已保存图: figures/ae_loss_curve.png, ae_error_distribution.png, ae_reconstruction.png")


if __name__ == "__main__":
    main()
