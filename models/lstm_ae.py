# -*- coding: utf-8 -*-
"""
算法模块 2：LSTM 自编码器（对应《制造智能技术》"无监督学习/异常检测"技术方向）。

作用：只用"正常"设备样本训练重构能力。正常样本重构误差小，异常样本
     重构误差大。推理时以重构误差是否超过阈值 τ 判断是否异常。

结构：编码器 LSTM 将序列压缩为低维向量 z，解码器 LSTM 再由 z 重构
     原始序列。阈值 τ = 验证集正常样本重构误差的 mean + 3 * std。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 32,
        latent_size: int = 16,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # 编码器：序列 -> 隐状态 -> 低维向量 z
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.latent_fc = nn.Linear(hidden_size, latent_size)

        # 解码器：z -> 隐状态序列 -> 重构序列
        self.decoder_fc = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.out_fc = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数：
            x : (batch, seq_len, input_size)
        返回：
            recon : (batch, seq_len, input_size) 重构序列
        """
        batch, seq_len, _ = x.shape

        # 编码：取最后一层最后时刻隐状态作为序列压缩表示
        _, (h_n, _) = self.encoder(x)
        z = self.latent_fc(h_n[-1])  # (batch, latent_size)

        # 解码：把 z 展开为 seq_len 个时间步的输入
        z = self.decoder_fc(z).unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden)
        out, _ = self.decoder(z)
        recon = self.out_fc(out)  # (batch, seq_len, input_size)
        return recon


def reconstruction_error(recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """逐样本重构误差 = 每个样本在时间维和特征维上的均方误差 MSE。返回 (batch,)。"""
    return torch.mean((recon - x) ** 2, dim=(1, 2))


def compute_threshold(errors: torch.Tensor, k: float = 3.0) -> float:
    """
    阈值 τ = mean + k * std。默认 k=3（3σ 准则）。
    errors : (n,) 验证集正常样本的重构误差。
    """
    return float(errors.mean().item() + k * errors.std().item())


def is_anomaly(error: torch.Tensor, threshold: float) -> torch.Tensor:
    """重构误差 > 阈值则判为异常。返回 bool 张量 (batch,)。"""
    return error > threshold
