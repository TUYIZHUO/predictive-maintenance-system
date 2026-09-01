# -*- coding: utf-8 -*-
"""
算法模块 1：LSTM 分类器（对应《制造智能技术》"深度神经网络/监督学习"技术方向）。

作用：输入一段设备运行状态序列，同时输出两个预测——
    1. 二分类头：设备是否将发生故障（sigmoid）
    2. 六分类头：具体是哪种故障类型（0=正常 + 5 种故障，softmax）

结构：LSTM(输入8 -> 隐藏64 -> 2层) 取最后一层最后时刻隐状态，
     分接两个全连接头。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 6,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        # 多层 LSTM：dropout 仅在 num_layers > 1 时有效，否则 PyTorch 会报警告
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # 双头输出
        self.binary_head = nn.Linear(hidden_size, 1)          # 是否故障
        self.multi_head = nn.Linear(hidden_size, num_classes)  # 故障类型

    def forward(self, x: torch.Tensor):
        """
        参数：
            x : (batch, seq_len, input_size)
        返回：
            bin_logit  : (batch,)            二分类 logit
            multi_logit: (batch, num_classes) 多分类 logit
        """
        out, (h_n, _) = self.lstm(x)
        last = h_n[-1]  # 取最后一层、最后时刻的隐状态 (batch, hidden_size)

        bin_logit = self.binary_head(last).squeeze(-1)
        multi_logit = self.multi_head(last)
        return bin_logit, multi_logit


# ---------------------------------------------------------------------------
# 损失函数
# ---------------------------------------------------------------------------
def binary_loss(bin_logit: torch.Tensor, y_bin: torch.Tensor) -> torch.Tensor:
    """二分类：BCE with logits。"""
    return nn.functional.binary_cross_entropy_with_logits(bin_logit, y_bin.float())


def multi_loss(multi_logit: torch.Tensor, y_multi: torch.Tensor) -> torch.Tensor:
    """多分类：交叉熵。"""
    return nn.functional.cross_entropy(multi_logit, y_multi.long())


# ---------------------------------------------------------------------------
# 推理辅助：把 logit 转成可解释输出
# ---------------------------------------------------------------------------
def predict(bin_logit: torch.Tensor, multi_logit: torch.Tensor):
    """
    返回：
        failure_prob : (batch,)       故障概率
        failure_pred : (batch,)       二分类预测 0/1
        class_prob   : (batch, C)     各类别概率
        class_pred   : (batch,)       预测类别索引
    """
    failure_prob = torch.sigmoid(bin_logit)
    failure_pred = (failure_prob >= 0.5).long()
    class_prob = torch.softmax(multi_logit, dim=-1)
    class_pred = class_prob.argmax(dim=-1)
    return failure_prob, failure_pred, class_prob, class_pred
