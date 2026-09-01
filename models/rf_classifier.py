# -*- coding: utf-8 -*-
"""
算法模块 1：随机森林分类器（对应《制造智能技术》"集成学习"技术方向）。

作用：对设备运行特征做故障分类——
    1. 二分类：设备是否将发生故障（返回故障概率）
    2. 多分类：具体故障类型（6 类：0=正常 + 5 种故障）

模型选型说明（答辩要点）：
    通过对照实验发现，AI4I 2020 每行是独立设备运行快照、并非严格时序，
    LSTM 序列建模会丢失温度/转速/扭矩等特征的直接判别力（故障 F1 仅 0.32）。
    改用随机森林后故障 F1 提升到 0.62、召回率 71%，故分类模块选用随机森林。
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

NUM_CLASSES = 6  # 0=No_Failure, 1=TWF, 2=HDF, 3=PWF, 4=OSF, 5=RNF


def train_binary(
    X: np.ndarray, y: np.ndarray, n_estimators: int = 200, random_state: int = 42
) -> RandomForestClassifier:
    """训练二分类随机森林（class_weight 缓解类别不平衡）。"""
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X, y)
    return rf


def train_multi(
    X: np.ndarray, y: np.ndarray, n_estimators: int = 200, random_state: int = 42
) -> RandomForestClassifier:
    """训练多分类随机森林（6 类）。"""
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X, y)
    return rf


def save_model(model, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(binary_model, multi_model, X: np.ndarray, threshold: float = 0.5):
    """
    推理接口，返回：
        failure_prob : (n,)       故障概率
        failure_pred : (n,)       二分类预测 0/1
        class_prob   : (n, 6)     各类别概率
        class_pred   : (n,)       预测类别 0~5
    """
    failure_prob = binary_model.predict_proba(X)[:, 1]
    failure_pred = (failure_prob >= threshold).astype(int)
    class_prob = multi_model.predict_proba(X)
    class_pred = multi_model.predict(X)
    return failure_prob, failure_pred, class_prob, class_pred
