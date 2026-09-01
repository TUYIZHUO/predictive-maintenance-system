# -*- coding: utf-8 -*-
"""算法模块 1（随机森林分类器）的自动化测试：训练、保存/加载、推理接口。"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.datasets import make_classification

from models.rf_classifier import (
    load_model,
    predict,
    save_model,
    train_binary,
    train_multi,
)


class TestRandomForestClassifier(unittest.TestCase):
    def setUp(self):
        self.X_bin, self.y_bin = make_classification(
            n_samples=400, n_features=8, n_informative=5,
            weights=[0.85, 0.15], random_state=42,
        )
        self.X_mul, self.y_mul = make_classification(
            n_samples=400, n_features=8, n_classes=6,
            n_informative=5, n_clusters_per_class=1, random_state=1,
        )
        self.bin_model = train_binary(self.X_bin, self.y_bin, n_estimators=20)
        self.mul_model = train_multi(self.X_mul, self.y_mul, n_estimators=20)
        self.X_test = self.X_bin[:5]  # 特征维度相同，可喂给两个模型

    def test_binary_prob_shape(self):
        prob = self.bin_model.predict_proba(self.X_bin)[:, 1]
        self.assertEqual(prob.shape, (len(self.X_bin),))
        self.assertTrue(((prob >= 0) & (prob <= 1)).all())

    def test_multi_output(self):
        pred = self.mul_model.predict(self.X_mul)
        self.assertEqual(pred.shape, (len(self.X_mul),))
        self.assertTrue(set(pred).issubset(set(range(6))))

    def test_predict_interface(self):
        prob, pred, class_prob, class_pred = predict(
            self.bin_model, self.mul_model, self.X_test, threshold=0.5
        )
        self.assertEqual(prob.shape, (5,))
        self.assertEqual(pred.shape, (5,))
        self.assertEqual(class_prob.shape, (5, 6))
        self.assertEqual(class_pred.shape, (5,))
        self.assertTrue(((prob >= 0) & (prob <= 1)).all())

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "rf_bin.pkl")
            save_model(self.bin_model, p)
            loaded = load_model(p)
            np.testing.assert_array_equal(loaded.predict(self.X_bin), self.bin_model.predict(self.X_bin))


if __name__ == "__main__":
    unittest.main()
