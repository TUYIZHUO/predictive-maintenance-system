# -*- coding: utf-8 -*-
"""算法模块 1、2 的自动化测试：模型输出形状、前向传播、损失、阈值判定。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.lstm_ae import LSTMAutoencoder, compute_threshold, is_anomaly, reconstruction_error
from models.lstm_classifier import LSTMClassifier, binary_loss, multi_loss, predict


class TestLSTMClassifier(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.batch, self.seq, self.feat = 4, 10, 8
        self.model = LSTMClassifier(input_size=self.feat)

    def test_output_shapes(self):
        x = torch.randn(self.batch, self.seq, self.feat)
        bin_logit, multi_logit = self.model(x)
        self.assertEqual(bin_logit.shape, (self.batch,))
        self.assertEqual(multi_logit.shape, (self.batch, 6))

    def test_forward_is_finite(self):
        x = torch.randn(self.batch, self.seq, self.feat)
        self.model.eval()
        with torch.no_grad():
            bin_logit, multi_logit = self.model(x)
        self.assertTrue(torch.isfinite(bin_logit).all())
        self.assertTrue(torch.isfinite(multi_logit).all())

    def test_losses(self):
        bin_logit = torch.randn(self.batch)
        y_bin = torch.randint(0, 2, (self.batch,)).float()
        self.assertTrue(torch.isfinite(binary_loss(bin_logit, y_bin)))

        multi_logit = torch.randn(self.batch, 6)
        y_multi = torch.randint(0, 6, (self.batch,)).long()
        self.assertTrue(torch.isfinite(multi_loss(multi_logit, y_multi)))

    def test_predict(self):
        bin_logit = torch.randn(self.batch)
        multi_logit = torch.randn(self.batch, 6)
        prob, pred, class_prob, class_pred = predict(bin_logit, multi_logit)
        self.assertEqual(prob.shape, (self.batch,))
        self.assertEqual(pred.shape, (self.batch,))
        self.assertEqual(class_prob.shape, (self.batch, 6))
        self.assertEqual(class_pred.shape, (self.batch,))
        self.assertTrue(((prob >= 0) & (prob <= 1)).all())


class TestLSTMAutoencoder(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.batch, self.seq, self.feat = 4, 10, 8
        self.model = LSTMAutoencoder(input_size=self.feat)

    def test_recon_shape(self):
        x = torch.randn(self.batch, self.seq, self.feat)
        recon = self.model(x)
        self.assertEqual(recon.shape, x.shape)

    def test_reconstruction_error(self):
        x = torch.randn(self.batch, self.seq, self.feat)
        with torch.no_grad():
            recon = self.model(x)
        err = reconstruction_error(recon, x)
        self.assertEqual(err.shape, (self.batch,))
        self.assertTrue((err >= 0).all())

    def test_threshold_and_anomaly(self):
        errors = torch.tensor([0.10, 0.20, 0.15, 0.12, 0.18])
        tau = compute_threshold(errors, k=3.0)
        # 正常误差应低于阈值
        self.assertGreater(tau, errors.max().item())
        # 远超阈值的大误差应被判为异常
        self.assertTrue(is_anomaly(torch.tensor([tau + 1.0]), tau).item())


if __name__ == "__main__":
    unittest.main()
