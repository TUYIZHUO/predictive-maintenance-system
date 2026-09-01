# -*- coding: utf-8 -*-
"""算法模块 3（遗传算法）的自动化测试：辅助函数、适应度、收敛性、计划输出。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimizer.ga_scheduler import (
    MaintenanceScheduler,
    build_devices,
    risk_to_level,
    schedule_to_plan,
)


class TestHelpers(unittest.TestCase):
    def test_risk_to_level(self):
        self.assertEqual(risk_to_level(0.1), 0)
        self.assertEqual(risk_to_level(0.4), 1)
        self.assertEqual(risk_to_level(0.7), 2)

    def test_build_devices(self):
        devices = build_devices(["a", "b"], [0.1, 0.9], [100, 200], [2, 4], [1, 3])
        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[1].device_id, "b")
        self.assertAlmostEqual(devices[1].risk, 0.9)


class TestMaintenanceScheduler(unittest.TestCase):
    def setUp(self):
        self.devices = build_devices(
            ids=[f"dev-{i}" for i in range(6)],
            risks=[0.1, 0.9, 0.3, 0.8, 0.2, 0.7],
            downtime_costs=[100, 500, 200, 400, 150, 350],
            resources=[2, 4, 3, 5, 2, 4],
            maintenance_times=[1, 3, 2, 4, 1, 3],
        )
        self.scheduler = MaintenanceScheduler(
            self.devices, resource_capacity=6, pop_size=30, n_generations=80, seed=42
        )

    def test_fitness_positive(self):
        f = self.scheduler._fitness(list(range(6)))
        self.assertIsInstance(f, float)
        self.assertGreater(f, 0)

    def test_high_risk_first_reduces_loss(self):
        # 把高风险设备排前面的排序，总损失应小于逆序
        risk_sorted = sorted(range(6), key=lambda i: -self.devices[i].risk)
        reverse = risk_sorted[::-1]
        self.assertLess(
            self.scheduler._fitness(risk_sorted), self.scheduler._fitness(reverse)
        )

    def test_run_returns_valid_permutation(self):
        order, fitness, history = self.scheduler.run()
        self.assertEqual(sorted(order), list(range(6)))
        self.assertIsInstance(fitness, float)
        self.assertEqual(len(history), self.scheduler.n_generations + 1)

    def test_ga_converges(self):
        _, _, history = self.scheduler.run()
        # 精英保留保证历史最优单调不增
        self.assertLessEqual(history[-1], history[0])

    def test_schedule_to_plan(self):
        order, _, _ = self.scheduler.run()
        plan = schedule_to_plan(self.devices, order, resource_capacity=6)
        self.assertEqual(len(plan), 6)
        for item in plan:
            self.assertIn("priority", item)
            self.assertIn("risk_level", item)
            self.assertIn("batch_day", item)


if __name__ == "__main__":
    unittest.main()
