# -*- coding: utf-8 -*-
"""后端 API 自动化测试：/predict、/anomaly、/schedule 三个接口。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


class TestAPI(unittest.TestCase):
    def test_predict(self):
        r = client.post("/predict", json={
            "air_temperature": 300, "process_temperature": 310,
            "rotational_speed": 1500, "torque": 40, "tool_wear": 100, "type": "L",
        })
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertIn("failure_prob", d)
        self.assertIn("failure_type", d)
        self.assertIn("class_probs", d)
        self.assertGreaterEqual(d["failure_prob"], 0)
        self.assertLessEqual(d["failure_prob"], 1)

    def test_anomaly(self):
        seq = [{"air_temperature": 300, "process_temperature": 310, "rotational_speed": 1500,
                "torque": 40, "tool_wear": 100 + i, "type": "L"} for i in range(10)]
        r = client.post("/anomaly", json={"sequence": seq})
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertIn("reconstruction_error", d)
        self.assertIn("is_anomaly", d)

    def test_schedule(self):
        devices = [{"device_id": f"dev-{i}", "risk": 0.1 + 0.1 * i, "downtime_cost": 100 + 50 * i,
                    "resource": 2 + i, "maintenance_time": 1 + i} for i in range(5)]
        r = client.post("/schedule", json={"devices": devices, "resource_capacity": 6})
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertIn("total_loss", d)
        self.assertEqual(len(d["plan"]), 5)


if __name__ == "__main__":
    unittest.main()
