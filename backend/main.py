# -*- coding: utf-8 -*-
"""后端服务：FastAPI 加载算法模块，提供 /predict、/anomaly、/schedule 接口。

启动（在项目根目录）：
    uvicorn backend.main:app --reload
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from fastapi import Depends, FastAPI
from pydantic import BaseModel
from sqlalchemy.orm import Session

# 让 models / optimizer / backend 可导入（无论从哪个目录启动）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import database  # noqa: E402
from models.lstm_ae import LSTMAutoencoder, reconstruction_error  # noqa: E402
from models.rf_classifier import load_model, predict  # noqa: E402
from optimizer.ga_scheduler import MaintenanceScheduler, build_devices, schedule_to_plan  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
CKPT_DIR = BASE_DIR / "checkpoints"
DATA_DIR = BASE_DIR / "data" / "processed"

CLASS_NAMES = ["No_Failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
TYPE_ONEHOT = {"L": [1, 0, 0], "M": [0, 1, 0], "H": [0, 0, 1]}


def _read_kv(path: Path, key: str, default: float) -> float:
    """从 config 文件读 key=value 行，找不到返回 default。"""
    if not path.exists():
        return default
    for line in path.read_text().splitlines():
        if line.startswith(f"{key}="):
            return float(line.split("=")[1])
    return default


# ---- 启动时加载模型与配置 ----
rf_binary = load_model(str(CKPT_DIR / "rf_binary.pkl"))
rf_multi = load_model(str(CKPT_DIR / "rf_multi.pkl"))
cls_threshold = _read_kv(CKPT_DIR / "classifier_config.txt", "threshold", 0.5)

with open(DATA_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

ae_model = LSTMAutoencoder(input_size=8)
ae_model.load_state_dict(torch.load(CKPT_DIR / "ae_best.pth", map_location="cpu", weights_only=True))
ae_model.eval()
ae_threshold = _read_kv(CKPT_DIR / "ae_config.txt", "threshold", 0.6665)

database.init_db()
app = FastAPI(title="设备预测性维护系统", version="1.0")


def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


def prepare_features(air: float, process: float, speed: float, torque: float, wear: float, type_: str) -> np.ndarray:
    """标准化 5 个数值特征 + Type one-hot，拼接为 8 维特征。"""
    nums = scaler.transform([[air, process, speed, torque, wear]])[0].astype(np.float32)
    oh = np.array(TYPE_ONEHOT[type_.upper()], dtype=np.float32)
    return np.concatenate([nums, oh])


# ---- 请求模型 ----
class FeatureIn(BaseModel):
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float
    type: str


class SequenceIn(BaseModel):
    sequence: List[FeatureIn]


class DeviceIn(BaseModel):
    device_id: str
    risk: float
    downtime_cost: float
    resource: float
    maintenance_time: float


class ScheduleIn(BaseModel):
    devices: List[DeviceIn]
    resource_capacity: float = 6.0


# ---- 路由 ----
@app.get("/")
def root():
    return {"service": "设备预测性维护系统后端", "docs": "/docs"}


@app.post("/predict")
def predict_failure(feat: FeatureIn, db: Session = Depends(get_db)):
    x = prepare_features(feat.air_temperature, feat.process_temperature,
                         feat.rotational_speed, feat.torque, feat.tool_wear, feat.type)
    prob, pred, class_prob, class_id = predict(rf_binary, rf_multi, x.reshape(1, -1), cls_threshold)
    prob, pred, cid = float(prob[0]), int(pred[0]), int(class_id[0])
    ftype = CLASS_NAMES[cid]

    db.add(database.PredictionRecord(
        air_temperature=feat.air_temperature, process_temperature=feat.process_temperature,
        rotational_speed=feat.rotational_speed, torque=feat.torque, tool_wear=feat.tool_wear,
        type=feat.type, failure_prob=prob, failure_type=ftype,
    ))
    db.commit()

    return {
        "failure_prob": round(prob, 4),
        "is_failure": bool(pred),
        "failure_type": ftype,
        "failure_type_id": cid,
        "class_probs": {CLASS_NAMES[i]: round(float(class_prob[0][i]), 4) for i in range(len(CLASS_NAMES))},
    }


@app.post("/anomaly")
def detect_anomaly(seq: SequenceIn):
    x = np.stack([prepare_features(s.air_temperature, s.process_temperature, s.rotational_speed,
                                   s.torque, s.tool_wear, s.type) for s in seq.sequence])
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        recon = ae_model(x_t)
    err = float(reconstruction_error(recon, x_t).item())
    return {"reconstruction_error": round(err, 6), "threshold": round(ae_threshold, 6), "is_anomaly": err > ae_threshold}


@app.post("/schedule")
def make_schedule(sch: ScheduleIn, db: Session = Depends(get_db)):
    devices = build_devices(
        ids=[d.device_id for d in sch.devices],
        risks=[d.risk for d in sch.devices],
        downtime_costs=[d.downtime_cost for d in sch.devices],
        resources=[d.resource for d in sch.devices],
        maintenance_times=[d.maintenance_time for d in sch.devices],
    )
    scheduler = MaintenanceScheduler(devices, resource_capacity=sch.resource_capacity)
    order, total_loss, _ = scheduler.run()
    plan = schedule_to_plan(devices, order, sch.resource_capacity)

    db.add(database.ScheduleRecord(total_loss=round(float(total_loss), 2),
                                   plan_json=json.dumps(plan, ensure_ascii=False)))
    db.commit()

    return {"total_loss": round(float(total_loss), 2), "plan": plan}
