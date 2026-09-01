# -*- coding: utf-8 -*-
"""后端数据库层：SQLite + SQLAlchemy，存储预测记录与维护计划。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import Column, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "predictive_maintenance.db"

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class PredictionRecord(Base):
    """一次故障预测的存档。"""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(String, default=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    air_temperature = Column(Float)
    process_temperature = Column(Float)
    rotational_speed = Column(Float)
    torque = Column(Float)
    tool_wear = Column(Float)
    type = Column(String)
    failure_prob = Column(Float)
    failure_type = Column(String)


class ScheduleRecord(Base):
    """一次维护排程的存档。"""
    __tablename__ = "schedules"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(String, default=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    total_loss = Column(Float)
    plan_json = Column(Text)


def init_db() -> None:
    """建表（幂等）。"""
    Base.metadata.create_all(bind=engine)
