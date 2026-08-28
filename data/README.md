# 数据说明

本目录存放课程设计所用数据：原始数据 `raw/` 与预处理数据 `processed/`。

## 数据集概况

- **名称**：AI4I 2020 Predictive Maintenance Dataset
- **来源**：UCI Machine Learning Repository（数据集 ID：601）
- **下载链接**：https://archive.ics.uci.edu/dataset/601/predictive+maintenance+dataset
- **原始论文**：S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications", AI4I 2020, doi:10.1109/AI4I49448.2020.00023
- **规模**：10,000 条记录 × 14 列，无缺失值
- **场景**：铣床设备运行，用于预测性维护（故障分类 / 异常检测 / 维护排程）

## 原始字段（14 列）

| 字段 | 含义 | 单位 |
|---|---|---|
| UDI | 唯一标识 | — |
| Product ID | 产品编号 | — |
| Type | 产品类型（L=低 / M=中 / H=高质量等级） | — |
| Air temperature | 空气温度 | K |
| Process temperature | 工艺温度 | K |
| Rotational speed | 转速 | rpm |
| Torque | 扭矩 | Nm |
| Tool wear | 刀具磨损 | min |
| Machine failure | 是否发生故障（二分类标签） | 0/1 |
| TWF | 刀具磨损故障（Tool Wear Failure） | 0/1 |
| HDF | 散热故障（Heat Dissipation Failure） | 0/1 |
| PWF | 电源故障（Power Failure） | 0/1 |
| OSF | 过载故障（Overstrain Failure） | 0/1 |
| RNF | 随机故障（Random Failures） | 0/1 |

## 预处理后特征（8 维）

| 维度 | 特征 | 说明 |
|---|---|---|
| 1–5 | air_temperature, process_temperature, rotational_speed, torque, tool_wear | 数值特征，StandardScaler 标准化 |
| 6–8 | type_L, type_M, type_H | 产品类型 One-Hot 编码 |

## 标签分布

- **二分类**（Machine failure）：正常 9661（96.61%）/ 故障 339（3.39%）
- **多分类**（6 类）：

| 类别 | 含义 | 数量 |
|---|---|---|
| 0 No_Failure | 无故障 | 9652 |
| 1 TWF | 刀具磨损故障 | 42 |
| 2 HDF | 散热故障 | 106 |
| 3 PWF | 电源故障 | 83 |
| 4 OSF | 过载故障 | 98 |
| 5 RNF | 随机故障 | 19 |

## 目录结构

```
data/
├── raw/                  # 原始数据
│   └── ai4i2020.csv      # 10,000 × 14
├── processed/            # 预处理产物
│   ├── splits_2d.npz     # 2D 数据（传统 ML：随机森林）
│   ├── splits_3d.npz     # 3D 序列（LSTM：自编码器）
│   ├── scaler.pkl        # StandardScaler
│   ├── label_encoder.pkl # 标签编码器
│   ├── dataset_stats.json# 数据统计汇总
│   └── feature_stats.csv # 数值特征描述统计
└── preprocess.py         # 预处理脚本（清洗/编码/标准化/划分/序列/SMOTE）
```

## 许可

该数据集为公开学术数据集，由 S. Matzka 等人于 AI4I 2020 发布，仅供学术研究与教学使用，
具体许可以 UCI 数据集页面标注为准。
