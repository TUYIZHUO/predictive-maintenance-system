# 设备预测性维护智能应用

面向制造"设备运维"场景的预测性维护系统。基于公开工业数据集 **AI4I 2020**
（UCI），实现故障分类、异常检测、维护排程优化三个技术方向，采用 B/S 架构。

## 技术方向映射表

| 技术方向 | 课程章节 | 在本系统中的实际作用 |
|---------|---------|-------------------|
| 集成学习（随机森林） | 监督学习/集成学习 | 随机森林分类器：识别设备是否将发生故障及故障类型 |
| 无监督学习 / 异常检测 | 无监督学习 | LSTM 自编码器：以重构误差做设备异常早期预警 |
| 智能优化 / 进化计算 | 优化方法 | 遗传算法：求解维护优先级排程，最小化停机损失 |

## 目录结构

```
predictive-maintenance/
├── data/
│   ├── preprocess.py        # 数据清洗、编码、标准化、序列构建、划分
│   ├── raw/                 # 原始数据集 ai4i2020.csv
│   └── processed/           # 划分后的 npz 与 scaler.pkl
├── models/
│   ├── rf_classifier.py     # 算法模块1：随机森林分类器（二分类+多分类）
│   ├── lstm_classifier.py   # 对照实验保留：LSTM 分类器（双头输出）
│   └── lstm_ae.py           # 算法模块2：LSTM 自编码器（异常检测）
├── optimizer/
│   └── ga_scheduler.py      # 算法模块3：遗传算法维护排程
├── backend/
│   ├── database.py          # SQLite + SQLAlchemy 数据层
│   └── main.py              # FastAPI 服务：/predict /anomaly /schedule
├── frontend/
│   └── app.py               # Streamlit 看板（支持 Excel 导入）
├── tests/
│   ├── test_models.py       # 模块1、2 的自动化测试
│   ├── test_ga.py           # 模块3 的自动化测试
│   ├── test_rf.py           # 随机森林模块测试
│   └── test_api.py          # 后端三个接口测试
├── train_classifier.py      # 训练随机森林分类器
├── train_ae.py              # 训练自编码器 + 阈值确定
├── checkpoints/             # 训练产出（模型与阈值配置）
├── 启动说明.txt             # 启动系统 + 进入网站 + 使用说明
├── requirements.txt
└── README.md
```

## 数据来源

本系统使用公开数据集 **AI4I 2020 Predictive Maintenance Dataset**：

- **来源**：UCI Machine Learning Repository（数据集 ID 601）
- **下载链接**：https://archive.ics.uci.edu/dataset/601/predictive+maintenance+dataset
- **原始论文**：S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications", AI4I 2020, doi:10.1109/AI4I49448.2020.00023
- **规模**：10,000 条记录 × 14 列，无缺失值
- **场景**：铣床设备运行快照，含 5 个数值传感特征 + 产品类型，标注 5 类故障模式

字段含义、标签分布、目录结构详见 `data/README.md`。

## 数据预处理

预处理程序 `data/preprocess.py` 完成以下工作：

1. **清洗**：缺失值检查（无缺失）、删除 ID 列、统一列名
2. **编码**：产品类型 Type(L/M/H) One-Hot；故障标签编码为二分类(0/1)与多分类(6 类)
3. **标准化**：5 个数值特征做 StandardScaler（仅训练集拟合，避免数据泄漏）
4. **划分**：按 7:1.5:1.5 划分训练/验证/测试集（7000 / 1500 / 1500）
5. **序列构建**：滑动窗口（长度 10）构建 3D 序列供 LSTM 使用
6. **过采样**：手动实现 SMOTE，仅训练集缓解 3.39% 故障不平衡

产出文件位于 `data/processed/`：`splits_2d.npz`（传统 ML）、`splits_3d.npz`（LSTM）、
`scaler.pkl`、`label_encoder.pkl`、`dataset_stats.json`（数据统计）、
`feature_stats.csv`（数值特征描述统计）。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据

从 UCI（https://archive.ics.uci.edu/dataset/601/predictive+maintenance+dataset）
下载 AI4I 2020 数据集，保存为 `data/raw/ai4i2020.csv`，然后运行预处理：

```bash
python data/preprocess.py
```

### 3. 训练模型

```bash
python train_classifier.py
python train_ae.py
```

产出：`checkpoints/rf_binary.pkl`、`checkpoints/rf_multi.pkl`、
`checkpoints/classifier_config.txt`、`checkpoints/ae_best.pth`、
`checkpoints/ae_config.txt`。

### 4. 运行自动化测试

```bash
python -m unittest discover -s tests -v
```

## 启动系统

后端 + 前端的启动命令、进入网站的网址、三个功能及 Excel 导入格式，
详见 **启动说明.txt**。

## 关键设计决策

1. **为什么分类用随机森林而非 LSTM**：AI4I 2020 每行是独立设备快照、并非
   严格时序，LSTM 序列建模会丢失温度/转速/扭矩等特征的直接判别力。对照实验
   显示故障 F1：LSTM 0.32 vs 随机森林 0.62（阈值 0.23，召回率 71%），故分类
   模块选用随机森林。LSTM 分类器代码保留在 `models/lstm_classifier.py` 作对照。

2. **为什么用 class_weight 而非 SMOTE**：随机森林在二维特征上用
   `class_weight="balanced"` 直接缓解 3.4% 故障率的不平衡，避免 SMOTE 对
   多分类极少数类（如 OSF 仅 9 条）采样失败的问题，更简洁可解释。

3. **遗传算法的真实作用**：适应度 = 按优先级贪心调度后，各设备
   `风险 × 停机损失 × (维护耗时 + 推迟天数 × 惩罚)` 之和。排在前面的设备
   推迟天数少，因此把高风险、高停机损失的设备优先维护能降低总损失。
   这是量化的优化目标，非装饰性调用。
