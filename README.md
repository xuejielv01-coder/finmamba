# FinMamba

金融市场预测模型，基于 Mamba 架构的深度学习模型。

## 项目结构

- `data/`: 数据处理模块
- `models/`: 模型定义
- `train/`: 训练相关
- `config/`: 配置文件
- `utils/`: 工具函数

## 环境要求

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.7+ (推荐 A800 GPU)

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 下载数据
```bash
python -m data.downloader
```

### 开始训练
```bash
python -m train.trainer
```

## 配置说明

配置文件位于 `config/config.py`，主要配置项：

- `SEQ_LEN`: 回看天数
- `D_MODEL`: 模型隐藏维度
- `BATCH_SIZE`: 批次大小
- `TRAIN_YEARS`: 训练数据年数
- `MAX_EPOCHS`: 最大训练轮数

## A800 GPU 优化

本项目已针对 A800 GPU 进行了优化：
- 更大的模型维度和批次大小
- 混合精度训练 (FP16)
- 多尺度时序分析
- 行业嵌入和关系建模
