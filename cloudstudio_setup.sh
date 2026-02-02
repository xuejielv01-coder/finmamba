
#!/bin/bash

# Cloud Studio 环境设置脚本

echo "=== FinMamba 环境设置 ==="
echo "设置时间: $(date)"

# 更新系统
echo "
1. 更新系统包"
sudo apt-get update -y

# 安装依赖
echo "
2. 安装系统依赖"
sudo apt-get install -y git curl wget build-essential

# 安装 Python 依赖
echo "
3. 安装 Python 依赖"
pip install --upgrade pip
pip install -r requirements.txt

# 配置 GPU 环境
echo "
4. 配置 GPU 环境"
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU 检测到: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"
    echo "✅ GPU 显存: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
else
    echo "⚠️  未检测到 GPU，将使用 CPU 模式"
fi

# 下载数据
echo "
5. 下载历史数据"
python -m data.downloader

# 测试模型初始化
echo "
6. 测试模型初始化"
python -c "
import sys
sys.path.insert(0, '.')
from config.config import Config
from models.finmamba import FinMamba

print('创建模型实例...')
model = FinMamba(
    seq_len=Config.SEQ_LEN,
    feature_dim=Config.FEATURE_DIM,
    d_model=Config.D_MODEL,
    n_layers=Config.N_LAYERS,
    n_transformer_layers=Config.N_TRANSFORMER_LAYERS,
    n_heads=Config.N_HEADS,
    d_state=Config.D_STATE,
    levels=Config.MAMBA_LEVELS,
    n_industries=Config.N_INDUSTRIES,
    use_industry=Config.USE_GRAPH,
    dropout=Config.DROPOUT
)
print('模型创建成功!')
print(f'模型参数: {model.count_parameters():,}')
"

echo "
=== 设置完成 ==="
echo "使用以下命令开始训练:"
echo "python -m train.trainer"
