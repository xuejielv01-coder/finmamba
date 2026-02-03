#!/bin/bash

# Cloud Studio 训练启动脚本
# 优化内存使用，确保在Cloud Studio环境中稳定运行

echo "=== Cloud Studio 训练启动脚本 ==="
echo "启动时间: $(date)"
echo "工作目录: $(pwd)"

# 1. 环境检查
echo "\n1. 环境检查"

# 检查 Python 版本
python --version

# 检查 GPU 可用性
if command -v nvidia-smi &> /dev/null; then
    echo "\nGPU 状态:"
    nvidia-smi | head -10
else
    echo "\n⚠️  未检测到 GPU，将使用 CPU 模式"
    echo "提示: 请在 Cloud Studio 中选择带有 GPU 的规格"
fi

# 2. 依赖安装
echo "\n2. 依赖安装"

# 升级 pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt

# 3. 内存优化配置
echo "\n3. 内存优化配置"

# 设置内存优化环境变量
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export TORCH_NVML_DIR="/usr/local/cuda/nvml/lib"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"

# 清理 CUDA 缓存
echo "清理 CUDA 缓存..."
python -c "import torch; torch.cuda.empty_cache()"

# 4. 数据准备
echo "\n4. 数据准备"

# 检查数据目录
if [ ! -d "data/raw" ]; then
    echo "创建数据目录..."
    mkdir -p data/raw data/processed data/cache models/checkpoints logs
fi

# 下载数据
echo "下载历史数据..."
python -m data.downloader || echo "⚠️  数据下载可能需要较长时间，可在训练前手动运行"

# 5. 模型初始化测试
echo "\n5. 模型初始化测试"

python -c "
import sys
import torch

sys.path.insert(0, '.')

# 测试内存使用
print('Python 内存使用:', torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 'N/A')

# 尝试导入模型
print('导入模型...')
try:
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
    
    # 测试内存使用
    if torch.cuda.is_available():
        model = model.cuda()
        print('模型移动到 GPU 成功')
        print('GPU 内存使用:', torch.cuda.memory_allocated() / 1e9, 'GB')
        print('GPU 内存总量:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
        
        # 测试前向传播
        batch_size = min(Config.BATCH_SIZE, 32)
        dummy_input = torch.randn(batch_size, Config.SEQ_LEN, Config.FEATURE_DIM).cuda()
        dummy_industry = torch.randint(0, Config.N_INDUSTRIES, (batch_size,)).cuda()
        
        print(f'测试前向传播 (batch_size={batch_size})...')
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = model(dummy_input, dummy_industry)
        print('前向传播成功! 输出形状:', output.shape)
        print('前向传播后 GPU 内存使用:', torch.cuda.memory_allocated() / 1e9, 'GB')
        
        # 清理
        torch.cuda.empty_cache()
        print('内存清理完成')
    
    print('\n✅ 模型初始化测试通过!')
    print('配置适合 Cloud Studio 环境')
    
except Exception as e:
    print(f'\n❌ 模型初始化测试失败: {e}')
    import traceback
    traceback.print_exc()
    print('\n提示: 请检查配置文件和依赖安装')
"

# 6. 启动训练
echo "\n6. 启动训练"
echo "使用配置:"
echo "- 序列长度: $(python -c 'from config.config import Config; print(Config.SEQ_LEN)')"
echo "- 批次大小: $(python -c 'from config.config import Config; print(Config.BATCH_SIZE)')"
echo "- 梯度累积: $(python -c 'from config.config import Config; print(Config.GRAD_ACCUM_STEPS)')"
echo "- 训练年数: $(python -c 'from config.config import Config; print(Config.TRAIN_YEARS)')"

# 直接启动训练（移除交互式询问，便于在 Cloud Studio 中自动化运行）
echo "\n启动训练..."
echo "训练日志将输出到控制台和 logs/ 目录"
echo "==============================================="
echo "开始训练流程..."
echo "==============================================="

# 启动训练，添加详细的错误处理
python -m training.trainer || {
    echo "\n❌ 训练启动失败!"
    echo "\n错误分析:"
    echo "1. 检查数据文件是否存在"
    echo "2. 检查配置文件参数是否正确"
    echo "3. 检查 GPU 内存是否充足"
    echo "4. 检查依赖是否安装完整"
    echo "\n建议:"
    echo "- 尝试减小 config.py 中的 BATCH_SIZE"
    echo "- 确保 data/raw 目录中有数据文件"
    echo "- 运行 'python -m data.downloader' 下载数据"
    exit 1
}

# 7. 训练完成处理
echo "\n=== 训练启动脚本完成 ==="
echo "完成时间: $(date)"
echo "\n提示:"
echo "- 训练过程中可使用 'nvidia-smi' 监控 GPU 使用"
echo "- 训练日志保存在 logs/ 目录"
echo "- 模型检查点保存在 models/checkpoints/ 目录"
echo "- 如遇内存错误，请减小 config.py 中的 BATCH_SIZE"
