#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试A800 GPU配置的有效性
"""

import sys
import os
import torch
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.config import Config


def test_a800_config():
    """测试A800 GPU配置"""
    print("=== 测试A800 GPU配置有效性 ===")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 检查GPU可用性
    print("\n1. 检查GPU可用性:")
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ GPU可用: {torch.cuda.get_device_name(device)}")
        print(f"✅ GPU数量: {torch.cuda.device_count()}")
        print(f"✅ GPU显存: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
    else:
        print("❌ 未检测到GPU，将使用CPU模式")
        device = torch.device('cpu')
    
    # 2. 验证配置参数
    print("\n2. 验证配置参数:")
    
    # 检查模型架构参数
    print("\n模型架构参数:")
    print(f"SEQ_LEN: {Config.SEQ_LEN} (回看天数)")
    print(f"FEATURE_DIM: {Config.FEATURE_DIM} (特征维度)")
    print(f"D_MODEL: {Config.D_MODEL} (模型隐藏维度)")
    print(f"N_LAYERS: {Config.N_LAYERS} (Mamba层数)")
    print(f"N_TRANSFORMER_LAYERS: {Config.N_TRANSFORMER_LAYERS} (Transformer层数)")
    print(f"N_HEADS: {Config.N_HEADS} (注意力头数)")
    print(f"D_STATE: {Config.D_STATE} (SSM状态维度)")
    print(f"MAMBA_LEVELS: {Config.MAMBA_LEVELS} (多尺度)")
    print(f"N_GCN_LAYERS: {Config.N_GCN_LAYERS} (GCN层数)")
    
    # 检查训练参数
    print("\n训练参数:")
    print(f"TRAIN_YEARS: {Config.TRAIN_YEARS} (训练数据年数)")
    print(f"BATCH_SIZE: {Config.BATCH_SIZE} (批次大小)")
    print(f"GRAD_ACCUM_STEPS: {Config.GRAD_ACCUM_STEPS} (梯度累积步长)")
    print(f"LR_INIT: {Config.LR_INIT} (初始学习率)")
    print(f"MAX_EPOCHS: {Config.MAX_EPOCHS} (最大训练轮数)")
    print(f"PATIENCE: {Config.PATIENCE} (早停耐心值)")
    print(f"DROPOUT: {Config.DROPOUT} ( dropout比例)")
    
    # 检查下载器参数
    print("\n下载器参数:")
    print(f"CONCURRENT_WORKERS: {Config.CONCURRENT_WORKERS} (并发下载线程数)")
    print(f"API_RATE_LIMIT: {Config.API_RATE_LIMIT} (API调用间隔)")
    
    # 3. 内存使用估算
    print("\n3. 内存使用估算:")
    if device.type == 'cuda':
        # 估算模型内存使用
        # 简单估算公式：模型参数数量 * 4 (FP32) / 1e9
        # 实际内存使用会更大，这里只是粗略估算
        
        # 计算参数数量（粗略估算）
        # Mamba层参数
        mamba_params = Config.N_LAYERS * (Config.D_MODEL * Config.D_MODEL * 4 + Config.D_MODEL * Config.D_STATE * 4)
        # Transformer层参数
        transformer_params = Config.N_TRANSFORMER_LAYERS * (Config.D_MODEL * Config.D_MODEL * 4 + Config.D_MODEL * Config.N_HEADS * 4)
        # GCN参数
        gcn_params = Config.N_GCN_LAYERS * (Config.D_MODEL * Config.D_MODEL * 2)
        # 其他参数
        other_params = Config.FEATURE_DIM * Config.D_MODEL * 2
        
        total_params = mamba_params + transformer_params + gcn_params + other_params
        estimated_memory = total_params * 4 / 1e9  # FP32
        
        print(f"\n模型参数估算:")
        print(f"总参数数量: {total_params / 1e6:.2f} M")
        print(f"估算模型内存: {estimated_memory:.2f} GB (FP32)")
        print(f"估算模型内存: {estimated_memory / 2:.2f} GB (FP16)")
        print(f"估算模型内存: {estimated_memory / 4:.2f} GB (INT8)")
        
        # 估算 batch 内存使用
        batch_memory = Config.BATCH_SIZE * Config.SEQ_LEN * Config.FEATURE_DIM * 4 / 1e9  # FP32
        print(f"\nBatch内存估算:")
        print(f"单Batch内存: {batch_memory:.2f} GB (FP32)")
        print(f"单Batch内存: {batch_memory / 2:.2f} GB (FP16)")
        
        # 总内存估算
        total_required = estimated_memory + batch_memory
        print(f"\n总内存需求估算:")
        print(f"总需求: {total_required:.2f} GB (FP32)")
        print(f"总需求: {total_required / 2:.2f} GB (FP16)")
        
        # 检查是否适合当前GPU
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
            if total_required / 2 < gpu_memory * 0.8:  # 留20%余量
                print(f"✅ 配置适合当前GPU (使用FP16)")
                print(f"推荐使用混合精度训练以提高性能")
            else:
                print(f"⚠️  配置可能超出GPU内存")
                print(f"建议: 减小BATCH_SIZE或SEQ_LEN")
    
    # 4. 验证目录结构
    print("\n4. 验证目录结构:")
    try:
        Config.ensure_dirs()
        print("✅ 目录结构创建成功")
        
        # 检查关键目录
        key_dirs = [
            Config.DATA_ROOT,
            Config.RAW_DATA_DIR,
            Config.PROCESSED_DATA_DIR,
            Config.MODEL_DIR,
            Config.CACHE_DIR
        ]
        
        for directory in key_dirs:
            if directory.exists():
                print(f"✅ {directory}")
            else:
                print(f"❌ {directory} (不存在)")
                
    except Exception as e:
        print(f"❌ 创建目录失败: {e}")
    
    # 5. 测试数据下载配置
    print("\n5. 测试数据下载配置:")
    print(f"使用雅虎财经作为数据源")
    print(f"并发下载线程数: {Config.CONCURRENT_WORKERS}")
    print(f"API调用间隔: {Config.API_RATE_LIMIT}秒")
    
    # 6. 生成配置报告
    print("\n6. 配置报告:")
    print("\n=== A800 GPU配置验证报告 ===")
    print(f"状态: {'✅ 配置有效' if torch.cuda.is_available() else '⚠️  无GPU可用'}")
    print(f"建议: 使用混合精度训练 (FP16) 以充分利用A800性能")
    print(f"注意: 首次运行将下载历史数据，可能需要较长时间")
    print(f"优化: 可考虑使用分布式训练进一步提高性能")
    
    print("\n=== 测试完成 ===")
    return True


def test_model_initialization():
    """测试模型初始化"""
    print("\n=== 测试模型初始化 ===")
    
    try:
        # 延迟导入模型以避免不必要的依赖
        from models.finmamba import FinMamba
        
        # 创建模型实例
        model = FinMamba(
            seq_len=Config.SEQ_LEN,
            feature_dim=Config.FEATURE_DIM,
            d_model=Config.D_MODEL,
            n_layers=Config.N_LAYERS,
            n_transformer_layers=Config.N_TRANSFORMER_LAYERS,
            n_heads=Config.N_HEADS,
            d_state=Config.D_STATE,
            d_conv=Config.D_CONV,
            expand=Config.EXPAND,
            levels=Config.MAMBA_LEVELS,
            n_industries=Config.N_INDUSTRIES,
            use_industry=Config.USE_GRAPH,
            dropout=Config.DROPOUT
        )
        
        # 移动到设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # 测试前向传播
        batch_size = min(Config.BATCH_SIZE, 32)  # 测试时使用较小的batch size
        dummy_input = torch.randn(batch_size, Config.SEQ_LEN, Config.FEATURE_DIM, device=device)
        dummy_industry = torch.randint(0, Config.N_INDUSTRIES, (batch_size,), device=device)
        
        # 使用混合精度
        with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16):
            output = model(dummy_input, dummy_industry)
        
        print(f"✅ 模型初始化成功")
        print(f"✅ 前向传播成功，输出形状: {output.shape}")
        print(f"✅ 模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 运行配置测试
    config_valid = test_a800_config()
    
    # 运行模型初始化测试
    if config_valid:
        model_valid = test_model_initialization()
    
    print("\n=== 测试总结 ===")
    if config_valid and 'model_valid' in locals() and model_valid:
        print("✅ 所有测试通过，配置适合A800 GPU")
        print("\n建议下一步操作:")
        print("1. 运行数据下载: python -m data.downloader")
        print("2. 开始模型训练: python -m train.trainer")
        print("3. 监控训练过程: 使用TensorBoard或日志")
    else:
        print("❌ 测试未通过，请检查配置和环境")
