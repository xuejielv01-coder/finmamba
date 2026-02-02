#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cloud Studio 自动配置脚本
使用 Git 方式上传项目到 Cloud Studio
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime


class CloudStudioSetup:
    def __init__(self):
        self.project_root = os.path.abspath(os.path.dirname(__file__))
        self.git_repo = None
        self.workspace_name = "finmamba"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def check_requirements(self):
        """检查系统要求"""
        print("=== 检查系统要求 ===")
        
        # 检查 Git
        try:
            result = subprocess.run(["git", "--version"], capture_output=True, text=True, check=True)
            print(f"✅ Git 已安装: {result.stdout.strip()}")
        except Exception as e:
            print(f"❌ Git 未安装: {e}")
            return False
        
        # 检查 Python
        try:
            result = subprocess.run(["python", "--version"], capture_output=True, text=True, check=True)
            print(f"✅ Python 已安装: {result.stdout.strip()}")
        except Exception as e:
            print(f"❌ Python 未安装: {e}")
            return False
        
        return True
    
    def setup_git_repo(self):
        """设置 Git 仓库"""
        print("\n=== 设置 Git 仓库 ===")
        
        # 检查是否已有 .git 目录
        git_dir = os.path.join(self.project_root, ".git")
        if os.path.exists(git_dir):
            print("✅ 已有 Git 仓库")
            return True
        
        # 初始化 Git 仓库
        try:
            subprocess.run(["git", "init"], cwd=self.project_root, check=True)
            print("✅ 初始化 Git 仓库成功")
            
            # 创建 .gitignore 文件
            gitignore_content = """
# 依赖包
node_modules/
pip-wheel-metadata/
*.egg-info/
dist/
build/

# 环境文件
.env
.env.local
.env.*.local

# IDE 和编辑器
.vscode/
.idea/
*.swp
*.swo
*~

# 操作系统
.DS_Store
Thumbs.db

# 数据文件
data/raw/
data/processed/
data/cache/
models/checkpoints/
logs/

# 临时文件
*.tmp
*.temp
*.log
"""
            
            gitignore_path = os.path.join(self.project_root, ".gitignore")
            with open(gitignore_path, "w", encoding="utf-8") as f:
                f.write(gitignore_content)
            print("✅ 创建 .gitignore 文件成功")
            
            return True
            
        except Exception as e:
            print(f"❌ 设置 Git 仓库失败: {e}")
            return False
    
    def configure_project(self):
        """配置项目"""
        print("\n=== 配置项目 ===")
        
        # 检查并创建必要的配置文件
        config_files = [
            "requirements.txt",
            "README.md"
        ]
        
        for config_file in config_files:
            file_path = os.path.join(self.project_root, config_file)
            if not os.path.exists(file_path):
                print(f"⚠️  {config_file} 不存在，创建默认版本")
                self._create_default_file(config_file)
            else:
                print(f"✅ {config_file} 已存在")
        
        return True
    
    def _create_default_file(self, filename):
        """创建默认配置文件"""
        if filename == "requirements.txt":
            content = """
torch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
tqdm==4.65.0
matplotlib==3.7.2
seaborn==0.12.2
pyarrow==15.0.0
fastparquet==2023.7.0
yfinance==0.2.28
akshare==1.7.99
requests==2.31.0
python-dotenv==1.0.0
"""
        elif filename == "README.md":
            content = """# FinMamba

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
"""
        else:
            content = ""
        
        file_path = os.path.join(self.project_root, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
    
    def create_setup_script(self):
        """创建 Cloud Studio 设置脚本"""
        print("\n=== 创建 Cloud Studio 设置脚本 ===")
        
        setup_script_content = """
#!/bin/bash

# Cloud Studio 环境设置脚本

echo "=== FinMamba 环境设置 ==="
echo "设置时间: $(date)"

# 更新系统
echo "\n1. 更新系统包"
sudo apt-get update -y

# 安装依赖
echo "\n2. 安装系统依赖"
sudo apt-get install -y git curl wget build-essential

# 安装 Python 依赖
echo "\n3. 安装 Python 依赖"
pip install --upgrade pip
pip install -r requirements.txt

# 配置 GPU 环境
echo "\n4. 配置 GPU 环境"
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU 检测到: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"
    echo "✅ GPU 显存: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
else
    echo "⚠️  未检测到 GPU，将使用 CPU 模式"
fi

# 下载数据
echo "\n5. 下载历史数据"
python -m data.downloader

# 测试模型初始化
echo "\n6. 测试模型初始化"
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

echo "\n=== 设置完成 ==="
echo "使用以下命令开始训练:"
echo "python -m train.trainer"
"""
        
        script_path = os.path.join(self.project_root, "cloudstudio_setup.sh")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(setup_script_content)
        
        # 设置执行权限
        os.chmod(script_path, 0o755)
        print(f"✅ 创建设置脚本成功: {script_path}")
        
        return True
    
    def create_cloudstudio_config(self):
        """创建 Cloud Studio 配置文件"""
        print("\n=== 创建 Cloud Studio 配置文件 ===")
        
        config_content = {
            "name": self.workspace_name,
            "description": "FinMamba 金融模型训练",
            "spec": "8core-32g",
            "gpu": "A800",
            "env": "python",
            "startup_script": "./cloudstudio_setup.sh",
            "git_repository": {
                "url": "",  # 将在用户提供 Git 仓库后填写
                "branch": "main"
            }
        }
        
        config_path = os.path.join(self.project_root, "cloudstudio_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_content, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 创建配置文件成功: {config_path}")
        return True
    
    def run(self):
        """执行完整设置流程"""
        print("=== Cloud Studio 自动配置工具 ===")
        print(f"项目路径: {self.project_root}")
        print(f"时间戳: {self.timestamp}")
        
        # 检查系统要求
        if not self.check_requirements():
            print("\n❌ 系统要求检查失败，退出设置")
            return False
        
        # 设置 Git 仓库
        if not self.setup_git_repo():
            print("\n❌ Git 仓库设置失败，退出设置")
            return False
        
        # 配置项目
        if not self.configure_project():
            print("\n❌ 项目配置失败，退出设置")
            return False
        
        # 创建设置脚本
        if not self.create_setup_script():
            print("\n❌ 设置脚本创建失败，退出设置")
            return False
        
        # 创建 Cloud Studio 配置
        if not self.create_cloudstudio_config():
            print("\n❌ Cloud Studio 配置创建失败，退出设置")
            return False
        
        print("\n=== 设置完成 ===")
        print("\n下一步操作:")
        print("1. 在 GitHub/Gitee 创建新仓库")
        print("2. 将本地代码推送到远程仓库:")
        print("   git remote add origin <your_repository_url>")
        print("   git add .")
        print("   git commit -m 'Initial commit'")
        print("   git push -u origin main")
        print("3. 在 Cloud Studio 中:")
        print("   - 点击 '新建工作空间'")
        print("   - 选择 '从 Git 仓库创建'")
        print("   - 输入你的仓库地址")
        print("   - 选择 A800 GPU 和 8core-32g 规格")
        print("   - 启动工作空间")
        print("4. 工作空间启动后，脚本会自动:")
        print("   - 安装依赖")
        print("   - 下载历史数据")
        print("   - 测试模型初始化")
        print("\n5. 开始训练:")
        print("   python -m train.trainer")
        
        return True


if __name__ == "__main__":
    setup = CloudStudioSetup()
    success = setup.run()
    
    if success:
        print("\n🎉 Cloud Studio 自动配置完成！")
        print("请按照上述步骤完成项目上传和训练")
    else:
        print("\n❌ 配置过程中出现错误，请检查日志")
        sys.exit(1)
