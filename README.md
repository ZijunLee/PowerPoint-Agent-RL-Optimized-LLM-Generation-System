# RLTrainPPT - RL-based PPT Content Generation / 基于强化学习的PPT内容生成

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

[English](#english) | [中文说明](#chinese)

---

<a name="english"></a>

## 🌟 Introduction (English)

**RLTrainPPT** utilizes the **GSPO (Group Sequence Policy Optimization)** reinforcement learning method to fine-tune Large Language Models (LLMs). The project aims to automate the generation of high-quality, logically structured presentation outlines and detailed slide content.

## 📁 Project Structure

```text
backend/
├── .env                    # Environment config (API keys, model settings, etc.)
├── requirements.txt        # Python dependencies
├── README.md               # Documentation
│
├── outline/               # Phase 1: Outline Generation
│   ├── prompt.py          # System prompts for outlines
│   ├── train_trl.py       # GSPO training script for outlines
│   ├── model_test.py      # Inference/Test script
│   ├── topic.json         # Training dataset (topics)
│   ├── outline.jsonl      # Generated outline results
│   └── output/            # Model checkpoints
│
└── content/               # Phase 2: Content Generation
    ├── prompt.py          # System prompts for content
    ├── train_trl.py       # GSPO training script for content
    ├── model_test.py      # Inference/Test script
    ├── content.jsonl      # Final generated content
    └── output/            # Model checkpoints

🚀 Quick Start
1. Installation

# Create conda environment
conda create -n rlppt python=3.10
conda activate rlppt

# Install dependencies
cd backend
pip install -r requirements.txt
2. Configuration
Create and edit the backend/.env file:

# Model & API Config
ART_MODEL=Qwen/Qwen2.5-0.5B-Instruct
DEEPSEEK_API_KEY=your-api-key
DEEPSEEK_BASE_URL=[https://api.deepseek.com/v1/chat/completions](https://api.deepseek.com/v1/chat/completions)
USE_DEEPSEEK_JUDGE=true

📝 Workflow
Phase 1: Outline Generation

# 1. Start Training
cd backend/outline
python train_trl.py

# 2. Run Inference Test
python model_test.py
Phase 2: Content Generation

# 1. Start Training (Requires outline.jsonl from Phase 1)
cd ../content
python train_trl.py

# 2. Run Inference Test
python model_test.py
<a name="chinese"></a>

🌟 项目简介 (中文)
RLTrainPPT 是一个基于 GSPO (Group Sequence Policy Optimization) 强化学习方法的研究项目。通过训练大语言模型（LLM），实现从单一主题到完整、专业 PPT 大纲及详细内容的自动化生成。

🚀 快速开始
1. 环境准备

# 创建并激活环境
conda create -n rlppt python=3.10
conda activate rlppt

# 安装依赖
cd backend
pip install -r requirements.txt
2. 配置环境变量
编辑 backend/.env 文件，填入您的 DeepSeek API Key：

# 编辑 .env 文件
nano .env  # 或者使用 VS Code 打开

📝 使用流程
阶段一：大纲生成

# 进入大纲模块执行训练
cd backend/outline
python train_trl.py

# 测试生成效果
python model_test.py

阶段二：内容生成

# 进入内容模块执行训练
cd ../content
python train_trl.py

# 生成最终 PPT 内容
python model_test.py
