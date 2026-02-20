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
