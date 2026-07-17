# LLM Post-Training & Reasoning Lab (SSP-Framework)

This is a personal research workspace and playground dedicated to reproducing, extending, and experimenting with post-training methods for Small and Large Language Models (SLMs/LLMs). 

The primary starting point is the **Spectrum-to-Signal Principle (SSP)** introduced in the VibeThinker papers, along with various other post-training alignment and scaling techniques.

> [!IMPORTANT]
> **Initialization Phase**: This repository is currently in the setup phase. Directory structures and placeholders are established, but no training, reinforcement learning, or model code has been implemented yet.

---

## 🎯 Design Principles

- **Simplicity & Agility**: Prioritize fast iteration, readable code, and minimal boilerplate over heavy software engineering abstractions.
- **Model-Agnostic**: Design core wrappers to easily swap between models (e.g. Qwen, SmolLM, TinyLlama, Gemma, Llama, Mistral, Phi, or custom models) without restructuring the code.
- **Dataset-Interchangeable**: Decouple data loading from training loops to facilitate running experiments across diverse reasoning and alignment datasets.
- **Stage-Oriented Structure**: Organize code around key research and post-training stages rather than model types.

---

## 📂 Workspace Layout

```text
SSP-Framework/
├── papers/                 # Research papers PDFs, math derivations, and reading notes
├── configs/                # Hyperparameter and run configuration files
├── datasets/               # Dataset downloaders, preprocessors, and tokenizers
├── models/                 # Model wrappers, interfaces, and architecture wrappers
├── training/               # Supervised Fine-Tuning (SFT) loops and loss functions
├── rl/                     # Reinforcement Learning alignment (GRPO, PPO, DPO, MGPO)
├── evaluation/             # Reasoning benchmarks and evaluation pipelines
├── notebooks/              # Jupyter notebooks for data visualization and state analysis
├── scripts/                # Helper scripts for automation
├── outputs/                # Local training outputs, plots, and logs (ignored by Git)
├── checkpoints/            # Model weight checkpoints (ignored by Git)
├── logs/                   # TensorBoard / Weights & Biases logs (ignored by Git)
├── Makefile                # Shortcuts for setup and cleaning
├── pyproject.toml          # Light metadata package configuration
└── requirements.txt        # Python dependency manifest
```

---

## 🚀 Future Research Roadmap

This lab will be populated with implementations of the following stages:
1. **Spectrum SFT**: Alignment of sequential token-generation hidden trajectories with lower-dimensional spectrum priors.
2. **Signal RL (GRPO & MGPO)**: Rejection-free reinforcement learning alignment using group-relative metrics.
3. **Alternative Alignment**: Baseline PPO and DPO algorithms.
4. **Offline Self-Distillation**: Compressing longer reasoning traces into condensed models.
5. **Test-Time Scaling**: Modulating reasoning compute paths at inference time.

---

## 🛠️ Quick Start

1. Install dependencies:
   ```bash
   make install
   ```

2. Open research notebooks:
   ```bash
   jupyter notebook
   ```
