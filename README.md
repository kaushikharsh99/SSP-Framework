# SSP Framework (Spectrum-to-Signal Principle)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A modular, production-quality PyTorch research framework designed to study and scale the **Spectrum-to-Signal Principle (SSP)** for deep learning models and agentic reasoners, inspired by the VibeThinker papers.

> [!IMPORTANT]
> **Current Status**: This project is under **active development**. Only the codebase skeleton, configuration structures, documentation, and quality workflows have been initialized. Core algorithms, SFT training, and Reinforcement Learning features are **not yet implemented**.

---

## 🌌 Project Vision

The **Spectrum-to-Signal Principle (SSP)** proposes mapping high-dimensional state trajectories generated during model inference or reasoning (the "signal") to lower-dimensional latent spaces representing discrete cognitive modes, constraints, or optimization targets (the "spectrum"). By parameterizing and projecting between these representations, we aim to design architectures that:
- Seamlessly transition between fast intuitive response and slow deliberate reasoning (Test-Time Scaling).
- Learn dense cognitive priors that generalize better than raw text tokens.
- Allow robust model alignment using low-dimensional optimization manifolds.

---

## 🎯 Project Goals & Future Scope

The framework is architected to scale and support a variety of research efforts, including:
1. **Spectrum Supervised Fine-Tuning (Spectrum SFT)**: Embedding spectrum projection targets during supervised sequence generation.
2. **Reinforcement Learning Alignment**:
   - **GRPO (Group Relative Policy Optimization)**: Rejection-free alignment without a critic network.
   - **MGPO (Multi-Group Policy Optimization)**: Extending GRPO for multiple reasoning sub-objectives.
   - **PPO & DPO**: Classic alignment baselines adapted to the spectrum projection space.
3. **Offline Self-Distillation**: Allowing reasoning models to distill long-form reasoning trajectories (signals) into compact spectrum priors.
4. **Test-Time Scaling**: Algorithms that modulate inference compute dynamically based on spectrum projections.

---

## 📂 Repository Structure

The project follows standard open-source best practices, separating concerns across distinct modular packages:

```text
SSP-Framework/
├── configs/                # Hydra YAML configuration folders
│   ├── datasets/           # Dataset loader configs
│   ├── evaluation/         # Benchmarking configs
│   ├── inference/          # Generation & decoding configs
│   ├── models/             # Transformer and projection configurations
│   ├── rl/                 # PPO, GRPO, MGPO parameters
│   └── training/           # Optimizer & scheduler parameters
├── docs/                   # Documentation source pages
│   ├── api_reference.md    # API indexing
│   ├── architecture.md     # System design diagrams
│   ├── installation.md     # Package requirements and setup
│   ├── quick_start.md      # Minimal executable examples
│   ├── research_notes.md   # Mathematical background
│   └── tutorials.md        # Walkthrough guides
├── examples/               # Standalone pipeline demonstration scripts
├── experiments/            # Sandbox for running and tracking local sweeps
├── scripts/                # Launch scripts for CLI automation
│   ├── convert_checkpoint.py
│   ├── evaluate.py
│   ├── export_model.py
│   ├── prepare_dataset.py
│   └── train.py
├── ssp_framework/          # Main python package
│   ├── core/               # Model base classes, projections, config schema
│   ├── data/               # Collation, streaming, dataset tokenization
│   ├── evaluation/         # Benchmark evaluation loops
│   ├── rl/                 # Alignment trainers (PPO, GRPO, MGPO, DPO)
│   ├── supervised/         # Supervised fine-tuning (SFT) trainers
│   └── utils/              # Hardware config, logging, and seed helpers
├── tests/                  # Pytest verification suites
└── Makefile                # Entry points for development setup
```

---

## 🛠️ Getting Started

For detailed guidelines, please refer to:
- [Installation Guide](file:///home/harsh/coding/SSP-Framework/docs/installation.md)
- [Quick Start Guide](file:///home/harsh/coding/SSP-Framework/docs/quick_start.md)

### Quick Setup

1. Clone and enter the workspace:
   ```bash
   git clone https://github.com/kaushikharsh99/SSP-Framework.git
   cd SSP-Framework
   ```

2. Build environment and install developer dependencies:
   ```bash
   make install-dev
   ```

3. Run verification tests:
   ```bash
   make test
   ```

---

## 🧪 Development Quality Assurance

This framework is configured with a robust linting, formatting, and type-checking suite:
- **Formatters**: [Black](https://github.com/psf/black) & [isort](https://github.com/pycqa/isort) (integrated via Makefile)
- **Linter**: [Ruff](https://github.com/astral-sh/ruff)
- **Static Type Checker**: [Mypy](https://github.com/python/mypy)
- **Test Runner**: [Pytest](https://github.com/pytest-dev/pytest) with test coverage reports

Run style checks and test commands via the provided Makefile targets:
```bash
make format    # Auto-formats imports and python styles
make lint      # Runs ruff and type-checks with mypy
make test      # Runs pytest
```

---

## 🤝 Contributing

We welcome research collaborations! Please read our [Contributing Guidelines](file:///home/harsh/coding/SSP-Framework/CONTRIBUTING.md) and adhere to the [Code of Conduct](file:///home/harsh/coding/SSP-Framework/CODE_OF_CONDUCT.md).
