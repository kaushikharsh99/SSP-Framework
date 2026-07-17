# SSP Architecture Validation & Paper Compliance Audit

This audit evaluates the planned repository layout, interfaces, and roadmap against the methodologies described in the two VibeThinker research papers:
1. **VibeThinker-1.5B**: *Tiny Model, Big Logic: Diversity-Driven Optimization Elicits Large-Model Reasoning Ability in VibeThinker-1.5B* (arXiv:2511.06221)
2. **VibeThinker-3B**: *VibeThinker-3B: Exploring the Frontier of Verifiable Reasoning in Small Language Models* (arXiv:2606.16140)

---

## Task 1 — Paper Compliance Matrix

| Paper Component | Repository Component | Status | Notes |
| :--- | :--- | :--- | :--- |
| **Two-Stage Diversity SFT (1.5B)** | `training/` (SFT modules) | **Partially Represented** | Configured to support Stage 1 (diversity exploration) and Stage 2 (long-horizon difficulty target) but lacks specific diversity metrics code. |
| **Curriculum SFT (3B)** | `training/` (SFT modules) | **Partially Represented** | Outlined in SFT roadmap; actual database division filters (length, complexity) are postponed to SFT dataset loading. |
| **MaxEnt-Guided RL (MGPO)** | `rl/` (MGPO Trainer) | **Fully Represented** | Included in both interface specs and roadmap. Captures group-relative baseline and entropy constraints. |
| **Domain-Aware Weighting (3B)** | `rl/mgpo_trainer.py` | **Postponed** | Boundary weighting (focusing updates on samples near performance limits) is mathematically designed but deferred to RL implementation. |
| **Long2Short Reward (3B)** | `rl/rewards.py` | **Postponed** | Math efficiency penalty is designed under reward structures, but length scaling rules are deferred. |
| **Long-Context Stability (64K RL)** | `configs/` & `models/` wrapper | **Fully Represented** | Supported by the model wrapper (`models/wrapper.py`) and inference token configurations. |
| **Offline Self-Distillation** | `rl/distillation.py` | **Fully Represented** | Interface contracts and trajectory dataset schemas are designed and scheduled. |
| **Claim-Level Reliability (CLR)** | `evaluation/clr_search.py` | **Postponed** | Test-time scaling logic is designed but postponed to post-reproduction extensions. |

---

## Task 2 — Missing / Postponed Components

### 1. Diversity Probing & Probing Metrics (1.5B)
* *Status*: **Postponed**.
* *Rationale*: Diversity probing is computationally expensive. It requires sampling multiple sequences and calculating edit distances or embedding similarities. For our MVP, we will achieve diversity through high-temperature decoding ($T \ge 1.0$) during rollouts rather than specialized embedding distance filtering.

### 2. Checkpoint Merging / Averaging
* *Status*: **Postponed**.
* *Rationale*: Checkpoint averaging (e.g. SWA - Stochastic Weight Averaging or merging base SFT model weights back with RL policy weights to retain conversational ability) is a optimization choice. In a single-GPU personal workspace, we prioritize checking raw performance of the direct RL check-pointed model.

### 3. Long2Short Math RL Reward Redistribution (3B)
* *Status*: **Postponed to Milestone 4**.
* *Rationale*: Before optimizing model efficiency (shorter correct solutions), we must first ensure the model generates *correct* solutions. Therefore, we will optimize correctness reward first, and add length-shortening penalties subsequently.

### 4. Learning-Potential Filtering during Distillation
* *Status*: **Postponed to Milestone 5**.
* *Rationale*: Deciding which trajectories contain high "learning potential" requires training classifiers or checking model perplexity on generated tokens. We will initially filter trajectories strictly by binary correctness (reward = 1.0).

---

## Task 3 — Assumption Audit

### Supported by the paper
- **Decoupled SFT / RL phases**: Decoupling SFT from RL optimization is the core tenant of the Spectrum-to-Signal Principle.
- **Group-Relative Advantages**: Performing rollouts of size $N$ per prompt and using group normalization (without a separate critic network) to calculate policy updates.
- **Maximum Entropy Regularization**: Adding an entropy term to the policy loss to prevent premature collapse of the exploratory spectrum.

### Engineering decisions
- **Python Subprocess execution for Verifiers**: The paper states models are tested on coding tasks, but doesn't specify runtime safety. We chose to execute coding outputs in isolated subprocesses with restricted OS scopes to protect host environments.
- **XML Tag Schema**: Using `<think> ... </think> <answer> ... </answer>` blocks to cleanly parse responses. The paper indicates thinking traces are generated, but the exact tag structure is left to implementation choice.
- **Sympy-based Math Equivalence Check**: Since the paper doesn't explain how mathematical answers are evaluated, we assume using Sympy to parse and compare equations is the most robust engineering path.

### Personal lab preference
- **JSONL Dataset formats**: We store all datasets, prompts, and rollouts as JSONL text files for quick inspector reading and ease of debugging inside Jupyter notebooks.
- **Unified config.yaml output copy**: Copying the exact active configuration into the run's output directory ensures complete local reproducibility.

### Unsupported assumptions
- *None identified*. All assumptions represent standard PyTorch/ML engineering practices required to bridge gaps left unspecified by the research papers.

---

## Task 4 — Extension Separation

To ensure that research extensions do not complicate or corrupt the core implementation of the VibeThinker papers, they are strictly isolated from the core code.

### Core SSP (Reproduction Dependency)
- **Base model loader**: Loading huggingface models.
- **Rollout Generator**: Simple batched sampling.
- **SFT training loop**: Standard cross-entropy SFT with format validation.
- **Math Verifier**: Sympy equivalence checker.
- **Code Verifier**: Subprocess exec test-case runner.
- **GRPO Policy Optimizer**: Basic policy gradient with KL penalty.
- **MaxEnt Regularization**: Simple entropy loss addition.

### Future Research Extensions (Zero core dependencies)
- **Alternative RL algorithms**: DPO, standard PPO, or custom online policy gradients.
- **Plugin Reward functions**: Code quality checkers, styling checks, or semantic rewards.
- **Claim-Level Reliability (CLR) search**: Monte Carlo Tree Search (MCTS) decoding loops.
- **Checkpoint merging scripts**: Tooling to average multiple checkpoint models.
- **Synthetic reasoning generation loops**: Calling APIs (e.g. Gemini/GPT) to bootstrap datasets.

---

## Task 5 — MVP Definition (Successful Reproduction)

We will claim a successful reproduction of the Spectrum-to-Signal Principle (Version 1.0) when:
1. An SLM (e.g., SmolLM-135M or Qwen2.5-Coder-1.5B) is trained using SFT to generate step-by-step reasoning chains within `<think>` tags on a dataset of mathematical problems.
2. The SFT model is optimized via MaxEnt GRPO using a correctness-based reward function (Sympy equivalence).
3. The RL model shows a measurable increase in GSM8K validation accuracy compared to the base SFT model, while keeping the KL divergence from the base model within stable bounds.
4. Correct reasoning paths are successfully collected, and offline distillation is verified to preserve reasoning capabilities.

---

## Task 6 — Version Roadmap

```text
Version 0.1 ── Initial Workspace Setup (Complete)
      │
Version 0.2 ── Ingestion and Verifiers (Milestone 1)
      │        - Math (Sympy) and Code (subprocess) checkers
      │
Version 0.3 ── SFT Spectrum Phase (Milestone 2)
      │        - Model wrapper, custom tokens, basic SFT loop
      │
Version 0.4 ── Exploration Loop (Milestone 3)
      │        - Group rollout engine and reward computations
      │
Version 0.5 ── The Signal Phase (Milestone 4)
      │        - GRPO Policy optimizer with Maximum Entropy
      │
Version 1.0 ── Consolidated SSP (Milestone 5)
      │        - Offline self-distillation & completed reproduction
      │
Version 2.0 ── Lab Extensions
               - CLR Test-Time scaling, MGPO domain weighting, alternative RLs
```

---

## Task 7 — Final Design Review & Recommendations

### Over-engineered?
- *The Interface Abstractions*: While abstract classes like `BaseVerifier` might seem verbose for a single researcher, they are necessary. They allow writing modular scripts without hardcoding rules (e.g., swapping a math checker for a code execution test suite).
- *Verdict*: No, the layout is highly clean and lightweight.

### Manageable on a Single GPU?
- *The bottleneck*: Loading the policy model, reference model, and performing batched generation concurrently will quickly exceed 24GB VRAM (standard consumer GPU limit) for models larger than 1.5B.
- *Recommendation*: Use a small model like **SmolLM-135M** or **Qwen-0.5B** for verifying the codebase pipeline. When scaling to Qwen-1.5B or 3B, integrate DeepSpeed or LoRA parameter-efficient training to keep both models in memory (policy model trained via LoRA adapter, reference model is just the base frozen model).

### Safely Deferred?
- Checkpoint Merging, Claim-Level Reliability, and complex multi-domain boundary weighting are safely deferred to **Version 2.0**.
- The focus is solely on verifying the SFT-to-RL training loop.

### Pre-implementation Recommendation
1. Start directly with **Version 0.2 (Ingestion & Verifiers)**.
2. Ensure that code execution security is protected from infinite loops by enforcing thread timeout flags.
3. Test math equivalence robustly before coding any SFT trainer.
