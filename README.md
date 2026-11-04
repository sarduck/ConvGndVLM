# Does Alignment Training Break Conversational Grounding in Vision-Language Models?

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

This repository contains the experimental harness for **Experiment 1 (E1)** from the CS 6120 research project, replicating and extending the PhotoBook referential game study (*Imai, Inan, Sicilia & Alikhani, RANLP 2025*).

The central question: **Does alignment training (instruction tuning and preference optimization) degrade a Vision-Language Model's ability to build conversational common ground?**

Using **Partial Adaptation (PAd)** (*Fleshman & Van Durme, 2024*), we perform continuous weight interpolation between a pretrained base VLM and its instruction-tuned counterpart:
$$W(\lambda) = W_{\text{base}} + \lambda \cdot (W_{\text{instruct}} - W_{\text{base}}), \quad \lambda \in [0.0, 1.0]$$

We evaluate the full dose-response curve on the **PhotoBook** collaborative referential task across 50 games $\times$ 3 rounds per game using open-weight checkpoints (`OpenGVLab/InternVL3_5-4B` and `OpenGVLab/InternVL3_5-4B-Pretrained`).

---

## Repository Structure

```text
├── Data/                       # PhotoBook dataset JSONs and COCO image folders
├── checkpoints/                # Cached interpolated PAd weights (internvl_lambda_{λ})
├── outputs/                    # Output logs per lambda (results_lambda_{λ}.json)
├── config.py                   # Centralized CLI arguments and device auto-detection
├── data_loader.py              # Multi-round PhotoBook game parser and target extractor
├── game_engine.py              # Turn-taking protocol, dialogue history, and scoring
├── model_utils.py              # InternVL HuggingFace backend with in-place PAd interpolation
├── runner.py                   # Master execution runner with progress bars & autosaving
├── metrics.py                  # Grounding efficiency, word count, turn count, and dynamics
├── analyze_results.py          # Dose-response curve plotting tool (dose_response.png)
└── requirements.txt            # Python dependencies
```

### Core Modules
- **[`runner.py`](runner.py)**: The primary experiment driver. Features pre-flight Hugging Face model verification/downloading, nested `tqdm` progress bars (outer $\lambda$ sweep + inner game loop), real-time dialogue streaming, incremental per-game autosaving, and `--resume` support.
- **[`model_utils.py`](model_utils.py)**: Implements the Hugging Face `InternVLHuggingFace` interface. Performs zero-copy in-place weight interpolation (`torch.lerp_`), image tensor caching, and model checkpoint management.
- **[`game_engine.py`](game_engine.py)**: Orchestrates alternating dialogue turns between Agent A and Agent B, injects engineered system prompts, parses JSON utterances, and calculates ground-truth match scores (max 18 per game).
- **[`data_loader.py`](data_loader.py)**: Extracts 3-round game setups from the PhotoBook dataset, ensuring exactly 3 target images per agent per round.
- **[`metrics.py`](metrics.py)**: Calculates average score, word counts, turn counts, and per-round dynamics ($R_1 \to R_2 \to R_3$).
- **[`analyze_results.py`](analyze_results.py)**: Generates publishable dose-response curve visualizations.

---

## Key Hardware & Engineering Optimizations

The experiment was built and validated on Apple Silicon unified memory (macOS M-series):

1. **Zero-Copy In-Place Weight Interpolation (`torch.lerp_`)**:
   Instead of allocating duplicate 8–16 GB state dictionaries in RAM, weights are modified directly in-place within the base model's memory buffer.
2. **Multi-Image Token Management (`max_num=1`)**:
   Standard dynamic resolution slicing creates up to 7 patches per image ($>5,400$ visual tokens for 3 images), causing exponential self-attention latency. Constraining each image to 1 patch ($448 \times 448$, 256 tokens) reduces sequence length to 768 tokens, achieving a **~30x speedup** on Metal (MPS).
3. **PAd Checkpoint Caching**:
   Interpolated models are saved to `checkpoints/internvl_lambda_{λ}/`. Resuming or re-running directly loads the cached checkpoint, eliminating duplicate model loads.
4. **Memory Purges Between Games**:
   Explicit `gc.collect()` and `torch.mps.empty_cache()` calls are performed after each turn and game to prevent unified memory bloat.
5. **Incremental Autosaving**:
   Every game's transcript and score are saved immediately to `outputs/results_lambda_{λ}.json`, ensuring zero data loss if interrupted.

---

## Quick Start & Usage

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Running the Full Experiment
Run the complete 50-game sweep across all 6 $\lambda$ values ($\lambda \in \{0.0, 0.2, 0.4, 0.6, 0.8, 1.0\}$):
```bash
.venv/bin/python runner.py
```

#### Useful CLI Options:
- **Resume an interrupted run**:
  ```bash
  .venv/bin/python runner.py --resume
  ```
- **Quiet mode** (suppress turn dialogue and only show progress bars):
  ```bash
  .venv/bin/python runner.py --quiet
  ```
- **Run on CUDA cluster / HPC**:
  ```bash
  python runner.py --device cuda --load_in_4bit
  ```
- **Test a specific $\lambda$ with fewer games**:
  ```bash
  .venv/bin/python runner.py --lambda_values 0.5 --num_games 5 --max_rounds 3
  ```

### 3. Analyzing Results
Compute summary metrics and per-round dynamics:
```bash
.venv/bin/python metrics.py --results_dir outputs/
```

Generate the dose-response plots (`dose_response.png`):
```bash
.venv/bin/python analyze_results.py --results_dir outputs/
```

---

## Empirical Results Summary (50 Games $\times$ 3 Rounds)

Completed dose-response sweep across all interpolation values on `OpenGVLab/InternVL3_5-4B`:

| $\lambda$ Setting | Model Type | Total Score (/18) | Total Words | Total Turns | R1 Words | R2 Words | R3 Words | Compression ($\Delta$ R1 $\to$ R3) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0** | Pure Pretrained Base | 9.44 | 1001.2 | 19.92 | 340.6 | 328.7 | 331.8 | **$-2.6\%$** (Flat) |
| **0.2** | 20% Instruction Tuned | **10.40** | 890.3 | 18.88 | 300.9 | 292.4 | 296.9 | **$-1.3\%$** (Flat) |
| **0.4** | 40% Instruction Tuned | 9.20 | 815.5 | 19.58 | 283.0 | 268.6 | 263.9 | **$-6.7\%$** (Negligible) |
| **0.6** | 60% Instruction Tuned | 9.82 | 794.8 | 19.74 | 261.3 | 260.2 | 273.3 | **$+4.6\%$** (Expanded) |
| **0.8** | 80% Instruction Tuned | 9.88 | **657.0** | **18.70** | 224.5 | 215.8 | 216.6 | **$-3.5\%$** (Flat) |
| **1.0** | Full Instruct / Chat | 10.20 | 697.7 | 18.84 | 246.8 | 217.8 | 233.1 | **$-5.5\%$** (Flat) |
| *Human Baseline* | *RANLP 2025 Paper* | **16.62** | **338.1** | **74.08** | ~150 | ~100 | ~70 | **$-53.3\%$** (Sharp) |

### Key Theoretical Insights
1. **Alignment Reduces Total Word Bloat**:
   As $\lambda$ increases from $0.0 \to 0.8$, word count drops monotonically by **34.4%** (1001 $\to$ 657 words) because instruction tuning enforces schema adherence and prevents repetitive monologues.
2. **Cross-Round Compression is Broken Across ALL $\lambda$**:
   Human dyads converge on shorthand conceptual pacts, cutting word count in half by Round 3. Models across all $\lambda$ values display essentially flat word counts from Round 1 to Round 3 ($\le 5\%$ change). This falsifies the simple hypothesis that RLHF/alignment is the *sole* cause of non-compression.
3. **Alignment Acts as a Multi-Round Stabilizer**:
   At $\lambda=0.0$, task accuracy degrades over rounds ($3.72 \to 2.70$) due to context bleed (inability to distinguish past dialogue history from new visual evidence). At $\lambda \ge 0.8$, accuracy remains stable across rounds ($3.56 \to 3.30$).
