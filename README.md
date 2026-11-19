# Does Alignment Training Break Conversational Grounding in Vision-Language Models?

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

This repository contains the full experimental harness for the CS 6120 research project, replicating and extending the PhotoBook referential game study (*Imai, Inan, Sicilia & Alikhani, RANLP 2025*).

The central question: **Does alignment training (instruction tuning and preference optimization) degrade a Vision-Language Model's ability to build conversational common ground?**

Using **Partial Adaptation (PAd)** (*Fleshman & Van Durme, 2024*), we perform continuous weight interpolation between a pretrained base VLM and its instruction-tuned counterpart:
$$W(\lambda) = W_{\text{base}} + \lambda \cdot (W_{\text{instruct}} - W_{\text{base}}), \quad \lambda \in [0.0, 1.0]$$

We evaluate the full dose-response curve on the **PhotoBook** collaborative referential task across 50 games $\times$ 3 rounds per game using open-weight checkpoints (`OpenGVLab/InternVL3_5-4B` and `OpenGVLab/InternVL3_5-4B-Pretrained`).

---

## Repository Structure

```text
├── Data/                       # PhotoBook dataset JSONs and COCO image folders
├── Experiments/
│   ├── output_E_1/             # E1 output JSONs (results_lambda_{λ}.json)
│   ├── output_E_2/             # E2 output JSONs (results_{condition}.json)
│   └── output_E_3/             # E3 output JSONs (results_{budget}.json)
├── checkpoints/                # Cached interpolated PAd weights (internvl_lambda_{λ})
├── outputs/                    # Legacy E1 output mirror (same content as output_E_1)
├── figures/                    # All generated figures + analysis_summary.txt + all_metrics.json
├── config.py                   # Centralized CLI arguments and device auto-detection
├── data_loader.py              # Multi-round PhotoBook game parser and target extractor
├── game_engine.py              # Turn-taking protocol, dialogue history, and scoring
├── model_utils.py              # InternVL HuggingFace backend with in-place PAd interpolation
├── runner1.py                  # Experiment 1 runner (Alignment sweep: λ ∈ {0.0 … 1.0})
├── runner2.py                  # Experiment 2 runner (History ablation: oracle summary)
├── runner3.py                  # Experiment 3 runner (Token budget pressure test)
├── metrics.py                  # Per-game grounding metrics (used by runners)
├── analyze_results.py          # ⭐ Comprehensive analysis suite (all 3 experiments)
└── requirements.txt            # Python dependencies
```

### Core Modules
- **[`runner1.py`](runner1.py)**: Experiment 1 runner (Alignment sweep). Features pre-flight Hugging Face model verification/downloading, nested `tqdm` progress bars (outer $\lambda$ sweep + inner game loop), real-time dialogue streaming, incremental per-game autosaving, and `--resume` support. Output saved to `Experiments/output_E_1/`.
- **[`runner2.py`](runner2.py)**: Experiment 2 runner (History ablation). Injects deterministic oracle memory summaries between rounds to isolate memory failure vs. structural failure. Output saved to `Experiments/output_E_2/`.
- **[`runner3.py`](runner3.py)**: Experiment 3 runner (Token budget). Evaluates compression under varying word budgets per turn (unlimited, 75, 50, 25 words). Output saved to `Experiments/output_E_3/`.
- **[`model_utils.py`](model_utils.py)**: Implements the Hugging Face `InternVLHuggingFace` interface. Performs zero-copy in-place weight interpolation (`torch.lerp_`), image tensor caching, and model checkpoint management.
- **[`game_engine.py`](game_engine.py)**: Orchestrates alternating dialogue turns between Agent A and Agent B, injects engineered system prompts, parses JSON utterances, and calculates ground-truth match scores (max 18 per game).
- **[`data_loader.py`](data_loader.py)**: Extracts 3-round game setups from the PhotoBook dataset, ensuring exactly 3 target images per agent per round.
- **[`metrics.py`](metrics.py)**: Calculates per-game metrics (scores, word counts, turn counts, per-round dynamics). Used by all three runners during live experiment runs.
- **[`analyze_results.py`](analyze_results.py)**: ⭐ Comprehensive post-hoc analysis suite. Loads raw per-game JSON for all 3 experiments, computes every metric, runs Mann-Whitney U tests + Bootstrap 95% CIs, prints a full statistical report, saves structured JSON, and generates 4 publication-quality figures.

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

### 2. Running Experiments
Run Experiment 1 (50-game sweep across all 6 $\lambda$ values):
```bash
.venv/bin/python runner1.py
```

Run Experiment 2 (History ablation: oracle summary):
```bash
.venv/bin/python runner2.py
```

Run Experiment 3 (Token budget constraint test):
```bash
.venv/bin/python runner3.py
```

#### Useful CLI Options (E1):
- **Resume an interrupted run**:
  ```bash
  .venv/bin/python runner1.py --resume
  ```
- **Quiet mode** (suppress turn dialogue and only show progress bars):
  ```bash
  .venv/bin/python runner1.py --quiet
  ```
- **Run on CUDA cluster / HPC**:
  ```bash
  python runner1.py --device cuda --load_in_4bit
  ```
- **Test a specific $\lambda$ with fewer games**:
  ```bash
  .venv/bin/python runner1.py --lambda_values 0.5 --num_games 5 --max_rounds 3
  ```

### 3. Analyzing Results

Run the comprehensive analysis suite (all 3 experiments, all metrics, statistical tests, all figures):
```bash
.venv/bin/python analyze_results.py
```

This single command:
- Loads raw per-game JSON from `Experiments/output_E_{1,2,3}/`
- Computes score, compression (R3/R1), Word Novelty Rate, efficiency, score trajectory, per-round breakdowns
- Runs **Mann-Whitney U** (non-parametric pairwise significance tests) with rank-biserial correlation effect size
- Computes **Bootstrap 95% CIs** (10,000 resamples, percentile method)
- Saves a full text report → `figures/analysis_summary.txt`
- Saves structured results → `figures/all_metrics.json`
- Generates 4 figures → `figures/e1_dose_response.png`, `figures/e2_history_ablation.png`, `figures/e3_token_budget.png`, `figures/summary_all.png`

You may also point it at custom directories:
```bash
.venv/bin/python analyze_results.py \
    --e1_dir Experiments/output_E_1 \
    --e2_dir Experiments/output_E_2 \
    --e3_dir Experiments/output_E_3 \
    --out_dir figures
```

---

## Empirical Results Summary (50 Games $\times$ 3 Rounds per Condition)

### E1 — Alignment Dose-Response Sweep

Completed dose-response sweep across all interpolation values on `OpenGVLab/InternVL3_5-4B`:

| $\lambda$ Setting | Model Type | Score (/18) | 95% CI | Total Words | R3/R1 | Sig. vs λ=0.0 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0.0** | Pure Pretrained Base | 9.44 | [8.84–10.04] | 1001.2 | 0.978 | — |
| **0.2** | 20% Instruction Tuned | **10.40** | [9.82–10.98] | 890.3 | 0.992 | * p=0.041 |
| **0.4** | 40% Instruction Tuned | 9.20 | [8.68–9.74] | 815.5 | 0.949 | ns |
| **0.6** | 60% Instruction Tuned | 9.82 | [9.16–10.48] | 794.8 | 1.055 | ns |
| **0.8** | 80% Instruction Tuned | 9.88 | [9.20–10.58] | 657.0 | 0.980 | ns |
| **1.0** | Full Instruct / Chat | 10.20 | [9.56–10.84] | 697.7 | 0.992 | ns |
| *Human Baseline* | *RANLP 2025 Paper* | **16.62** | — | **338.1** | **~0.47** | — |

> All compression comparisons (λ vs λ=1.0) are **ns** (p=0.23–0.83). Alignment level has no effect on cross-round compression.

### E2 — History Ablation (Oracle Summary)

| Condition | Score (/18) | 95% CI | R3/R1 | 95% CI | C-Rate% | Sig. (compression) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **no_summary** | 10.28 | [9.66–10.88] | 0.993 | [0.91–1.07] | 52% | — |
| **oracle_summary** | 10.10 | [9.50–10.70] | **0.840** | [0.76–0.92] | 64% | * p=0.028, r=−0.26 |
| *Human Baseline* | **16.62** | — | **~0.47** | — | ~100% | — |

> Score difference is **ns** (p=0.666). Compression improvement is **significant** (p=0.028), word reduction borderline (p=0.059).

### E3 — Token Budget Pressure Test

| Budget | Score (/18) | 95% CI | Compress (R3/R1) | Util-Drop | Sig. score vs unlimited |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **unlimited** | 9.60 | [8.92–10.26] | 0.932 | N/A | — |
| **75 wds/turn** | 8.61 | [8.14–9.10] | 1.030 | +0.007 | * p=0.033 |
| **50 wds/turn** | 8.48 | [7.90–9.06] | 0.990 | +0.017 | * p=0.019 |
| **25 wds/turn** | 8.24 | [7.76–8.78] | 0.926 | −0.012 | ** p=0.003 |

> Compression comparisons vs unlimited: all **ns** (p=0.19–0.64). Budget hurts accuracy but does **not** induce compression at any level.

### Key Findings

1. **Alignment does NOT cause non-compression (E1)** — compression ratios across all λ values are statistically indistinguishable (Mann-Whitney U, all p=0.23–0.83). The failure precedes alignment.

2. **Alignment reduces verbosity (E1)** — word count drops monotonically by 34% (1001→657) as λ increases from 0.0→0.8.

3. **λ=0.2 gives the best score — marginally significant (E1)** — the only significant score difference in E1: λ=0.2 vs λ=0.0, p=0.041, r=+0.235.

4. **Alignment prevents accuracy degradation across rounds (E1)** — λ=0.0 loses −1.02 pts R1→R3; λ=1.0 loses only −0.26.

5. **Oracle memory scaffolding significantly reduces compression (E2)** — R3/R1 drops 0.993→0.840 (p=0.028, r=−0.26). But score is unchanged (p=0.666) and adherence is 0.8% — the model uses context implicitly, not through explicit pact formation.

6. **Budget constraints significantly hurt accuracy but do NOT induce compression (E3)** — score drops are significant at all budgets (p=0.003–0.033); compression changes are all ns (p=0.19–0.64). The 25-word budget falls below random baseline.

7. **The failure is structural** — three independent interventions (alignment, memory, token cost) all fail to unlock human-like compression (human R3/R1 ≈ 0.47 vs best model 0.84).
