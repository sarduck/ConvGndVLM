# VLM Grounding Alignment

This repository contains the experimental harness for investigating whether post-training (instruction tuning and preference optimization) degrades a Vision-Language Model's ability to build conversational common ground. 

By utilizing **Partial Adaptation (PAd)**, we interpolate the weights between a base VLM and its instruct-tuned counterpart to create a continuous dose-response curve ($\lambda \in [0, 1]$) on the **PhotoBook** referential game.

## Project Structure

- **`config.py`**: Centralized setup using `argparse`. Defines variables like `--lambda_values`, `--num_games`, `--device`, and allows toggling between HuggingFace and Ollama backends.
- **`data_loader.py`**: Parses the raw PhotoBook dataset JSONs to extract the shared and unique target images for each agent's reference list, along with their ground-truth labels.
- **`game_engine.py`**: The core multi-turn orchestration logic. It manages the conversational protocol, passes system prompts and visual context to the models, parses JSON responses, and tracks dialogue history until both agents submit their guesses.
- **`model_utils.py`**: Contains the `VLMInterface`. 
  - Supports `OllamaVLM` for quick local testing.
  - Supports `HuggingFaceVLM` which implements the PAd tensor arithmetic: $W(\lambda) = W_{base} + \lambda \cdot (W_{instruct} - W_{base})$.
- **`run_e1.py`**: The main execution script for Experiment 1. It sweeps over the specified $\lambda$ values, runs the model-model self-play, and saves the dialogue transcripts.
- **`metrics.py`**: Analyzes the transcripts to calculate Grounding Efficiency metrics, including Task Score (accuracy), Word Count, and Turn Count.
- **`analyze_results.py`**: Generates the final dose-response plots (`dose_response.png`) comparing performance across the different instruction-tuning strengths.

## Quickstart

### Local Testing (Ollama)
For a quick sanity check of the game engine without heavy GPU requirements:
```bash
python run_e1.py --backend ollama --model_family llava --num_games 1 --lambda_values 1.0
```

### HPC GPU Deployment (HuggingFace)
To run the full dose-response weight interpolation sweep on a cluster:
```bash
python run_e1.py --backend huggingface \
                 --model_family Qwen/Qwen2.5-VL-7B-Instruct \
                 --base_model Qwen/Qwen2.5-VL-7B \
                 --lambda_values 0.0,0.2,0.4,0.6,0.8,1.0 \
                 --num_games 50 \
                 --device cuda \
                 --load_in_4bit
```

### Evaluation
Compute metrics and plot the results:
```bash
python metrics.py --results_dir outputs/
python analyze_results.py --results_dir outputs/
```

