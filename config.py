import argparse
import sys

def get_default_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"

def get_config():
    parser = argparse.ArgumentParser(description="VLM Grounding Alignment Experiment")
    
    # Model Configuration
    parser.add_argument("--backend", type=str, default="huggingface", choices=["huggingface", "ollama"],
                        help="Backend to use for model inference.")
    parser.add_argument("--model_family", type=str, default="OpenGVLab/InternVL3_5-4B",
                        help="HuggingFace model ID or Ollama model name.")
    parser.add_argument("--base_model", type=str, default="OpenGVLab/InternVL3_5-4B-Pretrained",
                        help="HuggingFace base model ID (only used for huggingface backend PAd interpolation).")
    
    # Infrastructure Configuration
    default_dev = get_default_device()
    parser.add_argument("--device", type=str, default=default_dev,
                        help=f"Device to load the model on (cuda, mps, cpu). Default detected: {default_dev}")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Use 4-bit quantization (requires CUDA/bitsandbytes; ignored on MPS).")
    
    # Experiment Configuration
    parser.add_argument("--lambda_values", type=str, default="0.0,0.5,1.0",
                        help="Comma-separated list of lambda values for PAd interpolation.")
    parser.add_argument("--num_games", type=int, default=50,
                        help="Number of games to sample and play.")
    parser.add_argument("--max_rounds", type=int, default=3,
                        help="Maximum number of rounds to play per game (paper capped at 3).")
    parser.add_argument("--prompt_type", type=str, default="engineered", choices=["original", "engineered"],
                        help="Which prompt to use from the baseline paper (default: engineered).")
    
    # Paths
    parser.add_argument("--data_dir", type=str, default="Data",
                        help="Path to the Data directory.")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Path to save the generated logs.")
    
    return parser.parse_args()

