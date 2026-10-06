import argparse

def get_config():
    parser = argparse.ArgumentParser(description="VLM Grounding Alignment Experiment")
    
    # Model Configuration
    parser.add_argument("--backend", type=str, default="huggingface", choices=["huggingface", "ollama"],
                        help="Backend to use for model inference.")
    parser.add_argument("--model_family", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="HuggingFace model ID or Ollama model name.")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-7B",
                        help="HuggingFace base model ID (only used for huggingface backend PAd interpolation).")
    
    # Infrastructure Configuration
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to load the model on (e.g., cuda, mps, cpu).")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Use 4-bit quantization (requires bitsandbytes).")
    
    # Experiment Configuration
    parser.add_argument("--lambda_values", type=str, default="0.0,0.5,1.0",
                        help="Comma-separated list of lambda values for PAd interpolation.")
    parser.add_argument("--num_games", type=int, default=50,
                        help="Number of games to sample and play.")
    parser.add_argument("--max_rounds", type=int, default=3,
                        help="Maximum number of rounds to play per game (paper capped at 3).")
    parser.add_argument("--prompt_type", type=str, default="original", choices=["original", "engineered"],
                        help="Which prompt to use from the baseline paper.")
    
    # Paths
    parser.add_argument("--data_dir", type=str, default="Data",
                        help="Path to the Data directory.")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Path to save the generated logs.")
    
    return parser.parse_args()

