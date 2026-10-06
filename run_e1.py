import os
import json
from config import get_config
from data_loader import load_photobook_games
from model_utils import get_vlm
from game_engine import play_game

def run_experiment():
    args = get_config()
    
    # 1. Load the dataset
    print(f"Loading {args.num_games} games from {args.data_dir}...")
    games = load_photobook_games(args.data_dir, num_games=args.num_games)
    print(f"Loaded {len(games)} games.")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse lambda values
    lambda_list = [float(l) for l in args.lambda_values.split(',')]
    print(f"Running sweep over lambda values: {lambda_list}")
    
    image_base_dir = os.path.join(args.data_dir, "images")
    
    for lambda_val in lambda_list:
        print(f"\n{'='*50}\nStarting Lambda = {lambda_val}\n{'='*50}")
        
        # 2. Load the VLM for the current lambda
        # If using Ollama, lambda interpolation is skipped conceptually but we can 
        # still use it to drive the loop for testing.
        vlm_instance = get_vlm(args, lambda_val=lambda_val)
        
        results_for_lambda = []
        
        # 3. Play the games
        for idx, game_config in enumerate(games):
            print(f"\n[Lambda {lambda_val}] Playing Game {idx + 1}/{len(games)} (Game ID: {game_config['game_id']})")
            
            try:
                result = play_game(
                    game_config=game_config,
                    vlm_A=vlm_instance,
                    vlm_B=vlm_instance,
                    image_base_dir=image_base_dir,
                    max_rounds=args.max_rounds,
                    prompt_type=args.prompt_type
                )
                results_for_lambda.append(result)
            except Exception as e:
                print(f"Error during game {game_config['game_id']}: {e}")
                
        # 4. Save results for this lambda
        output_file = os.path.join(args.output_dir, f"results_lambda_{lambda_val}.json")
        with open(output_file, 'w') as f:
            json.dump(results_for_lambda, f, indent=2)
            
        print(f"Saved results to {output_file}")
        
        # Free memory if using HF
        if args.backend == "huggingface":
            import torch
            import gc
            del vlm_instance
            gc.collect()
            torch.cuda.empty_cache()
            
    print("Experiment complete!")

if __name__ == "__main__":
    run_experiment()

