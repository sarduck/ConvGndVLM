#!/usr/bin/env python3
"""
PhotoBook Grounding Experiment Runner (E1)
===========================================
Executes the dose-response alignment experiment across lambda interpolation values
with real-time tqdm progress bars, incremental autosaving, and memory management.

Usage:
    .venv/bin/python runner1.py
    .venv/bin/python runner1.py --lambda_values 0.0,0.5,1.0 --num_games 10
    .venv/bin/python runner1.py --verbose  # To print turn-by-turn dialogue
"""

import os
import gc
import sys
import json
import time
import argparse
from tqdm import tqdm

import torch
from config import get_default_device
from data_loader import load_photobook_games
from model_utils import get_vlm
from game_engine import play_game


def parse_args():
    parser = argparse.ArgumentParser(description="PhotoBook Grounding Experiment Runner")
    
    # Model Configuration
    parser.add_argument("--backend", type=str, default="huggingface", choices=["huggingface", "ollama"],
                        help="Backend for model inference (default: huggingface)")
    parser.add_argument("--model_family", type=str, default="OpenGVLab/InternVL3_5-4B",
                        help="Instruct/Chat model checkpoint ID (default: OpenGVLab/InternVL3_5-4B)")
    parser.add_argument("--base_model", type=str, default="OpenGVLab/InternVL3_5-4B-Pretrained",
                        help="Base model checkpoint ID for PAd interpolation (default: OpenGVLab/InternVL3_5-4B-Pretrained)")
    
    # Infrastructure Configuration
    default_dev = get_default_device()
    parser.add_argument("--device", type=str, default=default_dev,
                        help=f"Device to run inference on (cuda, mps, cpu). Default: {default_dev}")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Use 4-bit quantization (CUDA only; ignored on Apple Silicon/MPS)")
    
    # Experiment Configuration
    parser.add_argument("--lambda_values", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0",
                        help="Comma-separated lambda interpolation values (default: 0.0,0.2,0.4,0.6,0.8,1.0)")
    parser.add_argument("--num_games", type=int, default=50,
                        help="Number of games to sample and play per lambda (default: 50)")
    parser.add_argument("--max_rounds", type=int, default=3,
                        help="Number of rounds per game (default: 3, max score 18)")
    parser.add_argument("--prompt_type", type=str, default="engineered", choices=["engineered", "original"],
                        help="Prompt style to use (default: engineered)")
    
    # Logging & UX
    parser.add_argument("--quiet", action="store_true",
                        help="Silence turn-by-turn dialogue and only show the progress bar")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing partial outputs in output_dir instead of restarting")
    parser.add_argument("--data_dir", type=str, default="Data",
                        help="Path to PhotoBook Data directory (default: Data)")
    parser.add_argument("--output_dir", type=str, default="Experiments/output_E_1",
                        help="Path to directory for saving results (default: Experiments/output_E_1)")
    
    return parser.parse_args()


def preflight_download(args):
    """Ensure Hugging Face models are downloaded with progress bars before running."""
    if args.backend != "huggingface":
        return

    from huggingface_hub import snapshot_download
    
    print("\n" + "=" * 60)
    print("Pre-flight check: Verifying / downloading models from HuggingFace...")
    print("=" * 60)
    
    models_to_check = [args.model_family]
    # Check if base model is needed (needed for lambda < 1.0)
    lambda_list = [float(l) for l in args.lambda_values.split(",")]
    if any(l < 1.0 for l in lambda_list):
        models_to_check.append(args.base_model)
        
    for repo_id in models_to_check:
        print(f"\n[Checking Repo] {repo_id}")
        try:
            snapshot_download(
                repo_id=repo_id,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
            )
            print(f"✓ Ready: {repo_id}")
        except Exception as e:
            print(f"Warning during snapshot download of {repo_id}: {e}")
            print("Attempting to proceed with AutoModel dynamic loading...")


def main():
    args = parse_args()
    
    print("=" * 60)
    print(" PHOTOBOOK VLM GROUNDING EXPERIMENT (E1)")
    print("=" * 60)
    print(f"Backend:        {args.backend}")
    print(f"Instruct Model: {args.model_family}")
    print(f"Base Model:     {args.base_model}")
    print(f"Device:         {args.device}")
    print(f"Games/Lambda:   {args.num_games}")
    print(f"Rounds/Game:    {args.max_rounds} (Max score: {args.max_rounds * 6})")
    print(f"Lambdas:        {args.lambda_values}")
    print(f"Output Dir:     {args.output_dir}")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Download / Verify models with progress bar
    preflight_download(args)
    
    # 2. Load dataset
    print(f"\nLoading {args.num_games} PhotoBook games from '{args.data_dir}'...")
    games = load_photobook_games(args.data_dir, num_games=args.num_games, max_rounds=args.max_rounds)
    print(f"✓ Loaded {len(games)} valid games.")
    
    image_base_dir = os.path.join(args.data_dir, "images")
    lambda_list = [float(l.strip()) for l in args.lambda_values.split(",")]
    
    summary_stats = []
    
    # 3. Outer Progress Bar: Lambda Sweep
    lambda_pbar = tqdm(lambda_list, desc="Overall Progress (Lambdas)", position=0, unit="lambda")
    
    for lambda_val in lambda_pbar:
        output_file = os.path.join(args.output_dir, f"results_lambda_{lambda_val}.json")
        
        # Check for resume
        existing_results = []
        start_game_idx = 0
        if args.resume and os.path.exists(output_file):
            try:
                with open(output_file, "r") as f:
                    existing_results = json.load(f)
                start_game_idx = len(existing_results)
                if start_game_idx >= len(games):
                    tqdm.write(f"\n[λ = {lambda_val}] All {len(games)} games already completed. Skipping.")
                    scores = [g["total_score"] for g in existing_results]
                    summary_stats.append({
                        "lambda": lambda_val,
                        "games": len(existing_results),
                        "avg_score": sum(scores) / len(scores) if scores else 0,
                    })
                    continue
                else:
                    tqdm.write(f"\n[λ = {lambda_val}] Resuming from game {start_game_idx + 1}/{len(games)}...")
            except Exception as e:
                tqdm.write(f"Warning reading existing results for λ = {lambda_val}: {e}. Starting fresh.")
                existing_results = []
                start_game_idx = 0
        
        tqdm.write(f"\n{'='*60}\n>>> Initializing Model for λ = {lambda_val}\n{'='*60}")
        
        # Load / interpolate model
        t0 = time.time()
        vlm_instance = get_vlm(args, lambda_val=lambda_val)
        load_time = time.time() - t0
        tqdm.write(f"Model loaded and ready in {load_time:.1f}s.")
        
        results_for_lambda = list(existing_results)
        running_scores = [g["total_score"] for g in results_for_lambda]
        
        # Inner Progress Bar: Games for current lambda
        games_to_play = games[start_game_idx:]
        game_pbar = tqdm(
            enumerate(games_to_play, start=start_game_idx + 1),
            total=len(games),
            initial=start_game_idx,
            desc=f"Games (λ={lambda_val})",
            position=1,
            leave=False,
            unit="game"
        )
        
        verbose = not args.quiet
        for game_num, game_config in game_pbar:
            gid = game_config["game_id"]
            if verbose:
                tqdm.write(f"\n--- [λ={lambda_val}] Game {game_num}/{len(games)} (ID: {gid}) ---")
            
            try:
                game_result = play_game(
                    game_config=game_config,
                    vlm_A=vlm_instance,
                    vlm_B=vlm_instance,
                    image_base_dir=image_base_dir,
                    max_rounds=args.max_rounds,
                    prompt_type=args.prompt_type,
                    verbose=verbose
                )
                results_for_lambda.append(game_result)
                running_scores.append(game_result["total_score"])
                
                # Autosave incrementally after each game
                with open(output_file, "w") as f:
                    json.dump(results_for_lambda, f, indent=2)
                    
            except Exception as e:
                tqdm.write(f"Error during game {gid}: {e}")
                
            # Update inner progress bar info
            cur_avg = sum(running_scores) / len(running_scores) if running_scores else 0.0
            game_pbar.set_postfix({
                "game_id": gid,
                "score": f"{running_scores[-1] if running_scores else 0}/18",
                "avg_score": f"{cur_avg:.2f}"
            })
            
            # Clean intermediate memory
            if args.backend == "huggingface":
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                    
        game_pbar.close()
        
        avg_score = sum(running_scores) / len(running_scores) if running_scores else 0.0
        tqdm.write(f"\n✓ Completed λ = {lambda_val}: {len(results_for_lambda)} games, Avg Score: {avg_score:.2f}/18")
        tqdm.write(f"  Results saved to: {output_file}")
        
        summary_stats.append({
            "lambda": lambda_val,
            "games": len(results_for_lambda),
            "avg_score": avg_score
        })
        
        # Free model before next lambda to prevent memory accumulation
        if args.backend == "huggingface":
            del vlm_instance
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    lambda_pbar.close()
    
    # Print Final Summary Table
    print("\n" + "=" * 60)
    print(" EXPERIMENT COMPLETE — SUMMARY OF RESULTS")
    print("=" * 60)
    print(f"{'Lambda':<10} | {'Games Completed':<18} | {'Avg Score (/18)':<16}")
    print("-" * 60)
    for stat in summary_stats:
        print(f"{stat['lambda']:<10.2f} | {stat['games']:<18} | {stat['avg_score']:<16.2f}")
    print("=" * 60)
    print(f"All outputs saved in: {args.output_dir}/")
    print(f"\nNext steps:")
    print(f"  Run metrics:   .venv/bin/python metrics.py --results_dir {args.output_dir}/")
    print(f"  Plot curves:   .venv/bin/python analyze_results.py --e1_dir {args.output_dir}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

