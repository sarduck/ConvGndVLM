import json
import os
import argparse

def calculate_score(ground_truth, guesses):
    """
    Score is 1 point for each correct label (max 3 per agent, 6 per round).
    For a 3-round game, the maximum score is 18.
    """
    score = 0
    if not guesses or len(guesses) != len(ground_truth):
        return score
        
    for gt, guess in zip(ground_truth, guesses):
        if gt == guess:
            score += 1
    return score

def compute_grounding_efficiency(transcript_file):
    with open(transcript_file, 'r') as f:
        data = json.load(f)
        
    metrics = {
        "total_games": len(data),
        "avg_score_A": 0,
        "avg_score_B": 0,
        "avg_total_score": 0, # Max 6 per game (since we only guess once at the end of the round in our simplified model-model self-play)
        "avg_word_count": 0,
        "avg_turn_count": 0
    }
    
    if len(data) == 0:
        return metrics
        
    total_score = 0
    total_words = 0
    total_turns = 0
    
    for game in data:
        # Task Score
        gt_A = game["agent_A"]["ground_truth"]
        guesses_A = game["agent_A"]["guesses"]
        score_A = calculate_score(gt_A, guesses_A)
        
        gt_B = game["agent_B"]["ground_truth"]
        guesses_B = game["agent_B"]["guesses"]
        score_B = calculate_score(gt_B, guesses_B)
        
        total_score += (score_A + score_B)
        metrics["avg_score_A"] += score_A
        metrics["avg_score_B"] += score_B
        
        # Word Count & Turn Count
        game_words = 0
        turns = len(game["transcript"])
        total_turns += turns
        
        for turn in game["transcript"]:
            msg = turn["output"].get("message", "")
            if msg:
                game_words += len(msg.split())
                
        total_words += game_words
        
    metrics["avg_score_A"] /= len(data)
    metrics["avg_score_B"] /= len(data)
    metrics["avg_total_score"] = total_score / len(data)
    metrics["avg_word_count"] = total_words / len(data)
    metrics["avg_turn_count"] = total_turns / len(data)
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="outputs")
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Directory {args.results_dir} does not exist.")
        exit(1)
        
    for filename in sorted(os.listdir(args.results_dir)):
        if filename.endswith(".json"):
            filepath = os.path.join(args.results_dir, filename)
            eff_metrics = compute_grounding_efficiency(filepath)
            
            print(f"\n--- Metrics for {filename} ---")
            print(f"Total Games: {eff_metrics['total_games']}")
            print(f"Avg Score (max 6): {eff_metrics['avg_total_score']:.2f} (A: {eff_metrics['avg_score_A']:.2f}, B: {eff_metrics['avg_score_B']:.2f})")
            print(f"Avg Word Count: {eff_metrics['avg_word_count']:.2f}")
            print(f"Avg Turn Count: {eff_metrics['avg_turn_count']:.2f}")


