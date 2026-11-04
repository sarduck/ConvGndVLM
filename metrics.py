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
        "avg_total_score": 0.0,
        "max_possible_score": 18,
        "avg_word_count": 0.0,
        "avg_turn_count": 0.0,
        "round_metrics": {}
    }
    
    if len(data) == 0:
        return metrics
        
    total_score = 0
    total_words = 0
    total_turns = 0
    
    round_stats = {} # r_idx -> {score, words, turns, count}
    
    for game in data:
        # Check if new multi-round structure exists
        if "rounds" in game and isinstance(game["rounds"], list):
            game_score = game.get("total_score", 0)
            total_score += game_score
            metrics["max_possible_score"] = game.get("max_score", 18)
            
            game_words = 0
            game_turns = len(game.get("transcript", []))
            for turn in game.get("transcript", []):
                msg = turn["output"].get("message", "")
                if msg:
                    game_words += len(msg.split())
                    
            total_words += game_words
            total_turns += game_turns
            
            # Per round stats
            for r in game["rounds"]:
                r_num = r["round_nr"]
                if r_num not in round_stats:
                    round_stats[r_num] = {"score": 0, "words": 0, "turns": 0, "count": 0}
                round_stats[r_num]["score"] += r.get("round_score", 0)
                r_words = sum(len(t["output"].get("message", "").split()) for t in r.get("transcript", []))
                round_stats[r_num]["words"] += r_words
                round_stats[r_num]["turns"] += len(r.get("transcript", []))
                round_stats[r_num]["count"] += 1
                
        else: # Legacy single-round format fallback
            gt_A = game["agent_A"]["ground_truth"]
            guesses_A = game["agent_A"]["guesses"]
            score_A = calculate_score(gt_A, guesses_A)
            
            gt_B = game["agent_B"]["ground_truth"]
            guesses_B = game["agent_B"]["guesses"]
            score_B = calculate_score(gt_B, guesses_B)
            
            total_score += (score_A + score_B)
            metrics["max_possible_score"] = 6
            
            game_words = 0
            for turn in game["transcript"]:
                msg = turn["output"].get("message", "")
                if msg:
                    game_words += len(msg.split())
                    
            total_words += game_words
            total_turns += len(game["transcript"])
            
    metrics["avg_total_score"] = total_score / len(data)
    metrics["avg_word_count"] = total_words / len(data)
    metrics["avg_turn_count"] = total_turns / len(data)
    
    for r_num, s in sorted(round_stats.items()):
        c = s["count"]
        metrics["round_metrics"][f"Round_{r_num}"] = {
            "avg_score": s["score"] / c,
            "avg_words": s["words"] / c,
            "avg_turns": s["turns"] / c
        }
        
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
            print(f"Avg Score: {eff_metrics['avg_total_score']:.2f} / {eff_metrics['max_possible_score']}")
            print(f"Avg Word Count: {eff_metrics['avg_word_count']:.2f}")
            print(f"Avg Turn Count: {eff_metrics['avg_turn_count']:.2f}")
            if eff_metrics["round_metrics"]:
                print("Per-Round Dynamics:")
                for r_name, r_data in eff_metrics["round_metrics"].items():
                    print(f"  {r_name}: Score={r_data['avg_score']:.2f}/6, Words={r_data['avg_words']:.1f}, Turns={r_data['avg_turns']:.1f}")


