import os
import json
import matplotlib.pyplot as plt
from metrics import compute_grounding_efficiency

def analyze_dose_response(results_dir, output_plot="dose_response.png"):
    lambda_values = []
    scores = []
    words = []
    turns = []
    
    # Parse all result files
    files = [f for f in os.listdir(results_dir) if f.startswith("results_lambda_") and f.endswith(".json")]
    
    for filename in sorted(files):
        # Extract lambda value from filename (results_lambda_0.5.json)
        lambda_val_str = filename.replace("results_lambda_", "").replace(".json", "")
        try:
            l_val = float(lambda_val_str)
        except ValueError:
            continue
            
        filepath = os.path.join(results_dir, filename)
        metrics = compute_grounding_efficiency(filepath)
        
        lambda_values.append(l_val)
        scores.append(metrics["avg_total_score"])
        words.append(metrics["avg_word_count"])
        turns.append(metrics["avg_turn_count"])
        
    if not lambda_values:
        print("No results found to plot.")
        return
        
    # Sort by lambda
    sorted_indices = sorted(range(len(lambda_values)), key=lambda k: lambda_values[k])
    lambda_values = [lambda_values[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    words = [words[i] for i in sorted_indices]
    turns = [turns[i] for i in sorted_indices]
    
    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Task Score
    axs[0].plot(lambda_values, scores, marker='o', linestyle='-', color='b')
    axs[0].set_title('Avg Task Score vs Lambda')
    axs[0].set_xlabel('Lambda (Instruction Strength)')
    axs[0].set_ylabel('Score (max 6)')
    axs[0].grid(True)
    
    # Word Count
    axs[1].plot(lambda_values, words, marker='s', linestyle='-', color='g')
    axs[1].set_title('Avg Word Count vs Lambda')
    axs[1].set_xlabel('Lambda (Instruction Strength)')
    axs[1].set_ylabel('Words')
    axs[1].grid(True)
    
    # Turn Count
    axs[2].plot(lambda_values, turns, marker='^', linestyle='-', color='r')
    axs[2].set_title('Avg Turn Count vs Lambda')
    axs[2].set_xlabel('Lambda (Instruction Strength)')
    axs[2].set_ylabel('Turns')
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="outputs")
    parser.add_argument("--output_plot", type=str, default="dose_response.png")
    args = parser.parse_args()
    
    analyze_dose_response(args.results_dir, args.output_plot)

