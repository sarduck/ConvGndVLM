import os
import json
import random

def get_target_images_and_labels(images_list, highlighted_list, partner_images_list):
    """
    Extracts the 3 highlighted target images for an agent and determines their ground truth labels.
    """
    target_images = []
    labels = []
    for img, is_highlighted in zip(images_list, highlighted_list):
        if is_highlighted:
            target_images.append(img)
            if img in partner_images_list:
                labels.append("C")
            else:
                labels.append("D")
    return target_images, labels

def load_photobook_games(data_dir, num_games=50, seed=42):
    """
    Loads a subset of games from the PhotoBook dataset.
    Returns a list of game configurations.
    """
    logs_dir = os.path.join(data_dir, "logs")
    all_files = [f for f in os.listdir(logs_dir) if f.endswith(".json")]
    
    random.seed(seed)
    random.shuffle(all_files)
    
    games = []
    
    for filename in all_files:
        if len(games) >= num_games:
            break
            
        filepath = os.path.join(logs_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Use the first round's setup
            round_data = data['rounds'][0]
            images = round_data['images']
            highlighted = round_data['highlighted']
            
            images_A = images['A']
            images_B = images['B']
            
            highlighted_A = highlighted['A']
            highlighted_B = highlighted['B']
            
            targets_A, labels_A = get_target_images_and_labels(images_A, highlighted_A, images_B)
            targets_B, labels_B = get_target_images_and_labels(images_B, highlighted_B, images_A)
            
            # Ensure we only have 3 targets as per the original PhotoBook design
            if len(targets_A) == 3 and len(targets_B) == 3:
                games.append({
                    "game_id": data["game_id"],
                    "log_file": filename,
                    "agent_A": {
                        "images": targets_A,
                        "ground_truth": labels_A
                    },
                    "agent_B": {
                        "images": targets_B,
                        "ground_truth": labels_B
                    }
                })
        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")
            
    return games

if __name__ == "__main__":
    games = load_photobook_games("Data", num_games=2)
    print(json.dumps(games, indent=2))

