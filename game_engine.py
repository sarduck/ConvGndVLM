import json
import re
import os

PROMPT_ORIGINAL = """You will have a conversation with another player to determine which of the images on your display are shared between both of you, and which are unique to your list. You are Player {player}. Objective: Identify the common and different images as quickly as possible!

Your images:
• Image 1
• Image 2
• Image 3

Important: The numbering (Image 1, Image 2, Image 3) is local to you. Your partner cannot see your screen. Describe the visual content of your images to find matches.

Instructions: When you are ready, return a JSON field "guesses" containing exactly three letters:
"guesses": ["C", "D", "C"]
where each position corresponds to:
• Image 1
• Image 2
• Image 3
with the following meanings:
• C: The picture is present in both your list and your partner's list.
• D: The picture appears only in your list.

Output format: Every turn must consist of a single valid JSON object containing:
• "message" - your utterance as plain text describing images or asking questions
• "reference" - "Image 1", "Image 2", "Image 3", or null
• "guesses" - null while still discussing; only provide a list like ["C", "D", "C"] when you are ready to submit your choices
"""

PROMPT_ENGINEERED = """Goal: find which images are COMMON between you and your partner’s list. You are Player {player}. Refer to images only as “Image 1”, “Image 2”, or “Image 3” – never mention file names or IDs. Important: the number (Image 1/2/3) is local to you. Your Image 1 could be your partner’s Image 3. Decide only from what you SEE, not from the number.
– Describe what you see in each image you are currently discussing.
– Ask short, precise questions about your partner’s view.
– Answer their questions clearly.

Your images:
• Image 1
• Image 2
• Image 3

When – and only when – you are ready to identify all three images, return a JSON field "guesses" containing exactly three letters:
"guesses": ["C", "D", "C"]
where each position corresponds to Image 1 Image 2 Image 3 in that order:
• C = the picture is present in both your list and your partner’s list
• D = the picture appears only in your list

The physical position of the matching image in your partner’s list does not matter. Once you send a non-null guesses, you may not revise them. Sending a final guesses automatically signals that you are ready to complete the round.

Output format – every turn must be a single valid JSON object with:
• "message" – your utterance as plain text
• "reference" – “Image 1”, “Image 2”, “Image 3”, or null
• "guesses" – null while discussing, or a list like ["C", "D", "C"]
"""

class Agent:
    def __init__(self, name, vlm_interface, prompt_template):
        self.name = name
        self.vlm = vlm_interface
        self.prompt_template = prompt_template
        self.system_prompt = prompt_template.format(player=name)
        self.images_list = []
        self.history = []
        self.current_round_guesses = None
        
    def start_new_round(self, round_nr, images_list):
        self.images_list = images_list
        self.current_round_guesses = None
        
        if round_nr == 1:
            self.history = []
            self.history.append({
                "role": "user",
                "content": f"Round 1 begins. You are Player {self.name}. Start by describing one or more of your 3 images visually to your partner. Reply with a valid JSON object."
            })
        else:
            self.history.append({
                "role": "user",
                "content": f"Round {round_nr} begins with a new set of 3 images on your display. Some may have appeared in previous rounds. Continue communicating to identify which images are common (C) or different (D). Reply with a valid JSON object."
            })

    def extract_json(self, response_text):
        try:
            match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            
            start = response_text.find('{')
            end = response_text.rfind('}')
            if start != -1 and end != -1:
                return json.loads(response_text[start:end+1])
        except Exception as e:
            print(f"[{self.name}] Failed to parse JSON: {e}", flush=True)
            
        return {"message": response_text, "reference": None, "guesses": None}

    def take_turn(self, image_base_dir, force_guess=False):
        image_paths = [os.path.join(image_base_dir, img) for img in self.images_list]
        
        if force_guess and self.current_round_guesses is None:
            self.history.append({
                "role": "user",
                "content": "Time to finalize this round. Based on the discussion, submit your final guesses for Image 1, Image 2, and Image 3 as ['C'/'D', 'C'/'D', 'C'/'D']. Reply with a valid JSON object."
            })
            
        response_text = self.vlm.generate(self.system_prompt, self.history, image_paths)
        parsed = self.extract_json(response_text)
        
        self.history.append({"role": "assistant", "content": json.dumps(parsed)})
        
        # Check if valid guesses were submitted
        guesses = parsed.get("guesses")
        if isinstance(guesses, list) and len(guesses) == 3 and all(g in ["C", "D"] for g in guesses):
            self.current_round_guesses = guesses
            
        return parsed
        
    def receive_message(self, parsed_response):
        partner_msg = f"Partner said: {parsed_response.get('message', '')}"
        if parsed_response.get('reference'):
            partner_msg += f" (Reference: {parsed_response.get('reference')})"
            
        partner_guessed = parsed_response.get('guesses') is not None
        if partner_guessed and self.current_round_guesses is None:
            prompt_text = f"{partner_msg}\n\nYour partner has submitted their final guesses for this round. What are your final guesses? Provide your choices as 'guesses': ['C'/'D', 'C'/'D', 'C'/'D']."
        else:
            prompt_text = f"{partner_msg}\n\nRespond to your partner based on visual content. Reply with a valid JSON object."
            
        self.history.append({"role": "user", "content": prompt_text})


def play_round(round_setup, agent_A, agent_B, image_base_dir, round_nr, max_turns_per_round=6, verbose=True):
    """
    Executes one round of PhotoBook:
    - Agents alternate turns until both submit non-null guesses or turn cap is reached.
    """
    if verbose:
        print(f"\n>>> Starting Round {round_nr} <<<", flush=True)
    agent_A.start_new_round(round_nr, round_setup["agent_A"]["images"])
    agent_B.start_new_round(round_nr, round_setup["agent_B"]["images"])
    
    round_transcript = []
    
    turn_count = 0
    while turn_count < max_turns_per_round:
        # Check if both have already guessed
        if agent_A.current_round_guesses is not None and agent_B.current_round_guesses is not None:
            break
            
        # Agent A's turn (if A hasn't guessed, or if B just guessed)
        if agent_A.current_round_guesses is None:
            force_A = (turn_count >= max_turns_per_round - 2)
            parsed_A = agent_A.take_turn(image_base_dir, force_guess=force_A)
            round_transcript.append({"round": round_nr, "turn": turn_count, "speaker": "A", "output": parsed_A})
            if verbose:
                msg = parsed_A.get("message", "")
                guess = f" -> Guesses: {parsed_A['guesses']}" if parsed_A.get('guesses') else ""
                print(f"  [R{round_nr} T{turn_count}] Agent A: {msg[:100]}{'...' if len(msg)>100 else ''}{guess}", flush=True)
            agent_B.receive_message(parsed_A)
            turn_count += 1
            
        # If both guessed after A's turn, round ends
        if agent_A.current_round_guesses is not None and agent_B.current_round_guesses is not None:
            break
            
        # Agent B's turn
        if agent_B.current_round_guesses is None:
            force_B = (turn_count >= max_turns_per_round - 1)
            parsed_B = agent_B.take_turn(image_base_dir, force_guess=force_B)
            round_transcript.append({"round": round_nr, "turn": turn_count, "speaker": "B", "output": parsed_B})
            if verbose:
                msg = parsed_B.get("message", "")
                guess = f" -> Guesses: {parsed_B['guesses']}" if parsed_B.get('guesses') else ""
                print(f"  [R{round_nr} T{turn_count}] Agent B: {msg[:100]}{'...' if len(msg)>100 else ''}{guess}", flush=True)
            agent_A.receive_message(parsed_B)
            turn_count += 1
            
    # Final check: if an agent still hasn't guessed, force a final turn
    if agent_A.current_round_guesses is None:
        parsed_A = agent_A.take_turn(image_base_dir, force_guess=True)
        round_transcript.append({"round": round_nr, "turn": turn_count, "speaker": "A", "output": parsed_A})
        if verbose:
            print(f"  [R{round_nr} Final] Agent A: {parsed_A.get('message', '')} -> Guesses: {parsed_A.get('guesses')}", flush=True)
        turn_count += 1
        
    if agent_B.current_round_guesses is None:
        parsed_B = agent_B.take_turn(image_base_dir, force_guess=True)
        round_transcript.append({"round": round_nr, "turn": turn_count, "speaker": "B", "output": parsed_B})
        if verbose:
            print(f"  [R{round_nr} Final] Agent B: {parsed_B.get('message', '')} -> Guesses: {parsed_B.get('guesses')}", flush=True)
        turn_count += 1
        
    # Fallback default if model still refused
    if agent_A.current_round_guesses is None:
        agent_A.current_round_guesses = ["D", "D", "D"]
    if agent_B.current_round_guesses is None:
        agent_B.current_round_guesses = ["D", "D", "D"]
        
    # Compute round score
    gt_A = round_setup["agent_A"]["ground_truth"]
    gt_B = round_setup["agent_B"]["ground_truth"]
    score_A = sum(1 for g, t in zip(agent_A.current_round_guesses, gt_A) if g == t)
    score_B = sum(1 for g, t in zip(agent_B.current_round_guesses, gt_B) if g == t)
    if verbose:
        print(f"  --> Round {round_nr} Score: Agent A = {score_A}/3, Agent B = {score_B}/3 (Total: {score_A + score_B}/6)", flush=True)
    
    return {
        "round_nr": round_nr,
        "transcript": round_transcript,
        "agent_A": {
            "ground_truth": gt_A,
            "guesses": agent_A.current_round_guesses,
            "score": score_A
        },
        "agent_B": {
            "ground_truth": gt_B,
            "guesses": agent_B.current_round_guesses,
            "score": score_B
        },
        "round_score": score_A + score_B
    }


def play_game(game_config, vlm_A, vlm_B, image_base_dir, max_rounds=3, prompt_type="engineered", verbose=True):
    prompt = PROMPT_ENGINEERED if prompt_type == "engineered" else PROMPT_ORIGINAL
    
    agent_A = Agent("A", vlm_A, prompt)
    agent_B = Agent("B", vlm_B, prompt)
    
    num_rounds = min(max_rounds, len(game_config["rounds"]))
    completed_rounds = []
    total_score = 0
    full_transcript = []
    
    for r_idx in range(num_rounds):
        round_setup = game_config["rounds"][r_idx]
        round_res = play_round(
            round_setup=round_setup,
            agent_A=agent_A,
            agent_B=agent_B,
            image_base_dir=image_base_dir,
            round_nr=r_idx + 1,
            verbose=verbose
        )
        completed_rounds.append(round_res)
        total_score += round_res["round_score"]
        full_transcript.extend(round_res["transcript"])
        
    if verbose:
        print(f"=== Game {game_config['game_id']} Completed! Total Score: {total_score}/{num_rounds * 6} ===", flush=True)
    
    return {
        "game_id": game_config["game_id"],
        "num_rounds": num_rounds,
        "total_score": total_score,
        "max_score": num_rounds * 6,
        "transcript": full_transcript,
        "rounds": completed_rounds
    }

