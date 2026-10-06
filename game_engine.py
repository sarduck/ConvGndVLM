import json
import re
import os

PROMPT_ORIGINAL = """You will have a conversation with another player to determine which of the six images on your display are shared between both of you, and which are unique to your list. Objective: Identify the common and different images as quickly as possible!

Your images:
- Image 1: {image_1}
- Image 2: {image_2}
- Image 3: {image_3}

Instructions: Return a JSON field "guesses" containing exactly three letters:
"guesses": ["C", "D", "C"]
where each position corresponds to:
- Image 1
- Image 2
- Image 3
with the following meanings:
- C: The picture is present in both your list and your partner's list.
- D: The picture appears only in your list.

Output format: Every turn must consist of a single valid JSON object containing:
- "message" - your utterance as plain text
- "reference" - "Image 1", "Image 2", "Image 3", or null
- "guesses" - either null or a list like ["C", "D", "C"]
"""

class Agent:
    def __init__(self, name, vlm_interface, images_list, prompt_template):
        self.name = name
        self.vlm = vlm_interface
        self.images_list = images_list
        self.history = []
        self.guesses = None
        
        # Build the system prompt
        self.system_prompt = prompt_template.format(
            image_1=os.path.basename(images_list[0]),
            image_2=os.path.basename(images_list[1]),
            image_3=os.path.basename(images_list[2])
        )
        
    def extract_json(self, response_text):
        try:
            # Try to find a JSON block in the response
            match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            
            # Fallback: try to parse the raw text as JSON
            # Find first { and last }
            start = response_text.find('{')
            end = response_text.rfind('}')
            if start != -1 and end != -1:
                return json.loads(response_text[start:end+1])
        except Exception as e:
            print(f"[{self.name}] Failed to parse JSON: {e}")
            
        return {"message": response_text, "reference": None, "guesses": None}

    def take_turn(self, image_base_dir):
        # We pass the images visually only on the first turn or attach them to the system prompt.
        # For simplicity, we attach the images to the user's current turn to ensure the VLM sees them.
        image_paths = [os.path.join(image_base_dir, img) for img in self.images_list]
        
        # If history is empty, this is the first turn
        if not self.history:
            self.history.append({"role": "user", "content": "Start the game. Remember to output ONLY a valid JSON object."})
            
        # Add visual context only to the most recent user prompt
        response_text = self.vlm.generate(self.system_prompt, self.history, image_paths)
        parsed = self.extract_json(response_text)
        
        # Save our own output to history
        self.history.append({"role": "assistant", "content": json.dumps(parsed)})
        
        if parsed.get("guesses") is not None and isinstance(parsed.get("guesses"), list):
            self.guesses = parsed["guesses"]
            
        return parsed
        
    def receive_message(self, parsed_response):
        # Construct what the partner said
        partner_msg = f"Partner said: {parsed_response.get('message')}"
        if parsed_response.get('reference'):
            partner_msg += f" (Reference: {parsed_response.get('reference')})"
            
        self.history.append({"role": "user", "content": partner_msg + "\nReply with a valid JSON object."})


def play_game(game_config, vlm_A, vlm_B, image_base_dir, max_rounds=3, prompt_type="original"):
    prompt = PROMPT_ORIGINAL
    
    agent_A = Agent("A", vlm_A, game_config["agent_A"]["images"], prompt)
    agent_B = Agent("B", vlm_B, game_config["agent_B"]["images"], prompt)
    
    transcript = []
    
    for round_num in range(max_rounds):
        # Agent A's turn
        print(f"--- Round {round_num + 1} | Agent A ---")
        parsed_A = agent_A.take_turn(image_base_dir)
        transcript.append({"turn": round_num*2, "speaker": "A", "output": parsed_A})
        print(f"A: {parsed_A}")
        
        agent_B.receive_message(parsed_A)
        
        # Agent B's turn
        print(f"--- Round {round_num + 1} | Agent B ---")
        parsed_B = agent_B.take_turn(image_base_dir)
        transcript.append({"turn": round_num*2 + 1, "speaker": "B", "output": parsed_B})
        print(f"B: {parsed_B}")
        
        agent_A.receive_message(parsed_B)
        
        # Check termination condition: both agents have submitted non-null guesses
        if agent_A.guesses is not None and agent_B.guesses is not None:
            print("Both agents submitted guesses. Game over.")
            break
            
    # Compile results
    results = {
        "game_id": game_config["game_id"],
        "transcript": transcript,
        "agent_A": {
            "ground_truth": game_config["agent_A"]["ground_truth"],
            "guesses": agent_A.guesses
        },
        "agent_B": {
            "ground_truth": game_config["agent_B"]["ground_truth"],
            "guesses": agent_B.guesses
        }
    }
    
    return results

