import json
import torch
import requests
import base64
from abc import ABC, abstractmethod

class VLMInterface(ABC):
    @abstractmethod
    def generate(self, system_prompt, messages, images) -> str:
        """
        Generate a response given the conversation history and images.
        images is a list of absolute paths to the image files.
        """
        pass

class OllamaVLM(VLMInterface):
    def __init__(self, model_name="llama3.2-vision"):
        self.model_name = model_name
        self.url = "http://localhost:11434/api/chat"
        
    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate(self, system_prompt, messages, images) -> str:
        # Prepare the payload for Ollama
        ollama_messages = []
        if system_prompt:
            ollama_messages.append({"role": "system", "content": system_prompt})
            
        for msg in messages:
            ollama_msg = {"role": msg["role"], "content": msg["content"]}
            if msg.get("images"):
                # Ollama expects base64 encoded images
                ollama_msg["images"] = [self._encode_image(img) for img in msg["images"]]
            ollama_messages.append(ollama_msg)
            
        # The latest message from the user gets the current turn's images
        if images and ollama_messages and ollama_messages[-1]["role"] == "user":
            if "images" not in ollama_messages[-1]:
                ollama_messages[-1]["images"] = []
            ollama_messages[-1]["images"].extend([self._encode_image(img) for img in images])

        payload = {
            "model": self.model_name,
            "messages": ollama_messages,
            "stream": False,
            "options": {"temperature": 0.0} # Greedy decoding for reproducibility
        }
        
        response = requests.post(self.url, json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]

class HuggingFaceVLM(VLMInterface):
    def __init__(self, model_family, base_model=None, lambda_val=1.0, device="cuda", load_in_4bit=False):
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        print(f"Loading processor for {model_family}...")
        self.processor = AutoProcessor.from_pretrained(model_family, trust_remote_code=True)
        self.device = device
        
        kwargs = {"trust_remote_code": True, "device_map": device}
        if load_in_4bit:
            kwargs["load_in_4bit"] = True
            
        if lambda_val == 1.0:
            print(f"Loading instruct model {model_family}...")
            self.model = AutoModelForCausalLM.from_pretrained(model_family, **kwargs)
        elif lambda_val == 0.0:
            print(f"Loading base model {base_model}...")
            self.model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
        else:
            print(f"Interpolating weights with lambda={lambda_val}...")
            # Load base
            base = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, torch_dtype=torch.float16)
            # Load instruct
            instruct = AutoModelForCausalLM.from_pretrained(model_family, trust_remote_code=True, torch_dtype=torch.float16)
            
            # PAd interpolation: W(lambda) = W_base + lambda * (W_instruct - W_base)
            interpolated_state_dict = {}
            base_dict = base.state_dict()
            instruct_dict = instruct.state_dict()
            
            for key in base_dict.keys():
                if key in instruct_dict:
                    diff = instruct_dict[key] - base_dict[key]
                    interpolated_state_dict[key] = base_dict[key] + (lambda_val * diff)
                else:
                    interpolated_state_dict[key] = base_dict[key]
                    
            base.load_state_dict(interpolated_state_dict)
            self.model = base
            
            if load_in_4bit:
                print("Warning: 4-bit quantization after interpolation requires saving and reloading. Using float16 instead.")
                self.model = self.model.to(device)
            else:
                self.model = self.model.to(device)
                
        self.model.eval()

    def generate(self, system_prompt, messages, images) -> str:
        from PIL import Image
        
        # Format conversation for the specific model's chat template
        # Note: This is a generic implementation. Specific VLMs might require custom handling
        # of image tokens in the prompt.
        
        # Prepare text prompt using processor's chat template if available
        # Fallback to simple concatenation if not
        try:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        except Exception:
            prompt = ""
            if system_prompt:
                prompt += f"System: {system_prompt}\n"
            for msg in messages:
                prompt += f"{msg['role']}: {msg['content']}\n"
            prompt += "assistant:\n"
            
        pil_images = [Image.open(img).convert("RGB") for img in images] if images else None
        
        inputs = self.processor(text=prompt, images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=256, temperature=0.0)
            
        # Strip the input prompt from the generated output
        generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        response = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()

def get_vlm(args, lambda_val=1.0):
    if args.backend == "ollama":
        return OllamaVLM(model_name=args.model_family)
    else:
        return HuggingFaceVLM(
            model_family=args.model_family,
            base_model=args.base_model,
            lambda_val=lambda_val,
            device=args.device,
            load_in_4bit=args.load_in_4bit
        )

