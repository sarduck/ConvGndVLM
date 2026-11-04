import json
import requests
import base64
import os
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
        self._image_cache = {}
        
    def _encode_image(self, image_path):
        if image_path not in self._image_cache:
            with open(image_path, "rb") as image_file:
                self._image_cache[image_path] = base64.b64encode(image_file.read()).decode('utf-8')
        return self._image_cache[image_path]

    def generate(self, system_prompt, messages, images) -> str:
        ollama_messages = []
        if system_prompt:
            ollama_messages.append({"role": "system", "content": system_prompt})
            
        first_user_idx = None
        for msg in messages:
            ollama_msg = {"role": msg["role"], "content": msg["content"]}
            if msg.get("images"):
                ollama_msg["images"] = [self._encode_image(img) for img in msg["images"]]
            ollama_messages.append(ollama_msg)
            if first_user_idx is None and msg["role"] == "user":
                first_user_idx = len(ollama_messages) - 1

        if images and first_user_idx is not None:
            if "images" not in ollama_messages[first_user_idx]:
                ollama_messages[first_user_idx]["images"] = [self._encode_image(img) for img in images]

        payload = {
            "model": self.model_name,
            "messages": ollama_messages,
            "stream": False,
            "options": {"temperature": 0.0}
        }
        
        response = requests.post(self.url, json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]


# ---------------------------------------------------------------------------
# InternVL image preprocessing (from official OpenGVLab example)
# Lazy-loaded to avoid requiring torchvision when using Ollama backend.
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def _build_transform(input_size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_ar)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_ar = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_ar[0]
    target_height = image_size * target_ar[1]
    blocks = target_ar[0] * target_ar[1]
    resized_img = image.resize((target_width, target_height))
    processed = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed.append(resized_img.crop(box))
    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))
    return processed

def load_internvl_image(image_file, input_size=448, max_num=1):
    """Load and preprocess a single image for InternVL. Returns pixel_values tensor."""
    import torch
    from PIL import Image
    image = Image.open(image_file).convert('RGB')
    transform = _build_transform(input_size)
    images = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=False, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class InternVLHuggingFace(VLMInterface):
    """
    HuggingFace backend for InternVL3 models.
    Uses AutoModel + model.chat() as per the official OpenGVLab API.
    Supports PAd weight interpolation between base and instruct checkpoints.
    """
    def __init__(self, model_family, base_model=None, lambda_val=1.0, device="cuda", load_in_4bit=False):
        import torch
        from transformers import AutoModel, AutoTokenizer
        
        # Compatibility fix: Newer versions of transformers/accelerate expect 'all_tied_weights_keys'
        # dictionary on models when inspecting tied weights. Custom remote code models (like InternVL)
        # only define '_tied_weights_keys'. Providing a default empty dict prevents AttributeError.
        if not hasattr(torch.nn.Module, "all_tied_weights_keys"):
            torch.nn.Module.all_tied_weights_keys = {}

        # Determine dtype
        if device == "mps":
            torch_dtype = torch.float16
            if load_in_4bit:
                print("Note: bitsandbytes 4-bit quantization requires CUDA; using float16 for Apple Silicon.")
                load_in_4bit = False
        elif device == "cuda":
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = torch.float32
            
        self.device = device
        self.torch_dtype = torch_dtype
        
        # Load tokenizer (always from local cache if available)
        print(f"Loading tokenizer from {model_family}...")
        tok_kwargs = {"trust_remote_code": True, "use_fast": False, "fix_mistral_regex": True}
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_family, local_files_only=True, **tok_kwargs)
        except Exception:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_family, **tok_kwargs)
            except Exception:
                tok_kwargs.pop("fix_mistral_regex", None)
                self.tokenizer = AutoTokenizer.from_pretrained(model_family, **tok_kwargs)
        
        # Build common kwargs — use 'dtype' (not deprecated 'torch_dtype').
        # Always try local cache first; fall back to HuggingFace Hub on a cache miss.
        load_kwargs_base = {
            "trust_remote_code": True,
            "dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }
        
        # On CUDA, use device_map="auto" to distribute layers or support 4-bit.
        # On Apple Silicon (MPS), unified memory allows direct loading and .to("mps")
        # without triggering accelerate's device_map placement logic.
        if device == "cuda":
            load_kwargs_base["device_map"] = "auto"
            if load_in_4bit:
                load_kwargs_base["load_in_4bit"] = True

        def _load_model(model_id, **extra_kwargs):
            """Try local cache first; fall through to hub download if not cached."""
            kwargs = {**load_kwargs_base, **extra_kwargs}
            try:
                return AutoModel.from_pretrained(model_id, local_files_only=True, **kwargs)
            except Exception:
                print(f"  Model not in local cache — downloading {model_id} from HuggingFace Hub...")
                return AutoModel.from_pretrained(model_id, **kwargs)

        if lambda_val == 1.0:
            print(f"Loading instruct model: {model_family}")
            self.model = _load_model(model_family).eval()
            if "device_map" not in load_kwargs_base:
                self.model = self.model.to(device)
            
        elif lambda_val == 0.0:
            print(f"Loading base model: {base_model}")
            self.model = _load_model(base_model).eval()
            if "device_map" not in load_kwargs_base:
                self.model = self.model.to(device)
            
        else:
            save_dir = f"checkpoints/internvl_lambda_{lambda_val}"
            if os.path.exists(save_dir):
                print(f"Loading pre-saved interpolated checkpoint from disk: {save_dir}")
                self.model = _load_model(save_dir).eval()
                if "device_map" not in load_kwargs_base:
                    self.model = self.model.to(device)
            else:
                print(f"PAd interpolation: lambda={lambda_val}")
                print(f"  Loading base:     {base_model}")
                base_m = _load_model(base_model)
                print(f"  Loading instruct: {model_family}")
                inst_m = _load_model(model_family)
                
                # Memory-optimal in-place PAd: W(λ) = W_base + λ·(W_instruct − W_base)
                # Uses torch.lerp_ in-place so NO extra 16GB dictionary is allocated in RAM.
                inst_params = dict(inst_m.named_parameters())
                with torch.no_grad():
                    for name, param in base_m.named_parameters():
                        if name in inst_params and param.shape == inst_params[name].shape:
                            param.data.lerp_(inst_params[name].data, lambda_val)
                            
                # Save the interpolated model to disk so we never have to interpolate it again!
                print(f"Saving interpolated checkpoint to disk for instant reuse: {save_dir}")
                os.makedirs(save_dir, exist_ok=True)
                base_m.save_pretrained(save_dir)
                self.tokenizer.save_pretrained(save_dir)

                # Ensure custom remote code python files are copied to save_dir
                import glob, shutil
                hf_cache = os.path.expanduser("~/.cache/huggingface")
                custom_py_files = (
                    glob.glob(f"{hf_cache}/hub/models--OpenGVLab--InternVL3_5-4B/snapshots/*/*.py") or
                    glob.glob(f"{hf_cache}/modules/transformers_modules/OpenGVLab/*/*/*.py")
                )
                for py_file in custom_py_files:
                    dst = os.path.join(save_dir, os.path.basename(py_file))
                    if not os.path.exists(dst):
                        shutil.copy(py_file, dst)
                
                # Immediately free the instruct model and parameter dict from memory
                del inst_m, inst_params
                import gc; gc.collect()
                if device == "mps" and hasattr(torch, "mps"):
                    torch.mps.empty_cache()
                elif device == "cuda":
                    torch.cuda.empty_cache()
                    
                self.model = base_m.to(device).eval()
                
        # Generation config
        self.generation_config = dict(
            num_beams=1,
            max_new_tokens=256,
            do_sample=False,
        )
        
        # Image pixel cache (capped to avoid memory bloat)
        self._pixel_cache = {}
        
    def _load_pixels(self, image_path):
        """Load and cache preprocessed pixel values for an image."""
        if len(self._pixel_cache) > 18:
            self._pixel_cache.clear()
        if image_path not in self._pixel_cache:
            self._pixel_cache[image_path] = load_internvl_image(image_path, max_num=1)
        return self._pixel_cache[image_path]

    def generate(self, system_prompt, messages, images) -> str:
        import torch
        
        # Prepare pixel values for images
        pixel_values = None
        num_patches_list = None
        pixel_list = []
        if images:
            num_patches_list = []
            for img_path in images:
                pv = self._load_pixels(img_path).to(dtype=self.torch_dtype, device=self.device)
                pixel_list.append(pv)
                num_patches_list.append(pv.size(0))
            pixel_values = torch.cat(pixel_list, dim=0)
        
        # Build the question string with image placeholders
        # InternVL expects <image>\n tokens at the start for each image
        image_prefix = ""
        if images:
            for i in range(len(images)):
                image_prefix += f"Image {i+1}: <image>\n"
                
        # Construct the full question from system prompt + message history
        question_parts = []
        if system_prompt:
            question_parts.append(system_prompt)
        if image_prefix:
            question_parts.append(image_prefix)
        for msg in messages:
            role = "You" if msg["role"] == "assistant" else "Partner" if msg["role"] == "user" else msg["role"]
            question_parts.append(f"{role}: {msg['content']}")
        question_parts.append("You:")
        
        question = "\n".join(question_parts)
        
        with torch.no_grad():
            response = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=self.generation_config,
                history=None,
                return_history=False,
                num_patches_list=num_patches_list,
            )
            
        # Free inference tensors immediately
        del pixel_values, pixel_list
        if self.device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
            
        return response.strip()


def get_vlm(args, lambda_val=1.0):
    if args.backend == "ollama":
        return OllamaVLM(model_name=args.model_family)
    else:
        return InternVLHuggingFace(
            model_family=args.model_family,
            base_model=args.base_model,
            lambda_val=lambda_val,
            device=args.device,
            load_in_4bit=args.load_in_4bit
        )
