"""
screening/hf_backend.py
-----------------------
Inference backend using HuggingFace Transformers + CUDA.
Designed for the office Windows machine (RTX 2000 Ada, 16GB VRAM).

Usage:
    from screening.hf_backend import HuggingFaceBackend
    backend = HuggingFaceBackend(model_name="Qwen/Qwen2.5-7B-Instruct")
    result = backend.classify(title, abstract)
"""

import re
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from screening.prompts import SYSTEM_PROMPT, build_user_message

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class HuggingFaceBackend:
    def __init__(self, model_name: str = DEFAULT_MODEL, quantization: str = "none"):
        """
        Args:
            model_name: HuggingFace model identifier.
            quantization: "none" (bf16), "8bit", or "16bit".
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model(quantization)

    def _load_model(self, quantization: str):
        print(f"Loading {self.model_name} (quantization={quantization})...")

        load_kwargs = {
            "low_cpu_mem_usage": True,
            "device_map": "auto",
        }

        if quantization == "8bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == "16bit":
            load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
        self.model.eval()
        print(f"Model loaded. Device: {self._get_device()}")

    def _get_device(self):
        try:
            return self.model.get_input_embeddings().weight.device
        except Exception:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def classify(self, title: str, abstract: str) -> dict:
        """Classify a single paper. Same interface as OllamaBackend."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(title, abstract)},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._get_device())

        t0 = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        elapsed = time.time() - t0

        prediction = self._parse_label(raw)

        return {
            "prediction": prediction,
            "raw_response": raw,
            "elapsed_s": round(elapsed, 2),
        }

    @staticmethod
    def _parse_label(text: str):
        match = re.search(r"\b([01])\b", text)
        return int(match.group(1)) if match else None