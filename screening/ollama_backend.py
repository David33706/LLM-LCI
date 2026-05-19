"""
screening/ollama_backend.py
---------------------------
Inference backend using Ollama's local HTTP API.
Designed for Apple Silicon Macs (no CUDA required).

Usage:
    from screening.ollama_backend import OllamaBackend
    backend = OllamaBackend(model="qwen3:8b-q8_0")
    result = backend.classify(title, abstract)
"""

import re
import time
import requests

from screening.prompts import SYSTEM_PROMPT, build_user_message

DEFAULT_MODEL = "qwen3:8b-q8_0"
OLLAMA_URL = "http://localhost:11434"


class OllamaBackend:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.api_url = f"{OLLAMA_URL}/api/chat"

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is pulled."""
        try:
            resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            # Ollama tags can include :latest, so check prefix
            return any(self.model in m for m in models)
        except Exception:
            return False

    def classify(self, title: str, abstract: str) -> dict:
        """Classify a single paper. Returns dict with prediction, raw_response, elapsed_s."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT + "\n/no_think"},
                {"role": "user", "content": build_user_message(title, abstract)},
            ],
            "stream": False,
            "options": {
                "num_predict": 10,
                "temperature": 0.0,
            },
        }

        t0 = time.time()
        try:
            resp = requests.post(self.api_url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("message", {}).get("content", "").strip()
        except Exception as e:
            raw = f"ERROR: {e}"

        elapsed = time.time() - t0
        prediction = self._parse_label(raw)

        return {
            "prediction": prediction,
            "raw_response": raw,
            "elapsed_s": round(elapsed, 2),
        }

    @staticmethod
    def _parse_label(text: str):
        """Extract 0 or 1 from model output. Returns None if unparseable."""
        match = re.search(r"\b([01])\b", text)
        return int(match.group(1)) if match else None