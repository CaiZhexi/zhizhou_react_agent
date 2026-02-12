import requests
from typing import List, Dict
from config import (
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    SILICONFLOW_MODEL
)

class SiliconFlowLLM:
    def __init__(self):
        self.api_key = SILICONFLOW_API_KEY
        self.base_url = SILICONFLOW_BASE_URL.rstrip("/")
        self.model = SILICONFLOW_MODEL

    def chat(self, messages: List[Dict]) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
