import requests
from typing import List

from config import (
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    SILICONFLOW_EMBEDDING_MODEL,
)


class SiliconFlowEmbeddings:
    def __init__(self):
        self.api_key = SILICONFLOW_API_KEY
        self.base_url = SILICONFLOW_BASE_URL.rstrip("/")
        self.model = SILICONFLOW_EMBEDDING_MODEL

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": texts,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        data = sorted(data, key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in data]
