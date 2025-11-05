# -*- coding: utf-8 -*-
from __future__ import annotations
import os, requests
from typing import List, Dict, Any

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "").strip()
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
# 常用中文多域模型（可换成你偏好的）：BAAI/bge-m3 或 intfloat/multilingual-e5-large
EMBEDDING_MODEL = os.getenv("SILICONFLOW_EMBEDDING_MODEL", "BAAI/bge-m3")

class EmbedConfigError(RuntimeError): ...
class EmbedHTTPError(RuntimeError): ...

def embed_texts(texts: List[str], model: str | None = None) -> List[List[float]]:
    """
    调硅基流动 Embeddings：POST /v1/embeddings
    请求体: {"model": "...", "input": [text1, text2, ...]}
    文档: https://docs.siliconflow.cn/cn/api-reference/embeddings/create-embeddings
    """
    if not SILICONFLOW_API_KEY:
        raise EmbedConfigError("缺少 SILICONFLOW_API_KEY 环境变量")
    url = f"{SILICONFLOW_BASE_URL}/embeddings"
    payload = {"model": model or EMBEDDING_MODEL, "input": texts}
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
    except requests.RequestException as e:
        raise EmbedHTTPError(f"Embeddings request error: {e!r}") from e
    if resp.status_code >= 400:
        raise EmbedHTTPError(f"Embeddings HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    # 期望结构：{"data":[{"embedding":[...], "index":0},...]}
    try:
        return [row["embedding"] for row in data["data"]]
    except Exception as e:
        raise EmbedHTTPError(f"Bad embeddings response: {data}") from e
