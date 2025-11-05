# core/providers/llm_silicon.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, requests
from typing import List, Dict, Any

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "").strip()
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
DEFAULT_MODEL = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2-7B-Instruct")  # 文档/型号见官网

class LLMConfigError(RuntimeError): ...
class LLMHTTPError(RuntimeError): ...

def chat(messages: List[Dict[str, str]],
         model: str | None = None,
         temperature: float = 0.2,
         max_tokens: int = 1024,
         **kwargs) -> Dict[str, Any]:
    """
    直调硅基流动 Chat Completions，返回完整 JSON。
    文档：/v1/chat/completions（OpenAI 兼容）。参见官方说明。  # :contentReference[oaicite:6]{index=6}
    """
    if not SILICONFLOW_API_KEY:
        raise LLMConfigError("缺少 SILICONFLOW_API_KEY 环境变量")
    url = f"{SILICONFLOW_BASE_URL}/chat/completions"
    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        # 需要时可以透传：top_p, presence_penalty, frequency_penalty, stream 等
    }
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
    except requests.RequestException as e:
        raise LLMHTTPError(f"SiliconFlow request error: {e!r}") from e
    if resp.status_code >= 400:
        raise LLMHTTPError(f"SiliconFlow HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        data = resp.json()
    except ValueError as e:
        raise LLMHTTPError(f"SiliconFlow returned non-JSON: {resp.text[:200]}") from e
    # 期望结构：choices[0].message.content
    if not isinstance(data, dict) or "choices" not in data:
        raise LLMHTTPError(f"SiliconFlow bad response: {data}")
    return data

def simple_answer(q: str,
                  system_prompt: str = "你是一个中文智能助手，请用简洁、准确的中文回答用户问题。") -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": q}
    ]
    data = chat(messages)
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    return content.strip()
