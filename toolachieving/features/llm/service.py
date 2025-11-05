# features/llm/service.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List
try:
    from core.providers.llm_silicon import simple_answer, LLMConfigError, LLMHTTPError
except Exception:
    # 兜底定义，避免导入期报错
    def simple_answer(q: str, system_prompt: str = "") -> str:
        return "（LLM 未配置，返回占位回答）"
    class LLMConfigError(Exception): ...
    class LLMHTTPError(Exception): ...

def run(payload: Dict) -> Dict:
    q = (payload.get("q") or "").strip()
    if not q:
        return {"feature": "llm", "items": [], "error": "missing q"}

    sys_prompt = payload.get("system_prompt") or "你是一个中文智能助手，请用简洁中文回答。"
    try:
        text = simple_answer(q, sys_prompt)
        items: List[Dict] = [{"title": "LLM Answer", "url": "", "snippet": text}]
        return {"feature": "llm", "text": text, "items": items}
    except (LLMConfigError, LLMHTTPError) as e:
        return {"feature": "llm", "items": [], "error": str(e)}
    except Exception as e:
        return {"feature": "llm", "items": [], "error": f"llm error: {e}"}
