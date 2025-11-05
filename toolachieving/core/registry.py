# -*- coding: utf-8 -*-
from features.f1.service import run as f1_run
from features.llm.service import run as llm_run  # 默认 LLM（硅基流动）
from features.f2.service import run as f2_run
from features.f3.service import run as f3_run

REGISTRY = {
    "f1":  {"handler": f1_run,  "desc": "Web Search (联网搜索)"},
    "llm": {"handler": llm_run, "desc": "Default LLM (SiliconFlow)"},
    "f2":  {"handler": f2_run, "desc": "Knowledge Base (RAG)"},
    "f3":  {"handler": f3_run, "desc": "Python Math (代码求解)"},
}


def dispatch(target: str, payload: dict) -> dict:
    item = REGISTRY.get(target)
    if not item:
        return {"error": f"unknown feature: {target}", "items": []}
    return item["handler"](payload)
