# -*- coding: utf-8 -*-
"""
f1（联网搜索）业务编排：默认走秘塔 API（provider=metaso）
- 不做重写：始终使用原始问题 q
- 调用 provider 进行搜索
- 去重、兜底，并把错误透传到响应
"""
from __future__ import annotations
from typing import Dict, List

# 兼容：如果你的 provider 暂未导出 MetasoHTTPError/MetasoConfigError，也能正常工作
try:
    from core.providers.websearch import web_search, MetasoHTTPError, MetasoConfigError  # type: ignore
except Exception:  # noqa
    from core.providers.websearch import web_search  # type: ignore
    class MetasoHTTPError(Exception):
        pass
    class MetasoConfigError(Exception):
        pass

def _dedup(items: List[Dict]) -> List[Dict]:
    seen = set()
    out: List[Dict] = []
    for it in items or []:
        key = it.get("url") or it.get("title")
        if key and key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out

# 允许通过 slots 透传到 provider 的字段（其余忽略，用 provider 默认）
_PASSTHROUGH_KEYS = {
    "scope",
    "page",
    "includeSummary",
    "includeRawContent",
    "conciseSnippet",
}

def run(payload: Dict) -> Dict:
    q = payload.get("q", "") or ""
    if not q:
        # 通常由上层路由返回 400，这里再防一层
        return {"feature": "f1", "query": q, "used_query": q, "provider": "metaso", "items": [], "error": "missing q"}

    # 方案：前端只传问题，其余参数后端固定
    k = 5
    slots: Dict = payload.get("slots") or {}

    # 不做重写：始终使用原始问题
    used_query = q
    provider = (slots.get("provider") or "metaso").lower()

    # 最简跑通：不默认打开任何可选开关；仅在 slots 显式给出时透传
    kwargs: Dict = {}
    # 仅把允许的键透传给 provider
    kwargs.update({k: slots[k] for k in _PASSTHROUGH_KEYS if k in slots})
    # page 要转 int；布尔字段保持 True/False
    if "page" in kwargs:
        try:
            kwargs["page"] = int(kwargs["page"])
        except Exception:
            kwargs["page"] = 1

    items: List[Dict]
    error = None

    try:
        items = web_search(used_query, k=k, provider=provider, **kwargs)
        items = _dedup(items)

        # 由于不做重写，上述回退逻辑不再触发（used_query == q）

    except MetasoHTTPError as e:
        # 业务/HTTP 层面的可读错误（如 1000/2005），透传给前端
        items, error = [], str(e)
    except MetasoConfigError as e:
        # 配置问题（如缺少 METASO_API_KEY）
        items, error = [], str(e)
    except Exception as e:  # 兜底：网络等异常
        items, error = [], f"f1 error: {e}"

    return {
        "feature": "f1",
        "query": q,
        "provider": provider,
        "items": items,
        **({"error": error} if error else {}),
    }
