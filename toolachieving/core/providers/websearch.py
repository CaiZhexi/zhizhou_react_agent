# -*- coding: utf-8 -*-
"""
联网搜索 Provider（Metaso，使用 v1 接口）
- 统一入口: web_search(q, k=None, provider="metaso", **kwargs)
- 固定路径 /api/v1/search（与提供的 curl 示例一致）
- 仅发送必要字段：q/size/page/scope（include* 仅在显式传入时发送）
"""
from __future__ import annotations
import os, requests
from typing import Any, Mapping, List, Dict

# === 默认配置（可被环境变量覆写） ===
METASO_API_KEY   = os.getenv("METASO_API_KEY", "").strip()
METASO_BASE_URL  = os.getenv("METASO_BASE_URL", "https://metaso.cn").rstrip("/")
METASO_API_PATH  = "/api/v1/search"
DEFAULT_SCOPE    = "webpage"
DEFAULT_SIZE     = int(os.getenv("SEARCH_SIZE_DEFAULT", "5"))
DEFAULT_PAGE     = int(os.getenv("SEARCH_PAGE_DEFAULT", "1"))
class MetasoConfigError(RuntimeError): ...
class MetasoHTTPError(RuntimeError): ...

def _normalize_items(obj: Mapping | List) -> List[Dict[str, str]]:
    """从常见容器键中提取列表，然后映射到 title/url/snippet（v1 优先）。"""
    def find_list(node):
        if isinstance(node, list):
            return node
        if isinstance(node, dict):
            for key in ("items", "list", "docs", "documents", "webpages"):
                v = node.get(key)
                if isinstance(v, list):
                    return v
            for key in ("results", "data", "result"):
                v = node.get(key)
                lst = find_list(v)
                if isinstance(lst, list):
                    return lst
        return []
    raw = find_list(obj) or []
    out: List[Dict[str, str]] = []
    for it in raw:
        if not isinstance(it, dict):
            continue
        title = (it.get("title") or it.get("name") or it.get("rawTitle")
                 or it.get("headline") or it.get("pageTitle"))
        url   = (it.get("url") or it.get("link") or it.get("sourceUrl")
                 or it.get("pageUrl") or it.get("href"))
        snip  = (it.get("snippet") or it.get("summary") or it.get("content")
                 or it.get("abstract") or it.get("description") or "")
        if title and url:
            out.append({"title": str(title), "url": str(url), "snippet": str(snip)})
    return out

def search_metaso(q: str, k: int | None = None, **kwargs) -> List[Dict[str, str]]:
    """直连秘塔 v1：/api/v1/search，传入 q 并返回标准 items。

    认证仅使用环境变量 METASO_API_KEY。
    """
    api_key = METASO_API_KEY
    if not api_key:
        raise MetasoConfigError("缺少 METASO_API_KEY 环境变量")

    # 按照 v1 curl 示例构造参数（最小差异对齐成功样例）
    # 固定 scope=webpage，size 由 k 控制；page 仅在显式传入时发送
    size = int(k or DEFAULT_SIZE)
    scope = str(kwargs.get("scope", DEFAULT_SCOPE) or DEFAULT_SCOPE)
    includeSummary    = bool(kwargs.get("includeSummary", False))
    includeRawContent = bool(kwargs.get("includeRawContent", False))
    conciseSnippet    = bool(kwargs.get("conciseSnippet", False))

    url = f"{METASO_BASE_URL}{METASO_API_PATH}"
    payload: Dict[str, Any] = {"q": q, "scope": scope, "size": size}
    if "page" in kwargs:
        try:
            payload["page"] = int(kwargs["page"])  # 仅在显式给出时发送
        except Exception:
            pass
    # 显式包含 include* 字段（与 curl 示例一致）
    payload["includeSummary"] = includeSummary
    payload["includeRawContent"] = includeRawContent
    payload["conciseSnippet"] = conciseSnippet
    # 主题搜索参数仅用于 v2，这里不再透传

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=12)
    except requests.RequestException as e:
        raise MetasoHTTPError(f"Metaso request error: {e!r}") from e
    if resp.status_code >= 400:
        raise MetasoHTTPError(f"Metaso HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        data = resp.json()
    except ValueError as e:
        raise MetasoHTTPError(f"Metaso returned non-JSON: {resp.text[:200]}") from e
    # 兼容业务错误（部分场景 200 但 errCode!=0）
    if isinstance(data, dict) and ("errCode" in data) and str(data.get("errCode")) not in ("", "0"):
        raise MetasoHTTPError(f"Metaso API error {data.get('errCode')}: {data.get('errMsg')}")
    items = _normalize_items(data)
    return items[:size] if items else []

def web_search(q: str, k: int | None = None, provider: str = "metaso", **kwargs) -> List[Dict[str, str]]:
    """对外统一入口；目前只有 metaso。"""
    provider = (provider or "metaso").lower()
    return search_metaso(q, k=k, **kwargs)
