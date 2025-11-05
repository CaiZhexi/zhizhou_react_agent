# -*- coding: utf-8 -*-
import os
import re
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

try:
    from core.vectorstore.faiss_store import KBIndex  # type: ignore
    _KB_AVAILABLE = True
except Exception:  # pragma: no cover
    KBIndex = None  # type: ignore
    _KB_AVAILABLE = False

# 可选：用于路由语义判定的 LLM（硅基流动）
try:
    from core.providers.llm_silicon import chat as llm_chat  # type: ignore
    _LLM_AVAILABLE = True
except Exception:  # pragma: no cover
    llm_chat = None  # type: ignore
    _LLM_AVAILABLE = False

TIME_WORDS = r"(今天|明天|后天|现在|本周|本月)"
WEATHER_HINT = r"(天气|气温|温度|下雨|降雨|晴|阴|多云|台风|空气质量)"
NEWS_PRICE = r"(新闻|价格|发布|最新)"
KB_HINT = r"(知识库|文档|文件|资料|报告|手册|说明书|方案|总结|教程|笔记)"
GEN_HINT = r"(写|短诗|诗|故事|续写|润色|摘要|总结|概括|定义|解释|推荐|建议|如何|怎么|一句话|短文|评论|点评|文案|口播|对联|打油诗)"
GREET_HINT = r"^(你好|您好|哈喽|嗨|hi|hello|hey|早上好|中午好|下午好|晚上好)[！!。\.\s]*$"

# —— KB 预探测缓存 ——
_KB_CACHE = {
    "kb_id": None,
    "idx": None,
    "mt_meta": 0.0,
    "mt_index": 0.0,
}

def _kb_paths(kb_id: str):
    root = Path(os.getenv("KB_ROOT", "data/kb")).resolve()
    idx_dir = root / kb_id / "index"
    return idx_dir, idx_dir/"meta.jsonl", idx_dir/"index.faiss"

def _kb_probe(q: str, kb_id: str | None = None) -> Tuple[bool, float]:
    if not _KB_AVAILABLE:
        return False, 0.0
    kb_id = kb_id or os.getenv("KB_ROUTE_KB_ID", "default")
    idx_dir, meta_path, index_path = _kb_paths(kb_id)
    if not (meta_path.exists() and index_path.exists()):
        return False, 0.0
    try:
        mt_meta = meta_path.stat().st_mtime
        mt_index = index_path.stat().st_mtime
        need_reload = (
            _KB_CACHE["idx"] is None or
            _KB_CACHE["kb_id"] != kb_id or
            _KB_CACHE["mt_meta"] != mt_meta or
            _KB_CACHE["mt_index"] != mt_index
        )
        if need_reload:
            idx = KBIndex(str(idx_dir))  # type: ignore
            idx.load()
            _KB_CACHE.update({
                "kb_id": kb_id,
                "idx": idx,
                "mt_meta": mt_meta,
                "mt_index": mt_index,
            })
        idx = _KB_CACHE["idx"]
        hits = idx.query(q, top_k=1)  # type: ignore
        max_score = float(hits[0]["score"]) if hits else 0.0
        return True, max_score
    except Exception:
        return False, 0.0


def _llm_route(q: str) -> Optional[Dict[str, Any]]:
    """调用 LLM 给出工具建议：
    single: {mode: "single", target, confidence, reasons}
    multi:  {mode: "multi", segments: [{id,q,tool,confidence,reasons,needs_context?,q_template?}], confidence?, reasons?}
    失败则返回 None。
    """
    if not _LLM_AVAILABLE:
        return None
    sys_prompt = (
        '你是工具路由器，只输出 JSON。\n'
        '工具：f1=联网搜索；f2=知识库；f3=Python数学；llm=纯对话/创作/解释。\n'
        '若问题包含多个子问，返回 multi：{"mode":"multi","segments":[{id,q,tool,confidence,reasons,needs_context?,q_template?}],"confidence":..,"reasons":[]};\n'
        '否则返回 single：{"mode":"single","target":"f1|f2|f3|llm","confidence":..,"reasons":[]}。\n'
        '注意：不需要外部事实的定义/解释/总结/一句话描述/推荐/建议/写作（诗歌/故事/文案等）优先选择 llm；需要实时/新闻/天气等才选 f1；数学求解选 f3；企业内文档选 f2。'
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"根据问题选择工具并可拆分子问：{q}"},
    ]
    try:
        data = llm_chat(messages)  # type: ignore
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        s = content.strip()
        if "```" in s:
            parts = s.split("```")
            if len(parts) >= 3:
                s = parts[1]
                if s.lstrip().lower().startswith("json"):
                    s = s.split("\n", 1)[1] if "\n" in s else "{}"
        obj = json.loads(s)
        mode = str(obj.get("mode", "single")).lower()
        if mode == "multi" and isinstance(obj.get("segments"), list):
            segs_in: List[Dict[str, Any]] = obj["segments"]
            segs: List[Dict[str, Any]] = []
            for i, seg in enumerate(segs_in, 1):
                tool = str(seg.get("tool", "")).lower()
                if tool not in ("f1", "f2", "f3", "llm"):
                    continue
                segs.append({
                    "id": seg.get("id") or f"s{i}",
                    "q": seg.get("q") or "",
                    "tool": tool,
                    "confidence": float(seg.get("confidence", 0.0)),
                    "reasons": seg.get("reasons") if isinstance(seg.get("reasons"), list) else [],
                    "needs_context": bool(seg.get("needs_context", False)),
                    "q_template": seg.get("q_template") or "",
                })
            conf = float(obj.get("confidence", 0.0))
            reasons = obj.get("reasons") if isinstance(obj.get("reasons"), list) else []
            return {"mode": "multi", "segments": segs, "confidence": conf, "reasons": reasons}
        # single
        tgt = str(obj.get("target", "")).lower()
        if tgt not in ("f1", "f2", "f3", "llm"):
            return None
        conf = float(obj.get("confidence", 0.0))
        reasons = obj.get("reasons") or []
        if not isinstance(reasons, list):
            reasons = [str(reasons)]
        return {"mode": "single", "target": tgt, "confidence": conf, "reasons": reasons}
    except Exception:
        return None

def _extract_slots(q: str) -> dict:
    slots = {}
    m_time = re.search(TIME_WORDS, q)
    if m_time:
        slots["when"] = m_time.group(1)

    # 优先匹配“城市+天气/气温/温度”结构，如“佛山天气”“上海的天气”
    m_loc = re.search(r"([一-龥]{2,8})(?:市|省|区|县)?(?:的)?(?:天气|气温|温度)", q)
    if not m_loc:
        # 次选：城市+行政后缀
        m_loc = re.search(r"([一-龥]{2,8})(市|省|区|县)", q)
        if m_loc:
            slots["location"] = "".join(m_loc.groups())
    else:
        slots["location"] = m_loc.group(1)

    # 不做重写：仅标注 provider，保留可用的语义槽位（如时间/地点）
    slots["provider"] = "metaso"
    return slots

def _prefer_llm(q: str) -> bool:
    """是否偏好用 LLM 直接生成/解释：
    命中生成/解释类提示，且不含 web/math 强信号时返回 True。
    """
    gen = bool(re.search(GEN_HINT, q))
    web = bool(re.search(TIME_WORDS, q) or re.search(WEATHER_HINT, q) or re.search(NEWS_PRICE, q))
    math = bool(
        re.search(r"(计算|求值|求最值|求和|方程|解方程|积分|微分|导数|极限|排列|组合|概率|方差|标准差|矩阵|行列式|特征值|阶乘)", q)
        or re.search(r"[\d\)\(\+\-\*/\^=]{3,}", q)
    )
    return gen and (not web) and (not math)

def _is_greet(q: str) -> bool:
    return bool(re.match(GREET_HINT, q.strip(), flags=re.IGNORECASE))

def _is_short_query(q: str) -> bool:
    # 粗略短句判定：去空白后长度 <= 6 视为短句
    return len((q or "").strip()) <= 6

def _detect_explicit(q: str) -> dict | None:
    """显式工具指令优先：
    - f2: 用/使用/调用 + (知识库|文档|资料|报告|手册|说明书)
    - f1: (联网|上网|搜索|搜一下|搜一搜|查一查)
    - llm: (不联网|直接回答|纯对话|闲聊)
    命中后立即返回路由决策。
    """
    qn = q.lower()
    # f2 显式
    if re.search(r"(用|使用|调用).*(知识库|文档|资料|报告|手册|说明书)", q):
        return {"target": "f2", "confidence": 0.98, "reasons": ["explicit:f2"], "slots": {}}
    # f1 显式
    if re.search(r"(联网|上网|搜索|搜一下|搜一搜|查一查)", q):
        return {"target": "f1", "confidence": 0.98, "reasons": ["explicit:f1"], "slots": _extract_slots(q)}
    # llm 显式
    if re.search(r"(不联网|直接回答|纯对话|闲聊)", q):
        return {"target": "llm", "confidence": 0.98, "reasons": ["explicit:llm"], "slots": {}}
    return None

def route(q: str) -> dict:
    # 0) 显式工具指令优先
    exp = _detect_explicit(q)
    if exp:
        return exp

    # 1) 规则与信号（用于校验）
    web_signal = bool(
        re.search(TIME_WORDS, q) or
        re.search(WEATHER_HINT, q) or
        re.search(NEWS_PRICE, q)
    )
    math_signal = bool(
        re.search(r"(计算|求值|求最值|求和|方程|解方程|积分|微分|导数|极限|排列|组合|概率|方差|标准差|矩阵|行列式|特征值|阶乘)", q)
        or re.search(r"[\d\)\(\+\-\*/\^=]{3,}", q)
    )
    try:
        th = float(os.getenv("KB_ROUTE_THRESHOLD", "0.35"))
    except Exception:
        th = 0.35
    kb_ready, kb_score = _kb_probe(q)
    kb_hint = bool(re.search(KB_HINT, q))

    # 2) LLM 语义建议
    sug = _llm_route(q)

    # 3) 合并策略
    def by_rules():
        # 简短问候优先走 LLM，避免被 KB 误触发
        if _is_greet(q):
            return {"target": "llm", "confidence": 0.9, "reasons": ["greet"], "slots": {}}
        # 生成/解释类优先 LLM（无 web/math 信号时）
        if _prefer_llm(q):
            return {"target": "llm", "confidence": 0.85, "reasons": ["gen"], "slots": {}}
        if math_signal:
            return {"target": "f3", "confidence": 0.85, "reasons": ["math"], "slots": {}}
        if web_signal:
            rs = []
            if re.search(TIME_WORDS, q): rs.append("time")
            if re.search(WEATHER_HINT, q): rs.append("weather")
            if re.search(NEWS_PRICE, q): rs.append("news/price")
            return {"target": "f1", "confidence": 0.85, "reasons": rs, "slots": _extract_slots(q)}
        if kb_ready and kb_score >= th:
            return {"target": "f2", "confidence": 0.85, "reasons": [f"kb:score={kb_score:.3f}"], "slots": {}}
        if kb_hint:
            return {"target": "f2", "confidence": 0.75, "reasons": ["kb-hint"], "slots": {}}
        return {"target": "llm", "confidence": 0.7, "reasons": ["default-llm"], "slots": {}}

    if not sug:
        return by_rules()

    # 若 LLM 返回 multi，选择主段（最高置信度）作为当前工具，同时把完整 plan 附到决策里（保持向后兼容）
    plan: List[Dict[str, Any]] = []
    if sug.get("mode") == "multi" and isinstance(sug.get("segments"), list) and sug["segments"]:
        # 复制并按规则对每个子问做二次校验（例如生成/解释类优先 llm）
        plan_in = sug["segments"]  # type: ignore
        plan = []
        for seg in plan_in:
            q_seg = seg.get("q") or ""
            tool = str(seg.get("tool", "llm")).lower()
            reasons_seg = seg.get("reasons") if isinstance(seg.get("reasons"), list) else []
            if tool in ("f1", "f2") and _prefer_llm(str(q_seg)):
                tool = "llm"
                reasons_seg = reasons_seg + ["rule:prefer-llm-gen"]
            plan.append({
                "id": seg.get("id") or None,
                "q": q_seg,
                "tool": tool,
                "confidence": float(seg.get("confidence", 0.0)),
                "reasons": reasons_seg,
                "needs_context": bool(seg.get("needs_context", False)),
                "q_template": seg.get("q_template") or "",
            })
        primary = sorted(plan, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)[0]
        tgt = str(primary.get("tool", "llm"))
        conf = float(primary.get("confidence", 0.0))
        reasons = ["llm:multi-primary"] + (primary.get("reasons") or [])
    else:
        tgt = sug["target"]  # type: ignore
        conf = float(sug.get("confidence", 0.0))
        reasons = list(sug.get("reasons", []))

    strong_kb = (kb_ready and kb_score >= th)

    # 若 LLM 置信度低，直接用规则
    if conf < 0.6:
        r = by_rules()
        r["reasons"].append("rule:override-llm-lowconf")
        if plan:
            r["plan"] = plan
        return r

    # 一致性加权
    if tgt == "f1":
        if web_signal:
            out = {"target": "f1", "confidence": max(conf, 0.9), "reasons": reasons + ["rule:web"], "slots": _extract_slots(q)}
            if plan: out["plan"] = plan
            return out
        if math_signal:
            out = {"target": "f3", "confidence": 0.85, "reasons": reasons + ["rule:math"], "slots": {}}
            if plan: out["plan"] = plan
            return out
        if strong_kb or kb_hint:
            # KB 强信号与 LLM 冲突时，倾向 KB
            out = {"target": "f2", "confidence": 0.8, "reasons": reasons + ([f"kb:score={kb_score:.3f}"] if strong_kb else ["kb-hint"]), "slots": {}}
            if plan: out["plan"] = plan
            return out
        out = {"target": "f1", "confidence": conf, "reasons": reasons, "slots": _extract_slots(q)}
        if plan: out["plan"] = plan
        return out

    if tgt == "f2":
        if strong_kb:
            out = {"target": "f2", "confidence": max(conf, 0.9), "reasons": reasons + [f"kb:score={kb_score:.3f}"], "slots": {}}
            if plan: out["plan"] = plan
            return out
        if web_signal:
            out = {"target": "f1", "confidence": 0.85, "reasons": reasons + ["rule:web"], "slots": _extract_slots(q)}
            if plan: out["plan"] = plan
            return out
        if math_signal:
            out = {"target": "f3", "confidence": 0.85, "reasons": reasons + ["rule:math"], "slots": {}}
            if plan: out["plan"] = plan
            return out
        if kb_hint:
            out = {"target": "f2", "confidence": max(conf, 0.8), "reasons": reasons + ["kb-hint"], "slots": {}}
            if plan: out["plan"] = plan
            return out
        out = {"target": "f2", "confidence": conf, "reasons": reasons, "slots": {}}
        if plan: out["plan"] = plan
        return out

    if tgt == "f3":
        if math_signal:
            out = {"target": "f3", "confidence": max(conf, 0.9), "reasons": reasons + ["rule:math"], "slots": {}}
            if plan: out["plan"] = plan
            return out
        if web_signal:
            out = {"target": "f1", "confidence": 0.85, "reasons": reasons + ["rule:web"], "slots": _extract_slots(q)}
            if plan: out["plan"] = plan
            return out
        if strong_kb or kb_hint:
            out = {"target": "f2", "confidence": 0.8, "reasons": reasons + ([f"kb:score={kb_score:.3f}"] if strong_kb else ["kb-hint"]), "slots": {}}
            if plan: out["plan"] = plan
            return out
        out = {"target": "f3", "confidence": conf, "reasons": reasons, "slots": {}}
        if plan: out["plan"] = plan
        return out

    # tgt == llm
    # 问候或短句（且无强信号）不被 KB 覆盖
    if _is_greet(q) or (_is_short_query(q) and not web_signal and not math_signal and not kb_hint):
        out = {"target": "llm", "confidence": max(conf, 0.85), "reasons": reasons + (["greet"] if _is_greet(q) else ["short"]), "slots": {}}
        if plan: out["plan"] = plan
        return out
    if web_signal:
        out = {"target": "f1", "confidence": 0.85, "reasons": reasons + ["rule:web"], "slots": _extract_slots(q)}
        if plan: out["plan"] = plan
        return out
    if math_signal:
        out = {"target": "f3", "confidence": 0.85, "reasons": reasons + ["rule:math"], "slots": {}}
        if plan: out["plan"] = plan
        return out
    if strong_kb or kb_hint:
        out = {"target": "f2", "confidence": 0.8, "reasons": reasons + ([f"kb:score={kb_score:.3f}"] if strong_kb else ["kb-hint"]), "slots": {}}
        if plan: out["plan"] = plan
        return out
    out = {"target": "llm", "confidence": max(conf, 0.7), "reasons": reasons, "slots": {}}
    if plan: out["plan"] = plan
    return out
