# tools/registry.py
from typing import Any, Dict, Optional
from utils.trace import now_ms, log_tool_call

from tools.metaso_search import metaso_search
from tools.python_calc import python_calc
from tools.kb_search import kb_search

TOOLS = {
    "kb_search": kb_search,
    "search": metaso_search,
    "python": python_calc,
}

def call_tool(
    name: str,
    args: dict,
    *,
    run_id: str,
    step_id: str,
    attempt: int,
) -> str:
    if name not in TOOLS:
        return f"Tool {name} not found"

    t0 = now_ms()
    ok = True
    err: Optional[str] = None
    raw: Any = None

    try:
        raw = TOOLS[name](**args)  # ✅ 真正工具调用发生在这里
    except Exception as e:
        ok = False
        err = repr(e)
        raw = f"error: {err}"

    elapsed = now_ms() - t0
    trace_path = log_tool_call(
        run_id=run_id,
        step_id=step_id,
        attempt=attempt,
        tool_name=name,
        args=args,
        ok=ok,
        elapsed_ms=elapsed,
        raw_result=raw,   # ✅ 原始返回完整记录
        error=err,
    )

    # ✅ 控制台一眼可见到底有没有调用工具
    print(f"[TOOL_CALL] name={name} ok={ok} ms={elapsed} args={args} trace={trace_path}")

    # 返回给 LLM 的 observation 也要是字符串
    return raw if isinstance(raw, str) else str(raw)
