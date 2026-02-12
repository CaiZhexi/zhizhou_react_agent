# utils/trace.py
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

TRACE_DIR = os.getenv("TRACE_DIR", "traces")
RAW_MAX_CHARS = int(os.getenv("TRACE_RAW_MAX_CHARS", "20000"))  # 原始返回最多保存多少字符

def ensure_trace_dir() -> None:
    os.makedirs(TRACE_DIR, exist_ok=True)

def now_ms() -> int:
    return int(time.time() * 1000)

def utc_ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")

def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"...(truncated, total={len(s)} chars)"

def append_jsonl(run_id: str, record: Dict[str, Any]) -> str:
    """
    追加写入 traces/run_<run_id>.jsonl，返回文件路径
    """
    ensure_trace_dir()
    path = os.path.join(TRACE_DIR, f"run_{run_id}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path

def log_tool_call(
    *,
    run_id: str,
    step_id: str,
    attempt: int,
    tool_name: str,
    args: Dict[str, Any],
    ok: bool,
    elapsed_ms: int,
    raw_result: Any,
    error: Optional[str] = None,
) -> str:
    # raw_result 尽量序列化为字符串（保留“原始返回”）
    if isinstance(raw_result, str):
        raw_str = raw_result
    else:
        try:
            raw_str = json.dumps(raw_result, ensure_ascii=False)
        except Exception:
            raw_str = repr(raw_result)

    record = {
        "ts_utc": utc_ts(),
        "run_id": run_id,
        "step_id": step_id,
        "attempt": attempt,
        "tool": tool_name,
        "args": args,
        "ok": ok,
        "elapsed_ms": elapsed_ms,
        "error": error,
        "raw_result": _truncate(raw_str, RAW_MAX_CHARS),  # ✅ 这里就是你要的“原始返回”
    }
    return append_jsonl(run_id, record)
