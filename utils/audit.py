import json
import os
from datetime import datetime
from typing import Any, Dict

DEFAULT_AUDIT_LOG_PATH = os.getenv("PYTHON_EXECUTOR_AUDIT_LOG_PATH", "logs/executor_audit.log")
DEFAULT_FAILURE_LOG_PATH = os.getenv("PYTHON_EXECUTOR_FAILURE_LOG_PATH", "logs/executor_failures.log")


def _utc_ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _truncate(value: Any, max_chars: int) -> str:
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"...(truncated, total={len(s)} chars)"


def append_audit_log(record: Dict[str, Any], *, path: str, max_chars: int) -> None:
    record = dict(record)
    record["ts_utc"] = _utc_ts()
    if "code" in record:
        record["code"] = _truncate(record["code"], max_chars)
    if "result_preview" in record:
        record["result_preview"] = _truncate(record["result_preview"], max_chars)
    if "error" in record and record["error"] is not None:
        record["error"] = _truncate(record["error"], max_chars)

    _ensure_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_executor_event(
    record: Dict[str, Any],
    *,
    audit_path: str = DEFAULT_AUDIT_LOG_PATH,
    failure_path: str = DEFAULT_FAILURE_LOG_PATH,
    max_chars: int = 2000,
) -> None:
    append_audit_log(record, path=audit_path, max_chars=max_chars)
    if not record.get("ok", True):
        append_audit_log(record, path=failure_path, max_chars=max_chars)
