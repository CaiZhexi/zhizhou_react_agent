import ast
import contextlib
import hashlib
import importlib
import io
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from utils.audit import log_executor_event

# ---------------------------
# Defaults (override in config.py)
# ---------------------------
DEFAULT_EXECUTOR_TYPE = "default"  # or "process_isolated"
DEFAULT_TIMEOUT_SEC = 10.0
DEFAULT_MAX_OUTPUT_CHARS = 5000
DEFAULT_MAX_CODE_LENGTH = 10000
DEFAULT_MAX_AST_NODES = 2000
DEFAULT_RECURSION_LIMIT = 1000
DEFAULT_SANITIZE_ENV = True
DEFAULT_ENABLE_AUDIT = True
DEFAULT_AUDIT_LOG_PATH = "logs/executor_audit.log"
DEFAULT_FAILURE_LOG_PATH = "logs/executor_failures.log"
DEFAULT_AUDIT_MAX_CHARS = 2000
DEFAULT_AUDIT_LOG_CODE = True

DEFAULT_ALLOWED_MODULES: Dict[str, List[str]] = {
    "math": ["*"],
    "statistics": ["*"],
    "decimal": ["*"],
    "fractions": ["*"],
    "random": ["*"],
    "datetime": ["*"],
    "collections": ["*"],
    "itertools": ["*"],
    "re": ["*"],
    "json": ["*"],
}

DEFAULT_ALLOWED_BUILTINS = [
    "abs",
    "round",
    "sum",
    "min",
    "max",
    "pow",
    "int",
    "float",
    "str",
    "bool",
    "len",
    "range",
    "enumerate",
    "zip",
    "map",
    "filter",
    "list",
    "dict",
    "set",
    "tuple",
    "print",
    "format",
    "type",
    "isinstance",
]

SENSITIVE_ENV_TOKENS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "ACCESS", "API")


class UnsafeExpressionError(ValueError):
    pass


class LimitedStringIO(io.StringIO):
    def __init__(self, max_chars: int):
        super().__init__()
        self._max = max_chars
        self._written = 0
        self._truncated = False

    @property
    def truncated(self) -> bool:
        return self._truncated

    def write(self, s: str) -> int:
        if not s:
            return 0
        remain = self._max - self._written
        if remain <= 0:
            self._truncated = True
            return 0
        if len(s) > remain:
            super().write(s[:remain])
            self._written += remain
            self._truncated = True
            return remain
        super().write(s)
        self._written += len(s)
        return len(s)


@dataclass(frozen=True)
class PythonExecutorConfig:
    executor_type: str
    timeout_sec: float
    max_output_chars: int
    max_code_length: int
    max_ast_nodes: int
    recursion_limit: int
    sanitize_env: bool
    enable_audit: bool
    audit_log_path: str
    failure_log_path: str
    audit_max_chars: int
    audit_log_code: bool
    allowed_modules: Dict[str, List[str]]
    allowed_builtins: List[str]

    @staticmethod
    def from_app_config() -> "PythonExecutorConfig":
        try:
            import config as app_config
        except Exception:
            app_config = None

        def get(name: str, default: Any) -> Any:
            return getattr(app_config, name, default) if app_config else default

        return PythonExecutorConfig(
            executor_type=str(get("PYTHON_EXECUTOR_TYPE", DEFAULT_EXECUTOR_TYPE)),
            timeout_sec=float(get("PYTHON_EXECUTOR_TIMEOUT", DEFAULT_TIMEOUT_SEC)),
            max_output_chars=int(get("PYTHON_EXECUTOR_MAX_OUTPUT", DEFAULT_MAX_OUTPUT_CHARS)),
            max_code_length=int(get("PYTHON_EXECUTOR_MAX_CODE_LENGTH", DEFAULT_MAX_CODE_LENGTH)),
            max_ast_nodes=int(get("PYTHON_EXECUTOR_MAX_AST_NODES", DEFAULT_MAX_AST_NODES)),
            recursion_limit=int(get("PYTHON_EXECUTOR_RECURSION_LIMIT", DEFAULT_RECURSION_LIMIT)),
            sanitize_env=bool(get("PYTHON_EXECUTOR_SANITIZE_ENV", DEFAULT_SANITIZE_ENV)),
            enable_audit=bool(get("PYTHON_EXECUTOR_ENABLE_AUDIT", DEFAULT_ENABLE_AUDIT)),
            audit_log_path=str(get("PYTHON_EXECUTOR_AUDIT_LOG_PATH", DEFAULT_AUDIT_LOG_PATH)),
            failure_log_path=str(get("PYTHON_EXECUTOR_FAILURE_LOG_PATH", DEFAULT_FAILURE_LOG_PATH)),
            audit_max_chars=int(get("PYTHON_EXECUTOR_AUDIT_MAX_CHARS", DEFAULT_AUDIT_MAX_CHARS)),
            audit_log_code=bool(get("PYTHON_EXECUTOR_AUDIT_LOG_CODE", DEFAULT_AUDIT_LOG_CODE)),
            allowed_modules=dict(get("PYTHON_ALLOWED_MODULES", DEFAULT_ALLOWED_MODULES)),
            allowed_builtins=list(get("PYTHON_ALLOWED_BUILTINS", DEFAULT_ALLOWED_BUILTINS)),
        )


ALLOWED_EXPR_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.IfExp,
    ast.Dict,
    ast.Set,
    ast.List,
    ast.Tuple,
    ast.Subscript,
    ast.Slice,
    ast.Compare,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Attribute,
    ast.keyword,
    # operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.FloorDiv,
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.IsNot,
    ast.In,
    ast.NotIn,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
    ast.LShift,
    ast.RShift,
)


class _AstValidator(ast.NodeVisitor):
    def __init__(self, allowed_names: Iterable[str], allowed_modules: Dict[str, List[str]], max_nodes: int):
        self.allowed_names = set(allowed_names)
        self.allowed_modules = allowed_modules
        self.max_nodes = max_nodes
        self._node_count = 0
        self.errors: List[str] = []

    def _touch(self, node: ast.AST) -> bool:
        self._node_count += 1
        if self._node_count > self.max_nodes:
            self.errors.append("AST too large")
            return False
        if not isinstance(node, ALLOWED_EXPR_NODES):
            self.errors.append(f"Disallowed node: {type(node).__name__}")
            return False
        return True

    def generic_visit(self, node: ast.AST) -> None:
        if not self._touch(node):
            return
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if not self._touch(node):
            return
        if node.id.startswith("_"):
            self.errors.append(f"Disallowed name: {node.id}")
            return
        if node.id not in self.allowed_names:
            self.errors.append(f"Unknown name: {node.id}")
            return

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if not self._touch(node):
            return
        if node.attr.startswith("_"):
            self.errors.append(f"Disallowed attribute: {node.attr}")
            return
        if isinstance(node.value, ast.Name):
            base = node.value.id
            if base not in self.allowed_modules:
                self.errors.append(f"Attribute access not allowed: {base}.{node.attr}")
                return
            allowed_attrs = self.allowed_modules.get(base, [])
            if "*" not in allowed_attrs and node.attr not in allowed_attrs:
                self.errors.append(f"Attribute not allowed: {base}.{node.attr}")
                return
        else:
            self.errors.append("Attribute access only allowed on module names")
            return
        self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        if not self._touch(node):
            return
        if isinstance(node.func, ast.Name):
            if node.func.id not in self.allowed_names:
                self.errors.append(f"Call not allowed: {node.func.id}")
                return
        elif isinstance(node.func, ast.Attribute):
            # Attribute validation handled in visit_Attribute
            if not isinstance(node.func.value, ast.Name):
                self.errors.append("Call target must be module attribute")
                return
        else:
            self.errors.append("Call target not allowed")
            return
        super().generic_visit(node)


def _sanitize_env() -> None:
    for key in list(os.environ.keys()):
        upper = key.upper()
        if any(token in upper for token in SENSITIVE_ENV_TOKENS):
            os.environ.pop(key, None)


def _build_safe_globals(config: PythonExecutorConfig) -> Dict[str, Any]:
    builtins_obj = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
    safe_builtins: Dict[str, Any] = {}
    for name in config.allowed_builtins:
        if name in builtins_obj:
            safe_builtins[name] = builtins_obj[name]

    safe_globals: Dict[str, Any] = {"__builtins__": safe_builtins}

    for mod_name in config.allowed_modules.keys():
        try:
            safe_globals[mod_name] = importlib.import_module(mod_name)
        except Exception:
            continue

    return safe_globals


def _validate_expression(code: str, config: PythonExecutorConfig) -> ast.Expression:
    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError as e:
        raise UnsafeExpressionError(f"syntax error: {e.msg}") from e

    allowed_names = set(config.allowed_builtins) | set(config.allowed_modules.keys())
    validator = _AstValidator(allowed_names, config.allowed_modules, config.max_ast_nodes)
    validator.visit(tree)
    if validator.errors:
        raise UnsafeExpressionError("; ".join(validator.errors))
    return tree


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"...(truncated, total={len(text)} chars)"


def _evaluate_expression(code: str, config: PythonExecutorConfig) -> str:
    if not code.strip():
        return "error: empty expression"
    if len(code) > config.max_code_length:
        return "error: expression too long"

    tree = _validate_expression(code, config)
    safe_globals = _build_safe_globals(config)

    buf = LimitedStringIO(config.max_output_chars)
    result: Any = None
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        result = eval(compile(tree, "<python>", "eval"), safe_globals, {})

    output = buf.getvalue()
    if buf.truncated:
        output = output.rstrip() + "\n...(output truncated)"

    result_str = "" if result is None else str(result)

    if output and result_str:
        combined = output.rstrip() + "\n" + result_str
    elif output:
        combined = output.rstrip()
    else:
        combined = result_str if result_str else "None"

    return _truncate(combined, config.max_output_chars)


def _worker_execute(code: str, cfg: Dict[str, Any], out_queue: "mp.Queue[Dict[str, Any]]") -> None:
    config = PythonExecutorConfig(
        executor_type=cfg["executor_type"],
        timeout_sec=cfg["timeout_sec"],
        max_output_chars=cfg["max_output_chars"],
        max_code_length=cfg["max_code_length"],
        max_ast_nodes=cfg["max_ast_nodes"],
        recursion_limit=cfg["recursion_limit"],
        sanitize_env=cfg["sanitize_env"],
        allowed_modules=cfg["allowed_modules"],
        allowed_builtins=cfg["allowed_builtins"],
    )

    if config.sanitize_env:
        _sanitize_env()

    if config.recursion_limit > 0:
        sys.setrecursionlimit(config.recursion_limit)

    try:
        result = _evaluate_expression(code, config)
        out_queue.put({"ok": True, "result": result})
    except Exception as e:
        out_queue.put({"ok": False, "error": str(e)})


class PythonExecutor:
    def __init__(self, config: Optional[PythonExecutorConfig] = None):
        self.config = config or PythonExecutorConfig.from_app_config()

    def execute(self, code: str) -> str:
        start = time.time()
        ok, result, error, error_type = self._execute_with_status(code)
        elapsed_ms = int((time.time() - start) * 1000)
        if self.config.enable_audit:
            try:
                self._audit(code, ok, result, error, error_type, elapsed_ms)
            except Exception:
                pass
        return result

    def _execute_with_status(self, code: str) -> Tuple[bool, str, Optional[str], Optional[str]]:
        if self.config.executor_type == "process_isolated":
            return self._execute_isolated(code)
        return self._execute_in_process(code)

    def _execute_in_process(self, code: str) -> Tuple[bool, str, Optional[str], Optional[str]]:
        try:
            return True, _evaluate_expression(code, self.config), None, None
        except UnsafeExpressionError as e:
            return False, f"error: unsafe expression ({e})", str(e), "unsafe_expression"
        except Exception as e:
            return False, f"error: {e}", str(e), "exception"

    def _execute_isolated(self, code: str) -> Tuple[bool, str, Optional[str], Optional[str]]:
        ctx = mp.get_context("spawn")
        queue: "mp.Queue[Dict[str, Any]]" = ctx.Queue()
        cfg = {
            "executor_type": self.config.executor_type,
            "timeout_sec": self.config.timeout_sec,
            "max_output_chars": self.config.max_output_chars,
            "max_code_length": self.config.max_code_length,
            "max_ast_nodes": self.config.max_ast_nodes,
            "recursion_limit": self.config.recursion_limit,
            "sanitize_env": self.config.sanitize_env,
            "allowed_modules": self.config.allowed_modules,
            "allowed_builtins": self.config.allowed_builtins,
        }
        proc = ctx.Process(target=_worker_execute, args=(code, cfg, queue))
        proc.start()
        proc.join(self.config.timeout_sec)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            return False, "error: timeout", "timeout", "timeout"

        if queue.empty():
            return False, "error: executor failed", "executor failed", "executor_failed"
        payload = queue.get()
        if payload.get("ok"):
            return True, payload.get("result", ""), None, None
        err = payload.get("error", "unknown error")
        return False, f"error: {err}", err, "worker_error"

    def _audit(
        self,
        code: str,
        ok: bool,
        result: str,
        error: Optional[str],
        error_type: Optional[str],
        elapsed_ms: int,
    ) -> None:
        code_hash = hashlib.sha256(code.encode("utf-8", errors="ignore")).hexdigest()
        record: Dict[str, Any] = {
            "ok": ok,
            "executor_type": self.config.executor_type,
            "elapsed_ms": elapsed_ms,
            "error_type": error_type,
            "error": error,
            "code_len": len(code),
            "code_hash": code_hash,
            "result_preview": result,
        }
        if self.config.audit_log_code:
            record["code"] = code
        log_executor_event(
            record,
            audit_path=self.config.audit_log_path,
            failure_path=self.config.failure_log_path,
            max_chars=self.config.audit_max_chars,
        )
