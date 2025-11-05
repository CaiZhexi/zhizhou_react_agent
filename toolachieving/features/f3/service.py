# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
import io, sys, math, random, datetime, calendar

try:
    import numpy as np  # noqa: F401
except Exception:  # pragma: no cover
    np = None  # type: ignore

# 可选：用于代码生成的 LLM（硅基流动）
try:
    from core.providers.llm_silicon import chat, LLMConfigError, LLMHTTPError  # type: ignore
    _LLM_AVAILABLE = True
except Exception:  # pragma: no cover
    chat = None  # type: ignore
    LLMConfigError = Exception  # type: ignore
    LLMHTTPError = Exception  # type: ignore
    _LLM_AVAILABLE = False


def _safe_print(*args, **kwargs):
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    try:
        sys.stdout.write(sep.join(str(x) for x in args) + end)
    except Exception:
        pass

SAFE_BUILTINS = {
    # 仅暴露常用且相对安全的内置函数
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "range": range,
    "sorted": sorted,
    "print": _safe_print,
    # 常见类型与工具
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "zip": zip,
    "enumerate": enumerate,
    "pow": pow,
    "any": any,
    "all": all,
    "map": map,
    "filter": filter,
    "reversed": reversed,
    "divmod": divmod,
}

# 受限全局：注入 math 及其成员到全局，支持直接调用 sin/cos 等；可用时注入 np 标识。
SAFE_GLOBALS = {
    "__builtins__": {},  # 禁止默认内置（不允许 import/open 等）
    "math": math,
}

# 将 math.* 常用函数/常量直接暴露为全局名（sin/ cos / pi ...）
for _name in dir(math):
    if _name.startswith("_"):  # 跳过私有
        continue
    try:
        SAFE_GLOBALS[_name] = getattr(math, _name)
    except Exception:
        pass
if np is not None:
    SAFE_GLOBALS["np"] = np
SAFE_GLOBALS["random"] = random
SAFE_GLOBALS["datetime"] = datetime
SAFE_GLOBALS["calendar"] = calendar


# —— 安全策略 ——
SAFE_POLICY_MSG = (
    "出于安全策略，拒绝执行可能读写文件、网络访问、系统命令、进程/线程控制、"
    "动态代码执行（eval/exec/compile）或尝试获取底层对象/魔法属性的代码。"
)

FORBIDDEN_CODE_TOKENS = [
    # 文件/系统/进程/线程/网络/序列化/动态执行等敏感能力
    "open(", "os.", "sys.", "subprocess", "shutil", "socket", "requests", "http.client",
    "urllib", "pathlib", "pickle", "marshal", "ctypes", "multiprocessing", "threading",
    "signal", "resource", "psutil", "paramiko", "pexpect", "pty", "ptyprocess",
    "system(", "popen(", "spawn(", "kill(", "remove(", "unlink(", "rmdir(", "rmtree(",
    "chdir(", "chmod(", "chown(", "rename(", "mkdir(", "makedirs(", "getenv(", "putenv(",
    "environ", "sys.path", "sys.modules", "sys.argv", "sys.exit", "importlib",
    # 动态执行/反射/逃逸
    "__import__", "eval(", "exec(", "compile(", "globals(", "locals(",
    "__class__", "__mro__", "__subclasses__(", "__getattribute__", "__reduce__",
    "__reduce_ex__", "__globals__", "__code__", "__closure__", "__dict__", "__get__",
]

FORBIDDEN_QUERY_HINTS = [
    # 中文危险意图关键词（非穷尽）
    "删除文件", "写文件", "覆盖系统", "修改系统", "注册表", "运行系统命令", "执行命令",
    "rm -rf", "del /f /q", "关机", "重启", "结束进程", "扫描端口", "网络扫描", "注入",
    "提权", "清空", "格式化", "下载并执行", "读取敏感文件", "读取系统文件",
]

def _is_unsafe_query(q: str) -> bool:
    qs = (q or "").lower()
    return any(h.lower() in qs for h in FORBIDDEN_QUERY_HINTS)

def _is_unsafe_code(code: str) -> bool:
    s = (code or "").lower()
    return any(tok.lower() in s for tok in FORBIDDEN_CODE_TOKENS)


def _gen_code_from_expr(expr: str) -> str:
    # 简单清洗：替换 ^ 为 **；允许常见函数名；去除非法字符
    e = expr.strip()
    e = e.replace("^", "**")
    # 生成代码：把表达式求值到 ans，并打印
    code = (
        "import math\n"
        + ("import numpy as np\n" if np is not None else "")
        + f"ans = ({e})\n"
        + "print(ans)\n"
    )
    return code


def _extract_expr(q: str) -> str:
    # 朴素提取：尝试从问题中抓取数学表达式；若失败就直接用整句尝试（可能触发语法错误并在外层捕获）
    # 提取中文关键词后的内容
    import re
    m = re.search(r"(计算|求值|求|结果|等于|=)[:：]?\s*(.+)$", q)
    if m:
        return m.group(2)
    # 否则保守返回原句（由外层捕获语法错误）
    return q


def _strip_code_fence(txt: str) -> str:
    s = txt.strip()
    # 提取 ```python ... ``` 或 ``` ... ``` 之间的内容
    if "```" in s:
        parts = s.split("```")
        if len(parts) >= 3:
            inner = parts[1]
            # 可能以 python 开头
            if inner.lstrip().lower().startswith("python"):
                inner = inner.split("\n", 1)[1] if "\n" in inner else ""
            return inner.strip()
    return s


def _gen_code_via_llm(q: str) -> str:
    """调用 LLM 生成安全的 Python 代码（将结果赋值给 ans 并 print(ans)）。"""
    if not _LLM_AVAILABLE:
        raise LLMConfigError("LLM not available for codegen")
    sys_prompt = (
        "你是 Python 数学助手。只输出可直接运行的 Python 代码，不要解释。"
        "要求：1) 仅使用 math（可选 numpy，作为 np）、datetime、calendar；"
        "2) 不要进行文件/网络/系统操作；3) 最终将结果赋值给 ans，并 print(ans)；"
        "4) 不要写 import 语句，直接使用 math/np/datetime/calendar。"
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"根据问题生成求解代码：{q}"},
    ]
    data = chat(messages)  # type: ignore
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    code = _strip_code_fence(content)
    if "ans" not in code:
        # 兜底：若模型未包含 ans，附加一行（不覆盖现有变量）
        code = code.rstrip() + "\n\n# ensure result variable\ntry:\n    ans\nexcept NameError:\n    pass\n"
    return code


def _sanitize_code(code: str) -> str:
    """移除 import/from 语句，避免 __import__ 依赖；我们已在全局注入 math/np。"""
    cleaned: list[str] = []
    for line in code.splitlines():
        s = line.strip()
        if s.startswith("import ") or s.startswith("from "):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _prepend_helpers_if_needed(q: str, code: str) -> str:
    """针对常见数学任务缺失的辅助函数进行注入（例如 is_prime）。"""
    q_l = q.lower()
    need_prime = ("质数" in q) or ("prime" in q_l)
    has_prime_def = ("def is_prime" in code)
    if need_prime and not has_prime_def:
        helper = (
            "def is_prime(n):\n"
            "    if n < 2: return False\n"
            "    if n % 2 == 0: return n == 2\n"
            "    i = 3\n"
            "    while i * i <= n:\n"
            "        if n % i == 0: return False\n"
            "        i += 2\n"
            "    return True\n\n"
        )
        return helper + code
    return code


def _exec_code(code: str) -> Dict[str, Any]:
    # 在受限环境中执行代码，捕获 stdout，返回 ans 值和输出
    stdout = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = stdout
        # 关键：使用同一个命名空间作为 globals/locals，避免推导式作用域找不到标识符
        env: Dict[str, Any] = dict(SAFE_GLOBALS)
        env.update(SAFE_BUILTINS)
        exec(code, env, env)
        out = stdout.getvalue()
        ans = env.get("ans")
        return {"stdout": out, "ans": ans}
    finally:
        sys.stdout = old_stdout


def run(payload: Dict) -> Dict:
    """
    f3（Python 工具求解）：
    - 输入：{ q?, code? }
    - 若提供 code，则按受限环境执行；否则从 q 提取表达式并生成代码执行
    - 输出：{ feature, code, stdout, result, error? }
    """
    q = (payload.get("q") or "").strip()
    code = payload.get("code") or ""
    use_codegen = bool(payload.get("codegen", True))  # 默认开启 LLM 代码生成（若可用）

    if not q and not code:
        return {"feature": "f3", "items": [], "error": "missing q or code"}

    # —— 预安全检查：若疑似危险意图或代码，直接拒绝 ——
    if _is_unsafe_query(q) or _is_unsafe_code(code):
        return {"feature": "f3", "items": [], "error": f"f3 blocked: {SAFE_POLICY_MSG}"}

    try:
        if code:
            use_code = code
            codegen = "manual"
        else:
            if use_codegen:
                try:
                    use_code = _gen_code_via_llm(q)
                    codegen = "llm"
                except (LLMConfigError, LLMHTTPError, Exception):  # 回退到规则生成
                    use_code = _gen_code_from_expr(_extract_expr(q))
                    codegen = "rule"
            else:
                use_code = _gen_code_from_expr(_extract_expr(q))
                codegen = "rule"
        # 执行前再次检查（包含 LLM 生成的代码）
        if _is_unsafe_code(use_code):
            return {"feature": "f3", "code": "", "items": [], "error": f"f3 blocked: {SAFE_POLICY_MSG}"}
        use_code = _sanitize_code(use_code)
        use_code = _prepend_helpers_if_needed(q, use_code)
        try:
            res = _exec_code(use_code)
        except Exception as e:
            msg = str(e)
            # 针对未定义 is_prime 的回退：注入 helper 后重试
            if "is_prime" in msg:
                use_code = _prepend_helpers_if_needed(q + " 质数", use_code)
                res = _exec_code(use_code)
            else:
                raise
        ans = res.get("ans")
        items = []
        if ans is not None:
            items = [{"title": "Python 计算结果", "url": "", "snippet": str(ans)}]
        else:
            so = (res.get("stdout") or "").strip()
            if so:
                items = [{"title": "Python 输出", "url": "", "snippet": so[:500]}]
        return {
            "feature": "f3",
            "code": use_code,
            "codegen": codegen,
            "stdout": res.get("stdout", ""),
            "result": ans if ans is not None else "",
            "items": items,
        }
    except Exception as e:
        # 返回已生成的 use_code 便于前端查看问题
        try:
            uc = use_code  # type: ignore
        except Exception:
            uc = code or ""
        return {"feature": "f3", "code": uc, "stdout": "", "result": "", "items": [], "error": f"f3 error: {e}"}
