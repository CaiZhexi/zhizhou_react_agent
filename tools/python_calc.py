# tools/python_calc.py
from tools.python_executor import PythonExecutor

_EXECUTOR = PythonExecutor()


def python_calc(expr: str) -> str:
    return _EXECUTOR.execute(expr)
