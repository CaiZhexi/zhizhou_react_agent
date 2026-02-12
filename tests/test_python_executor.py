import json
import os
import tempfile
import unittest

from tools.python_executor import PythonExecutor, PythonExecutorConfig


def _make_config(tmp_dir: str) -> PythonExecutorConfig:
    return PythonExecutorConfig(
        executor_type="default",
        timeout_sec=1.0,
        max_output_chars=50,
        max_code_length=200,
        max_ast_nodes=200,
        recursion_limit=1000,
        sanitize_env=False,
        enable_audit=True,
        audit_log_path=os.path.join(tmp_dir, "executor_audit.log"),
        failure_log_path=os.path.join(tmp_dir, "executor_failures.log"),
        audit_max_chars=2000,
        audit_log_code=True,
        allowed_modules={"math": ["*"]},
        allowed_builtins=[
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
            "list",
            "dict",
            "set",
            "tuple",
            "print",
        ],
    )


def _read_jsonl(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TestPythonExecutor(unittest.TestCase):
    def test_basic_math(self):
        with tempfile.TemporaryDirectory() as tmp:
            ex = PythonExecutor(_make_config(tmp))
            out = ex.execute("1 + 2 * 3")
            self.assertEqual(out, "7")

    def test_module_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            ex = PythonExecutor(_make_config(tmp))
            out = ex.execute("math.sqrt(9)")
            self.assertEqual(out, "3.0")

    def test_disallowed_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            ex = PythonExecutor(_make_config(tmp))
            out = ex.execute("__import__('os')")
            self.assertTrue(out.startswith("error: unsafe expression"))

    def test_disallowed_attribute(self):
        with tempfile.TemporaryDirectory() as tmp:
            ex = PythonExecutor(_make_config(tmp))
            out = ex.execute("(1).__class__")
            self.assertTrue(out.startswith("error: unsafe expression"))

    def test_output_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_config(tmp)
            cfg = cfg.__class__(**{**cfg.__dict__, "max_output_chars": 10})
            ex = PythonExecutor(cfg)
            out = ex.execute("'a' * 200")
            self.assertIn("truncated", out)

    def test_audit_logs_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            ex = PythonExecutor(_make_config(tmp))
            ex.execute("1 + 1")
            ex.execute("__import__('os')")

            audit = _read_jsonl(os.path.join(tmp, "executor_audit.log"))
            failure = _read_jsonl(os.path.join(tmp, "executor_failures.log"))

            self.assertGreaterEqual(len(audit), 2)
            self.assertGreaterEqual(len(failure), 1)

            last = audit[-1]
            self.assertIn("code_hash", last)
            self.assertIn("code_len", last)
            self.assertIn("executor_type", last)
            self.assertIn("ok", last)


if __name__ == "__main__":
    unittest.main()
