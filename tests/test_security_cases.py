import unittest

from tools.python_executor import PythonExecutor, PythonExecutorConfig
from prompts.planner_prompt import PLANNER_PROMPT
from tools.registry import call_tool
from utils.parser import parse_action


def _config_no_audit() -> PythonExecutorConfig:
    return PythonExecutorConfig(
        executor_type="default",
        timeout_sec=1.0,
        max_output_chars=100,
        max_code_length=200,
        max_ast_nodes=200,
        recursion_limit=1000,
        sanitize_env=False,
        enable_audit=False,
        audit_log_path="logs/executor_audit.log",
        failure_log_path="logs/executor_failures.log",
        audit_max_chars=2000,
        audit_log_code=False,
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


class TestSecurityCases(unittest.TestCase):
    def test_sensitive_prompt_rules_present(self):
        required_keywords = [
            "敏感话题管控规则",
            "政治敏感",
            "色情暴力",
            "宗教极端",
            "民族矛盾",
            "地域歧视",
            "网络暴力",
            "违法犯罪",
            "毒品枪支",
            "赌博诈骗",
            "个人隐私泄露",
        ]
        for kw in required_keywords:
            self.assertIn(kw, PLANNER_PROMPT)

    def test_sensitive_refusal_phrase_present(self):
        phrase = "你所请求的操作涉及Python代码安全风险管控范围"
        self.assertIn(phrase, PLANNER_PROMPT)

    def test_reject_import(self):
        ex = PythonExecutor(_config_no_audit())
        out = ex.execute("__import__('os')")
        self.assertTrue(out.startswith("error: unsafe expression"))

    def test_reject_open(self):
        ex = PythonExecutor(_config_no_audit())
        out = ex.execute("open('x.txt', 'w')")
        self.assertTrue(out.startswith("error: unsafe expression"))

    def test_reject_dunder_attribute(self):
        ex = PythonExecutor(_config_no_audit())
        out = ex.execute("math.__dict__")
        self.assertTrue(out.startswith("error: unsafe expression"))

    def test_reject_lambda(self):
        ex = PythonExecutor(_config_no_audit())
        out = ex.execute("(lambda x: x)(1)")
        self.assertTrue(out.startswith("error: unsafe expression"))

    def test_reject_comprehension(self):
        ex = PythonExecutor(_config_no_audit())
        out = ex.execute("[x for x in range(3)]")
        self.assertTrue(out.startswith("error: unsafe expression"))

    def test_reject_attribute_chain(self):
        ex = PythonExecutor(_config_no_audit())
        out = ex.execute("math.sin.__globals__")
        self.assertTrue(out.startswith("error: unsafe expression"))

    def test_code_length_limit(self):
        cfg = _config_no_audit()
        cfg = cfg.__class__(**{**cfg.__dict__, "max_code_length": 5})
        ex = PythonExecutor(cfg)
        out = ex.execute("123456")
        self.assertEqual(out, "error: expression too long")

    def test_ast_node_limit(self):
        cfg = _config_no_audit()
        cfg = cfg.__class__(**{**cfg.__dict__, "max_ast_nodes": 5})
        ex = PythonExecutor(cfg)
        out = ex.execute("1 + 2 + 3 + 4 + 5")
        self.assertTrue(out.startswith("error: unsafe expression"))

    def test_unknown_tool_not_executed(self):
        out = call_tool("rm", {}, run_id="test", step_id="s1", attempt=1)
        self.assertEqual(out, "Tool rm not found")

    def test_action_input_requires_json(self):
        text = "Action: python\nAction Input: expr=1+1"
        action, args = parse_action(text)
        self.assertIsNone(action)
        self.assertIsNone(args)

    def test_action_parse_ok(self):
        text = 'Action: python\\nAction Input: {"expr": "1+1"}'
        action, args = parse_action(text)
        self.assertEqual(action, "python")
        self.assertEqual(args, {"expr": "1+1"})


if __name__ == "__main__":
    unittest.main()
