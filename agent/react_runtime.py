import uuid
from llm.siliconflow import SiliconFlowLLM
from prompts.react_prompt import REACT_PROMPT
from tools.registry import call_tool
from utils.parser import parse_action, parse_final_answer

class ReActStepExecutor:
    def __init__(self, max_steps=8):
        self.llm = SiliconFlowLLM()
        self.max_steps = max_steps
        self.run_id = uuid.uuid4().hex[:8]  # 每次程序启动一个 run_id
        self.step_id = "step"
        self.attempt = 1

    def run_step(self, step_instruction: str, *, step_id: str = "step", attempt: int = 1) -> str:
        self.step_id = step_id
        self.attempt = attempt

        tools_desc = """
kb_search(query: str, kb_id: str) - 知识库检索（优先）
search(query: str) - 联网搜索
python(expr: str) - Python 计算（仅表达式）
"""
        system_prompt = REACT_PROMPT.format(tools=tools_desc)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": step_instruction},
        ]

        for i in range(self.max_steps):
            text = self.llm.chat(messages)

            final = parse_final_answer(text)
            if final:
                return final

            action, args = parse_action(text)
            if not action:
                return text.strip()

            obs = call_tool(
                action,
                args,
                run_id=self.run_id,
                step_id=self.step_id,
                attempt=self.attempt,
            )

            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Observation: {obs}"})

        return "error: step executor reached max_steps without final answer"
