from llm.siliconflow import SiliconFlowLLM
from prompts.planner_prompt import PLANNER_PROMPT
from utils.json_parser import extract_json_object

class Planner:
    def __init__(self):
        self.llm = SiliconFlowLLM()

    def make_plan(self, user_question: str) -> list[str]:
        prompt = PLANNER_PROMPT.format(input=user_question)
        messages = [
            {"role": "system", "content": "你只输出 JSON，不要输出任何多余文本。"},
            {"role": "user", "content": prompt},
        ]
        text = self.llm.chat(messages)
        obj = extract_json_object(text)
        plan = obj.get("plan")
        if not isinstance(plan, list) or not all(isinstance(x, str) for x in plan):
            raise ValueError(f"Invalid plan JSON: {obj}")
        return plan

    def finalize(self, user_question: str, past_steps: list[tuple[str, str]]) -> str:
        """
        past_steps: [(step, result_text), ...]
        让模型基于已完成步骤输出最终答案（不再调用工具）。
        """
        steps_text = "\n".join([f"- {s}\n  结果: {r}" for s, r in past_steps])

        messages = [
            {"role": "system", "content": "你是一个严谨的助手。根据已完成步骤的结果，给出最终答案。不要再调用工具。"},
            {"role": "user", "content": f"用户问题：{user_question}\n\n已完成步骤：\n{steps_text}\n\n请输出最终答案（自然语言）。"},
        ]
        return self.llm.chat(messages).strip()
