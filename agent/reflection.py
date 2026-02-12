from llm.siliconflow import SiliconFlowLLM
from prompts.reflection_prompt import REFLECTION_PROMPT
from utils.json_parser import extract_json_object

class StepReflector:
    def __init__(self):
        self.llm = SiliconFlowLLM()

    def reflect(self, step: str, result: str) -> dict:
        prompt = REFLECTION_PROMPT.format(step=step, result=result)
        messages = [
            {"role": "system", "content": "你只输出 JSON，不要输出其他文本。"},
            {"role": "user", "content": prompt},
        ]
        text = self.llm.chat(messages)
        return extract_json_object(text)
