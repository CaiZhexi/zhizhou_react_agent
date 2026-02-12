import json
import re

def extract_json_object(text: str) -> dict:
    """
    尽量从模型输出中提取第一个 JSON 对象。
    """
    # 先找最外层 {...}
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("No JSON object found in text.")
    raw = m.group(0)
    return json.loads(raw)
