import json
import re
import ast

def parse_inline_tool_call(text: str):
    """
    Parse inline tool calls like:
      python(1+1)
      python('math.sqrt(9)')
      python({"expr":"1+1"})
      search("北京天气")
      search({"query":"北京天气"})
    Returns (tool_name, args) or (None, None).
    """
    m = re.fullmatch(r"\s*(\w+)\s*\((.*)\)\s*", text, re.S)
    if not m:
        return None, None
    tool = m.group(1)
    raw = m.group(2).strip()
    if not raw:
        return tool, {}

    # JSON object form
    if raw.startswith("{"):
        try:
            return tool, json.loads(raw)
        except Exception:
            return None, None

    # Quoted string form
    if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
        try:
            val = ast.literal_eval(raw)
            if not isinstance(val, str):
                return None, None
            return tool, _infer_args(tool, val)
        except Exception:
            return None, None

    # Raw expression/string form
    return tool, _infer_args(tool, raw)


def _infer_args(tool: str, value: str) -> dict:
    if tool == "python":
        return {"expr": value}
    if tool == "search":
        return {"query": value}
    if tool == "kb_search":
        return {"query": value}
    return {"input": value}

def parse_action(text: str):
    action = re.search(r"Action:\s*(\w+)", text)
    action_input = re.search(r"Action Input:\s*(\{.*?\})", text, re.S)

    if not action or not action_input:
        return None, None

    return action.group(1), json.loads(action_input.group(1))

def parse_final_answer(text: str):
    match = re.search(r"Final Answer:\s*(.*)", text, re.S)
    return match.group(1).strip() if match else None
