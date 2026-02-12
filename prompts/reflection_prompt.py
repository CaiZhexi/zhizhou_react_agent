REFLECTION_PROMPT = """
你是一个严格的结果审查员。

当前步骤目标：
{step}

模型给出的结果：
{result}

请判断该结果是否【可用于继续下一步】。

判断标准：
- 是否包含明确、有用的信息（如数值、结论）
- 是否满足该步骤的目标
- 是否避免空泛、概括性描述

请用 JSON 格式回答：
{{"ok": true/false, "reason": "...", "suggestion": "..."}}

说明：
- ok=true 表示可以继续
- ok=false 表示需要优化并重试当前步骤
- suggestion 给出如何优化当前步骤的建议
"""
