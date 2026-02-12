class AgentState:
    def __init__(self, question: str):
        self.messages = [
            {"role": "user", "content": question}
        ]
        self.steps = 0
