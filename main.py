from agent.plan_execute import PlanAndExecuteAgent

if __name__ == "__main__":
    agent = PlanAndExecuteAgent()
    answer = agent.run("给出20的阶乘")
    print("\n✅ 最终答案：")
    print(answer)
