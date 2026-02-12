from agent.planner import Planner
from agent.react_runtime import ReActStepExecutor
from agent.reflection import StepReflector
from tools.registry import call_tool, TOOLS
from utils.parser import parse_inline_tool_call

class PlanAndExecuteAgent:
    def __init__(self):
        self.planner = Planner()
        self.executor = ReActStepExecutor()
        self.reflector = StepReflector()

    def run(self, user_question: str, *, return_trace: bool = False):
        plan = self.planner.make_plan(user_question)
        past_steps = []
        trace_steps = []

        print("==== PLAN ====")
        for i, s in enumerate(plan, 1):
            print(f"{i}. {s}")

        print("\n==== EXECUTE ====")

        for idx, step in enumerate(plan, 1):
            max_retry = 3
            attempt = 0
            step_instruction = step
            step_trace = {"step": step, "attempts": [], "final_ok": False, "final_result": None}

            while attempt < max_retry:
                print(f"\n[STEP {idx} | Attempt {attempt+1}] {step_instruction}")

                tool_name, tool_args = (None, None)
                if attempt == 0:
                    tool_name, tool_args = parse_inline_tool_call(step_instruction)

                if tool_name and tool_name in TOOLS and isinstance(tool_args, dict):
                    result = call_tool(
                        tool_name,
                        tool_args,
                        run_id=self.executor.run_id,
                        step_id=f"step{idx}",
                        attempt=attempt+1,
                    )
                else:
                    result = self.executor.run_step(
                        f"请完成以下步骤：{step_instruction}\n"
                        f"完成后用 Final Answer 输出结果。",
                        step_id=f"step{idx}",
                        attempt=attempt+1
                    )
                print("Result:", result)

                reflection = self.reflector.reflect(step, result)

                print("Reflection:", reflection)
                step_trace["attempts"].append(
                    {
                        "attempt": attempt + 1,
                        "instruction": step_instruction,
                        "result": result,
                        "reflection": reflection,
                    }
                )

                if reflection.get("ok") is True:
                    past_steps.append((step, result))
                    step_trace["final_ok"] = True
                    step_trace["final_result"] = result
                    break
                else:
                    attempt += 1
                    step_instruction = (
                        f"{step}\n"
                        f"请注意：上一次结果不可用，原因是：{reflection.get('reason')}。\n"
                        f"改进建议：{reflection.get('suggestion')}。"
                    )

            else:
                past_steps.append((step, "步骤多次失败，未获取可靠结果"))
                step_trace["final_ok"] = False
                step_trace["final_result"] = "步骤多次失败，未获取可靠结果"

            if return_trace:
                trace_steps.append(step_trace)

        print("\n==== FINALIZE ====")
        answer = self.planner.finalize(user_question, past_steps)
        if return_trace:
            return {
                "answer": answer,
                "plan": plan,
                "steps": trace_steps,
                "run_id": self.executor.run_id,
            }
        return answer
