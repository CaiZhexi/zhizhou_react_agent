from flask import Blueprint, jsonify, request

from agent.plan_execute import PlanAndExecuteAgent


agent_bp = Blueprint("agent", __name__)


@agent_bp.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question required"}), 400

    kb_id = (payload.get("kb_id") or "default").strip() or "default"
    if kb_id:
        question = (
            f"{question}\n\n"
            f"如问题明显与知识库相关，优先使用 kb_search（kb_id={kb_id}）；"
            "若为常识或简单对话，可直接回答。"
        )

    agent = PlanAndExecuteAgent()
    try:
        want_trace = bool(payload.get("trace"))
        if want_trace:
            result = agent.run(question, return_trace=True)
            result["kb_id"] = kb_id
            return jsonify(result)
        answer = agent.run(question)
    except Exception as e:
        return jsonify({"error": f"agent failed: {e}"}), 500

    return jsonify({"answer": answer, "kb_id": kb_id})
