# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from flask import redirect
from flask_cors import CORS
import json

from router.intent_router import route as route_intent
from core.registry import dispatch
from core.middleware import install_middlewares
from core.errors import install_error_handlers

from features.f1.bp import bp as f1_bp
from features.f2.bp import bp as f2_bp
from features.f3.bp import bp as f3_bp

app = Flask(__name__)
CORS(app)

install_middlewares(app)
install_error_handlers(app)

app.register_blueprint(f1_bp)
app.register_blueprint(f2_bp)
app.register_blueprint(f3_bp)

@app.get("/")
def index():
    # 提供一个简单前端页面用于测试
    return redirect("/static/index.html", code=302)

@app.get("/favicon.ico")
def favicon():
    # 避免 500 噪声日志：没有图标时返回空内容
    return ("", 204)

@app.get("/v1/health")
def health():
    return {"status": "ok", "version": "v1", "features": ["f1:websearch"]}

@app.post("/v1/answer")
def answer():
    data = request.get_json(force=True) or {}
    # 兼容前端传 question
    q = (data.get("q") or data.get("question") or "").strip()
    mode = (data.get("mode") or "auto").lower()
    k = int(data.get("k", 5))

    if not q:
        return jsonify({"error":"missing q"}), 400

    decision = {"target": mode, "confidence": 1.0, "reasons": [], "slots": {}}                if mode != "auto" else route_intent(q)

    target = decision["target"] if decision["target"] != "hybrid" else "f1"
    payload = {"q": q, "k": k, "slots": decision.get("slots", {}), **data}

    # Orchestrator: 若有多步 plan，按顺序执行，并将前步结果注入模板
    steps_out = []
    primary_items = []
    primary_tool = target
    try:
        plan = decision.get("plan") if isinstance(decision, dict) else None
        if isinstance(plan, list) and plan:
            # 选择主段：最高置信度
            try:
                primary = sorted(plan, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)[0]
                primary_tool = str(primary.get("tool", target))
            except Exception:
                primary_tool = target

            ctx = {}
            for step in plan:
                sid = step.get("id") or "s"
                tool = str(step.get("tool", "llm"))
                step_q = step.get("q") or q
                needs_ctx = bool(step.get("needs_context", False))
                q_tmpl = step.get("q_template") or ""
                if needs_ctx and q_tmpl:
                    # 简单模板替换：{sX.ans}
                    def _repl(m):
                        key = m.group(1)
                        if "." in key:
                            sid2, var = key.split(".", 1)
                            return str(ctx.get(sid2, {}).get(var, m.group(0)))
                        return m.group(0)
                    import re as _re
                    step_q = _re.sub(r"\{([^\}]+)\}", _repl, q_tmpl)
                elif needs_ctx and not q_tmpl:
                    # 尝试用 LLM 做 query rewrite（仅输出查询词）
                    try:
                        from core.providers.llm_silicon import chat as _llm_chat  # type: ignore
                        sys_prompt = (
                            "你是查询改写器。给定上下文变量与子问题，请输出一个可直接用于中文搜索的简洁查询词。\n"
                            "只输出查询词本身，不要解释、不要换行。"
                        )
                        ctx_json = json.dumps(ctx, ensure_ascii=False)
                        messages = [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": f"上下文: {ctx_json}\n子问题: {step_q}\n输出查询词："},
                        ]
                        _data = _llm_chat(messages)  # type: ignore
                        step_q = (_data.get("choices") or [{}])[0].get("message", {}).get("content", step_q).strip()
                    except Exception:
                        # 失败则退回原始子问题
                        step_q = step_q

                # 构造每步 payload（保守地沿用上层部分参数）
                step_payload = {"q": step_q}
                if tool == "f1":
                    step_payload.update({"k": k, "slots": payload.get("slots", {})})
                elif tool == "f2":
                    if "kb_id" in data: step_payload["kb_id"] = data["kb_id"]
                    if "top_k" in data: step_payload["top_k"] = int(data["top_k"])
                    if "gen" in data: step_payload["gen"] = bool(data["gen"])
                elif tool == "f3":
                    # 保持默认 codegen=True
                    if "code" in data: step_payload["code"] = data["code"]

                out = dispatch(tool, step_payload)
                # 记录上下文（供后续模板）
                ctx[sid] = {
                    "ans": out.get("result"),
                    "text": out.get("text"),
                }
                rec = {
                    "id": sid,
                    "tool": tool,
                    "input": {"q": step_q},
                    "output": out,
                }
                steps_out.append(rec)

            # 兼容老结构：把主段 items 也放到 results 下对应键
            try:
                for r in steps_out:
                    if r["tool"] == primary_tool:
                        primary_items = r.get("output", {}).get("items", []) or []
                        break
            except Exception:
                primary_items = []

            resp = {
                "query": q,
                "decision": decision,
                "results": {primary_tool: primary_items, "steps": steps_out}
            }
            if isinstance(out, dict) and out.get("error"):
                resp["error"] = out.get("error")
            return jsonify(resp)
    except Exception:
        # 若编排异常，回退单步执行
        pass

    # 单步执行（原逻辑）
    result = dispatch(target, payload)

    resp = {
        "query": q,
        "decision": decision,
        "results": {target: result.get("items", [])}
    }
    if isinstance(result, dict) and result.get("error"):
        resp["error"] = result.get("error")
    return jsonify(resp)

if __name__ == "__main__":
    app.run(port=8000, debug=True)
