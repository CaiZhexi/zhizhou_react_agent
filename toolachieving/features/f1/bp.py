# -*- coding: utf-8 -*-
from flask import Blueprint, request, jsonify
from .service import run as f1_run

bp = Blueprint("f1", __name__, url_prefix="/v1/f1")

@bp.post("/run")
def run():
    data = request.get_json(force=True) or {}
    # 兼容前端传 question
    q = (data.get("q") or data.get("question") or "").strip()
    if not q:
        return jsonify({"error": "missing q"}), 400
    out = f1_run({"q": q, "k": int(data.get("k", 5)), "slots": data.get("slots", {})})
    return jsonify(out)
