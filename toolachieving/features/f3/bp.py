# -*- coding: utf-8 -*-
from __future__ import annotations
from flask import Blueprint, request, jsonify
from .service import run as f3_run

bp = Blueprint("f3", __name__, url_prefix="/v1/f3")

@bp.post("/run")
def run():
    data = request.get_json(force=True, silent=True) or {}
    q = (data.get("q") or data.get("question") or "").strip()
    code = data.get("code") or ""
    if not q and not code:
        return jsonify({"error": "missing q or code"}), 400
    out = f3_run({"q": q, "code": code})
    return jsonify(out)

