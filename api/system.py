from datetime import datetime

from flask import Blueprint, jsonify, render_template


system_bp = Blueprint("system", __name__)


@system_bp.get("/")
def root():
    return jsonify({"status": "ok"})


@system_bp.get("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat()})


@system_bp.get("/ui")
def ui():
    return render_template("ui.html")
