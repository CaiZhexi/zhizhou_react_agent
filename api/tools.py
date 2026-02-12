from flask import Blueprint, jsonify, request

from tools.registry import TOOLS


tools_bp = Blueprint("tools", __name__)


@tools_bp.get("/tools/list")
def list_tools():
    return jsonify({"tools": sorted(TOOLS.keys())})


@tools_bp.post("/tools/call")
def call_tool_route():
    payload = request.get_json(silent=True) or {}
    name = payload.get("name")
    return jsonify({"error": f"tool call not enabled in HTTP: {name}"}), 501
