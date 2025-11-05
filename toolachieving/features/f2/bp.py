# -*- coding: utf-8 -*-
from __future__ import annotations
from flask import Blueprint, request, jsonify
from .service import rebuild_index, run as f2_run, kb_status
from pathlib import Path
import os, time

bp = Blueprint("f2", __name__, url_prefix="/v1/f2")

@bp.route("/reindex", methods=["POST"])
def reindex():
    kb_id = (request.json or {}).get("kb_id", "default")
    return rebuild_index(kb_id)

@bp.route("/query", methods=["POST"])
def query():
    payload = request.json or {}
    return f2_run(payload)

@bp.route("/status", methods=["GET"])  # /v1/f2/status?kb_id=default
def status():
    kb_id = request.args.get("kb_id") or "default"
    return kb_status(kb_id)

def _kb_raw_dir(kb_id: str) -> Path:
    from .service import _kb_paths
    raw_dir, _ = _kb_paths(kb_id)
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir

def _safe_join(base: Path, name: str) -> Path:
    # 简易安全拼接，避免越权
    p = (base / name).resolve()
    if str(p).startswith(str(base.resolve())):
        return p
    raise ValueError("illegal path")

@bp.route("/docs", methods=["GET"])  # 列表
def list_docs():
    kb_id = request.args.get("kb_id") or "default"
    raw_dir = _kb_raw_dir(kb_id)
    items = []
    for p in raw_dir.rglob("*"):
        if not p.is_file():
            continue
        st = p.stat()
        items.append({
            "path": str(p.relative_to(raw_dir)).replace("\\", "/"),
            "size": st.st_size,
            "mtime": int(st.st_mtime),
            "mtime_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
        })
    return jsonify({"kb_id": kb_id, "root": str(raw_dir), "items": items})

@bp.route("/docs", methods=["DELETE"])  # 删除
def delete_doc():
    data = request.get_json(force=True, silent=True) or {}
    kb_id = data.get("kb_id") or request.args.get("kb_id") or "default"
    rel = data.get("path") or request.args.get("path")
    if not rel:
        return jsonify({"error": "missing path"}), 400
    raw_dir = _kb_raw_dir(kb_id)
    try:
        target = _safe_join(raw_dir, rel)
        if target.exists() and target.is_file():
            target.unlink()
            return jsonify({"kb_id": kb_id, "deleted": rel})
        return jsonify({"kb_id": kb_id, "deleted": False, "error": "not found"}), 404
    except Exception as e:
        return jsonify({"kb_id": kb_id, "deleted": False, "error": str(e)}), 400

@bp.route("/upload", methods=["POST"])  # 上传（multipart）
def upload():
    kb_id = request.args.get("kb_id") or (request.form.get("kb_id") if request.form else None) or "default"
    raw_dir = _kb_raw_dir(kb_id)
    files = []
    # 支持 files 或 file
    upfiles = request.files.getlist("files") or ([request.files.get("file")] if request.files.get("file") else [])
    if not upfiles:
        return jsonify({"error": "missing files"}), 400
    saved = []
    for f in upfiles:
        if not f or not f.filename:
            continue
        # 简易清理文件名
        name = os.path.basename(f.filename).replace("..", "_")
        dst = raw_dir / name
        f.save(str(dst))
        saved.append(name)
    return jsonify({"kb_id": kb_id, "saved": saved, "root": str(raw_dir)})
