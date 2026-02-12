import hashlib
import os
import uuid
from datetime import datetime

from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from config import ALLOWED_UPLOAD_EXTENSIONS, MAX_FILE_SIZE, UPLOAD_STORAGE_DIR
from utils.kb_db import ensure_kb_db
from utils.kb_store import insert_upload_record


upload_bp = Blueprint("upload", __name__)


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_UPLOAD_EXTENSIONS


@upload_bp.post("/upload")
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400

    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "empty filename"}), 400

    if request.content_length and request.content_length > MAX_FILE_SIZE:
        return jsonify({"error": "file too large"}), 413

    if not _allowed_file(f.filename):
        return jsonify({"error": "type not allowed"}), 400

    os.makedirs(UPLOAD_STORAGE_DIR, exist_ok=True)
    original = secure_filename(f.filename)
    ext = original.rsplit(".", 1)[1].lower()
    stored_name = f"{uuid.uuid4().hex}.{ext}"
    stored_path = os.path.join(UPLOAD_STORAGE_DIR, stored_name)

    h = hashlib.sha256()
    size = 0
    with open(stored_path, "wb") as out:
        while True:
            chunk = f.stream.read(8192)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_FILE_SIZE:
                out.close()
                try:
                    os.remove(stored_path)
                except Exception:
                    pass
                return jsonify({"error": "file too large"}), 413
            h.update(chunk)
            out.write(chunk)

    ensure_kb_db()
    record = {
        "upload_id": uuid.uuid4().hex,
        "filename": original,
        "stored_path": stored_path,
        "sha256": h.hexdigest(),
        "size": size,
        "content_type": f.content_type,
        "created_at": datetime.utcnow().isoformat(),
    }
    insert_upload_record(record)

    return jsonify(
        {
            "upload_id": record["upload_id"],
            "filename": original,
            "stored_as": stored_name,
            "sha256": record["sha256"],
            "size": size,
        }
    )
