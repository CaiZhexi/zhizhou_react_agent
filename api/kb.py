import json
import os
import tempfile
import uuid
from datetime import datetime
from typing import List

from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from config import (
    ALLOWED_UPLOAD_EXTENSIONS,
    KB_CHUNK_OVERLAP,
    KB_CHUNK_SIZE,
    KB_EMBED_BATCH_SIZE,
    KB_STORAGE_DIR,
    MAX_FILE_SIZE,
)
from llm.embeddings import SiliconFlowEmbeddings
from utils.kb_db import ensure_kb_db
from utils.kb_ingest import chunk_text, parse_file_to_text
from utils.kb_store import (
    ensure_kb_row,
    insert_chunks,
    insert_doc,
    upsert_kb_index,
)
from tools.kb_search import kb_search


kb_bp = Blueprint("kb", __name__)


def _allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    return bool(ext) and ext in ALLOWED_UPLOAD_EXTENSIONS


def _load_or_create_index(path: str, dim: int):
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError("faiss is required for kb upload but is not installed") from e
    if os.path.exists(path):
        try:
            return faiss.read_index(path)
        except Exception:
            # corrupted index; rebuild
            try:
                os.remove(path)
            except Exception:
                pass
            return faiss.IndexFlatIP(dim)
    return faiss.IndexFlatIP(dim)


def _normalize(vecs):
    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise RuntimeError("numpy is required for kb upload but is not installed") from e
    x = np.array(vecs, dtype="float32")
    norms = (x ** 2).sum(axis=1, keepdims=True) ** 0.5
    norms[norms == 0] = 1.0
    return x / norms


def _load_metadata_list(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        items = []
        for k in sorted(data.keys(), key=lambda x: int(x)):
            items.append(data[k])
        return items
    return []


def _save_metadata_list(path: str, items: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {str(i): item for i, item in enumerate(items)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@kb_bp.post("/kb/upload")
def upload_kb_file():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400

    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "empty filename"}), 400

    if request.content_length and request.content_length > MAX_FILE_SIZE:
        return jsonify({"error": "file too large"}), 413

    if not _allowed_file(f.filename):
        return jsonify({"error": "type not allowed"}), 400

    kb_id = request.form.get("kb_id", "default").strip() or "default"
    doc_name = request.form.get("doc_name", "").strip()
    original_name = f.filename or ""
    safe_name = secure_filename(original_name)
    if not doc_name:
        doc_name = original_name if original_name else safe_name
    ext = os.path.splitext(original_name)[1].lower().lstrip(".")
    if not ext:
        ext = os.path.splitext(safe_name)[1].lower().lstrip(".")
    if not ext:
        return jsonify({"error": "invalid filename"}), 400

    ensure_kb_db()
    ensure_kb_row(kb_id, name=kb_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = os.path.join(tmpdir, f"upload.{ext}")
        f.save(temp_path)

        text = parse_file_to_text(temp_path, ext)
        if not text.strip():
            return jsonify({"error": "no extractable text"}), 400

        chunks = chunk_text(text, KB_CHUNK_SIZE, KB_CHUNK_OVERLAP)
        if not chunks:
            return jsonify({"error": "no chunks created"}), 400

        embedder = SiliconFlowEmbeddings()
        vectors = []
        for i in range(0, len(chunks), KB_EMBED_BATCH_SIZE):
            batch = chunks[i : i + KB_EMBED_BATCH_SIZE]
            vectors.extend(embedder.embed(batch))

    if not vectors:
        return jsonify({"error": "embedding failed"}), 500

    vecs = _normalize(vectors)

    kb_dir = os.path.join(KB_STORAGE_DIR, kb_id)
    index_path = os.path.join(kb_dir, "faiss", "index.bin")
    meta_path = os.path.join(kb_dir, "chunks", "metadata.json")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    index = _load_or_create_index(index_path, vecs.shape[1])
    start_idx = index.ntotal
    index.add(vecs)
    try:
        import faiss  # type: ignore
    except Exception:
        faiss = None
    if faiss is not None:
        faiss.write_index(index, index_path)

    metadata = _load_metadata_list(meta_path)
    if len(metadata) != start_idx:
        # mismatch: fallback to rebuild metadata list length
        metadata = metadata[: start_idx]

    doc_id = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    insert_doc(
        {
            "doc_id": doc_id,
            "kb_id": kb_id,
            "doc_name": doc_name,
            "source_type": "file",
            "source_uri": "",
            "sha256": "",
            "size": 0,
            "status": "ready",
            "version": 1,
            "created_at": now,
            "updated_at": now,
        }
    )

    chunk_rows = []
    for i, content in enumerate(chunks):
        vector_id = start_idx + i
        chunk_id = uuid.uuid4().hex
        metadata.append(
            {
                "chunk_id": chunk_id,
                "content": content,
                "doc_name": doc_name,
            }
        )
        chunk_rows.append(
            {
                "chunk_id": chunk_id,
                "kb_id": kb_id,
                "doc_id": doc_id,
                "content": content,
                "start_offset": None,
                "end_offset": None,
                "token_count": None,
                "vector_id": vector_id,
                "metadata_json": "",
                "created_at": now,
            }
        )

    _save_metadata_list(meta_path, metadata)
    insert_chunks(chunk_rows)
    upsert_kb_index(
        {
            "kb_id": kb_id,
            "faiss_path": index_path,
            "dim": int(vecs.shape[1]),
            "metric": "ip",
            "updated_at": now,
        }
    )

    return jsonify(
        {
            "kb_id": kb_id,
            "doc_id": doc_id,
            "doc_name": doc_name,
            "chunks": len(chunks),
            "index_path": index_path,
            "metadata_path": meta_path,
        }
    )


@kb_bp.post("/kb/search")
def kb_search_route():
    payload = request.get_json(silent=True) or {}
    query = (payload.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query required"}), 400
    kb_id = (payload.get("kb_id") or "default").strip() or "default"
    top_k = int(payload.get("top_k") or 5)
    result = kb_search(query=query, kb_id=kb_id, top_k=top_k)
    try:
        return jsonify(json.loads(result))
    except Exception:
        return jsonify({"error": result}), 500
