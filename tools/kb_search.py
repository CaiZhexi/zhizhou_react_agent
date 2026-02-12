import json
import os
from typing import Any, Dict, List

from config import KB_DEFAULT_TOP_K, KB_STORAGE_DIR
from llm.embeddings import SiliconFlowEmbeddings
from utils.kb_db import ensure_kb_db


def _load_metadata(path: str) -> Dict[int, Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"metadata not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {i: item for i, item in enumerate(data)}
    if isinstance(data, dict):
        parsed: Dict[int, Dict[str, Any]] = {}
        for k, v in data.items():
            try:
                idx = int(k)
            except Exception:
                continue
            parsed[idx] = v
        return parsed
    raise ValueError("metadata.json must be list or dict")


def _load_faiss_index(path: str):
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError("faiss is required for kb_search but is not installed") from e
    if not os.path.exists(path):
        raise FileNotFoundError(f"faiss index not found: {path}")
    return faiss.read_index(path)


def kb_search(query: str, kb_id: str = "default", top_k: int = KB_DEFAULT_TOP_K) -> str:
    ensure_kb_db()
    kb_dir = os.path.join(KB_STORAGE_DIR, kb_id)
    index_path = os.path.join(kb_dir, "faiss", "index.bin")
    meta_path = os.path.join(kb_dir, "chunks", "metadata.json")

    embedder = SiliconFlowEmbeddings()
    vecs = embedder.embed([query])
    if not vecs:
        return "error: empty embedding"

    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise RuntimeError("numpy is required for kb_search but is not installed") from e

    index = _load_faiss_index(index_path)
    metadata = _load_metadata(meta_path)

    x = np.array(vecs, dtype="float32")
    norms = (x ** 2).sum(axis=1, keepdims=True) ** 0.5
    norms[norms == 0] = 1.0
    x = x / norms
    distances, indices = index.search(x, int(top_k))

    results: List[Dict[str, Any]] = []
    for rank, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        item = metadata.get(int(idx))
        if not item:
            continue
        results.append(
            {
                "chunk_id": item.get("chunk_id"),
                "content": item.get("content"),
                "doc_name": item.get("doc_name"),
                "score": float(distances[0][rank]),
                "rank": rank + 1,
            }
        )

    return json.dumps({"kb_id": kb_id, "query": query, "results": results}, ensure_ascii=False)
