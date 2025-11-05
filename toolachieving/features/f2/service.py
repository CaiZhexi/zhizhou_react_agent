# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List
from pathlib import Path
from core.vectorstore.faiss_store import KBIndex
from core.ingest.loader import walk_and_load
import os
import time

try:
    from core.providers.llm_silicon import simple_answer  # 复用你的 LLM
except Exception:
    def simple_answer(q: str, system_prompt: str = "") -> str:
        return "（LLM 未配置，返回占位回答）"

KB_ROOT = Path(os.getenv("KB_ROOT", "data/kb")).resolve()


def _kb_paths(kb_id: str):
    raw_dir = KB_ROOT / kb_id / "raw"
    idx_dir = KB_ROOT / kb_id / "index"
    return raw_dir, idx_dir

def rebuild_index(kb_id: str = "default") -> Dict:
    raw_dir, idx_dir = _kb_paths(kb_id)
    Path(idx_dir).mkdir(parents=True, exist_ok=True)  # 兜底

    if not raw_dir.exists():
        return {"kb_id": kb_id, "built": False, "error": f"raw dir not found: {raw_dir}"}
    chunks = list(walk_and_load(str(raw_dir)))
    if not chunks:
        return {"kb_id": kb_id, "built": False, "error": "no chunks in raw dir"}
    idx = KBIndex(str(idx_dir))
    try:
        idx.build_from_chunks(chunks)
    except Exception as e:
        return {
            "kb_id": kb_id,
            "built": False,
            "error": "faiss write index failed",
            "detail": str(e),
            "hint": "若为 Windows 非 ASCII 路径问题，请设置 $env:KB_ROOT 为纯英文路径并重试",
            "index_dir": str(idx_dir),
        }
    return {"kb_id": kb_id, "built": True, "chunks": len(chunks), "index_dir": str(idx_dir)}

def kb_status(kb_id: str = "default") -> Dict:
    """
    返回知识库索引状态：是否存在、chunk 数、来源文件数、索引维度、最后构建时间等。
    """
    raw_dir, idx_dir = _kb_paths(kb_id)
    meta_path = idx_dir / "meta.jsonl"
    index_path = idx_dir / "index.faiss"

    status: Dict = {
        "kb_id": kb_id,
        "raw_dir": str(raw_dir),
        "index_dir": str(idx_dir),
        "raw_exists": raw_dir.exists(),
        "index_exists": idx_dir.exists() and index_path.exists() and meta_path.exists(),
        "files": 0,
        "chunks": 0,
        "sources": 0,
        "dim": None,
        "last_built": None,
    }

    # 统计 raw 文件数
    if raw_dir.exists():
        try:
            status["files"] = sum(1 for p in raw_dir.rglob("*") if p.is_file())
        except Exception:
            pass

    # 统计 meta.jsonl 信息
    if meta_path.exists():
        try:
            sources = set()
            chunks = 0
            with meta_path.open("r", encoding="utf-8") as f:
                for line in f:
                    chunks += 1
                    try:
                        import json
                        m = json.loads(line)
                        src = m.get("source") or ""
                        if src:
                            sources.add(src)
                    except Exception:
                        pass
            status["chunks"] = chunks
            status["sources"] = len(sources)
            try:
                ts = meta_path.stat().st_mtime
                status["last_built"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
            except Exception:
                pass
        except Exception:
            pass

    # 读取索引维度
    try:
        idx = KBIndex(str(idx_dir))
        idx.load()
        status["dim"] = idx.dim
    except Exception:
        pass
    return status

def run(payload: Dict) -> Dict:
    """
    查询：输入 q，召回 top_k，并（可选）让 LLM 生成最终回答
    payload: { q, kb_id?, top_k?, gen? }
    """
    q = (payload.get("q") or "").strip()
    if not q:
        return {"feature":"f2","items": [], "error":"missing q"}

    kb_id  = payload.get("kb_id", "default")
    top_k  = int(payload.get("top_k", 5))
    gen    = bool(payload.get("gen", True))

    _, idx_dir = _kb_paths(kb_id)
    try:
        idx = KBIndex(str(idx_dir))
        idx.load()
    except Exception as e:
        return {"feature":"f2","items": [], "error": f"index not ready for kb={kb_id}: {e}"}

    hits = idx.query(q, top_k=top_k)
    items = [{"title": h["source"], "url": "", "snippet": h["text"], "score": h["score"]} for h in hits]

    answer = ""
    if gen and hits:
        ctx = "\n\n".join(f"【摘录{idx+1}】{h['text']}" for idx, h in enumerate(hits))
        prompt = f"基于以下资料回答：\n{ctx}\n\n问题：{q}\n请用中文给出简明、可信的答案。若资料不足，请明确说明。"
        try:
            answer = simple_answer(prompt)
        except Exception as e:
            answer = f"(生成失败，仅返回召回片段) {e}"

    return {
        "feature": "f2",
        "kb_id": kb_id,
        "items": items,
        **({"answer": answer} if answer else {})
    }
