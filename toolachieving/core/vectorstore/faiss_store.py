# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, faiss, numpy as np
from typing import List, Dict, Any
from pathlib import Path
from core.embeddings.silicon import embed_texts

class KBIndex:
    def __init__(self, kb_dir: str):
        self.kb_dir = Path(kb_dir)
        self.index_path = self.kb_dir / "index.faiss"
        self.meta_path  = self.kb_dir / "meta.jsonl"
        self.dim = None
        self.index = None
        self.meta: List[Dict[str, Any]] = []

    # —— 构建 / 重建 ——
    def build_from_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 64):

        # 确保索引目录存在（包含多级目录）
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        # 保险：若 index_path 带了子目录，额外确保其父目录存在
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self.kb_dir.mkdir(parents=True, exist_ok=True)
        vecs, metas = [], []
        buf = []

        def flush():
            nonlocal vecs, metas, buf
            if not buf: return
            emb = embed_texts([x["text"] for x in buf])
            vecs.extend(emb)
            metas.extend([x["meta"] | {"text": x["text"]} for x in buf])
            buf = []

        for ch in chunks:
            buf.append(ch)
            if len(buf) >= batch_size:
                flush()
        flush()
        if not vecs:
            raise RuntimeError("no chunks to index")

        arr = np.array(vecs, dtype="float32")
        self.dim = arr.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)  # 余弦相似：先归一化
        faiss.normalize_L2(arr)
        self.index.add(arr)
        self.meta = metas

        # 使用序列化规避 Windows 上非 ASCII 路径的 fopen 限制
        data = faiss.serialize_index(self.index)
        with open(self.index_path, "wb") as f:
            f.write(bytes(data))
        with self.meta_path.open("w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # —— 加载 ——
    def load(self):
        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError("index not built")
        # 使用反序列化以支持 Unicode 路径
        with open(self.index_path, "rb") as f:
            buf = f.read()
        arr = np.frombuffer(buf, dtype='uint8')
        self.index = faiss.deserialize_index(arr)
        with self.meta_path.open("r", encoding="utf-8") as f:
            self.meta = [json.loads(line) for line in f]
        self.dim = self.index.d

    # —— 查询 ——
    def query(self, q: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None: self.load()
        qv = np.array(embed_texts([q])[0], dtype="float32")[None, :]
        faiss.normalize_L2(qv)
        D, I = self.index.search(qv, top_k)
        out = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0: continue
            m = self.meta[idx]
            out.append({"score": float(score), "text": m["text"], "source": m["source"], "type": m["type"]})
        return out
