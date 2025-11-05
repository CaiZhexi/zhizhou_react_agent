# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, json
from typing import Iterable, Dict, List
from pathlib import Path

from pypdf import PdfReader
import docx
import pandas as pd

# 简单中文/英文混合分段：约 300~500 字一块
def split_text(text: str, max_len: int = 400) -> List[str]:
    text = re.sub(r'\s+', ' ', text).strip()
    if not text: return []
    segs = re.split(r'(?<=[。！？!?；;])', text)  # 句末分段
    out, buf = [], ''
    for s in segs:
        if len(buf) + len(s) <= max_len:
            buf += s
        else:
            if buf: out.append(buf.strip())
            buf = s
    if buf: out.append(buf.strip())
    return out

def load_txt(path: Path) -> Iterable[Dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    for i, chunk in enumerate(split_text(text)):
        yield {"text": chunk, "meta": {"source": str(path), "type": "txt", "chunk": i}}

def load_pdf(path: Path) -> Iterable[Dict]:
    reader = PdfReader(str(path))
    for pi, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        for ci, chunk in enumerate(split_text(raw)):
            yield {"text": chunk, "meta": {"source": f"{path}#page={pi+1}", "type": "pdf", "chunk": ci}}

def load_docx(path: Path) -> Iterable[Dict]:
    doc = docx.Document(str(path))
    content = "\n".join(p.text for p in doc.paragraphs)
    for i, chunk in enumerate(split_text(content)):
        yield {"text": chunk, "meta": {"source": str(path), "type": "docx", "chunk": i}}

def load_xlsx(path: Path) -> Iterable[Dict]:
    # 逐表逐行拼成句子再切分
    xls = pd.ExcelFile(str(path))
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        # 合并每行
        for ridx, row in df.fillna("").astype(str).iterrows():
            line = "；".join(v for v in row.tolist() if v)
            for ci, chunk in enumerate(split_text(line, 300)):
                yield {"text": chunk, "meta": {"source": f"{path}[{sheet}]#{ridx}", "type": "xlsx", "chunk": ci}}

LOADERS = {
    ".txt": load_txt,
    ".md": load_txt,
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".xlsx": load_xlsx,
}

def walk_and_load(root: str) -> Iterable[Dict]:
    rootp = Path(root)
    for p in rootp.rglob("*"):
        if not p.is_file(): continue
        func = LOADERS.get(p.suffix.lower())
        if not func: continue
        yield from func(p)
