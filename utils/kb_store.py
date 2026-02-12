import sqlite3
from typing import Dict, Iterable

from config import KB_DB_PATH
from utils.kb_db import ensure_kb_db


def _conn():
    ensure_kb_db()
    return sqlite3.connect(KB_DB_PATH)


def ensure_kb_row(kb_id: str, name: str) -> None:
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT kb_id FROM kb WHERE kb_id = ?", (kb_id,))
        row = cur.fetchone()
        if not row:
            cur.execute(
                "INSERT INTO kb (kb_id, name, description, owner_user_id, created_at, updated_at) "
                "VALUES (?, ?, '', '', datetime('now'), datetime('now'))",
                (kb_id, name),
            )
        conn.commit()
    finally:
        conn.close()


def insert_doc(record: Dict) -> None:
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO doc (doc_id, kb_id, doc_name, source_type, source_uri, sha256, size, status, version, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.get("doc_id"),
                record.get("kb_id"),
                record.get("doc_name"),
                record.get("source_type"),
                record.get("source_uri"),
                record.get("sha256"),
                record.get("size"),
                record.get("status"),
                record.get("version"),
                record.get("created_at"),
                record.get("updated_at"),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def insert_chunks(records: Iterable[Dict]) -> None:
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.executemany(
            "INSERT INTO chunk (chunk_id, kb_id, doc_id, content, start_offset, end_offset, token_count, vector_id, metadata_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    r.get("chunk_id"),
                    r.get("kb_id"),
                    r.get("doc_id"),
                    r.get("content"),
                    r.get("start_offset"),
                    r.get("end_offset"),
                    r.get("token_count"),
                    r.get("vector_id"),
                    r.get("metadata_json"),
                    r.get("created_at"),
                )
                for r in records
            ],
        )
        conn.commit()
    finally:
        conn.close()


def upsert_kb_index(record: Dict) -> None:
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT kb_id FROM kb_index WHERE kb_id = ?", (record.get("kb_id"),))
        row = cur.fetchone()
        if row:
            cur.execute(
                "UPDATE kb_index SET faiss_path=?, dim=?, metric=?, updated_at=? WHERE kb_id=?",
                (
                    record.get("faiss_path"),
                    record.get("dim"),
                    record.get("metric"),
                    record.get("updated_at"),
                    record.get("kb_id"),
                ),
            )
        else:
            cur.execute(
                "INSERT INTO kb_index (kb_id, faiss_path, dim, metric, updated_at) VALUES (?, ?, ?, ?, ?)",
                (
                    record.get("kb_id"),
                    record.get("faiss_path"),
                    record.get("dim"),
                    record.get("metric"),
                    record.get("updated_at"),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def insert_upload_record(record: Dict) -> None:
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO upload (upload_id, filename, stored_path, sha256, size, content_type, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                record.get("upload_id"),
                record.get("filename"),
                record.get("stored_path"),
                record.get("sha256"),
                record.get("size"),
                record.get("content_type"),
                record.get("created_at"),
            ),
        )
        conn.commit()
    finally:
        conn.close()
