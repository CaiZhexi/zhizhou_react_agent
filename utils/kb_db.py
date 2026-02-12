import os
import sqlite3
from typing import Optional

from config import KB_DB_PATH, KB_STORAGE_DIR


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS kb (
  kb_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  owner_user_id TEXT,
  created_at TEXT,
  updated_at TEXT
);

CREATE TABLE IF NOT EXISTS doc (
  doc_id TEXT PRIMARY KEY,
  kb_id TEXT NOT NULL,
  doc_name TEXT NOT NULL,
  source_type TEXT,
  source_uri TEXT,
  sha256 TEXT,
  size INTEGER,
  status TEXT,
  version INTEGER DEFAULT 1,
  created_at TEXT,
  updated_at TEXT,
  FOREIGN KEY (kb_id) REFERENCES kb(kb_id)
);

CREATE TABLE IF NOT EXISTS chunk (
  chunk_id TEXT PRIMARY KEY,
  kb_id TEXT NOT NULL,
  doc_id TEXT NOT NULL,
  content TEXT NOT NULL,
  start_offset INTEGER,
  end_offset INTEGER,
  token_count INTEGER,
  vector_id INTEGER,
  metadata_json TEXT,
  created_at TEXT,
  FOREIGN KEY (kb_id) REFERENCES kb(kb_id),
  FOREIGN KEY (doc_id) REFERENCES doc(doc_id)
);

CREATE TABLE IF NOT EXISTS kb_index (
  kb_id TEXT PRIMARY KEY,
  faiss_path TEXT NOT NULL,
  dim INTEGER,
  metric TEXT,
  updated_at TEXT,
  FOREIGN KEY (kb_id) REFERENCES kb(kb_id)
);

CREATE TABLE IF NOT EXISTS upload (
  upload_id TEXT PRIMARY KEY,
  filename TEXT NOT NULL,
  stored_path TEXT NOT NULL,
  sha256 TEXT,
  size INTEGER,
  content_type TEXT,
  created_at TEXT
);
"""


def ensure_kb_dirs() -> None:
    os.makedirs(KB_STORAGE_DIR, exist_ok=True)


def ensure_kb_db(path: Optional[str] = None) -> str:
    db_path = path or KB_DB_PATH
    ensure_kb_dirs()
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
    return db_path
