import io
import os
import sys
import tempfile
import types
import unittest

import app as app_module
import config as config_module
import tools.kb_search as kb_search_module
import utils.kb_db as kb_db_module
import utils.kb_store as kb_store_module
import api.kb as kb_api_module
from llm import embeddings as embeddings_module


class _FakeIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = []

    @property
    def ntotal(self) -> int:
        return len(self.vectors)

    def add(self, vecs):
        self.vectors.extend(list(vecs))

    def search(self, x, top_k: int):
        n = min(top_k, len(self.vectors))
        indices = list(range(n))
        distances = [1.0 - i * 0.01 for i in range(n)]
        return [distances], [indices]


def _install_fake_faiss():
    mod = types.ModuleType("faiss")
    store = {}

    def IndexFlatIP(dim):
        return _FakeIndex(dim)

    def write_index(index, path):
        store[path] = index
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"FAKE")

    def read_index(path):
        return store.get(path) or _FakeIndex(0)

    mod.IndexFlatIP = IndexFlatIP
    mod.write_index = write_index
    mod.read_index = read_index
    sys.modules["faiss"] = mod
    return mod


class TestKBRoutes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.kb_dir = os.path.join(self.tmp.name, "kb")
        self.upload_dir = os.path.join(self.tmp.name, "uploads")
        self.db_path = os.path.join(self.kb_dir, "metadata.db")

        # patch config + module constants
        self._old = {
            "KB_STORAGE_DIR": config_module.KB_STORAGE_DIR,
            "KB_DB_PATH": config_module.KB_DB_PATH,
            "KB_DEFAULT_TOP_K": config_module.KB_DEFAULT_TOP_K,
            "MAX_FILE_SIZE": config_module.MAX_FILE_SIZE,
            "UPLOAD_STORAGE_DIR": config_module.UPLOAD_STORAGE_DIR,
        }
        config_module.KB_STORAGE_DIR = self.kb_dir
        config_module.KB_DB_PATH = self.db_path
        config_module.KB_DEFAULT_TOP_K = 3
        config_module.MAX_FILE_SIZE = 1024 * 1024
        config_module.UPLOAD_STORAGE_DIR = self.upload_dir

        # patch imports that captured old values
        kb_db_module.KB_STORAGE_DIR = self.kb_dir
        kb_db_module.KB_DB_PATH = self.db_path
        kb_store_module.KB_DB_PATH = self.db_path
        kb_search_module.KB_STORAGE_DIR = self.kb_dir

        kb_api_module.KB_STORAGE_DIR = self.kb_dir
        kb_api_module.KB_CHUNK_SIZE = config_module.KB_CHUNK_SIZE
        kb_api_module.KB_CHUNK_OVERLAP = config_module.KB_CHUNK_OVERLAP
        kb_api_module.KB_EMBED_BATCH_SIZE = config_module.KB_EMBED_BATCH_SIZE
        kb_api_module.MAX_FILE_SIZE = config_module.MAX_FILE_SIZE
        kb_api_module.ALLOWED_UPLOAD_EXTENSIONS = config_module.ALLOWED_UPLOAD_EXTENSIONS

        app_module.MAX_FILE_SIZE = config_module.MAX_FILE_SIZE

        # fake embeddings
        self._orig_embed = embeddings_module.SiliconFlowEmbeddings.embed

        def _fake_embed(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

        embeddings_module.SiliconFlowEmbeddings.embed = _fake_embed

        # fake faiss
        self._fake_faiss = _install_fake_faiss()

        self.app = app_module.create_app()
        self.client = self.app.test_client()

    def tearDown(self):
        embeddings_module.SiliconFlowEmbeddings.embed = self._orig_embed
        for k, v in self._old.items():
            setattr(config_module, k, v)
        if "faiss" in sys.modules:
            del sys.modules["faiss"]
        self.tmp.cleanup()

    def test_kb_upload_and_search(self):
        data = {
            "file": (io.BytesIO("AI产品经理岗位调研内容".encode("utf-8")), "test.md"),
            "kb_id": "default",
        }
        resp = self.client.post("/kb/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json()
        self.assertEqual(payload["kb_id"], "default")
        self.assertGreater(payload["chunks"], 0)

        resp2 = self.client.post("/kb/search", json={"query": "产品经理", "kb_id": "default", "top_k": 2})
        self.assertEqual(resp2.status_code, 200)
        result = resp2.get_json()
        self.assertIn("results", result)
        self.assertGreaterEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["doc_name"], "test.md")

    def test_kb_upload_rejects_type(self):
        data = {
            "file": (io.BytesIO(b"hello"), "bad.exe"),
        }
        resp = self.client.post("/kb/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)

    def test_kb_search_requires_query(self):
        resp = self.client.post("/kb/search", json={"kb_id": "default"})
        self.assertEqual(resp.status_code, 400)

    def test_kb_upload_empty_text(self):
        data = {
            "file": (io.BytesIO(b""), "empty.txt"),
        }
        resp = self.client.post("/kb/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()
