import os

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
SILICONFLOW_MODEL = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2-7B-Instruct")
SILICONFLOW_EMBEDDING_MODEL = os.getenv("SILICONFLOW_EMBEDDING_MODEL", "BAAI/bge-m3")

METASO_API_KEY = os.getenv("METASO_API_KEY")

# ==================== 知识库 (RAG) 配置 ====================
KB_STORAGE_DIR = os.getenv("KB_STORAGE_DIR", "data/kb")
KB_DB_PATH = os.getenv("KB_DB_PATH", os.path.join(KB_STORAGE_DIR, "metadata.db"))
KB_DEFAULT_TOP_K = int(os.getenv("KB_DEFAULT_TOP_K", "5"))
KB_CHUNK_SIZE = int(os.getenv("KB_CHUNK_SIZE", "800"))
KB_CHUNK_OVERLAP = int(os.getenv("KB_CHUNK_OVERLAP", "100"))
KB_EMBED_BATCH_SIZE = int(os.getenv("KB_EMBED_BATCH_SIZE", "64"))
KB_MAX_CONTEXT_CHARS = int(os.getenv("KB_MAX_CONTEXT_CHARS", "4000"))

# ==================== 文件上传配置 ====================
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(20 * 1024 * 1024)))
ALLOWED_UPLOAD_EXTENSIONS = {"txt", "pdf", "docx", "xlsx", "md"}
UPLOAD_STORAGE_DIR = os.getenv("UPLOAD_STORAGE_DIR", "data/uploads")

# ==================== Python 执行器安全配置 ====================
PYTHON_EXECUTOR_TYPE = os.getenv("PYTHON_EXECUTOR_TYPE", "default")
PYTHON_EXECUTOR_TIMEOUT = float(os.getenv("PYTHON_EXECUTOR_TIMEOUT", "10"))
PYTHON_EXECUTOR_MAX_OUTPUT = int(os.getenv("PYTHON_EXECUTOR_MAX_OUTPUT", "5000"))
PYTHON_EXECUTOR_MAX_CODE_LENGTH = int(os.getenv("PYTHON_EXECUTOR_MAX_CODE_LENGTH", "10000"))
PYTHON_EXECUTOR_MAX_AST_NODES = int(os.getenv("PYTHON_EXECUTOR_MAX_AST_NODES", "2000"))
PYTHON_EXECUTOR_RECURSION_LIMIT = int(os.getenv("PYTHON_EXECUTOR_RECURSION_LIMIT", "1000"))
PYTHON_EXECUTOR_SANITIZE_ENV = os.getenv("PYTHON_EXECUTOR_SANITIZE_ENV", "true").lower() == "true"
PYTHON_EXECUTOR_ENABLE_AUDIT = os.getenv("PYTHON_EXECUTOR_ENABLE_AUDIT", "true").lower() == "true"
PYTHON_EXECUTOR_AUDIT_LOG_PATH = os.getenv("PYTHON_EXECUTOR_AUDIT_LOG_PATH", "logs/executor_audit.log")
PYTHON_EXECUTOR_FAILURE_LOG_PATH = os.getenv("PYTHON_EXECUTOR_FAILURE_LOG_PATH", "logs/executor_failures.log")
PYTHON_EXECUTOR_AUDIT_MAX_CHARS = int(os.getenv("PYTHON_EXECUTOR_AUDIT_MAX_CHARS", "2000"))
PYTHON_EXECUTOR_AUDIT_LOG_CODE = os.getenv("PYTHON_EXECUTOR_AUDIT_LOG_CODE", "true").lower() == "true"

PYTHON_ALLOWED_MODULES = {
    "math": ["*"],
    "statistics": ["*"],
    "decimal": ["*"],
    "fractions": ["*"],
    "random": ["*"],
    "datetime": ["*"],
    "collections": ["*"],
    "itertools": ["*"],
    "re": ["*"],
    "json": ["*"],
}

PYTHON_ALLOWED_BUILTINS = [
    "abs",
    "round",
    "sum",
    "min",
    "max",
    "pow",
    "int",
    "float",
    "str",
    "bool",
    "len",
    "range",
    "enumerate",
    "zip",
    "map",
    "filter",
    "list",
    "dict",
    "set",
    "tuple",
    "print",
    "format",
    "type",
    "isinstance",
]
