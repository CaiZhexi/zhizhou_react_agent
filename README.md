# ReactAgent（本地版）

一个基于 ReAct 流程的本地 Agent 实验项目，包含：
- 工具调用与执行审计
- Python 安全执行器
- RAG 知识库（FAISS + SQLite）
- Flask API 与简易前端调试页

---

## 运行方式

### 1) CLI 模式（原始 Agent）
```bash
python main.py
```

### 2) HTTP 模式（Flask）
```bash
python app.py
```

前端调试页：
```
http://127.0.0.1:5000/ui
```

---

## 主要接口

**上传**
- `POST /upload` 通用文档上传
- `POST /kb/upload` 知识库文档上传（解析→向量→索引）

**检索**
- `POST /kb/search` 知识库检索

**对话**
- `POST /chat` 走完整 Planner+ReAct 决策链（支持 trace）

**健康检查**
- `GET /`、`GET /health`

---

## 功能特点
- 知识库优先策略为**软规则**：明显相关时才优先 `kb_search`，常识/简单对话可直接回答。
- Python 工具仅允许**表达式**（禁止 import/多语句）。
- KB 文档上传后会写入本地索引与 SQLite 元数据。

---

## 上传限制
- 允许类型：`txt / pdf / docx / xlsx / md`
- 单文件大小：20MB（可在 `config.py` 调整）

---

## 快速示例（PowerShell）

**KB 上传**
```powershell
curl.exe -F "file=@tests/华南师范大学.txt" -F "kb_id=default" http://127.0.0.1:5000/kb/upload
```

**KB 检索**
```powershell
$body = @{query="华南师范大学"; kb_id="default"; top_k=5} | ConvertTo-Json -Compress
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:5000/kb/search" -ContentType "application/json" -Body $body
```

**对话（含 trace）**
```powershell
$body = @{question="华南师范大学简介"; kb_id="default"; trace=$true} | ConvertTo-Json -Compress
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:5000/chat" -ContentType "application/json" -Body $body
```

---

## 目录结构

```
agent/          # 规划与执行器
api/            # Flask 蓝图
llm/            # 模型调用与 Embeddings
tools/          # 工具注册与实现
utils/          # 通用工具（db/解析/日志）
templates/      # 前端页面
tests/          # 测试
data/           # 本地知识库数据（默认忽略）
```

---

## 依赖

```bash
pip install -r requirements.txt
```

Windows 上 `faiss-cpu` 可能需要用 conda：
```bash
conda install -c conda-forge faiss-cpu
```

解析依赖：
- `pypdf`（PDF）
- `python-docx`（DOCX）
- `openpyxl`（XLSX）

---

## 环境变量（示例）

```
SILICONFLOW_API_KEY=your_key
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_MODEL=Qwen/Qwen2-7B-Instruct
SILICONFLOW_EMBEDDING_MODEL=BAAI/bge-m3
```

---

## 备注
- `main.py` 为 CLI 入口，`app.py` 为 Flask 服务入口，二者互不冲突。
- KB 默认存储：`data/kb/`，索引文件 `faiss/index.bin`。

---

## 测试
```bash
python -m unittest tests/test_kb_routes.py
```
