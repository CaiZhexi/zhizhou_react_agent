# 路由说明（中文）

本文件描述系统的 HTTP 路由规划，覆盖上传、知识库（RAG）与健康检查等模块。

## 基础
- 服务地址：`http://127.0.0.1:5000`
- 默认返回 JSON
- 文件上传使用 `multipart/form-data`
- 预留认证头：`Authorization`

## 健康检查
1. `GET /`
   - 作用：服务存活检查
   - 返回：`{"status":"ok"}`

2. `GET /health`
   - 作用：就绪/存活检查
   - 返回：`{"status":"ok","time":"ISO-8601"}`

3. `GET /ui`
   - 作用：简单前端调试页面
   - 返回：HTML 页面

## 通用文件上传
1. `POST /upload`
   - 作用：上传普通文件（非知识库）
   - 表单字段：`file`
   - 返回：`{ upload_id, filename, stored_as, sha256, size }`

2. `GET /upload/{upload_id}`
   - 作用：查询上传文件元信息
   - 返回：`{ upload_id, filename, sha256, size, content_type, created_at }`

3. `DELETE /upload/{upload_id}`
   - 作用：删除上传文件与记录
   - 返回：`{ ok: true }`

## 知识库（KB / RAG）
1. `POST /kb/upload`
   - 作用：上传知识库文件（解析 → 分块 → 向量 → 索引）
   - 表单字段：`file`，可选 `kb_id`、`doc_name`
   - 返回：`{ kb_id, doc_id, doc_name, chunks, index_path, metadata_path }`

2. `POST /kb/search`
   - 作用：知识库检索
   - Body：`{ query, kb_id?, top_k?, filters? }`
   - 返回：`{ kb_id, query, results: [{ chunk_id, content, doc_name, score, rank }] }`

3. `POST /kb`
   - 作用：创建知识库
   - Body：`{ kb_id, name, description?, owner_user_id? }`
   - 返回：`{ kb_id, name }`

4. `GET /kb`
   - 作用：列出知识库
   - 返回：`[{ kb_id, name, description, owner_user_id, updated_at }]`

5. `GET /kb/{kb_id}`
   - 作用：知识库详情
   - 返回：`{ kb_id, name, description, owner_user_id, stats }`

6. `GET /kb/{kb_id}/docs`
   - 作用：列出 KB 文档
   - 返回：`[{ doc_id, doc_name, status, version, created_at }]`

7. `DELETE /kb/{kb_id}`
   - 作用：删除 KB（包含索引）
   - 返回：`{ ok: true }`

8. `DELETE /kb/{kb_id}/docs/{doc_id}`
   - 作用：删除指定文档
   - 返回：`{ ok: true }`

9. `POST /kb/{kb_id}/rebuild`
   - 作用：全量重建索引
   - 返回：`{ ok: true, reindexed: true }`

10. `GET /kb/{kb_id}/stats`
    - 作用：KB 使用统计
    - 返回：`{ doc_count, chunk_count, index_dim, last_updated }`

## Agent（后续）
1. `POST /chat`
   - 作用：完整 Agent 流程（规划 → 工具 → 回答）
   - Body：`{ question, kb_id?, user_id? }`
   - 返回：`{ answer, citations?, tool_trace? }`

## 统一注意事项
- 所有写操作建议记录审计日志
- KB 相关优先 `kb_search`，无结果再降级 `search`
- 多租户时所有 KB 操作必须校验 `owner_user_id`

## API 蓝图与依赖关系
1. `api/system.py`
   - 路由：`/`、`/health`
   - 依赖：无（仅基础健康检查）

2. `api/upload.py`
   - 路由：`/upload`
   - 依赖：`utils/kb_db.py`、`utils/kb_store.py`（写入 upload 表）

3. `api/kb.py`
   - 路由：`/kb/upload`、`/kb/search`
   - 依赖：
     - `utils/kb_ingest.py`（解析/分块）
     - `llm/embeddings.py`（向量化）
     - `utils/kb_store.py`、`utils/kb_db.py`（写入 SQLite）
     - `faiss`（向量索引）
     - `tools/kb_search.py`（检索）

4. `api/agent.py`
   - 路由：`/chat`
   - 依赖：
     - `tools/kb_search.py`（检索）
     - `llm/siliconflow.py`（生成回答）
     - `config.py`（上下文长度/检索参数）

5. `api/tools.py`
   - 路由：`/tools/list`、`/tools/call(占位)`
   - 依赖：`tools/registry.py`

6. `app.py`
   - 作用：注册所有蓝图（system/upload/kb/tools/agent）
