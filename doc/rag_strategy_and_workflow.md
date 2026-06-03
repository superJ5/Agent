# SuperBizAgent RAG 策略与运行流程说明

本文基于当前仓库 `ai_agent_competition` 的真实代码整理。写法尽量按“文件、函数、输入、输出、流程、降级”来说明，方便不熟悉项目的人直接顺着看。

---

## 1. 整体方案

### 1.1 这套 RAG 要解决什么

目标：

- 用户问说明书、手册、部件、步骤、保修、图片、OCR、原文追溯等问题时，系统不能靠模型猜。
- 系统要先去知识库找证据，再让大模型基于证据回答。
- 如果找不到可靠证据，要明确说信息不足，而不是编造。

当前方案：

```text
FastAPI 接收用户问题
  ↓
LangGraph Agent 判断是否需要工具
  ↓
调用 retrieve_knowledge()
  ↓
RAG 检索流水线找手册证据
  ↓
把证据上下文交给 Qwen
  ↓
Qwen 生成最终答案
```

核心技术：

| 模块 | 当前使用 |
|---|---|
| Web 服务 | FastAPI |
| Agent 编排 | LangGraph / LangChain `create_agent()` |
| 大模型 | DashScope Qwen，当前 `qwen3.5-plus` |
| Embedding | DashScope `text-embedding-v4` |
| 向量库 | Milvus |
| 向量索引 | HNSW |
| 相似度 | COSINE |
| 重排 | DashScope `qwen3-rerank` |
| 关键词召回 | jieba + rank-bm25 |

### 1.2 总链路

```text
用户问题 / 图片
  ↓
POST /chat
  ↓
ChatRequest 校验
  ↓
competition_chat()
  ↓
rag_agent_service.query()
  ↓
LangGraph Agent 调用 Qwen
  ↓
Qwen 按系统提示调用 retrieve_knowledge()
  ↓
orchestrator.retrieve()
  ↓
query_understanding → recall → reranker → evidence
  ↓
format_bundle() 生成证据上下文
  ↓
Qwen 基于证据输出答案
  ↓
API 返回 answer + session_id + metadata
```

### 1.3 当前重要配置

文件：`app/config.py`

类：`Settings`

当前代码默认配置和 `.env` 实际配置共同决定运行参数。本地 `.env` 中 RAG 相关配置如下：

| 配置项 | 当前值 | 含义 |
|---|---|---|
| `PORT` | `9900` | FastAPI 端口。 |
| `RAG_MODEL` | `qwen3.5-plus` | Agent 使用的大模型。 |
| `DASHSCOPE_EMBEDDING_MODEL` | `text-embedding-v4` | 文本向量模型。 |
| `RAG_TOP_K` | `3` | 最终主证据数量。 |
| `RAG_INTENT_STRATEGY` | `none` | 当前问题理解策略。代码支持 `profile/rules/summary/llm/hybrid`。 |
| `RAG_ENABLE_VECTOR_RECALL` | `true` | 启用向量召回。 |
| `RAG_ENABLE_BM25_RECALL` | `true` | 启用 BM25 召回。 |
| `RAG_ENABLE_SCAN_FALLBACK` | `true` | 启用扫描兜底。 |
| `RAG_SCAN_CANDIDATE_LIMIT` | `6000` | 扫描兜底候选上限。 |
| `RAG_RERANKER_PROVIDER` | `dashscope` | 使用 DashScope 重排。 |
| `RAG_RERANKER_MODEL` | `qwen3-rerank` | 重排模型。 |
| `RAG_RERANKER_TIMEOUT_MS` | `3000` | 重排超时时间。 |
| `MILVUS_HOST` | `localhost` | Milvus 地址。 |
| `MILVUS_PORT` | `19530` | Milvus 端口。 |

---

## 2. 仓库模块地图

### 2.1 代码分层

| 层级 | 文件 / 目录 | 作用 |
|---|---|---|
| API 层 | `app/api/chat.py` | 接收用户对话请求，返回接口响应。 |
| 请求模型 | `app/models/request.py` | 校验请求体，尤其是问题和图片。 |
| Agent 层 | `app/services/rag_agent_service.py` | 创建 Qwen Agent，管理工具和会话。 |
| 工具层 | `app/tools/knowledge_tool.py` | RAG 检索工具入口。 |
| 检索编排 | `app/retrieval/orchestrator.py` | 串起问题理解、召回、重排、证据组织。 |
| 问题理解 | `app/retrieval/query_understanding.py` | 判断意图、文档、关键词。 |
| 召回 | `app/retrieval/recall.py` | 向量召回、BM25 召回、扫描兜底。 |
| 重排 | `app/retrieval/reranker.py` | DashScope rerank 和本地词法兜底。 |
| 证据组织 | `app/retrieval/evidence.py` | 组织主证据、父级上下文、图片引用。 |
| 向量检索 | `app/services/vector_search_service.py` | 查询 Milvus。 |
| 向量写入 | `app/services/vector_store_manager.py` | 写入 Milvus、删除旧数据、构造 filter。 |
| 手册入库 | `app/services/vector_index_service.py` | 读取 JSONL chunks 并构造 Document。 |
| Milvus 管理 | `app/core/milvus_client.py` | 连接 Milvus、创建 collection 和索引。 |
| Embedding | `app/services/vector_embedding_service.py` | 调 DashScope embedding。 |
| 启动入口 | `app/main.py` | 创建 FastAPI app，注册路由，启动时连接 Milvus。 |

---

## 3. 服务启动流程

### 3.1 FastAPI 应用入口

文件：`app/main.py`

函数：`lifespan()`，当前在第 22 行附近。

应用对象：

```text
app = FastAPI(..., lifespan=lifespan)
```

启动时执行：

1. 记录服务启动日志。
2. 调用 `milvus_manager.connect()`。
3. 调用 `init_bm25_provider()`。
4. 等待请求进入。

对应代码位置：

- `app/main.py::lifespan()`：服务启动和关闭生命周期。
- `app/core/milvus_client.py::MilvusClientManager.connect()`：连接 Milvus。
- `app/retrieval/bm25_provider.py::init_bm25_provider()`：初始化 BM25。

启动流程：

```text
uvicorn app.main:app
  ↓
创建 FastAPI app
  ↓
进入 lifespan()
  ↓
milvus_manager.connect()
  ↓
init_bm25_provider()
  ↓
服务开始接收请求
```

注意：

- router 注册不是在 `lifespan()` 里面执行，而是在 `app/main.py` 模块加载时执行。
- 当前 router 注册在 `app/main.py` 第 65 到 69 行附近。

### 3.2 API Router 注册

文件：`app/main.py`

代码位置：第 65 到 69 行附近。

当前注册：

| Router | 路径前缀 | 作用 |
|---|---|---|
| `health.router` | 无 | 健康检查。 |
| `chat.competition_router` | 无 | 比赛标准接口，核心路径是 `/chat`。 |
| `chat.router` | `/api` | 旧版聊天接口，例如 `/api/chat`、`/api/chat_stream`。 |
| `file.router` | `/api` | 上传和入库接口。 |
| `aiops.router` | `/api` | AIOps 诊断接口。 |

所以当前主入口是：

```text
POST /chat
```

对应函数：

```text
app/api/chat.py::competition_chat()
```

当前在第 250 行附近。

### 3.3 Milvus 初始化

文件：`app/core/milvus_client.py`

类：`MilvusClientManager`，当前在第 44 行附近。

核心函数：

- `connect()`：第 59 行附近。
- `_create_collection()`：第 149 行附近。
- `_create_index()`：第 192 行附近。

当前 collection：

```text
biz
```

collection 字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `id` | `VARCHAR` | chunk 主键，优先使用 `chunk_id`。 |
| `vector` | `FLOAT_VECTOR(1024)` | embedding 向量。 |
| `content` | `VARCHAR` | chunk 展示正文。 |
| `metadata` | `JSON` | 文档名、chunk 类型、层级、图片路径、行号等。 |

索引算法：

- 向量索引：HNSW。
- 相似度指标：COSINE。
- 参数：`M=16`，`efConstruction=256`。

流程：

```text
milvus_manager.connect()
  ↓
connections.connect(alias="default")
  ↓
创建 MilvusClient
  ↓
检查 biz collection 是否存在
  ↓
不存在：_create_collection()
  ↓
创建 vector 字段索引：_create_index()
  ↓
加载 collection 到内存
```

特殊逻辑：

- 如果 `biz` 已存在，会检查 `vector` 维度。
- 如果已有 collection 的向量维度不是 1024，代码会删除旧 collection 并重建。

### 3.4 BM25 初始化

文件：`app/retrieval/bm25_provider.py`

函数：`init_bm25_provider()`，当前在第 204 行附近。

流程：

```text
init_bm25_provider()
  ↓
vector_search_service.query_all_documents()
  ↓
JiebaBM25Provider.build_index(all_results)
  ↓
recall.set_bm25_provider(provider)
```

输入：

- Milvus 中当前所有可查询 chunks。

输出：

- 一个进程内 BM25 provider。
- 后续供 `app/retrieval/recall.py::bm25_recall()` 使用。

分词策略：

- 中文：`jieba.cut()`。
- 英文：正则 token，并过滤常见停用词。
- 英文会保留 `not/no/use/set/run/start/stop` 这类操作词，避免手册问题丢关键信息。

降级：

- 如果初始化失败，记录 warning。
- 调用 `set_bm25_provider(None)` 清空 provider。
- 服务仍继续运行，后续依赖 vector recall 和 scan fallback。

---

## 4. 知识库入库流程

### 4.1 手册数据来源

当前真正用于 RAG 的数据目录：

```text
data/manuals/chunks/*.jsonl
```

示例：

```text
manual_60e374ac__电钻手册.jsonl
manual_9199cd53__冰箱手册.jsonl
manual_967d35e4__Camera.jsonl
```

每一行是一条 chunk 记录。常见字段：

| 字段 | 含义 |
|---|---|
| `doc_id` | 手册唯一 ID。 |
| `doc_name` | 手册名称。 |
| `chunk_id` | chunk 唯一 ID。 |
| `chunk_index` | chunk 顺序。 |
| `language` | 中文或英文。 |
| `retrieval_tier` | 检索层级，例如 `primary/support/big_support/auxiliary`。 |
| `chunk_type` | chunk 类型。 |
| `section_path` | 章节路径。 |
| `parent_chunk_id` | 父级 chunk。 |
| `pic_ids` | 图片 ID。 |
| `image_paths` | 图片路径，入库时会补充。 |
| `source_file` | 来源文件。 |
| `source_lines` | 来源行号。 |
| `source_issue_flags` | OCR 或源数据风险。 |
| `index_text` | 用于 embedding 和检索的文本。 |
| `text` | 展示给模型看的正文。 |

### 4.2 入库入口

API 入口：

文件：`app/api/file.py`

函数：`index_manual_chunks()`，当前在第 132 行附近。

路径：

```text
POST /api/index_manual_chunks
```

脚本入口：

文件：`scripts/index_manual_chunks.py`

函数：`main()`

常用命令：

```bash
python scripts/index_manual_chunks.py --directory data/manuals/chunks
```

### 4.3 手册 chunk 入库主流程

文件：`app/services/vector_index_service.py`

类：`VectorIndexService`

函数：

- `index_manual_chunks()`：第 94 行附近。
- `_index_chunk_jsonl()`：第 208 行附近。

流程：

```text
index_manual_chunks(directory)
  ↓
_collect_manual_chunk_files()
  ↓
_index_files()
  ↓
index_single_file()
  ↓
_index_chunk_jsonl(path)
  ↓
_load_chunk_records(path)
  ↓
delete_by_doc_id(doc_id)
  ↓
delete_by_source(source_path)
  ↓
_build_documents_from_chunk_records()
  ↓
vector_store_manager.add_documents(documents)
  ↓
reset_profile_caches()
```

输入：

- 一个 JSONL 文件，或者一个包含多个 JSONL 文件的目录。

输出：

- `IndexingResult`，包括成功数、失败数、耗时、失败文件等。

关键逻辑：

- 如果 chunk 里有 `doc_id`，入库前会先删除 Milvus 中同 `doc_id` 的旧数据。
- 同时会按 source 文件路径删除旧数据，避免重复入库。
- 入库完成后会刷新文档画像缓存，让后续路由能看到最新 metadata。

### 4.4 JSONL record 转 Document

文件：`app/services/vector_index_service.py`

函数：`_build_structured_document()`，当前在第 363 行附近。

输入：

- `record`：一条 JSONL chunk。
- `source_path`：当前 JSONL 文件路径。
- `image_path_lookup`：图片 ID 到图片路径的映射。
- `fallback_index`：兜底顺序。

输出：

- LangChain `Document`。

转换规则：

| Document 字段 | 来源 | 作用 |
|---|---|---|
| `page_content` | `record["text"]` | 给模型看的正文。 |
| `metadata["index_text"]` | `record["index_text"]` 或正文 | 给 embedding 和检索用。 |
| `metadata["chunk_id"]` | `record["chunk_id"]` | Milvus 主键优先来源。 |
| `metadata["retrieval_tier"]` | `record["retrieval_tier"]` | 控制召回和证据扩展。 |
| `metadata["chunk_type"]` | `record["chunk_type"]` | 控制意图路由。 |
| `metadata["parent_chunk_id"]` | `record["parent_chunk_id"]` | 找父级 support。 |
| `metadata["pic_ids"]` | `record["pic_ids"]` | 图片引用。 |
| `metadata["image_paths"]` | 根据 `metadata_image_path` 补充 | 最终回答中可引用图片。 |

### 4.5 写入 Milvus

文件：`app/services/vector_store_manager.py`

类：`VectorStoreManager`

函数：`add_documents()`，当前在第 73 行附近。

流程：

```text
add_documents(documents)
  ↓
_build_document_id(document)
  ↓
_get_raw_embedding_text(document)
  ↓
vector_embedding_service.embed_documents()
  ↓
构造 rows: id / vector / content / metadata
  ↓
collection.insert(rows)
  ↓
collection.flush()
```

输入：

- `list[Document]`

输出：

- 写入 Milvus 的 id 列表。

关键逻辑：

- 主键优先使用 `metadata["chunk_id"]`。
- embedding 文本优先使用 `metadata["index_text"]`。
- 展示正文使用 `page_content`。
- `content` 会按 Milvus 字段上限截断到 8000 bytes。
- metadata 会做 JSON 清洗，过大时压缩 `index_text/text`。

降级：

- 如果 embedding provider 因 8192 token 限制拒绝某条输入，代码会定位到单条 chunk。
- 然后截断 embedding 文本重试。
- 同时把这类情况写到 `logs/embedding_input_limit_report.jsonl`。

### 4.6 Embedding 服务

文件：`app/services/vector_embedding_service.py`

类：`DashScopeEmbeddings`，当前在第 12 行附近。

函数：

- `embed_documents()`：第 58 行附近。
- `embed_query()`：第 91 行附近。

当前模型：

```text
text-embedding-v4
```

当前维度：

```text
1024
```

输入输出：

| 函数 | 输入 | 输出 | 用途 |
|---|---|---|---|
| `embed_documents()` | `list[str]` | `list[list[float]]` | 入库时给 chunk 生成向量。 |
| `embed_query()` | `str` | `list[float]` | 查询时给用户问题生成向量。 |

---

## 5. 用户请求进入系统

### 5.1 请求体校验

文件：`app/models/request.py`

类：`ChatRequest`，当前在第 22 行附近。

输入字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `question` / `Question` | `str` | 用户问题，不能为空。 |
| `images` | `list[str]` | Base64 图片列表，最多 3 张。 |
| `session_id` / `Id` | `str | None` | 会话 ID，可选。 |
| `stream` | `bool` | 是否流式返回，默认 false。 |

校验函数：

- `validate_question()`：第 58 行附近。
- `validate_images()`：第 65 行附近。

图片校验规则：

- 支持 `png`、`jpg`、`jpeg`、`webp` 的 data URL。
- 每张图片必须是合法 base64。
- 单张图片不能超过 5MB。
- 最多 3 张。

真实运行流程：

```text
HTTP JSON body
  ↓
FastAPI 解析成 dict
  ↓
Pydantic 校验 ChatRequest
  ↓
question / Question → request.question
  ↓
session_id / Id → request.session_id
  ↓
validate_question()
  ↓
validate_images()
  ↓
校验成功：进入 competition_chat()
```

失败情况：

- 如果问题为空，FastAPI 直接返回 HTTP 422。
- 如果图片格式不合法，FastAPI 直接返回 HTTP 422。
- 如果图片超过 3 张或单张超过 5MB，FastAPI 直接返回 HTTP 422。
- 校验失败时不会进入 `competition_chat()`。

### 5.2 比赛接口入口

文件：`app/api/chat.py`

函数：`competition_chat()`，当前在第 250 行附近。

接口：

```text
POST /chat
```

输入：

- `request: ChatRequest`
- `Authorization` header

输出：

- 非流式：JSON。
- 流式：SSE `EventSourceResponse`。

非流式主流程：

```text
competition_chat(request, authorization)
  ↓
_require_bearer_token(authorization)
  ↓
_resolve_session_id(request.session_id)
  ↓
memory_service.append_message(role="user")
  ↓
_query_rag_agent(question, session_id, images)
  ↓
memory_service.append_message(role="assistant")
  ↓
rag_agent_service.get_last_retrieval_metadata(session_id)
  ↓
_competition_success_payload(answer, session_id, metadata)
```

关键函数说明：

| 函数 | 当前位置 | 输入 | 输出 | 作用 |
|---|---|---|---|---|
| `_require_bearer_token()` | 第 46 行附近 | Authorization header | 无返回；失败抛 401 | 检查 Bearer token。如果 `.env` 配了 `API_BEARER_TOKEN`，还会比对 token。 |
| `_resolve_session_id()` | 第 40 行附近 | 用户传入 session_id | session_id 字符串 | 有传入则去空格使用；没有则生成 `kf_session_` 前缀随机 ID。 |
| `_query_rag_agent()` | 第 237 行附近 | question、session_id、images | answer 字符串 | 调用 `rag_agent_service.query()`。 |
| `_competition_success_payload()` | 第 68 行附近 | answer、session_id、metadata | JSON dict | 包装比赛接口成功响应。 |

成功响应结构：

```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "answer": "最终答案",
    "session_id": "会话 ID",
    "timestamp": 1234567890,
    "metadata": {}
  }
}
```

流式分支：

```text
request.stream == true
  ↓
_build_stream_response()
  ↓
rag_agent_service.query_stream()
  ↓
EventSourceResponse
```

说明：

- 当前比赛主流程建议看非流式。
- 流式接口会按 SSE 逐段返回内容。

### 5.3 会话和记忆写入

文件：`app/api/chat.py`

调用位置：

- 用户消息写入：`competition_chat()` 内部。
- 助手消息写入：`competition_chat()` 内部。

调用函数：

```text
memory_service.append_message()
```

作用：

- 把用户问题保存到当前 session 的历史中。
- RAG Agent 后续可恢复最近历史。
- 回答成功后，也把助手答案写入历史。

写入用户消息的 metadata：

```text
source = "competition_chat"
stream = False
images_count = 图片数量
```

写入助手消息的 metadata：

```text
source = "competition_chat"
stream = False
status = "success" 或 "error"
```

---

## 6. Agent 执行流程

### 6.1 Agent 服务入口

文件：`app/services/rag_agent_service.py`

类：`RagAgentService`，当前在第 85 行附近。

核心函数：

```text
query(question, session_id, images=None)
```

当前在第 264 行附近。

输入：

- `question: str`
- `session_id: str`
- `images: list[str] | None`

输出：

- `answer: str`

非流式主流程：

```text
RagAgentService.query()
  ↓
_initialize_agent()
  ↓
_clear_last_retrieval_metadata()
  ↓
_build_effective_system_prompt()
  ↓
_build_persistent_history_messages()
  ↓
build_user_message(question, images)
  ↓
self.agent.ainvoke(agent_input, config)
  ↓
_read_last_retrieval_metadata()
  ↓
取最后一条 message.content
  ↓
_ensure_image_placeholders(answer)
  ↓
return answer
```

### 6.2 Agent 初始化

文件：`app/services/rag_agent_service.py`

函数：`_initialize_agent()`，当前在第 123 行附近。

流程：

```text
_initialize_agent()
  ↓
get_mcp_client_with_retry()
  ↓
mcp_client.get_tools()
  ↓
本地工具 + MCP 工具
  ↓
create_agent(model, tools, checkpointer)
```

本地基础工具：

| 工具 | 文件 | 作用 |
|---|---|---|
| `retrieve_knowledge` | `app/tools/knowledge_tool.py` | 查手册知识库。 |
| `memory_search` | `app/tools/memory_tool.py` | 查历史记忆。 |
| `get_current_time` | `app/tools/time_tool.py` | 获取当前时间。 |

降级：

- 如果 MCP 工具加载失败，记录 warning。
- 系统降级为只使用本地工具。
- RAG 问答仍可继续运行。

### 6.3 系统提示词

文件：`app/services/rag_agent_service.py`

函数：`_build_system_prompt()`，当前在第 158 行附近。

RAG 关键规则：

1. 只要问题涉及说明书、手册、部件、操作步骤、保修、图片、OCR 或原文追溯，必须先调用 `retrieve_knowledge`。
2. 优先依据 `retrieve_knowledge` 返回的证据回答。
3. 检索不到可靠内容时，要说明“当前检索到的信息不足”。
4. 如果检索结果包含图片标识或配图信息，回答时要结合这些证据。
5. 英文手册问题也要先调用 `retrieve_knowledge`。

这一步的实际作用：

- 它不是检索代码。
- 它是给大模型的行为约束。
- 它决定了手册类问题不能直接让模型自由发挥。

### 6.4 用户消息构造

文件：`app/services/multimodal_message_builder.py`

函数：`build_user_message()`，当前在第 8 行附近。

输入：

- `question`
- `images`

输出：

- LangChain `HumanMessage`

无图片时：

```text
HumanMessage(content=question)
```

有图片时：

```text
HumanMessage(
  content=[
    {"type": "text", "text": question},
    {"type": "image_url", "image_url": {"url": image_base64}}
  ]
)
```

说明：

- 用户图片直接给多模态 Qwen 看。
- RAG 知识库本身主要检索手册文本、图片 ID、图片路径和 metadata。

---

## 7. RAG 工具入口

### 7.1 retrieve_knowledge()

文件：`app/tools/knowledge_tool.py`

函数：`retrieve_knowledge()`，当前在第 571 行附近。

输入：

- `query: str`

输出：

- `context: str`
- `docs: list[Document]`

主流程：

```text
retrieve_knowledge(query)
  ↓
clear_last_retrieval_metadata()
  ↓
routed_retrieve(query)
  ↓
set_last_retrieval_metadata(bundle.metadata)
  ↓
search_result_to_document(result)
  ↓
format_bundle(bundle, query)
  ↓
return context, docs
```

如果没有命中：

```text
return "没有找到相关信息。", []
```

如果检索异常：

```text
return "检索知识时发生错误: ...", []
```

### 7.2 routed_retrieve()

文件：`app/tools/knowledge_tool.py`

函数：`routed_retrieve()`，当前在第 598 行附近。

流程：

```text
routed_retrieve(query)
  ↓
优先调用 app.retrieval.orchestrator.retrieve(query)
  ↓
如果模块化检索失败
  ↓
降级调用 _legacy_routed_retrieve(query)
```

输出：

- `RetrievalBundle`

`RetrievalBundle` 主要字段：

| 字段 | 含义 |
|---|---|
| `intent` | 检索意图。 |
| `retrieval_stage` | 检索阶段。 |
| `hits` | 主证据。 |
| `support_hits` | 补充上下文。 |
| `warnings` | 降级或风险提示。 |
| `metadata` | API 返回用的检索摘要。 |

---

## 8. 模块化检索流水线

### 8.1 总编排

文件：`app/retrieval/orchestrator.py`

函数：`retrieve()`，当前在第 32 行附近。

输入：

- `query: str`
- `options: RetrievalOptions | None`

输出：

- `RetrievalBundle`

流程：

```text
retrieve(query)
  ↓
load_retrieval_options_from_config()
  ↓
_run_query_understanding()
  ↓
_run_recall()
  ↓
_run_reranker()
  ↓
_run_evidence()
  ↓
_finalize_metadata()
  ↓
_write_trace()
  ↓
return RetrievalBundle
```

每一步的作用：

| 阶段 | 文件 | 函数 | 作用 |
|---|---|---|---|
| 问题理解 | `query_understanding.py` | `analyze_query()` | 判断意图、文档、关键词。 |
| 召回 | `recall.py` | `recall_candidates()` | 从 Milvus/BM25/scan 找候选。 |
| 重排 | `reranker.py` | `rerank_candidates()` | 对候选重新排序。 |
| 证据组织 | `evidence.py` | `build_retrieval_bundle()` | 整理主证据和上下文。 |
| 诊断摘要 | `diagnostics.py` | `build_summary_metadata()` | 生成 API 可返回 metadata。 |

### 8.2 问题理解

文件：`app/retrieval/query_understanding.py`

函数：`analyze_query()`，当前在第 53 行附近。

输入：

- 用户 query。
- RAG 配置 options。
- diagnostics。

输出：

- `QueryAnalysis`

`QueryAnalysis` 包含：

| 字段 | 含义 |
|---|---|
| `query` | 原始问题。 |
| `strategy` | 当前策略。 |
| `intent_candidates` | 意图候选。 |
| `doc_candidates` | 文档候选。 |
| `query_terms` | 关键词。 |
| `language` | `zh` 或 `en`。 |
| `warnings` | 降级提示。 |

当前配置：

```text
RAG_INTENT_STRATEGY=none
```

所以当前实际逻辑：

```text
analyze_query(query)
  ↓
识别语言 zh/en
  ↓
不做强意图路由
  ↓
extract_terms(query)
  ↓
返回 QueryAnalysis(strategy="none")
```

代码还支持：

| 策略 | 作用 |
|---|---|
| `profile` | 根据 Milvus 中已有 metadata 动态构建文档画像。 |
| `rules` | 用高置信关键词规则判断意图或文档。 |
| `summary` | 用离线 summary index 判断文档。 |
| `llm` | 预留 LLM analyzer。 |
| `hybrid` | 同时跑多种策略并融合。 |

### 8.3 召回

文件：`app/retrieval/recall.py`

函数：`recall_candidates()`，当前在第 122 行附近。

输入：

- `query`
- `QueryAnalysis`
- `RetrievalOptions`
- `RetrievalDiagnostics`

输出：

- `list[RecallCandidate]`

流程：

```text
recall_candidates()
  ↓
vector_recall()
  ↓
bm25_recall()
  ↓
merge_recall_results()
  ↓
should_trigger_scan()
  ↓
必要时 scan_recall()
  ↓
再次 merge_recall_results()
  ↓
return RecallCandidate 列表
```

召回通道：

| 通道 | 函数 | 当前位置 | 作用 |
|---|---|---|---|
| 向量召回 | `vector_recall()` | 第 188 行附近 | 用 embedding 语义检索 Milvus。 |
| BM25 召回 | `bm25_recall()` | 第 228 行附近 | 用关键词和 BM25 检索。 |
| 扫描兜底 | `scan_recall()` | 第 290 行附近 | 扫描 metadata 后本地重排。 |

候选合并：

文件：`app/retrieval/recall.py`

函数：`merge_recall_results()`，当前在第 372 行附近。

合并逻辑：

1. 不同召回通道的分数归一化。
2. 用 `chunk_id` 去重。
3. 同一个 chunk 记录命中过哪些通道。
4. 按权重计算 `merged_score`。

当前权重：

| 通道 | 权重 |
|---|---|
| vector | 0.6 |
| bm25 | 0.4 |
| scan | 0.2 |

### 8.4 向量召回

文件：`app/services/vector_search_service.py`

函数：`search_similar_documents()`，当前在第 91 行附近。

输入：

- `query`
- `top_k`
- `doc_id`
- `language`
- `retrieval_tiers`
- `chunk_types`
- `chunk_ids`

输出：

- `list[SearchResult]`

流程：

```text
search_similar_documents()
  ↓
vector_embedding_service.embed_query(query)
  ↓
milvus_manager.get_collection()
  ↓
build_metadata_filter_expr()
  ↓
collection.search()
  ↓
SearchResult(id, content, score, metadata)
```

Milvus 查询参数：

```text
anns_field = "vector"
metric_type = "COSINE"
params = {"ef": 64}
output_fields = ["id", "content", "metadata"]
```

### 8.5 BM25 召回

文件：`app/retrieval/recall.py`

函数：`bm25_recall()`，当前在第 228 行附近。

输入：

- query。
- doc_id、language、retrieval_tiers、chunk_types 等过滤条件。

输出：

- `list[SearchResult]`

流程：

```text
bm25_recall()
  ↓
读取进程内 _bm25_provider
  ↓
没有 provider：返回 []
  ↓
有 provider：provider.search(query, filters)
  ↓
返回 BM25 命中
```

降级：

- 如果启动时 BM25 初始化失败，`_bm25_provider` 为 `None`。
- 此时 `bm25_recall()` 记录 warning 并返回空列表。
- 不影响 vector recall 和 scan fallback。

### 8.6 扫描兜底

文件：`app/retrieval/recall.py`

函数：`scan_recall()`，当前在第 290 行附近。

输入：

- query。
- analysis。
- options。

输出：

- `list[SearchResult]`

流程：

```text
scan_recall()
  ↓
vector_search_service.query_all_documents(filters)
  ↓
filter_results_to_active_docs()
  ↓
本地 lexical_score 重排
  ↓
返回 top candidates
```

触发条件：

函数：`should_trigger_scan()`，当前在第 340 行附近。

会触发 scan 的情况：

- 前面没有召回候选。
- 意图是 `image_trace` 或 `ocr_audit`。
- 查询包含关键术语组，例如冷机、热机、加油、燃油混合、防护装备、OCR 缺失等。
- top candidate 的本地词法相关性不足。

### 8.7 重排

文件：`app/retrieval/reranker.py`

函数：`rerank_candidates()`，当前在第 147 行附近。

输入：

- `query`
- `list[RecallCandidate]`
- `RetrievalOptions`
- diagnostics

输出：

- `RerankResult`

主流程：

```text
rerank_candidates()
  ↓
读取 provider 配置
  ↓
provider == dashscope
  ↓
DashScopeReranker.rerank()
  ↓
调用 qwen3-rerank
  ↓
返回排序后的 candidates
```

DashScope rerank 输入：

- query。
- documents：候选 chunk 的 `index_text`、`content` 或 `text`。
- top_n：当前默认 32。

降级：

如果出现以下情况：

- provider 不可用。
- 缺少 API key。
- HTTP 调用失败。
- reranker 超时。
- reranker 返回空结果。

会降级到：

```text
lexical_fallback()
```

本地词法重排会调用 `knowledge_tool.lexical_score()`。

### 8.8 证据组织

文件：`app/retrieval/evidence.py`

函数：`build_retrieval_bundle()`，当前在第 58 行附近。

输入：

- query。
- `QueryAnalysis`。
- `RerankResult`。
- options。
- diagnostics。

输出：

- `RetrievalBundle`

流程：

```text
build_retrieval_bundle()
  ↓
取 rerank 后 top_k 个主候选
  ↓
fetch_parent_support_hits()
  ↓
必要时 expand_big_support_parent()
  ↓
记录图片 trace
  ↓
返回 RetrievalBundle
```

相关函数：

| 函数 | 当前位置 | 作用 |
|---|---|---|
| `fetch_parent_support_hits()` | 第 117 行附近 | 根据 primary 的 `parent_chunk_id` 查父级 support。 |
| `expand_big_support_parent()` | 第 280 行附近 | 对 big_support 父节点向下展开相关 primary 子节点。 |
| `format_bundle()` | 第 661 行附近 | 把 bundle 格式化成模型可读证据。 |
| `format_search_results()` | 第 691 行附近 | 格式化每条参考资料。 |

证据结构：

| 字段 | 含义 |
|---|---|
| `hits` | 主命中，通常 top 3。 |
| `support_hits` | 父级上下文。 |
| `warnings` | 检索提示或降级信息。 |
| `metadata` | 给 API 返回的摘要。 |

格式化后给模型的内容包括：

- 检索意图。
- 检索阶段。
- 主命中数量。
- 补充上下文数量。
- 图片数量。
- 每条证据的手册名、标题、章节路径、chunk ID、行号、图片 ID、正文。

---

## 9. 最终输出流程

### 9.1 模型如何生成答案

文件：`app/services/rag_agent_service.py`

函数：`query()`，当前在第 264 行附近。

在 `self.agent.ainvoke()` 内部：

```text
Qwen 读取系统提示词
  ↓
Qwen 读取用户问题和图片
  ↓
如果是手册类问题，调用 retrieve_knowledge()
  ↓
Qwen 读取 retrieve_knowledge 返回的证据 context
  ↓
Qwen 基于证据生成答案
```

代码取答案的位置：

```text
messages_result = result.get("messages", [])
last_message = messages_result[-1]
answer = last_message.content
```

然后执行：

```text
_ensure_image_placeholders(answer)
```

作用：

- 如果答案里有 `![](path)` 这种空 alt 图片引用，会尝试从路径中提取图片 ID。
- 转成 `![pic_id](path)`。

### 9.2 API 如何返回答案

文件：`app/api/chat.py`

函数：`_competition_success_payload()`，当前在第 68 行附近。

输入：

- `answer`
- `session_id`
- `metadata`

输出：

- JSON dict

结构：

```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "answer": "最终答案",
    "session_id": "会话 ID",
    "timestamp": 1234567890,
    "metadata": {
      "intent": "general",
      "retrieval_stage": "hybrid_search",
      "recall_channels": ["vector", "bm25", "scan"],
      "reranker_provider": "dashscope",
      "top_hits": []
    }
  }
}
```

metadata 来源：

```text
retrieve_knowledge()
  ↓
set_last_retrieval_metadata(bundle.metadata)
  ↓
rag_agent_service.get_last_retrieval_metadata(session_id)
  ↓
_competition_success_payload()
```

---

## 10. Legacy 检索兜底

### 10.1 什么时候触发

文件：`app/tools/knowledge_tool.py`

函数：`routed_retrieve()`，当前在第 598 行附近。

触发条件：

- `app.retrieval.orchestrator` 导入失败。
- `orchestrator.retrieve(query)` 执行失败。

触发后会调用：

```text
_legacy_routed_retrieve(query)
```

当前在第 647 行附近。

### 10.2 Legacy 流程

```text
_legacy_routed_retrieve(query)
  ↓
detect_intent(query)
  ↓
infer_doc_id(query)
  ↓
extract_query_terms(query)
  ↓
expand_queries(query, intent, query_terms)
  ↓
按 INTENT_STAGE_CONFIG 分阶段 run_search_stage()
  ↓
必要时 run_scan_stage()
  ↓
fetch_support_hits()
  ↓
collect_source_warnings()
  ↓
返回 RetrievalBundle
```

作用：

- 保证模块化检索异常时，系统仍能尝试回答。
- 保留手写领域规则，例如图片 ID、OCR、燃油混合、防护装备等。
- 保留全库扫描兜底。

---

## 11. 技术亮点

### 11.1 Agentic RAG

这个项目不是固定“先检索再回答”的死流程，而是 Agentic RAG。

特点：

- 大模型先看用户问题。
- 系统提示词约束手册类问题必须调用 `retrieve_knowledge()`。
- 工具返回证据。
- 大模型基于证据生成最终答案。

优势：

- 普通对话、手册问答、图片问答、记忆查询、AIOps 工具可以放在同一个 Agent 里。

### 11.2 混合召回

当前不是只用向量搜索。

实际召回组合：

```text
Vector Recall
  +
BM25 Recall
  +
Scan Fallback
```

各自价值：

| 方式 | 价值 |
|---|---|
| 向量召回 | 解决语义相近但字面不同的问题。 |
| BM25 | 解决产品名、部件名、型号、否定词、专有词。 |
| 扫描兜底 | 解决图片 ID、OCR、关键术语不能漏的问题。 |

### 11.3 层级证据

知识库不是简单平铺 chunk。

当前 chunk 有层级：

```text
primary
support
big_support
auxiliary
```

实际效果：

- primary 用来回答具体问题。
- support 用来补充章节上下文。
- big_support 可以向下展开相关 primary 子节点。
- auxiliary 可以保存图片路径、导航、OCR 辅助信息。

### 11.4 可降级

系统有多层降级：

| 位置 | 降级方式 |
|---|---|
| MCP 工具加载失败 | 只使用本地工具继续运行。 |
| BM25 初始化失败 | BM25 为空，继续使用 vector 和 scan。 |
| Reranker 失败 | 降级到本地 lexical fallback。 |
| 模块化检索失败 | 降级到 legacy 检索。 |
| 检索无结果 | 返回“没有找到相关信息”。 |

### 11.5 可观测

检索过程会生成 summary metadata。

主要字段：

```text
intent
doc_id
retrieval_stage
intent_strategy
recall_channels
reranker_provider
reranker_fallback
timeout
degraded
top_hits
warnings
```

如果 `APP_DEBUG=True`，详细 trace 会写入：

```text
logs/retrieval_trace.jsonl
```

---

## 12. 一条请求的完整示例

假设用户问：

```text
电钻充电灯闪烁是什么意思？
```

完整链路：

```text
POST /chat
  ↓
ChatRequest 校验 question
  ↓
competition_chat()
  ↓
_require_bearer_token()
  ↓
_resolve_session_id()
  ↓
memory_service.append_message(role="user")
  ↓
rag_agent_service.query()
  ↓
build_user_message()
  ↓
Agent 调用 Qwen
  ↓
Qwen 判断这是手册 / 部件 / 指示灯问题
  ↓
Qwen 调用 retrieve_knowledge()
  ↓
orchestrator.retrieve()
  ↓
analyze_query()
  ↓
recall_candidates()
  ↓
vector_recall() + bm25_recall() + 必要时 scan_recall()
  ↓
rerank_candidates()
  ↓
build_retrieval_bundle()
  ↓
format_bundle()
  ↓
Qwen 基于证据生成答案
  ↓
get_last_retrieval_metadata()
  ↓
_competition_success_payload()
  ↓
返回 JSON
```

---

## 13. 当前方案的注意事项

1. 当前 `RAG_INTENT_STRATEGY=none`，策略较保守，不会强行按意图或文档收窄。优点是误路由少，缺点是召回可能更散。
2. 如果后续希望增强文档识别，可以评测 `RAG_INTENT_STRATEGY=hybrid`。
3. BM25 是服务启动时从 Milvus 构建的进程内索引。运行中重新入库后，建议重启服务，让 BM25 同步新数据。
4. 图片理解由多模态 Qwen 完成；RAG 检索的是手册文本、图片 ID、图片路径和 metadata。
5. 如果 Milvus collection 维度或索引与代码配置不一致，需要重建 `biz` 并重新入库。

---

## 14. 总结

当前 RAG 主线可以理解成三句话：

1. 入库时，把结构化手册 chunk 转成 `Document`，用 `index_text` 做 embedding，连同 metadata 写入 Milvus。
2. 查询时，Agent 对手册类问题调用 `retrieve_knowledge()`，进入模块化检索流水线。
3. 检索流水线用向量召回、BM25、扫描兜底、rerank 和层级证据组织找到证据，再让 Qwen 基于证据回答。

整体架构已经拆成：

```text
API 层
  ↓
Agent 层
  ↓
RAG 工具层
  ↓
检索算法层
  ↓
向量库层
  ↓
数据入库层
```

这种拆法的好处是后续可以单独优化某一层，例如换 reranker、调召回权重、打开 hybrid intent strategy、增强图片 metadata，而不需要推翻整体架构。
