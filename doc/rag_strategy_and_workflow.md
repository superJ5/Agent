# 当前 RAG 策略与运行流程说明

本文档用于说明当前代码工作区中的 RAG 整体方案、核心算法、在线问答运行流程，以及知识库入库流程。

文档口径：

- 代码基准：当前工作区代码，包括尚未提交的本地改动。
- 主要链路：比赛标准非流式接口 `POST /chat`。
- 补充链路：流式接口、知识库入库、诊断与降级机制。
- 说明原则：只描述代码中已经存在的逻辑，不补充未实现的功能假设。

---

## 1. 整体方案概览

当前系统是一个基于 LangChain / LangGraph Agent 的 RAG 问答系统。在线问答时，用户问题先进入 FastAPI 接口，再交给 `RagAgentService` 创建的 Agent。Agent 根据系统提示词判断是否需要查询手册知识库；涉及说明书、手册、部件、操作步骤、保修、图片、OCR 或原文追溯的问题，会通过工具 `retrieve_knowledge` 进入 RAG 检索链路。

RAG 检索链路采用模块化流水线。由于当前配置为 `RAG_INTENT_STRATEGY=none`，所以在线真实执行时不会先做 profile/rules/summary/llm 意图路由，也不会根据 Query Understanding 产出的 doc_id 去限定某一本手册。当前真实检索路径是：

1. Query Understanding：只识别语言并抽取关键词；`intent_candidates=[]`，`doc_candidates=[]`。
2. Recall：按 `general` 路由做多路召回，默认查全库 `primary` 层 chunk，包括向量召回、BM25 召回、scan fallback。
3. Rerank：使用 DashScope reranker 对候选片段重排；失败时降级到本地词法重排。
4. Evidence：按 `general` intent 组织主证据、父级补充证据、图片引用、诊断元数据。
5. Answer：Agent 将证据上下文交给大模型，由大模型生成最终答案。

整体运行框架：

```text
用户 / 前端 / 测评程序
  ↓
POST /chat
  ↓
app/api/chat.py::competition_chat()
  ↓
app/services/rag_agent_service.py::RagAgentService.query()
  ↓
LangGraph Agent + ChatQwen
  ↓
Agent 按需调用 retrieve_knowledge
  ↓
app/tools/knowledge_tool.py::retrieve_knowledge()
  ↓
app/retrieval/orchestrator.py::retrieve()
  ↓
Query Understanding: strategy=none，只产出 language 和 query_terms
  ↓
Recall: general 路由，doc_id=None，retrieval_tier=primary，vector + BM25 + scan fallback
  ↓
Reranker: DashScope qwen3-rerank 或 lexical fallback
  ↓
Evidence: top hits + support hits + image refs + metadata
  ↓
证据上下文返回 Agent
  ↓
ChatQwen 基于证据生成最终回答
  ↓
POST /chat 返回 JSON
```

---

## 2. 代码地图

| 层级 | 文件 | 核心函数 / 类 | 职责 |
|---|---|---|---|
| 应用启动 | `app/main.py` | `lifespan()` | 服务启动时连接 Milvus，并初始化 BM25 provider。 |
| API 入口 | `app/api/chat.py` | `competition_chat()` | 处理 `POST /chat`，校验 token，分配 session，调用 RAG Agent。 |
| 请求模型 | `app/models/request.py` | `ChatRequest` | 校验问题、图片数量、图片 base64 格式、兼容旧字段。 |
| Agent 服务 | `app/services/rag_agent_service.py` | `RagAgentService.query()` | 构造系统提示词、历史消息、用户消息，执行 LangGraph Agent。 |
| RAG 工具 | `app/tools/knowledge_tool.py` | `retrieve_knowledge()` | LangChain 工具入口，返回模型可读的证据文本和 Document 列表。 |
| 检索编排 | `app/retrieval/orchestrator.py` | `retrieve()` | 串联 query understanding、recall、rerank、evidence。 |
| 查询理解 | `app/retrieval/query_understanding.py` | `analyze_query()` | 当前 `none` 下生成 `QueryAnalysis`，只填语言和关键词，意图/文档候选为空。 |
| 多路召回 | `app/retrieval/recall.py` | `recall_candidates()` | 调用 vector、BM25、scan，合并并打分候选 chunk。 |
| BM25 | `app/retrieval/bm25_provider.py` | `JiebaBM25Provider` | 进程内 BM25 词法索引，支持中英文分词和 metadata 过滤。 |
| 重排 | `app/retrieval/reranker.py` | `rerank_candidates()` | 调用 DashScope rerank，失败时本地词法降级。 |
| 证据组织 | `app/retrieval/evidence.py` | `build_retrieval_bundle()` | 生成最终证据包、补充上下文、图片引用和诊断摘要。 |
| 向量检索 | `app/services/vector_search_service.py` | `search_similar_documents()` | 查询 Milvus 向量相似度，返回 `SearchResult`。 |
| 向量入库 | `app/services/vector_index_service.py` | `index_manual_chunks()` | 读取结构化 chunk JSONL，转换为 Document 后写入 Milvus。 |
| Milvus 写入 | `app/services/vector_store_manager.py` | `add_documents()` | 批量 embedding，并将向量、正文、metadata 插入 Milvus。 |
| Embedding | `app/services/vector_embedding_service.py` | `DashScopeEmbeddings` | 调用 DashScope `text-embedding-v4` 生成 1024 维向量。 |
| Milvus 管理 | `app/core/milvus_client.py` | `MilvusClientManager` | 创建 / 加载 collection、建 HNSW 索引、管理连接。 |

---

## 3. 当前关键配置

配置来源是 `app/config.py` 的 `Settings`，本地可通过 `.env` 覆盖。下面列出当前代码默认值或模板值。

| 配置项 | 当前值 | 作用 |
|---|---:|---|
| `RAG_MODEL` | `qwen3.5-plus` | Agent 最终回答模型。 |
| `DASHSCOPE_EMBEDDING_MODEL` | `text-embedding-v4` | 文档和查询向量化模型。 |
| `RAG_TOP_K` | `3` | Evidence 阶段最终主命中数量。 |
| `RAG_INTENT_STRATEGY` | `none` | 当前默认不启用 profile/rules/summary/llm 混合意图策略，只保留关键词抽取。 |
| `RAG_ENABLE_VECTOR_RECALL` | `true` | 开启向量召回。 |
| `RAG_ENABLE_BM25_RECALL` | `true` | 开启 BM25 词法召回。 |
| `RAG_VECTOR_WEIGHT` | `0.6` | 合并召回分时向量通道权重。 |
| `RAG_BM25_WEIGHT` | `0.4` | 合并召回分时 BM25 通道权重。 |
| `RAG_ENABLE_SCAN_FALLBACK` | `true` | 开启 scan fallback。 |
| `RAG_SCAN_CANDIDATE_LIMIT` | `4096` | scan fallback 最大扫描候选数量。 |
| `RAG_RERANKER_PROVIDER` | `dashscope` | 使用 DashScope reranker。 |
| `RAG_RERANKER_MODEL` | `qwen3-rerank` | 重排模型。 |
| `RAG_RERANKER_TIMEOUT_MS` | `3000` | rerank 超时时间。 |
| `RAG_RERANKER_TOP_N` | `32` | 进入 reranker 的候选上限。 |
| `MILVUS_HOST` / `MILVUS_PORT` | `localhost` / `19530` | Milvus 连接地址。 |
| Milvus collection | `biz` | 当前知识库 collection 名。 |
| 向量维度 | `1024` | 与 `text-embedding-v4` 输出维度一致。 |

当前 `none` 配置对真实链路的直接影响：

| 项目 | 当前真实行为 |
|---|---|
| 意图候选 | 不生成，`intent_candidates=[]`。 |
| 文档候选 | 不生成，`doc_candidates=[]`。 |
| 主意图 | recall 阶段通过 `_primary_intent()` 兜底为 `general`。 |
| 文档过滤 | `_all_doc_ids()` 返回空，不按 `doc_id` 限制某本手册。 |
| 召回路由 | `_recall_routes()` 生成一条全库 `general` 路由。 |
| tier 过滤 | `general` 路由使用 `retrieval_tiers=["primary"]`。 |
| chunk type 过滤 | `chunk_families=None`，因此不限制具体 `chunk_type`。 |
| Evidence intent | `analysis.primary_intent` 为空，所以 evidence 中展示为 `general`。 |

### 3.1 `none` 配置下的一次真实端到端工作流

以下是当前真实执行链路，不包含未启用的 intent 策略：

```text
1. app/api/chat.py::competition_chat()
   输入: ChatRequest(question, images, session_id, stream=false)
   输出: 调用 rag_agent_service.query()

2. app/services/rag_agent_service.py::RagAgentService.query()
   输入: question, session_id, images
   动作: 构造 SystemMessage + 历史消息 + HumanMessage，执行 LangGraph Agent
   输出: Agent 最终 answer

3. Agent 调用 app/tools/knowledge_tool.py::retrieve_knowledge(query)
   输入: 模型生成的检索 query
   输出: context 文本 + Document artifact

4. app/tools/knowledge_tool.py::routed_retrieve(query)
   输入: query
   动作: 优先调用 app/retrieval/orchestrator.py::retrieve(query)
   输出: RetrievalBundle

5. app/retrieval/orchestrator.py::_run_query_understanding()
   输入: query, RetrievalOptions(intent_strategy="none")
   动作: analyze_query() 只抽 language 和 query_terms
   输出: QueryAnalysis(strategy="none", intent_candidates=[], doc_candidates=[])

6. app/retrieval/recall.py::recall_candidates()
   输入: query + 上一步 QueryAnalysis
   动作: primary_intent 兜底为 general；doc_id 为空；生成全库 primary 路由
   输出: vector / bm25 / scan 合并后的 RecallCandidate 列表

7. app/retrieval/recall.py::vector_recall()
   输入: query variants, doc_id=None, retrieval_tiers=["primary"], chunk_types=None, language=zh/en
   动作: embed query 后查 Milvus
   输出: SearchResult 列表

8. app/retrieval/recall.py::bm25_recall()
   输入: query, doc_id=None, retrieval_tiers=["primary"], chunk_types=None, language=zh/en
   动作: 使用进程内 JiebaBM25Provider 查询
   输出: SearchResult 列表；如果 provider 不存在则为空

9. app/retrieval/recall.py::scan_recall()
   触发: 候选为空、关键术语命中、general 词法分不足等
   输入: doc_id=None, retrieval_tiers=["primary"], chunk_types=None, language=zh/en
   动作: query_all_documents() 后本地词法排序
   输出: SearchResult 列表

10. app/retrieval/recall.py::merge_recall_results()
    输入: vector / bm25 / scan 通道结果
    动作: 按 chunk_id 去重，通道分归一化，加权求 merged_score
    输出: 排序后的 RecallCandidate 列表

11. app/retrieval/reranker.py::rerank_candidates()
    输入: RecallCandidate 列表
    动作: 调用 DashScope qwen3-rerank；失败则 lexical_fallback()
    输出: RerankResult

12. app/retrieval/evidence.py::build_retrieval_bundle()
    输入: QueryAnalysis(strategy="none") + RerankResult
    动作: 取 top_k 主证据；按 parent_chunk_id 查询 support；整理图片和 metadata
    输出: RetrievalBundle(intent="general", hits, support_hits, metadata)

13. app/retrieval/evidence.py::format_bundle()
    输入: RetrievalBundle
    输出: 模型可读 evidence context

14. Agent 基于 evidence context 生成 answer
    输出: 返回 app/api/chat.py，包装成 /chat JSON 响应
```

---

## 4. 服务启动流程

### 4.1 入口

文件：`app/main.py`

函数：`lifespan()`，当前在第 22 行附近。

启动时执行：

1. 记录服务启动日志。
2. 调用 `milvus_manager.connect()`。
3. 调用 `init_bm25_provider()`。
4. 注册 API router。
5. 挂载静态文件。

对应代码位置：

- `app/main.py::lifespan()`：启动生命周期。
- `app/core/milvus_client.py::MilvusClientManager.connect()`：连接 Milvus。
- `app/retrieval/bm25_provider.py::init_bm25_provider()`：初始化 BM25。

### 4.2 Milvus 初始化

文件：`app/core/milvus_client.py`

类：`MilvusClientManager`

当前 collection 设定：

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

如果 collection 已存在但向量维度不是 1024，代码会删除旧 collection 并重建。

### 4.3 BM25 初始化

文件：`app/retrieval/bm25_provider.py`

函数：`init_bm25_provider()`，当前在第 202 行附近。

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

- 一个进程内 BM25 provider，供 `recall.py` 的 `bm25_recall()` 使用。

降级：

- 如果初始化失败，记录 warning，并把 BM25 provider 清空。
- 服务仍可继续运行，后续依赖 vector recall 和 scan fallback。

---

## 5. 用户请求进入系统

### 5.1 请求体校验

文件：`app/models/request.py`

类：`ChatRequest`

输入字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `question` / `Question` | `str` | 用户问题，不能为空。 |
| `images` | `list[str]` | Base64 图片列表，最多 3 张。 |
| `session_id` / `Id` | `str | None` | 会话 ID，可选。 |
| `stream` | `bool` | 是否流式返回，默认 false。 |

图片校验规则：

- 支持 `png`、`jpg`、`jpeg`、`webp` 的 data URL。
- 每张图片必须是合法 base64。
- 单张图片不能超过 5MB。
- 最多 3 张。

### 5.2 比赛接口入口

文件：`app/api/chat.py`

函数：`competition_chat()`，当前在第 250 行附近。

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

| 函数 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `_require_bearer_token()` | Authorization header | 无返回；异常时抛 401 | 检查 Bearer token。如果 `.env` 配了 `API_BEARER_TOKEN`，还会比对 token。 |
| `_resolve_session_id()` | 用户传入 session_id | session_id 字符串 | 有传入则去空格使用；没有则生成 `kf_session_` 前缀随机 ID。 |
| `_query_rag_agent()` | question、session_id、images | answer 字符串 | 兼容调用 `rag_agent_service.query()`，如果 query 支持 images 就传入图片。 |
| `_competition_success_payload()` | answer、session_id、metadata | JSON dict | 包装比赛接口成功响应。 |

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

---

## 6. Agent 执行流程

### 6.1 Agent 服务入口

文件：`app/services/rag_agent_service.py`

类：`RagAgentService`

函数：`query()`，当前在第 264 行附近。

输入：

| 参数 | 类型 | 含义 |
|---|---|---|
| `question` | `str` | 用户问题。 |
| `session_id` | `str` | 会话 ID，也作为 LangGraph `thread_id`。 |
| `images` | `list[str] | None` | 用户上传的 Base64 图片。 |

输出：

- `str`：最终答案文本。

主流程：

```text
RagAgentService.query(question, session_id, images)
  ↓
_initialize_agent()
  ↓
清理上一轮 retrieval metadata
  ↓
构造 messages:
    SystemMessage(_build_effective_system_prompt())
    最近持久会话历史
    build_user_message(question, images)
  ↓
agent_input = {"messages": messages}
  ↓
config = {"configurable": {"thread_id": session_id}}
  ↓
self.agent.ainvoke(input=agent_input, config=config)
  ↓
读取工具侧最新 retrieval metadata
  ↓
取最后一条 message.content 作为 answer
  ↓
_ensure_image_placeholders(answer)
  ↓
return answer
```

### 6.2 Agent 初始化

函数：`_initialize_agent()`，当前在第 123 行附近。

流程：

1. 尝试通过 `get_mcp_client_with_retry()` 加载 MCP 工具。
2. 如果 MCP 不可用，降级为仅使用本地工具。
3. 合并工具列表：
   - `retrieve_knowledge`
   - `memory_search`
   - `get_current_time`
   - MCP tools
4. 调用 `create_agent(self.model, tools=all_tools, checkpointer=self.checkpointer)` 创建 LangGraph Agent。

使用模型：

- `ChatQwen`
- 模型名来自 `config.rag_model`，当前默认 `qwen3.5-plus`。
- `streaming=True`。

### 6.3 系统提示词策略

函数：`_build_system_prompt()`，当前在第 158 行附近。

核心规则：

- 只要问题涉及说明书、手册、部件、操作步骤、保修、图片、OCR 或原文追溯，必须先调用 `retrieve_knowledge`。
- 优先依据 `retrieve_knowledge` 返回的证据回答。
- 检索不到可靠内容时，要说明“当前检索到的信息不足”，不要编造。
- 如果检索结果带图片标识 PIC 或配图信息，回答时要结合证据。
- 英文手册问题也要求先调用 `retrieve_knowledge`，且检索 query 保持英文。
- 图片引用必须用非空 alt，例如 `![Manual01_5](data/manuals/raw/.../Manual01_5.jpg)`。

函数：`_build_effective_system_prompt()`，当前在第 210 行附近。

作用：

- 在基础系统提示词后追加长期记忆 `data/memory/MEMORY.md`，作为稳定背景参考。

函数：`_build_persistent_history_messages()`，当前在第 221 行附近。

作用：

- 如果 LangGraph 当前进程内没有该 session 的 checkpoint，则从 JSONL 持久记忆里恢复最近对话。
- 当前恢复条数来自 `config.memory_recent_limit`，默认 10。

### 6.4 Agent 如何调用 RAG 工具

`self.agent.ainvoke()` 内部由 LangGraph 执行。执行时模型会看到：

- 系统提示词。
- 当前用户问题。
- 用户图片消息。
- 可用工具描述。
- 当前 thread 的会话历史。

如果模型判断需要查知识库，会产生工具调用：

```text
retrieve_knowledge(query="模型改写后的检索问题")
```

然后 LangGraph 执行工具，把工具返回的证据上下文放回消息列表，再让模型基于证据生成最终答案。

---

## 7. RAG 工具入口

### 7.1 retrieve_knowledge

文件：`app/tools/knowledge_tool.py`

函数：`retrieve_knowledge()`，当前在第 571 行附近。

装饰器：

```python
@tool(response_format="content_and_artifact")
```

输入：

- `query: str`：模型传入的检索问题。通常不是原始用户问题，而是模型为了查询手册整理出的搜索句。

输出：

- `context: str`：格式化后的证据文本，给模型阅读。
- `docs: list[Document]`：LangChain Document 列表，作为工具 artifact。

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
docs = [search_result_to_document(result) for result in bundle.all_hits]
  ↓
如果 docs 为空，返回“没有找到相关信息。”
  ↓
context = format_bundle(bundle, query=query)
  ↓
return context, docs
```

### 7.2 routed_retrieve

函数：`routed_retrieve()`，当前在第 598 行附近。

当前优先执行模块化检索：

```text
retrieval_orchestrator.retrieve(query)
```

如果模块化检索失败：

```text
_legacy_routed_retrieve(query)
```

也就是说当前系统有两层兜底：

1. 正常路径：`app/retrieval/orchestrator.py` 模块化 RAG。
2. 兼容路径：`knowledge_tool.py` 中旧版 routed retrieval。

本文后续重点说明当前优先使用的模块化 RAG。

---

## 8. 模块化 RAG 编排

文件：`app/retrieval/orchestrator.py`

函数：`retrieve()`，当前在第 32 行附近。

输入：

| 参数 | 类型 | 说明 |
|---|---|---|
| `query` | `str` | 检索问题。 |
| `options` | `RetrievalOptions | None` | 检索配置；为空时从 `config` 加载。 |

输出：

- `RetrievalBundle`：包含主命中、补充命中、warnings、metadata。

主流程：

```text
retrieve(query, options=None)
  ↓
load_retrieval_options_from_config()
  ↓
RetrievalDiagnostics(request_id)
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

异常降级策略：

| 阶段 | 异常处理 |
|---|---|
| Query Understanding | 记录 warning，降级为 `strategy="none"`，保留语言判断。 |
| Recall | 记录 warning，返回空候选列表。 |
| Reranker | 记录 warning，尝试 `lexical_fallback()`。 |
| Evidence | 记录 warning，使用 `_fallback_bundle()` 直接取 rerank 后 top candidates。 |
| Trace 写入 | 写入失败只记录 warning，不影响用户回答。 |

---

## 9. 数据结构

文件：`app/retrieval/schemas.py`

### 9.1 QueryAnalysis

字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `query` | `str` | 检索问题。 |
| `strategy` | `str` | Query Understanding 策略。 |
| `intent_candidates` | `list[IntentCandidate]` | 意图候选。 |
| `doc_candidates` | `list[DocCandidate]` | 文档候选。 |
| `query_terms` | `list[QueryTerm]` | 关键词。 |
| `warnings` | `list[str]` | 警告信息。 |
| `language` | `str` | `zh` 或 `en`。 |

派生属性：

- `primary_intent`：最高分意图。
- `primary_doc_id`：最高分文档。
- `all_intents`：去重后的所有意图。
- `all_doc_ids`：去重后的所有文档 ID。

### 9.2 RecallCandidate

字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `result` | `SearchResult` | 原始搜索结果。 |
| `chunk_id` | `str` | 稳定 chunk 标识。 |
| `recall_channels` | `set[str]` | 命中的召回通道，例如 vector、bm25、scan。 |
| `channel_scores` | `dict[str, float]` | 各通道归一化分。 |
| `merged_score` | `float` | 合并后的召回分。 |
| `diagnostics` | `dict` | rerank 分、词法分、权重等诊断字段。 |

### 9.3 RerankResult

字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `candidates` | `list[RecallCandidate]` | 重排后的候选。 |
| `provider` | `str` | reranker provider。 |
| `fallback_used` | `bool` | 是否使用降级。 |
| `warnings` | `list[str]` | 重排警告。 |
| `score_field` | `str` | 排序分数字段名。 |

### 9.4 RetrievalBundle

字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `intent` | `str` | 本次检索主意图。 |
| `retrieval_stage` | `str` | 检索阶段标识，当前通常是 `hybrid_search`。 |
| `hits` | `list[SearchResult]` | 主证据。 |
| `support_hits` | `list[SearchResult]` | 补充证据。 |
| `warnings` | `list[str]` | 检索警告。 |
| `metadata` | `dict` | 给 API 层和 trace 使用的摘要元数据。 |

---

## 10. Query Understanding 当前真实流程：strategy=none

文件：`app/retrieval/query_understanding.py`

函数：`analyze_query()`，当前在第 53 行附近。

输入：

- `query: str`
- `options: RetrievalOptions`
- `diagnostics: RetrievalDiagnostics`

输出：

- `QueryAnalysis`

### 10.1 当前配置

当前 `app/config.py` 中：

```text
rag_intent_strategy = "none"
```

因此当前真实执行分支是：

```text
strategy == "none"
  ↓
QueryAnalysis(
  query=query,
  strategy="none",
  intent_candidates=[],
  doc_candidates=[],
  query_terms=extract_terms(query),
  language=zh/en
)
```

也就是说，当前默认不主动生成意图候选和文档候选，只做两件事：

1. 识别语言：中文返回 `zh`，否则返回 `en`。
2. 抽取关键词：调用 `extract_terms(query)`。

对后续召回的影响：

- `intent_candidates=[]`，所以 `QueryAnalysis.primary_intent` 为 `None`。
- `doc_candidates=[]`，所以 `QueryAnalysis.primary_doc_id` 为 `None`。
- recall 阶段 `_primary_intent()` 会把空意图兜底成 `general`。
- recall 阶段 `_all_doc_ids()` 会返回空列表，因此不按任何 `doc_id` 过滤。
- `query_terms` 仍会参与 query variants、scan fallback 判断、词法重排打分。

### 10.2 none 下的关键词抽取

函数调用：

```text
analyze_query()
  ↓
extract_terms(query)
  ↓
_legacy_extract_query_terms(query)
  ↓
app/tools/knowledge_tool.py::extract_query_terms(query)
```

中文问题的关键词抽取逻辑：

1. 使用 `jieba.analyse.extract_tags(query, topK=8)` 做 TF-IDF 关键词抽取。
2. 如果 query 中出现图片 ID，例如 `Manual01_5`，通过 `extract_pic_id()` 补到关键词最前面。
3. 调用 `match_profile_terms_in_query()` 从已索引文档 profile 中补充领域词。
4. 去重后最多保留 16 个词。

英文问题的关键词抽取逻辑：

1. 使用正则提取英文 / 数字 token。
2. 去掉英文停用词。
3. 保留 `not`、`no`、`use`、`set`、`run`、`turn`、`change`、`check`、`open`、`close`、`start`、`stop` 等对操作问题有意义的词。
4. 去重后最多保留 16 个词。

### 10.3 none 下的 QueryAnalysis 输出样例

例如用户问题：

```text
电钻电池怎么充电？
```

当前 `none` 策略下，输出结构是类似这样的：

```text
QueryAnalysis(
  query="电钻电池怎么充电？",
  strategy="none",
  language="zh",
  intent_candidates=[],
  doc_candidates=[],
  query_terms=[QueryTerm("电钻"), QueryTerm("电池"), QueryTerm("充电"), ...],
  warnings=[]
)
```

注意这里不会产生：

```text
primary_intent = procedure
primary_doc_id = manual_60e374ac
```

即使 query 里出现“怎么”“充电”等词，当前模块化 Query Understanding 也不会把它作为 `procedure` 意图输出；后续召回仍按 `general` 路由执行。

### 10.4 代码支持但当前未启用的策略

代码支持的策略：

```text
none, profile, rules, summary, llm, hybrid
```

函数：`resolve_enabled_strategies()`，当前在第 102 行附近。

| 策略 | 函数 | 算法 / 方法 |
|---|---|---|
| `none` | `analyze_query()` 内部直接处理 | 只抽关键词，不做意图和文档推断。 |
| `profile` | `analyze_with_profile()` | 根据 indexed document profiles 匹配 doc_name、doc_id、pic_prefix、profile terms。 |
| `rules` | `analyze_with_rules()` | 使用保守关键词规则判断意图，并使用硬编码 doc 映射。 |
| `summary` | `analyze_with_summary()` | 读取离线摘要索引，按 query terms 和摘要重合度推断文档。 |
| `llm` | `analyze_with_llm()` | 调用可注入的 `LLM_ANALYZER`，当前没有默认构造。 |
| `hybrid` | `resolve_enabled_strategies()` | 组合 profile、rules、summary、llm。 |

这些策略只有当 `.env` 或环境变量把 `RAG_INTENT_STRATEGY` 改成对应值时才会进入真实主链路。当前 `none` 下，它们不会参与在线召回路由。

### 10.5 当前未启用策略中的意图类别

当前系统中常见意图：

| 意图 | 含义 | 后续倾向 |
|---|---|---|
| `component` | 部件、接口、按钮、配件 | 优先 component 类 chunk。 |
| `procedure` | 安装、使用、调节、维护步骤 | 优先 procedure 类 chunk。 |
| `legal` | 保修、声明、法规 | 优先 legal 类 chunk。 |
| `safety` | 安全、警告、注意事项 | 优先 safety 类 chunk。 |
| `troubleshooting` | 故障、异常、排查 | 优先 troubleshooting 类 chunk。 |
| `image_trace` | 图片、图示、PIC、配图 | 优先 auxiliary / image / component。 |
| `ocr_audit` | OCR、原文、识别、缺失 | 优先 OCR / overview，并触发 scan。 |
| `overview` | 概览、目录、章节 | 优先 overview / support。 |
| `general` | 普通问题 | 走通用 primary 路由。 |

当前 `none` 下，上表这些意图不会由 `query_understanding.analyze_query()` 输出给 recall。唯一需要注意的是：如果 DashScope reranker 失败并进入 `lexical_fallback()`，fallback 内部会单独调用 `knowledge_tool.detect_intent(query)` 来辅助本地词法打分；这只影响 fallback 排序分，不改变 recall 的 `general` 路由，也不改变 evidence 中的 intent。

### 10.6 当前未启用策略中的分数融合

函数：`_fuse_scores()`，当前在第 649 行附近。

多个策略给同一个 intent 或 doc_id 打分时，使用概率并集式融合：

```text
fused = 1 - Π(1 - score_i)
```

特点：

- 多个策略都支持同一候选时，最终分更高。
- 分数被限制在 `[0, 1]`。

当前 `none` 下不会运行多策略融合，因为没有启用 profile/rules/summary/llm。

---

## 11. 多路召回策略

文件：`app/retrieval/recall.py`

主函数：`recall_candidates()`，当前在第 97 行附近。

输入：

- `query: str`
- `analysis: QueryAnalysis`
- `options: RetrievalOptions`
- `diagnostics: RetrievalDiagnostics`

输出：

- `list[RecallCandidate]`

主流程：

```text
recall_candidates()
  ↓
读取 analysis.language，当前来自 strategy=none 的 zh/en
  ↓
如果开启 vector，执行 vector_recall()
  ↓
如果开启 BM25，执行 bm25_recall()
  ↓
merge_recall_results(vector + bm25)
  ↓
should_trigger_scan()
  ↓
必要时执行 scan_recall()
  ↓
merge_recall_results(vector + bm25 + scan)
  ↓
写入 diagnostics.trace 和 diagnostics.summary
  ↓
return RecallCandidate 列表
```

### 11.1 none 下的真实召回路由

函数：`_recall_routes()`，当前在第 447 行附近。

在通用代码设计里，路由由两部分组成：

1. Query Understanding 得到的 intent。
2. Query Understanding 得到的高置信 doc_id。

但当前 `RAG_INTENT_STRATEGY=none`，所以：

```text
analysis.intent_candidates = []
analysis.doc_candidates = []
```

于是实际执行是：

```text
_all_intents(analysis)
  ↓
没有 all_intents，也没有 primary_intent
  ↓
返回 ["general"]

_all_doc_ids(analysis)
  ↓
没有 doc_candidates，也没有 primary_doc_id
  ↓
返回 []

_recall_routes()
  ↓
route_intent = "general"
  ↓
doc_ids = []
  ↓
添加全库安全路径: doc_id=None
```

函数：`_route_for_intent()`，当前在第 470 行附近。

它会读取 `app/tools/knowledge_tool.py` 中的 `INTENT_STAGE_CONFIG`。

当前 `general` 的第一个 stage 是：

```text
{"name": "primary_route", "tiers": ["primary"], "families": None}
```

因此 none 下最终只有一条真实召回路由：

```text
{
  "doc_id": None,
  "tiers": ("primary",),
  "chunk_families": ()
}
```

实际含义：

| 路由字段 | 当前值 | 含义 |
|---|---|---|
| `doc_id` | `None` | 不限定具体手册，全库检索。 |
| `retrieval_tiers` | `["primary"]` | 只查主内容 chunk。 |
| `chunk_families` | 空 | 不按 component/procedure/safety 等 family 过滤。 |
| `chunk_types` | `None` | `resolve_chunk_types()` 返回空，因此不限制具体 chunk type。 |

所以当前真实召回不是“识别意图后查 procedure/safety/component 专用 chunk”，而是“统一查全库 primary chunk，再靠向量、BM25、scan 和 reranker 排序”。

### 11.2 Vector Recall

函数：`vector_recall()`，当前在第 159 行附近。

调用：

```text
vector_search_service.search_similar_documents()
```

输入：

| 参数 | 来源 | 说明 |
|---|---|---|
| `query` | query variants | 原 query 或扩展 query。 |
| `top_k` | `_recall_fetch_limit()` | 至少 8，通常取 `reranker_top_n`。 |
| `doc_id` | recall route | 当前 none 下为 `None`，即全库。 |
| `language` | QueryAnalysis | 中英文过滤。 |
| `retrieval_tiers` | recall route | 当前 none 下为 `["primary"]`。 |
| `chunk_types` | family 解析 | 当前 none 下为 `None`。 |

当前 none 下的算法：

1. 使用唯一 route：`doc_id=None`、`retrieval_tiers=["primary"]`、`chunk_types=None`。
2. 对每个 route，取 query variants 的前 3 个。
3. 每个 variant 调用 Milvus 向量检索。
4. 按 active docs 过滤。
5. 去重。
6. 用本地词法规则做一次轻量 rerank。
7. 返回 fetch limit 内候选。

### 11.3 none 下的 query variants

函数：`knowledge_tool.expand_queries()`，当前在第 1064 行附近。

因为当前 recall intent 被兜底成 `general`，所以 query expansion 也走 `intent == "general"` 分支。

中文问题：

```text
variants = [
  原始 query,
  去掉“怎么 / 如何 / 请问 / 这个”等前缀和“是什么 / 在哪 / 怎么用”等后缀后的紧凑 query,
  query_terms 拼接字符串,
  长度 >= 2 的 query_terms 拼接字符串
]
```

英文问题：

```text
variants = [
  原始 query,
  query_terms 拼接字符串
]
```

然后去重，vector recall 只取前 3 个 variant 搜索。

当前 none 下不会追加这些 intent 专用扩展：

```text
procedure:  步骤 安装 设置 使用
component:  部件 位置 说明
safety:     安全 注意 警告 电源
troubleshooting: 故障 原因 排查 解决
image_trace: 图片 配图 / PIC 对应
ocr_audit: OCR 原文 识别错误 缺失
```

这些只有当 recall intent 真的变成对应类别时才会进入 variants；当前 `none` 主链路不会发生。

### 11.4 Vector Search 内部

文件：`app/services/vector_search_service.py`

函数：`search_similar_documents()`，当前在第 89 行附近。

流程：

```text
search_similar_documents(query, filters)
  ↓
vector_embedding_service.embed_query(query)
  ↓
milvus_manager.get_collection()
  ↓
vector_store_manager.build_metadata_filter_expr(...)
  ↓
collection.search(
    data=[query_vector],
    anns_field="vector",
    metric_type="COSINE",
    nprobe=10,
    output_fields=["id", "content", "metadata"]
  )
  ↓
SearchResult(id, content, score=hit.distance, metadata)
```

注意：

- Milvus 返回的 `hit.distance` 在后续 merge 阶段会转成相似度方向。
- 返回内容优先使用 `metadata["text"]`，没有时用 `content`。

### 11.5 BM25 Recall

函数：`bm25_recall()`，当前在第 195 行附近。

调用：

```text
_bm25_provider.search()
```

如果 provider 没有初始化，返回空列表并记录：

```text
BM25 recall unavailable: no provider configured
```

BM25 provider：

文件：`app/retrieval/bm25_provider.py`

类：`JiebaBM25Provider`

算法：

- 中文：使用 `jieba.cut()` 分词。
- 英文：使用正则抽 token，并移除英文停用词，但保留 `not/no/use/set/run/...` 等有操作意义的词。
- BM25：使用 `rank_bm25.BM25Okapi`。
- 索引文本：从 `metadata.text`、`content`、`title`、`section_path`、`index_text`、`summary`、`keywords` 等字段拼接。

搜索流程：

```text
JiebaBM25Provider.search(query, filters)
  ↓
_tokenize(query)
  ↓
bm25.get_scores(query_tokens)
  ↓
过滤 doc_id / retrieval_tier / chunk_type / language
  ↓
按 BM25 score 降序排序
  ↓
返回 top_k SearchResult
```

当前 none 下，BM25 使用的过滤条件与 vector recall 一致：

```text
doc_id=None
retrieval_tiers=["primary"]
chunk_types=None
language=zh/en
```

### 11.6 Scan Fallback

函数：`scan_recall()`，当前在第 253 行附近。

Scan fallback 不是向量搜索，而是 metadata 全量查询后做本地词法排序。

流程：

```text
scan_recall()
  ↓
vector_search_service.query_all_documents(filters)
  ↓
deduplicate_results()
  ↓
_rerank_scan_results()
  ↓
返回 fetch_k 候选
```

触发函数：

```text
should_trigger_scan()
```

当前触发条件：

1. `RAG_ENABLE_SCAN_FALLBACK=false` 时不触发。
2. 当前无候选时触发。
3. intent 是 `image_trace` 或 `ocr_audit` 时触发；当前 none 下 recall intent 是 `general`，所以这条通常不触发。
4. query 包含关键术语组合时触发。
5. top candidate 的词法分低于阈值时触发；当前 `general` 阈值是 24。
6. 没有 query_terms 时，如果 top candidate 的 `merged_score < 0.18`，触发。

Scan 通道权重：

```text
SCAN_CHANNEL_WEIGHT = 0.2
```

当前 none 下，scan 查询也使用同一条 `general` 路由：

```text
doc_id=None
retrieval_tiers=["primary"]
chunk_types=None
language=zh/en
```

### 11.7 召回合并打分

函数：`merge_recall_results()`，当前在第 332 行附近。

通道权重：

```text
vector = 0.6
bm25 = 0.4
scan = 0.2
```

合并流程：

1. 每个通道内部先归一化分数。
2. 按 `chunk_id` 去重合并。
3. 同一个 chunk 被多个通道召回时，保留每个通道最高分。
4. 计算 `merged_score`。
5. 按 `merged_score` 降序排序。

分数公式：

```text
merged_score = Σ(channel_score[channel] × channel_weight[channel])
```

归一化函数：

```text
normalize_channel_scores()
```

归一化规则：

| 通道 | 原始分处理 |
|---|---|
| `vector` | 先把距离转相似度：`1 / (1 + score)`，再 min-max 归一化。 |
| `bm25` | 直接对 BM25 分做 min-max 归一化。 |
| `scan` | 不用原始分，按排名生成递减分：rank scores。 |
| 所有分相等 | 用 rank scores 避免全 0。 |

---

## 12. Reranker 重排策略

文件：`app/retrieval/reranker.py`

主函数：`rerank_candidates()`，当前在第 147 行附近。

输入：

- `query`
- `candidates: list[RecallCandidate]`
- `options`
- `diagnostics`

输出：

- `RerankResult`

### 12.1 当前默认 provider

当前配置：

```text
RAG_RERANKER_PROVIDER=dashscope
RAG_RERANKER_MODEL=qwen3-rerank
RAG_RERANKER_ENDPOINT=https://dashscope.aliyuncs.com/compatible-api/v1/reranks
RAG_RERANKER_TIMEOUT_MS=3000
RAG_RERANKER_TOP_N=32
```

### 12.2 DashScope Reranker

类：`DashScopeReranker`，当前在第 70 行附近。

请求体：

```json
{
  "model": "qwen3-rerank",
  "query": "用户检索问题",
  "documents": ["候选 chunk 的 index_text 或 content", "..."],
  "top_n": 32
}
```

候选文本选择函数：

```text
_candidate_rerank_text()
```

优先级：

1. `metadata["index_text"]`
2. `result.content`
3. `metadata["text"]`

返回处理：

1. 从 DashScope 响应里读取 `results`。
2. 按返回的 `index` 找回原候选。
3. 读取 `relevance_score` / `score` / `reranker_score`。
4. 写入 `candidate.diagnostics["reranker_score"]`。
5. 写入 `candidate.diagnostics["reranker_rank"]`。
6. 返回模型重排后的候选。

### 12.3 超时控制

函数：`run_with_timeout()`，当前在第 291 行附近。

实现方式：

- 用 daemon thread 执行同步 rerank 调用。
- `join(timeout_seconds)` 等待。
- 超时则抛 `RerankerTimeoutError`。

### 12.4 Lexical Fallback

函数：`lexical_fallback()`，当前在第 240 行附近。

触发条件：

- provider 是 `none` 或 `lexical`。
- DashScope API key 缺失。
- DashScope 请求超时。
- DashScope 返回空结果。
- DashScope 响应格式无效。
- 任意 reranker 异常。

算法：

1. 抽取 query terms。
2. 调用 `knowledge_tool.detect_intent(query)` 做本地启发式 intent 判断。
3. 对每个候选调用 `knowledge_tool.lexical_score()`。
4. 把分数写入 `candidate.diagnostics["lexical_score"]`。
5. 按词法分降序排序。
6. 标记 `fallback_used=True`，provider 为 `lexical`。

注意：这一步的 intent 只服务于 reranker 失败后的本地词法打分。当前 `RAG_INTENT_STRATEGY=none` 下，recall 阶段已经按 `general` 路由完成，不会因为 fallback 内部判断出 `procedure` 或 `safety` 就重新召回。

---

## 13. Evidence 证据组织策略

文件：`app/retrieval/evidence.py`

主函数：`build_retrieval_bundle()`，当前在第 44 行附近。

输入：

- `query`
- `analysis`
- `rerank_result`
- `options`
- `diagnostics`

输出：

- `RetrievalBundle`

主流程：

```text
build_retrieval_bundle()
  ↓
top_k = options.top_k，默认 3
  ↓
primary_hits = rerank_result.candidates[:top_k].result
  ↓
language_filter()
  ↓
fetch_parent_support_hits(primary_hits)
  ↓
summarize_top_hits()
  ↓
写入 trace:
    evidence_primary_hits
    evidence_support_hits
    evidence_images
  ↓
return RetrievalBundle(...)
```

当前 none 下的 evidence intent：

```text
intent = analysis.primary_intent or "general"
```

因为 `analysis.intent_candidates=[]`，所以 `analysis.primary_intent=None`，最终 `RetrievalBundle.intent="general"`。这也是证据文本里 `[Retrieval intent] general` 的来源。

### 13.1 主证据

主证据来自 rerank 后的前 `RAG_TOP_K` 个候选。

当前默认：

```text
RAG_TOP_K=3
```

### 13.2 补充上下文

函数：`fetch_parent_support_hits()`，当前在第 97 行附近。

作用：

- 对每个 primary hit，读取 `metadata["parent_chunk_id"]`。
- 只对 `retrieval_tier == "primary"` 的 hit 取 parent。
- 调用 `vector_search_service.query_documents()` 查询 support 层父 chunk。
- 按 parent id 原顺序返回。

输入：

- primary hits
- diagnostics
- language

输出：

- support hits

查询条件：

```text
retrieval_tiers = ["support"]
chunk_ids = parent_ids
language = 当前 query 语言
limit = parent_ids 数量
```

### 13.3 图片证据

函数：

- `image_trace()`，当前在第 390 行附近。
- `image_reference_lines()`，当前在第 571 行附近。
- `display_path_list()`，当前在第 561 行附近。

处理内容：

1. 从 primary hits 和 support hits 的 metadata 里读取：
   - `pic_ids`
   - `image_paths`
2. 把本机绝对路径转换成项目相对路径。
3. 生成模型可直接引用的 markdown 图片行。

目标格式：

```markdown
![Manual01_5](data/manuals/raw/.../Manual01_5.jpg)
```

`RagAgentService._ensure_image_placeholders()` 还会在非流式答案返回前修正空 alt 图片：

```text
![](path) → ![从 path 推断出的 pic_id](path)
```

### 13.4 证据文本格式

函数：`format_bundle()`，当前在第 278 行附近。

输出给模型的 context 大致包括：

```text
[Evidence schema] retrieval_evidence_v1
[Retrieval intent] ...
[Retrieval stage] ...
[Evidence summary] ...
[Retrieval warnings]
[Primary hits]
[Support context]
```

每个参考资料包括：

- 手册名。
- 标题。
- 章节路径。
- tier / chunk type。
- chunk ID。
- parent chunk ID。
- source lines。
- picture IDs。
- 图片 markdown 引用。
- chunk 正文。

---

## 14. 最终回答与 API 元数据

### 14.1 工具结果回到 Agent

`retrieve_knowledge()` 返回的是：

```text
(context, docs)
```

其中 `context` 会进入 Agent 消息，让大模型基于证据回答。

回答生成后，`RagAgentService.query()`：

1. 读取 Agent 最后一条 message。
2. 取 `last_message.content`。
3. 调用 `_ensure_image_placeholders()` 修正图片 markdown。
4. 返回 answer。

### 14.2 检索 metadata

工具执行时：

```text
knowledge_tool.set_last_retrieval_metadata(bundle.metadata)
```

Agent 执行后：

```text
RagAgentService._read_last_retrieval_metadata()
```

API 返回前：

```text
rag_agent_service.get_last_retrieval_metadata(session_id)
```

最终 metadata 会被 `sanitize_summary_metadata()` 过滤后放进比赛接口响应。

metadata 主要包含：

- request_id。
- top hits。
- recall channels。
- reranker provider。
- reranker fallback 状态。
- warnings。
- degraded 状态。

---

## 15. 知识库入库流程

知识库入库是在线 RAG 的前置流程。当前主线使用结构化手册 chunk JSONL。

### 15.1 入库入口

脚本：`scripts/index_manual_chunks.py`

主函数：

```text
main()
  ↓
vector_index_service.index_manual_chunks(args.directory)
```

默认目录：

```text
./data/manuals/chunks
```

支持：

- 单个 `.jsonl` 文件。
- 包含 `.jsonl` 文件的目录。
- 包含嵌套 `chunks.jsonl` 的目录。

### 15.2 读取 chunk 文件

文件：`app/services/vector_index_service.py`

函数：`index_manual_chunks()`，当前在第 94 行附近。

流程：

```text
index_manual_chunks(directory_path)
  ↓
检查路径存在
  ↓
_collect_manual_chunk_files()
  ↓
_index_files()
  ↓
index_single_file(file_path)
```

函数：`index_single_file()`，当前在第 175 行附近。

如果文件是 `.jsonl`：

```text
_index_chunk_jsonl(path)
```

### 15.3 JSONL 转 Document

函数：`_index_chunk_jsonl()`，当前在第 208 行附近。

流程：

```text
_index_chunk_jsonl(path)
  ↓
_load_chunk_records(path)
  ↓
读取 doc_id
  ↓
vector_store_manager.delete_by_doc_id(doc_id)
  ↓
vector_store_manager.delete_by_source(path)
  ↓
_build_documents_from_chunk_records(records, path)
  ↓
vector_store_manager.add_documents(documents)
  ↓
_refresh_retrieval_metadata_cache()
```

删除策略：

- 如果 chunk 文件里有 `doc_id`，先删除 Milvus 中同 doc_id 的旧数据。
- 同时按 source 路径删除旧数据，避免重复入库。

函数：`_build_structured_document()`，当前在第 363 行附近。

结构化 chunk 转成 LangChain Document：

| Document 部分 | 来源 |
|---|---|
| `page_content` | record 的 `text`。 |
| `metadata.index_text` | record 的 `index_text`，没有则使用 `text`。 |
| `metadata.doc_id` | record 的 `doc_id`。 |
| `metadata.doc_name` | record 的 `doc_name` 或文件名推导。 |
| `metadata.language` | record 的 `language`。 |
| `metadata.chunk_id` | record 的 `chunk_id`。 |
| `metadata.retrieval_tier` | primary / support / auxiliary。 |
| `metadata.chunk_type` | chunk 类型。 |
| `metadata.section_path` | 章节路径。 |
| `metadata.parent_chunk_id` | 父级 support chunk ID。 |
| `metadata.pic_ids` | 图片 ID。 |
| `metadata.image_paths` | 图片路径。 |
| `metadata.source_lines` | 原文行号。 |
| `metadata.source_quality` | 源数据质量。 |
| `metadata.source_issue_flags` | OCR / 缺失 / 截断等风险标记。 |

图片路径处理：

- 先从 `metadata_image_path` chunk 中提取图片绝对路径。
- 通过 `pic_ids` 建立图片 ID 到路径的映射。
- 正文 chunk 如果带 `pic_ids`，会关联对应 `image_paths`。
- 路径会尽量转成项目相对路径。

### 15.4 写入 Milvus

文件：`app/services/vector_store_manager.py`

函数：`add_documents()`，当前在第 66 行附近。

流程：

```text
add_documents(documents)
  ↓
为每个 Document 构造 id
  ↓
按 EMBEDDING_BATCH_SIZE=10 分批
  ↓
_get_embedding_text(document)
  ↓
vector_embedding_service.embed_documents(embedding_inputs)
  ↓
collection.insert(rows)
  ↓
collection.flush()
```

重要策略：

| 函数 | 策略 |
|---|---|
| `_build_document_id()` | 优先使用 `metadata["chunk_id"]` 作为 Milvus 主键；超长则 hash。 |
| `_get_embedding_text()` | 优先使用 `metadata["index_text"]` 做 embedding；没有则用 `page_content`。 |
| `_truncate_varchar_bytes()` | 写入 `content` 前按 UTF-8 字节截断，避免超过 Milvus varchar 限制。 |
| `_sanitize_metadata()` | metadata 经 JSON 序列化清洗，保证可写入 Milvus JSON 字段。 |

### 15.5 Embedding

文件：`app/services/vector_embedding_service.py`

类：`DashScopeEmbeddings`

函数：

- `embed_documents()`：批量文档 embedding。
- `embed_query()`：单条 query embedding。

当前模型：

```text
text-embedding-v4
```

当前维度：

```text
1024
```

### 15.6 入库后对检索的影响

入库写入的是：

```text
id + vector + content + metadata
```

在线检索时：

- vector recall 用 `vector` 做相似度搜索。
- BM25 provider 启动时从 Milvus 拉取 `content + metadata` 建进程内词法索引。
- metadata filter 使用 `doc_id`、`language`、`retrieval_tier`、`chunk_type`、`chunk_id`。
- evidence 阶段用 metadata 组织手册名、章节、图片、行号和 source issue。

注意：

- BM25 是进程内索引。
- 如果重新入库后不重启服务，也没有重新执行 `init_bm25_provider()`，BM25 可能仍使用旧索引。
- 向量检索直接查询 Milvus，所以新写入 Milvus 后向量侧可用，前提是 collection 已 flush / load。

---

## 16. 核心算法汇总

| 模块 | 算法 / 方法 | 代码位置 | 说明 |
|---|---|---|---|
| 文本向量化 | DashScope `text-embedding-v4` | `vector_embedding_service.py` | 文档和 query 转 1024 维向量。 |
| 向量检索 | Milvus COSINE + HNSW | `milvus_client.py`, `vector_search_service.py` | 根据 query 向量找相似 chunk。 |
| Query Terms | jieba TF-IDF / 英文 token / PIC 提取 | `knowledge_tool.py::extract_query_terms()` | 为扩展 query、scan、词法分提供关键词。 |
| Query Understanding 当前真实策略 | `strategy=none` | `query_understanding.py::analyze_query()` | 只输出 `language` 和 `query_terms`，不输出 intent/doc candidates。 |
| 当前真实召回路由 | `general + primary + doc_id=None` | `recall.py::_recall_routes()` | 全库检索 primary chunk，不按手册或意图 family 过滤。 |
| 未启用意图策略 | rules / profile / summary / llm / hybrid | `query_understanding.py` | 代码支持，但当前 `RAG_INTENT_STRATEGY=none` 下不参与主链路。 |
| Query Expansion | `general` 分支扩展 | `knowledge_tool.py::expand_queries()` | 当前只做原 query、紧凑 query、query_terms 拼接；不追加 procedure/component 等意图专用词。 |
| BM25 | jieba + rank_bm25 | `bm25_provider.py` | 当前按 `doc_id=None`、`retrieval_tiers=["primary"]`、`language=zh/en` 召回。 |
| Scan Fallback | Milvus metadata full query + lexical rerank | `recall.py::scan_recall()` | 当前按同一条 general 路由扫描，防止向量 / BM25 没召回到关键 chunk。 |
| 多路融合 | 归一化 + 加权求和 | `recall.py::merge_recall_results()` | vector 0.6，BM25 0.4，scan 0.2。 |
| Rerank | DashScope `qwen3-rerank` | `reranker.py::DashScopeReranker` | 对候选 chunk 做语义相关性重排。 |
| Rerank 降级 | lexical fallback | `reranker.py::lexical_fallback()` | reranker 不可用时本地词法排序；内部 intent 只用于打分，不改变 recall 路由。 |
| Evidence 扩展 | parent support chunk 查询 | `evidence.py::fetch_parent_support_hits()` | 让主命中有章节级补充上下文。 |
| 图片引用 | PIC ID + image path 配对 | `evidence.py::image_reference_lines()` | 输出可直接引用的 markdown 图片。 |

---

## 17. 降级与边界

当前系统有多层降级，保证某个组件失败时尽量不让整条问答链路直接失败。

| 失败点 | 降级行为 |
|---|---|
| MCP 工具不可用 | Agent 只使用本地工具继续运行。 |
| Query Understanding 异常 | 降级为 `strategy=none`。 |
| Vector recall 异常 | 记录 warning，继续 BM25 / scan。 |
| BM25 provider 未初始化 | 记录 warning，BM25 返回空，继续 vector / scan。 |
| Scan fallback 异常 | 记录 warning，使用已有 recall candidates。 |
| DashScope reranker 超时 / 报错 | 降级到 lexical fallback。 |
| Evidence 组织失败 | 使用 `_fallback_bundle()` 直接包装 top candidates。 |
| Trace 写入失败 | 只记录 warning，不影响回答。 |
| 图片 markdown alt 为空 | 非流式返回前尝试从路径补 pic_id。 |

系统边界：

- RAG 回答质量依赖 Milvus 中已入库的 chunk 质量。
- BM25 是进程内索引，服务启动后不会自动感知 Milvus 新入库数据。
- `RAG_INTENT_STRATEGY=none` 时，当前默认不做强意图和文档路由，只做通用召回。
- DashScope reranker 依赖外部 API；超时或额度问题会降低重排质量，但不会中断检索。
- 流式 `query_stream()` 当前主要输出模型文本 token；非流式链路对最终答案 metadata 和图片占位修正更完整。

---

## 18. 排查与验证入口

常用测试和诊断文件：

| 文件 | 用途 |
|---|---|
| `tests/retrieval/test_query_understanding.py` | Query Understanding 单元测试。 |
| `tests/retrieval/test_recall.py` | 多路召回与合并逻辑测试。 |
| `tests/retrieval/test_bm25_provider.py` | BM25 provider 测试。 |
| `tests/retrieval/test_reranker.py` | reranker 和 fallback 测试。 |
| `tests/retrieval/test_evidence.py` | Evidence 组织测试。 |
| `tests/retrieval/test_orchestrator.py` | 模块化检索编排测试。 |
| `tests/test_rag_agent_prompt.py` | Agent 提示词相关测试。 |
| `scripts/trace_retrieval_pipeline.py` | 生成检索过程 trace 报告。 |
| `scripts/run_public_question_regression.py` | 跑真实 Chat 回归，输出 JSON / Markdown 报告。 |

当 `APP_DEBUG=True` 或相关 debug 开关启用时，检索 trace 会写入：

```text
logs/retrieval_trace.jsonl
```

重点关注字段：

- `query_understanding`
- `recall.channels.vector`
- `recall.channels.bm25`
- `recall.channels.scan`
- `recall.normalized_scores`
- `recall.post_merge`
- `reranker.pre_rank`
- `reranker.post_rank`
- `reranker.rank_changes`
- `evidence_primary_hits`
- `evidence_support_hits`
- `evidence_images`
- `warnings`

---

## 19. 一句话总结

当前 RAG 系统的核心策略是：用 Agent 判断是否需要查手册；用模块化 RAG 管线把用户问题转成可检索 query；通过向量召回、BM25 召回和 scan fallback 获得候选；用 DashScope reranker 或本地词法 fallback 重排；最后组织成带来源、章节、图片和诊断信息的 evidence，让大模型基于证据生成答案。
