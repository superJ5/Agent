# 3f3c 相对 main 的项目代码改动报告

对比对象：

- 当前工作树：`E:\.codex\worktrees\3f3c\ai_agent_competition`
- 当前分支：`codex/rag-delivery`
- 当前 HEAD：`e5ae617eb6098f652a11c1d5a5e3aed97b57546c`
- 对比基准：`main`，`d3bb2c386f960489db455f87ba793255009e2edc`

对比范围：只比较项目代码和工程配置，不比较 `data/`、`reports/`、临时文件、测试输出、文档说明稿。

## 1. 总览

相对 `main`，`3f3c` 的代码改动集中在三件事：

- 增加比赛标准 `/chat` 输入输出接口。
- 把 RAG 从简单 topK 向量召回，改成标准 chunk 入库 + metadata 过滤 + 路由检索 + 父块补齐 + fallback。
- 增加 baseline、召回评估、检索链路追踪等测试脚本，方便和主线 RAG 做效果对比。

代码层面统计：

- 修改文件：12 个
- 新增文件：10 个
- 删除文件：0 个
- 代码统计：约 `5960` 行新增，`478` 行删除

## 2. 修改了什么

### 2.1 接口层

`app/api/chat.py`

- 新增无前缀的比赛标准接口：`POST /chat`。
- 新增 Bearer Token 校验逻辑。
- 新增统一 session_id 生成逻辑。
- 新增比赛格式响应：`code/msg/data.answer/data.session_id/data.timestamp`。
- 保留旧接口：`/api/chat` 和 `/api/chat_stream`，但内部改成兼容新请求模型。
- 流式响应逻辑被整理成 `_build_stream_response(...)`，减少重复代码。

`app/api/file.py`

- 新增 `POST /api/index_manual_chunks`。
- 用于把标准结构化 chunk JSONL 入库到 Milvus。

`app/main.py`

- 注册新的比赛标准路由：`competition_router`。
- 因此 `/chat` 可以直接访问，不需要 `/api` 前缀。

### 2.2 请求和响应模型

`app/models/request.py`

- `ChatRequest` 从旧字段 `Id/Question` 扩展为同时兼容：
  - 比赛字段：`question`、`images`、`session_id`、`stream`
  - 旧字段：`Id`、`Question`
- 增加图片字段格式校验，最多 3 张，要求 base64 data URL 格式。
- 增加空问题校验。

`app/models/response.py`

- 新增比赛标准响应模型：`CompetitionChatData`、`CompetitionChatResponse`。
- 旧的通用响应模型仍保留。

### 2.3 配置和 Agent

`app/config.py`

- 新增 `api_bearer_token` 配置，用于比赛接口鉴权。

`app/services/rag_agent_service.py`

- MCP 工具加载失败时不再直接阻断 Agent，而是降级为只使用本地工具。
- 系统提示词新增 RAG 使用规则：涉及手册、部件、步骤、保修、图片、OCR、原文追溯时，应优先调用 `retrieve_knowledge`。

### 2.4 标准 chunk 入库

`app/services/vector_index_service.py`

- 从只支持普通 txt/md 入库，扩展为支持标准结构化 chunk JSONL 入库。
- 默认标准 chunk 目录为：`./data/manuals/chunks`。
- 支持单个 `.jsonl` 文件入库。
- 支持目录下多个 `.jsonl` 文件入库。
- 支持递归发现嵌套的 `chunks.jsonl`。
- 识别标准字段：`doc_id`、`chunk_id`、`chunk_type`、`retrieval_tier`、`parent_chunk_id`、`section_path`、`pic_ids`、`text`、`index_text`。
- 能从 `metadata_image_path` 类 chunk 中解析图片绝对路径，形成 `pic_id -> image_path` 映射。
- 入库时把 `pic_ids`、`image_paths`、`source_file`、`source_lines`、`source_issue_flags` 等信息写入 metadata。
- 保留 legacy JSONL 兼容逻辑。
- 入库完成后会刷新检索侧的文档 profile 缓存。

### 2.5 向量存储层

`app/services/vector_store_manager.py`

- 从 LangChain 默认 `add_documents`，改为手动构造 Milvus 插入 rows。
- 使用 `chunk_id` 作为优先主键，方便后续按 chunk 精确定位。
- 使用 `index_text` 做 embedding 输入。
- 使用 `text/page_content` 保存原始展示内容，保留标题、换行、列表和 `<PIC:...>` 标记。
- 新增 metadata 清洗和字段长度控制。
- 新增 metadata 过滤表达式构造能力，支持：
  - `doc_id`
  - `retrieval_tier`
  - `chunk_type`
  - `chunk_id`
- 新增 `delete_by_doc_id(...)`，方便重建同一手册索引。

### 2.6 向量检索层

`app/services/vector_search_service.py`

- 原来是简单向量 topK 搜索。
- 现在支持在向量检索时附加 metadata 过滤：`doc_id`、`retrieval_tier`、`chunk_type`、`chunk_id`。
- 新增 `query_documents(...)`，可以不走向量相似度，只按 metadata 查询。
- 搜索结果返回时优先使用 metadata 中保存的原始 `text`，而不是只返回 Milvus content 字段。

### 2.7 路由检索核心

`app/tools/knowledge_tool.py`

这是主线 RAG 改动最大的文件。

原来逻辑基本是：

```text
用户问题 -> vector_store.as_retriever(topK) -> 拼上下文 -> 给大模型
```

现在改成：

```text
用户问题
-> 意图判断
-> 文档 profile 推断
-> 查询扩展
-> 按 doc_id / retrieval_tier / chunk_type 过滤向量召回
-> 本地词面重排
-> 命中 primary 后补 support 父块
-> 弱命中时走 expanded route 或 fallback scan
-> 输出结构化 evidence 上下文
```

新增能力包括：

- 意图分类：部件定位、操作步骤、法务/保修、图片追踪、OCR/源数据、概览、通用。
- 文档 profile：从库中现有 chunk metadata 反推文档、章节、图片前缀、chunk 类型族。
- 泛化 type 映射：不是只认固定 `chunk_type`，而是把新 type 映射到 component/procedure/legal/image/ocr/overview 等类型族。
- primary/support/auxiliary 分层检索。
- primary 命中后自动补 support 父块。
- OCR 和法务类问题增加后过滤，减少旁支噪声。
- 本地 lexical rerank，把标题、section_path、index_text、pic_ids、source issue 等因素纳入排序。
- fallback scan：当向量主路命中弱时，按 metadata 扫描候选 chunk 后本地重排。
- 输出 evidence payload：包含 `chunk_id`、`doc_id`、`title`、`section_path`、`retrieval_tier`、`chunk_type`、`parent_chunk_id`、`pic_ids`、`image_paths`、`source_file`、`source_lines`、`source_issue_flags` 等。

## 3. 新增了什么

### 3.1 标准 chunk 入库脚本

`scripts/index_manual_chunks.py`

- 命令行把标准 chunk JSONL 入库到 Milvus。
- 默认读取 `./data/manuals/chunks`。
- 支持传入单个 JSONL 文件或目录。

### 3.2 主线端到端测试脚本

`scripts/test_e2e_rag.py`

- 用于快速跑主线 RAG：问题 -> 检索 -> 大模型回答。

### 3.3 检索链路追踪脚本

`scripts/trace_retrieval_pipeline.py`

- 用于输出每个问题在检索链路里的阶段结果。
- 方便看清楚：意图是什么、走了哪个 route、召回了哪些 chunk、为什么失败或命中弱。

### 3.4 baseline 对照脚本

`scripts/build_baseline_chunks.py`

- 构造简单字符切分 baseline chunk。
- 保留 `<PIC:...>` 引用，用来和结构化 chunk 对比。

`scripts/build_langchain_baseline.py`

- 构造 LangChain 风格 baseline chunk。
- 可选直接入库。

`scripts/baseline_e2e_test.py`

- 独立 baseline E2E 测试。
- 使用独立 Milvus collection：`biz_baseline`。
- 不经过主线 `knowledge_tool.py`。
- 支持 index、test、drop、单问查询。

`scripts/eval_baseline_recall.py`

- 离线评估 baseline 或结构化 chunk 的召回效果。
- 主要用于看 Recall / Hit@K。

`scripts/compare_baseline_retrieval.py`

- 对比结构化 chunk 和字符切分 baseline 的检索效果。

`scripts/build_retrieval_evidence_baseline.py`

- 把当前主线检索 evidence 固化成基准输出，方便后续回归比较。

`scripts/analyze_failures.py`

- 读取 baseline 评估结果，辅助查看失败案例。

## 4. 删除了什么

代码文件层面：没有删除文件。

逻辑层面：主线检索不再主要依赖旧的简单 topK retriever，而是替换为 `knowledge_tool.py` 中的路由检索机制。

## 5. 工程配置改动

`.gitignore`

- 新增忽略 `reports/`、`tmp_competition_pdf/`、`scripts/__pycache__/`。
- 新增手册数据目录的忽略规则。
- 明确把部分数据清洗/检查类脚本标记为本地脚本，不作为正式提交内容。
- 忽略若干 chunk 规则文档和计划文档，避免把讨论过程产物混入交付代码。

依赖文件：

- `pyproject.toml` 未变化。
- `uv.lock` 未变化。
- `Makefile` 未变化。
- `vector-database.yml` 未变化。
- `start-windows.bat`、`stop-windows.bat` 未变化。

## 6. 这版代码的定位

这版 `3f3c` 代码不是单纯的 baseline 实验分支，而是已经把主线 RAG 接入能力落到了交付分支上：

- 有比赛入口 `/chat`。
- 有标准 chunk 入库路径。
- 有主线 RAG 路由检索。
- 有 baseline 和 trace 脚本用于对照测试。

所以它比 `main` 多出来的核心价值是：可以让队友用标准 chunk 数据入库，再通过 `/chat` 做手册类问题问答，并能用脚本追踪召回问题。