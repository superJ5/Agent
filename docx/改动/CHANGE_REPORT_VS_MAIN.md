# 当前代码相对 main 的项目代码改动报告

对比范围：只看项目代码，不统计数据文件、报告输出、临时 PDF、向量库内容。

当前工作树状态：`HEAD` 与 `main` 指向同一提交，但工作区存在未提交改动。

## 1. 改动总览

相对 `main`，当前项目代码主要改了 RAG 入库、向量检索、路由检索三块：

- 修改 `app/services/vector_index_service.py`
- 修改 `app/services/vector_store_manager.py`
- 修改 `app/services/vector_search_service.py`
- 修改 `app/tools/knowledge_tool.py`
- 新增未跟踪脚本 `scripts/index_manual_chunks.py`

代码文件删除：未发现。

## 2. 新增了什么

### 2.1 标准 chunk 入库能力

新增了面向结构化手册 chunk 的入库流程：

- 默认读取 `./data/manuals/chunks`
- 支持单个 `.jsonl` 文件
- 支持目录下直接放多个 `.jsonl`
- 支持递归发现嵌套的 `chunks.jsonl`
- 能识别标准字段：`doc_id`、`chunk_id`、`chunk_type`、`retrieval_tier`、`parent_chunk_id`、`section_path`、`pic_ids`、`text`、`index_text`
- 能把图片路径类 chunk 解析成 `pic_id -> image_path` 映射，并写入 metadata

对应文件：`app/services/vector_index_service.py`

### 2.2 入库脚本

新增 `scripts/index_manual_chunks.py`，用于命令行把标准 chunk 灌入 Milvus：

```bash
python scripts/index_manual_chunks.py --directory ./data/manuals/chunks
```

也可以传入单个 chunk 文件路径。

### 2.3 更强的 Milvus metadata 过滤

新增 metadata 过滤表达式构造能力，支持按以下字段过滤：

- `doc_id`
- `retrieval_tier`
- `chunk_type`
- `chunk_id`

同时新增按 `doc_id` 删除已入库文档的能力，方便替换同一本手册的 chunk。

对应文件：`app/services/vector_store_manager.py`

### 2.4 index_text 嵌入，text 保留展示结构

入库时改成：

- 用 `index_text` 做 embedding 输入，提升检索表达质量
- 用 `text` 作为 `content/page_content` 保存，保留标题、换行、列表、`<PIC:...>` 等原始图文结构

这点对后续回答时恢复图文关系很重要。

对应文件：`app/services/vector_store_manager.py`、`app/services/vector_index_service.py`

### 2.5 查询时支持 metadata-only 查询

新增 `query_documents(...)`，可以不走向量相似度，只按 metadata 查询 chunk。

用途包括：

- 补父块
- fallback 扫描
- 查指定 chunk
- 查指定 `doc_id`、`retrieval_tier`、`chunk_type`

对应文件：`app/services/vector_search_service.py`

### 2.6 路由检索核心

`app/tools/knowledge_tool.py` 从原来的普通向量召回，改成了路由检索：

- 先判断问题意图：部件、步骤、法务、OCR、图片追踪、概览、通用
- 再根据意图选择优先检索的 `retrieval_tier` 和 `chunk_type`
- 优先召回 primary
- 命中 primary 后补 support 父块
- 对结果做本地词面重排
- 如果主路结果弱，再进入 expanded route 或 fallback scan
- 输出检索证据 payload，包含命中 chunk、metadata、图片 ID、图片路径、source 信息、warnings

这是当前主线 RAG 的核心文件。

## 3. 修改了什么

### 3.1 `vector_index_service.py`

从“上传目录里的 txt/md 普通入库”，扩展为同时支持：

- 普通文本入库
- 标准结构化 chunk JSONL 入库
- chunk 字段校验和兼容
- 图片路径 metadata 构建
- legacy JSONL 兼容

### 3.2 `vector_store_manager.py`

从 LangChain Milvus 的普通 `add_documents`，改成更可控的插入方式：

- 手动分批 embedding
- 手动构造 Milvus rows
- `chunk_id` 优先作为主键
- metadata JSON 序列化清洗
- 构造 Milvus metadata 过滤表达式
- 新增 `delete_by_doc_id`

### 3.3 `vector_search_service.py`

从简单 topK 向量搜索，扩展为：

- 支持 `doc_id` 过滤
- 支持 `retrieval_tier` 过滤
- 支持 `chunk_type` 过滤
- 支持 `chunk_id` 过滤
- 支持 metadata 查询
- 搜索返回时优先使用 metadata 中保留的原始 `text`

### 3.4 `knowledge_tool.py`

从单一路径：

```text
用户问题 -> 向量库 topK -> 拼接上下文
```

改成多阶段路径：

```text
用户问题
-> 意图判断
-> 查询扩展
-> 按 tier/type/doc_id 过滤向量召回
-> 本地重排
-> 补 support 父块
-> 弱命中 fallback scan
-> 输出带证据的上下文
```

## 4. 删除了什么

没有发现项目代码文件被删除。

但逻辑上替换掉了旧的简单检索路径：

- 原来主要依赖 `vector_store.as_retriever(k=config.rag_top_k)`
- 现在主要依赖 `knowledge_tool.py` 内的路由检索、metadata 过滤、重排、父块补齐、fallback

## 5. 未纳入本报告的内容

以下内容不作为本报告对比范围：

- `data/` 下的手册 chunk 数据
- `reports/` 下的测试报告
- `tmp_competition_pdf/` 临时解析文件
- Milvus 数据库中的实际 collection 内容
- `.env` 中的本地密钥或环境变量值
- `scripts/__pycache__/` 运行缓存

## 6. 当前代码风险

- 当前代码处于未提交状态，不是一个干净提交。
- `scripts/index_manual_chunks.py` 还是未跟踪文件，需要决定是否加入版本控制。
- `scripts/__pycache__/` 不应提交。
- 如果同一 `doc_id` 的多个 chunk 版本同时入库，可能互相覆盖或造成召回混乱。
- 当前回答阶段还没有专门解析 `<PIC:...>` 与周围文字的邻接关系，因此“图文结构”数据基础存在，但输出阶段没有完全利用。
