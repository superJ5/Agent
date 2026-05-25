# Retrieval System Report

本文档记录当前召回系统已经实现的功能、正规启动方式、测试方式和报告查看方式。内容基于当前代码状态整理。

> 注意：如果仓库的 `.gitignore` 中已经加入 `doc/`，本文件默认不会被 Git 提交；需要提交时要调整 `.gitignore` 或使用 `git add -f doc/retrieval_system_report.md`。

## 1. 当前召回链路概览

当前真实 Chat 链路是：

```text
HTTP /chat 或批量测试脚本
  -> rag_agent_service.query()
  -> Agent 调用 retrieve_knowledge
  -> knowledge_tool.routed_retrieve()
  -> app.retrieval.orchestrator.retrieve()
  -> query_understanding
  -> recall
  -> reranker
  -> evidence
  -> answer + metadata + trace
```

核心入口：

- `app/services/rag_agent_service.py`
- `app/tools/knowledge_tool.py`
- `app/retrieval/orchestrator.py`
- `app/retrieval/recall.py`
- `app/retrieval/reranker.py`
- `app/retrieval/evidence.py`
- `app/retrieval/diagnostics.py`

## 2. 已实现功能

### 2.1 Query Understanding(这部分暂时没用)

模块：`app/retrieval/query_understanding.py`

作用：

- 解析用户问题。
- 生成 intent 候选、doc_id 候选、query_terms。
- 将分析过程写入 diagnostics trace。

当前支持的策略包括：

```text
none, profile, rules, summary, llm, hybrid
```

当前 `.env.template` 中默认：

```env
RAG_INTENT_STRATEGY=none
```

如果要启用规则或混合意图识别，可以在 `.env` 中调整为 `rules` 或 `hybrid`。

### 2.2 多路召回

模块：`app/retrieval/recall.py`

当前召回通道：

- `vector`：向量召回，调用 `vector_search_service.search_similar_documents()`。
- `bm25`：BM25 词法召回，调用已注册的 `JiebaBM25Provider`。
- `scan`：兜底扫描召回，调用 `vector_search_service.query_all_documents()` 后做本地词法排序。

召回路线会结合：

- intent
- doc_id 候选
- retrieval_tier
- chunk family / chunk_type
- query variants

召回结果会按 chunk_id 去重，然后计算合并分数。

### 2.3 BM25 Provider

模块：`app/retrieval/bm25_provider.py`

当前实现：

- 使用 `jieba` 分词。
- 使用 `rank_bm25.BM25Okapi` 建索引。
- 从 Milvus 当前文档中拉取 chunks 建立进程内 BM25 索引。
- 支持按 `doc_id`、`retrieval_tiers`、`chunk_types` 过滤。

初始化位置：

- FastAPI 正规启动：`app/main.py` 的 lifespan 中调用 `init_bm25_provider()`。
- 批量 Chat 回归脚本：`scripts/run_public_question_regression.py` 中也会先初始化 Milvus 和 BM25。

如果 BM25 初始化失败，系统会记录 warning，并继续使用 vector / scan，不会阻塞服务启动。

### 2.4 召回合并打分（注意得在.env里配一下rerank模型的apikey）

模块：`app/retrieval/recall.py`

默认权重：

```env
RAG_VECTOR_WEIGHT=0.6
RAG_BM25_WEIGHT=0.4
```

scan 通道内部权重为代码常量：

```text
SCAN_CHANNEL_WEIGHT = 0.2
```

打分流程：

1. 每个召回通道先各自产生候选结果。
2. 每个通道内的原始分数会被归一化到 `[0, 1]`。
3. 同一个 chunk 如果被多个通道召回，会合并 channel score。
4. 最终 `merged_score` 按通道权重加权求和。

向量通道会先把 Milvus 距离类 score 转成相似度方向：

```text
1 / (1 + score)
```

然后再做 min-max 归一化。

### 2.5 Scan Fallback

模块：`app/retrieval/recall.py`

scan fallback 会在以下情况触发：

- vector / bm25 没有召回候选。
- intent 是 `image_trace` 或 `ocr_audit`。
- 问题包含关键术语组合。
- 当前 top candidate 的词法匹配分不足。
- top candidate 的 merged_score 过低。

默认扫描上限：

```env
RAG_SCAN_CANDIDATE_LIMIT=4096
```

scan 主要用于避免向量召回失败时整条链路直接断掉。

### 2.6 DashScope Reranker

模块：`app/retrieval/reranker.py`

当前默认配置：

```env
RAG_RERANKER_PROVIDER=dashscope
RAG_RERANKER_MODEL=qwen3-rerank
RAG_RERANKER_ENDPOINT=https://dashscope.aliyuncs.com/compatible-api/v1/reranks
RAG_RERANKER_TIMEOUT_MS=3000
RAG_RERANKER_TOP_N=32
```

请求体格式：

```json
{
  "model": "qwen3-rerank",
  "query": "用户问题",
  "documents": ["chunk 的 rerank/index 文本", "..."],
  "top_n": 32
}
```

reranker 会把模型返回的分数写入候选 diagnostics，并在 metadata / trace 中暴露 `reranker_score`。

失败策略：

- DashScope rerank 超时、报错、返回空结果时，不中断检索。
- 系统自动降级到 `lexical_fallback`。
- metadata 中会标记 `reranker_fallback=true`，trace 中也会记录 warning / fallback 信息。

### 2.7 Evidence 组织与图片输出

模块：`app/retrieval/evidence.py`

当前 evidence 做的事情：

- 取 rerank 后的 top candidates。
- 按 primary hit 查找 support parent chunks。
- 生成最终给模型看的 context。
- 生成 API metadata 中的 top_hits、warnings、image_paths 等信息。
- 生成 trace 中的 evidence_primary_hits、evidence_support_hits、evidence_images。

图片路径处理：

- Milvus metadata 中可能仍然存本机绝对路径。
- 输出给模型和报告时会转成项目相对路径。
- 回答中引用图片时要求使用 PIC ID 作为 markdown alt text。

目标格式：

```markdown
![Manual01_5](data/manuals/raw/空调手册/images/Manual01_5.jpg)
```

避免格式：

```markdown
![](data/manuals/raw/空调手册/images/Manual01_5.jpg)
```

非流式回答返回前还有兜底修正：如果模型生成了空 alt 的图片 markdown，会根据路径文件名补上 PIC ID。

## 3. 配置方式

不要提交真实 `.env`。推荐流程：

```powershell
Copy-Item .env.template .env
```

然后编辑 `.env`，至少配置：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
RAG_RERANKER_PROVIDER=dashscope
```

`.env.template` 是提交给其他开发者看的配置模板；`.env` 是本地私密配置。

## 4. 正规启动方式

### 4.1 Windows 一键启动

项目提供：

```powershell
.\start-windows.bat
```

该脚本会尝试：

- 创建或同步虚拟环境。
- 启动 Milvus Docker Compose。
- 启动 MCP 服务。
- 启动 FastAPI 服务。

### 4.2 手动启动 Milvus

```powershell
docker compose -f vector-database.yml up -d
```

Milvus 默认连接信息：

```env
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

### 4.3 手动启动 FastAPI

```powershell
D:\anaconda3\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 9900
```

启动时 `app/main.py` 会执行：

```text
milvus_manager.connect()
init_bm25_provider()
```

因此正规服务启动后，Milvus 连接和 BM25 provider 注册都会完成。

### 4.4 健康检查

```powershell
curl http://localhost:9900/health
```

API 文档地址：

```text
http://localhost:9900/docs
```

## 5. 正规 Chat 接口使用

比赛标准接口：

```text
POST /chat
```

请求示例：

```powershell
curl -X POST http://localhost:9900/chat `
  -H "Authorization: Bearer your_token" `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"如何给空调遥控器安装电池？\",\"session_id\":\"manual-test-001\",\"stream\":false}"
```

说明：

- `question`：用户问题。
- `session_id`：可选；不传会自动生成。
- `stream`：非流式建议设为 `false`。
- `images`：支持 base64 图片列表，最多 3 张。
- `Authorization` header 必须存在；如果 `.env` 中设置了 `API_BEARER_TOKEN`，则 token 必须匹配。

## 6. 批量 Chat 回归测试

脚本：

```text
scripts/run_public_question_regression.py
```

作用：

- 走真实 `rag_agent_service.query()`。
- 初始化 Milvus 和 BM25。
- 每题单独 session。
- 单题失败不中断整轮。
- 输出 JSON 和 Markdown 报告。

跑前 10 题示例：

```powershell
Remove-Item Env:DASHSCOPE_API_BASE -ErrorAction SilentlyContinue
Remove-Item Env:RAG_INTENT_STRATEGY -ErrorAction SilentlyContinue
$env:DEBUG = "true"

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
Start-Transcript -Path "reports\public_question_runs\chat\official_10_$ts.transcript.txt"

D:\anaconda3\python.exe scripts\run_public_question_regression.py `
  --queries-file reports\public_question_runs\question_public_64_234.jsonl `
  --output-dir reports\public_question_runs\chat `
  --case-prefix official_10_$ts `
  --limit 10 2>&1 | Tee-Object -FilePath "reports\public_question_runs\chat\official_10_$ts.console.log"

Stop-Transcript
```

输出位置：

```text
reports/public_question_runs/chat/*.json
reports/public_question_runs/chat/*.md
```

报告内容：

- `summary.total`
- `summary.succeeded`
- `summary.failed`
- 每题 `case_id`
- 每题 `query`
- 每题 `answer`
- 每题 `metadata`
- 每题 `duration_ms`
- 每题 `status`
- 每题 `error`

## 7. Retrieval Trace 查看方式

### 7.1 运行时 JSONL Trace

当 `DEBUG=true` 时，模块化检索链路会把 trace 追加写入：

```text
logs/retrieval_trace.jsonl
```

每一行对应一次 retrieval request。

重点字段：

- `request_id`
- `query`
- `options`
- `query_understanding`
- `recall.channels.vector`
- `recall.channels.bm25`
- `recall.channels.scan`
- `recall.normalized_scores`
- `recall.pre_merge`
- `recall.post_merge`
- `reranker.pre_rank`
- `reranker.post_rank`
- `reranker.rank_changes`
- `evidence_primary_hits`
- `evidence_support_hits`
- `evidence_images`
- `warnings`

查看最近 trace：

```powershell
Get-Content logs\retrieval_trace.jsonl -Tail 1
```

按关键词查找：

```powershell
Select-String -Path logs\retrieval_trace.jsonl -Pattern "Manual01_5"
```

### 7.2 Retrieval Trace 抽样脚本

脚本：

```text
scripts/trace_retrieval_pipeline.py
```

用途：

- 生成检索过程诊断报告。
- 适合排查某个问题为什么没召回正确 chunk。
- 这是诊断脚本，不是最终 Chat 回答链路。

示例：

```powershell
D:\anaconda3\python.exe scripts\trace_retrieval_pipeline.py `
  --queries-file reports\public_question_runs\question_public_64_234.jsonl `
  --top-k 5 `
  --output-dir reports\public_question_runs\trace
```

输出位置：

```text
reports/public_question_runs/trace/*.json
reports/public_question_runs/trace/*.md
```

## 8. 当前推荐测试命令

### 8.1 Reranker / Recall 单元测试

```powershell
D:\anaconda3\python.exe -m pytest -q `
  tests\retrieval\test_bm25_provider.py `
  tests\retrieval\test_recall.py `
  tests\retrieval\test_reranker.py `
  tests\retrieval\test_orchestrator.py `
  tests\retrieval\test_diagnostics.py `
  tests\retrieval\test_evidence.py `
  tests\retrieval\test_schemas.py
```

### 8.2 API 兼容测试

```powershell
D:\anaconda3\python.exe -m pytest -q tests\test_api_compat.py
```

### 8.3 全扫描相关测试

```powershell
D:\anaconda3\python.exe -m pytest -q tests\test_retrieval_full_scan.py
```

### 8.4 Ruff 检查

```powershell
D:\anaconda3\python.exe -m ruff check app scripts tests
```

### 8.5 Mypy 检查

```powershell
D:\anaconda3\python.exe -m mypy app
```

说明：`mypy app` 是类型检查；如果历史代码已有类型问题，可能需要分批处理，不等同于运行失败。

## 9. 常见问题与排查

### 9.1 Chat 端 DashScope 401

检查 `.env`：

```env
DASHSCOPE_API_KEY=...
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
```

`DASHSCOPE_API_BASE` 不要误设成 `none`。

### 9.2 Embedding 或 Rerank 超时

优先检查本机网络和代理。

如果开了 TUN，可能会把 DashScope 解析到代理地址，导致请求变慢。

可以用：

```powershell
Resolve-DnsName dashscope.aliyuncs.com
Test-NetConnection dashscope.aliyuncs.com -Port 443
```

### 9.3 BM25 没有生效

确认是正规启动或批量脚本启动。

正规启动会执行：

```text
init_bm25_provider()
```

如果直接单独调用 `orchestrator.retrieve()`，需要先手动初始化 Milvus 和 BM25。

### 9.4 图片路径问题

当前输出层会把本机绝对路径转为相对路径。

期望格式：

```markdown
![Manual01_5](data/manuals/raw/空调手册/images/Manual01_5.jpg)
```

如果仍看到：

```markdown
![](data/manuals/raw/空调手册/images/Manual01_5.jpg)
```

说明模型绕过了提示或走了流式链路；非流式 `rag_agent_service.query()` 当前有兜底修正。

### 9.5 看不到 reports 文件

`reports/` 在 `.gitignore` 中，默认不会提交。

本地查看：

```powershell
Get-ChildItem reports\public_question_runs\chat
Get-ChildItem reports\public_question_runs\trace
```

### 9.6 doc 文件不会提交

如果 `.gitignore` 中有：

```gitignore
doc/
```

那么 `doc/retrieval_system_report.md` 默认只是本地文档。

需要提交时：

```bash
git add -f doc/retrieval_system_report.md
```

## 10. 当前系统边界

- Reranker 依赖 DashScope 外部服务；失败时会降级到 lexical fallback。
- BM25 是进程内索引，服务启动时从 Milvus 拉取当前 chunks 构建；Milvus 内容更新后需要重新初始化或重启服务才能刷新 BM25 索引。
- `logs/retrieval_trace.jsonl` 只在 `DEBUG=true` 时写入文件。
- `scripts/trace_retrieval_pipeline.py` 是诊断工具；最终答案质量仍应以 Chat 回归报告为准。
- `.env` 不应提交；新增配置项通过 `.env.template` 告知其他开发者。
