# 代码变更报告 - 2026-06-02

## 范围

本文档记录当前工作区中与英文手册召回 / `big_support` 相关的变更，并重点说明两个 `6000` 限制值的改动。

## 明确的 6000 变更

### 1. 环境变量中的扫描兜底限制

文件：

```text
.env
```

当前设置：

```dotenv
RAG_SCAN_CANDIDATE_LIMIT=6000
```

作用：

- 控制普通扫描兜底路径使用的最大候选数量。
- 该值通过 RAG 召回选项 / 配置路径读取。
- 它不是文档画像扫描限制。

相关代码中的默认值仍然存在：

```text
app/config.py
app/retrieval/schemas.py
app/retrieval/recall.py
```

这些默认值是兜底值；`.env` 中的值在加载后会覆盖运行时配置。

### 2. 文档画像扫描限制

文件：

```text
app/tools/knowledge_tool.py
```

当前设置：

```python
PROFILE_SCAN_LIMIT = 6000
```

用法：

```python
results = vector_search_service.query_documents(limit=PROFILE_SCAN_LIMIT)
```

作用：

- 控制构建动态文档画像时扫描多少条已索引的元数据行。
- 这些画像会供 `load_active_doc_ids()` 使用。
- 如果这个限制太小，后续索引的手册可能会在活跃文档过滤中缺失。
- 观察到的问题是：当文档画像扫描只读取前 `4096` 行时，`manual_f769aafb` 可能已经存在于 Milvus 中，但没有出现在 `active_doc_ids` 里。

将该值改为 `6000` 后，本地验证结果如下：

```text
query_documents_6000 4988 has_boat True
active_doc_count 40
has_boat True
```

## 召回 / big_support 相关代码区域

当前工作区还包含以下区域中与 `big_support` 相关的变更：

```text
app/retrieval/tier_policy.py
app/retrieval/bm25_provider.py
app/retrieval/recall.py
app/retrieval/evidence.py
app/services/vector_search_service.py
app/services/vector_store_manager.py
scripts/apply_big_support_policy.py
scripts/index_manual_chunks.py
```

整体意图：

- 不让 `big_support` 进入普通召回。
- 只允许 `big_support` 在证据扩展阶段参与。
- 保持 `hits` 作为原始召回命中结果。
- 使用 `support_hits` 作为统一的补充上下文列表。
- 即使某个 chunk 被放入 `support_hits`，也保留它自身的 `retrieval_tier`。

## 测试 / 诊断文件

当前相关测试和诊断文件包括：

```text
tests/retrieval/test_bm25_provider.py
tests/retrieval/test_evidence.py
tests/retrieval/test_metadata_indexing_filter.py
tests/retrieval/test_recall.py
tests/retrieval/test_tier_policy.py
scripts/diagnose_doc_recall.py
```

当前已生成的本地测试数据 / 报告包括：

```text
data/test_boat_airfryer_241_250.csv
data/submission_boat_airfryer_241_250.csv
doc/manual_f769aafb_recall_diagnosis.md
doc/retrieval_trace_intermediate_report.md
```

## 重要区别

这两个限制值含义不同，不应混为一谈：

```text
RAG_SCAN_CANDIDATE_LIMIT=6000
PROFILE_SCAN_LIMIT = 6000
```

- `RAG_SCAN_CANDIDATE_LIMIT` 影响扫描兜底候选数量。
- `PROFILE_SCAN_LIMIT` 影响动态文档画像构建和活跃文档过滤。

对于 `manual_f769aafb` 召回缺失问题，直接相关的变更是：

```python
PROFILE_SCAN_LIMIT = 6000
```
