# 03 RAG 与数据资产

这一篇只讲：

```text
Agent 调用 retrieve_knowledge 后，RAG 如何查到项目自己的手册数据。
```

## 1. RAG 工具入口

对应文件：

```text
app/tools/knowledge_tool.py
```

核心函数：

```python
retrieve_knowledge(query: str)
```

输入：

```text
query
检索问题。通常由模型根据用户问题生成。
```

输出：

```text
context
格式化后的证据文本，给模型阅读。

docs
检索到的 Document 列表。
```

它的职责：

```text
根据 query 去手册知识库里找相关证据。
```

## 2. retrieve_knowledge 的主流程

核心逻辑：

```python
bundle = routed_retrieve(query)
docs = [search_result_to_document(result) for result in bundle.all_hits]
context = format_bundle(bundle, query=query)
return context, docs
```

可以翻译成：

```text
先执行带路由的检索
   ↓
把检索结果转成 Document
   ↓
把检索结果格式化成模型能读懂的证据文本
   ↓
返回给 Agent
```

## 3. routed_retrieve 做什么

核心函数：

```python
routed_retrieve(query)
```

它不是简单搜一下，而是先分析问题。

主要步骤：

```text
detect_intent(query)
判断问题意图。

infer_doc_id(query)
尝试判断用户问的是哪本文档。

extract_query_terms(query)
抽取关键词。

expand_queries(...)
扩展同义词和相关问法。
```

例如：

```text
用户问“吹管怎么安装？”
```

系统可能扩展为：

```text
吹管
吹风管
喷口
安装
步骤
```

这样更容易查到手册里的相关片段。

## 4. 意图路由

`detect_intent(query)` 会把问题分到不同类型。

常见类型：

```text
procedure
操作步骤类问题。

safety
安全注意事项类问题。

troubleshooting
故障排查类问题。

image_trace
图片、图标、配图追溯类问题。

ocr_audit
OCR、原文识别、文字缺失类问题。

general
普通问题。
```

不同意图会优先查不同类型的 chunk。

例如：

```text
问操作步骤
→ 优先查 procedure_step。

问安全
→ 优先查 safety_clause / caution_clause。

问图片
→ 优先查 text_image_atomic / metadata_image_path。
```

## 5. 向量搜索

对应文件：

```text
app/services/vector_search_service.py
```

核心函数：

```python
search_similar_documents(...)
```

输入：

```text
query
检索文本。

top_k
返回几条结果。

doc_id
限定哪本文档。

retrieval_tiers
限定 primary / support / auxiliary。

chunk_types
限定 chunk 类型。
```

输出：

```text
List[SearchResult]
```

每个 `SearchResult` 包含：

```text
id
content
score
metadata
```

## 6. embedding 是什么

对应文件：

```text
app/services/vector_embedding_service.py
```

核心函数：

```python
embed_query(text)
```

作用：

```text
把一段文字变成向量。
```

当前使用：

```text
text-embedding-v4
```

注意这里有两个模型：

```text
qwen3.5-plus
负责理解用户问题、看图、组织答案。

text-embedding-v4
负责把文本变成向量，用来检索。
```

## 7. Milvus 存什么

对应文件：

```text
app/core/milvus_client.py
```

Milvus 是向量数据库。

当前 collection 名字：

```text
biz
```

字段大概是：

```text
id
每个 chunk 的唯一编号。

vector
chunk 文本对应的向量。

content
chunk 文本内容。

metadata
手册名、章节、图片路径、chunk 类型等附加信息。
```

查询时：

```text
用户问题 query
   ↓
embedding 成 query_vector
   ↓
Milvus 找相似 vector
   ↓
返回相似 chunk
```

## 8. 项目数据资产

原始数据：

```text
data/manuals/raw/
```

主要放：

```text
原始手册 txt
原始图片 images/
原始汇总文件
```

人工整理后的文本：

```text
data/manuals/reviewed_md/
```

主要放：

```text
人工或脚本整理后的 Markdown 文本
```

RAG 真正索引用的数据：

```text
data/manuals/chunks/
```

这里是 JSONL 文件。

每一行通常是一段 chunk，带 metadata，例如：

```text
doc_id
doc_name
chunk_id
chunk_type
retrieval_tier
section_path
image_paths
source_lines
text/content
```

索引脚本：

```text
scripts/index_manual_chunks.py
```

作用：

```text
读取 data/manuals/chunks/*.jsonl
   ↓
对每个 chunk 做 embedding
   ↓
写入 Milvus 的 biz collection
```

## 9. RAG 总流程

```text
Agent 决定调用 retrieve_knowledge(query)
   ↓
detect_intent 判断问题类型
   ↓
infer_doc_id 判断可能是哪本手册
   ↓
expand_queries 扩展同义词
   ↓
vector_search_service.search_similar_documents()
   ↓
vector_embedding_service.embed_query()
   ↓
Milvus 在 biz collection 里查相似 chunk
   ↓
返回 SearchResult
   ↓
format_bundle() 格式化成证据文本
   ↓
证据返回给 Agent
   ↓
模型基于证据生成最终答案
```

## 10. 多模态和 RAG 的关系

这两个不要混在一起。

```text
多模态模型
负责看用户上传的图片，理解图片里是什么。

RAG
负责查项目提前索引好的手册数据。
```

例子：

```text
用户上传冰箱图标图片，问“这个是什么意思？”
   ↓
qwen3.5-plus 先看图，理解可能是“瓶子图标”
   ↓
Agent 调用 retrieve_knowledge("冰箱 瓶子图标 含义")
   ↓
Milvus 查冰箱手册相关 chunk
   ↓
模型根据查到的手册证据回答
```
