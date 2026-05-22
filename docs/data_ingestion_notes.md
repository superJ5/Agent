# Data Ingestion Notes

这份文档简要说明当前项目里 `uploads/` 和 `data/` 两条入库路径的区别。

## 1. 当前主线：data/manuals/chunks

比赛主线的手册知识库使用：

```text
data/manuals/chunks/*.jsonl
```

入库命令：

```bash
python scripts/index_manual_chunks.py
```

如果使用虚拟环境：

```bash
.venv/bin/python scripts/index_manual_chunks.py
```

流程：

```text
Python 读取 data/manuals/chunks/*.jsonl
        ↓
解析结构化 chunk、图片标识、metadata
        ↓
生成 embedding
        ↓
写入 Milvus 的 biz collection
```

注意：

```text
data 文件不会被复制进 Docker 容器。
Milvus 容器也不会主动读取 data 目录。
是本地 Python 读取 data 文件后，通过 Milvus API 写入向量库。
```

## 2. 旧上传入口：uploads

`uploads/` 是 `/api/upload` 上传接口的临时落盘目录。

旧流程：

```text
aiops-docs/*.md
        ↓
POST /api/upload
        ↓
保存副本到 uploads/
        ↓
切分文本
        ↓
写入 Milvus
```

对应命令：

```bash
make upload
```

这条路径适合：

```text
临时上传 txt/md 文件
调试上传接口
兼容旧版 AIOps 示例文档
```

它不是当前手册 RAG 主线。

## 3. 两者是否冲突

不冲突。

```text
data/manuals/chunks
当前主线结构化知识库。

uploads
旧上传接口的临时目录。
```

如果同一份内容通过两条路径都写入 Milvus，可能出现重复检索结果，所以建议比赛主线只使用：

```bash
python scripts/index_manual_chunks.py
```

## 4. 记忆 daily 入库

记忆系统的 daily 总结是另一条单独路径：

```text
data/memory/daily/*.md
```

入库命令：

```bash
python scripts/index_memory.py --rebuild
```

总结并顺便入库：

```bash
python scripts/summarize_memory.py --date YYYY-MM-DD --index
```

## 5. 简单结论

```text
手册知识库主线：
data/manuals/chunks/*.jsonl → scripts/index_manual_chunks.py → Milvus biz

临时上传/旧入口：
aiops-docs/*.md → /api/upload → uploads/ → Milvus biz

记忆系统：
data/memory/daily/*.md → scripts/index_memory.py → Milvus memory
```
