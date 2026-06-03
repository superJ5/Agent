# Manual Chunk Indexing Commands

本文整理两套命令：

- PowerShell：Windows 本机执行。
- Ubuntu / WSL：Linux shell 执行，Python 路径使用 `./.venv/bin/python`。

## PowerShell

### 1. 进入项目目录

```powershell
cd E:\.codex\worktrees\3f3c\ai_agent_competition
```

### 2. 删除当前 Milvus 里的 `biz` collection

```powershell
@'
from pymilvus import connections, utility
from app.config import config

connections.connect(
    alias="default",
    host=config.milvus_host,
    port=str(config.milvus_port),
    timeout=config.milvus_timeout / 1000,
)

if utility.has_collection("biz"):
    utility.drop_collection("biz")
    print("dropped: biz")
else:
    print("collection not exists: biz")

connections.disconnect("default")
'@ | .\.venv\Scripts\python.exe -
```

### 3. 重新入库，并输出最终汇总

```powershell
.\.venv\Scripts\python.exe scripts\index_manual_chunks.py `
  --directory data\manuals\chunks `
  --embedding-limit-report logs\embedding_input_limit_report.jsonl `
  --skipped-chunk-report logs\index_skipped_chunks_report.jsonl
```

跑完后看输出里的 `final_error_summary`。

### 4. 查看实际入库总条数和入库的 `.jsonl` 文件数

```powershell
@'
from collections import Counter
from app.services.vector_search_service import vector_search_service

rows = vector_search_service.query_all_documents(batch_size=1000)

sources = Counter()
doc_ids = Counter()

for row in rows:
    m = row.metadata or {}
    source = m.get("_source") or m.get("source_file") or "UNKNOWN_SOURCE"
    doc_id = m.get("doc_id") or "UNKNOWN_DOC"
    sources[source] += 1
    doc_ids[doc_id] += 1

print("total_chunks:", len(rows))
print("jsonl_file_count:", len(sources))
print("doc_id_count:", len(doc_ids))

print("\nFILES:")
for source, count in sorted(sources.items()):
    print(f"{count}\t{source}")
'@ | .\.venv\Scripts\python.exe -
```

### 5. 查看被跳过的单条 chunk

```powershell
if (Test-Path logs\index_skipped_chunks_report.jsonl) {
  Get-Content logs\index_skipped_chunks_report.jsonl |
    ForEach-Object { $_ | ConvertFrom-Json } |
    Select-Object doc_id,chunk_id,retrieval_tier,chunk_type,stage,error |
    Format-Table -AutoSize
} else {
  Write-Host "没有单条 chunk 被跳过：logs\index_skipped_chunks_report.jsonl 不存在"
}
```

### 6. 查看触发 8192 后截断重试的 chunk

```powershell
if (Test-Path logs\embedding_input_limit_report.jsonl) {
  Get-Content logs\embedding_input_limit_report.jsonl |
    ForEach-Object { $_ | ConvertFrom-Json } |
    Select-Object doc_id,chunk_id,retrieval_tier,chunk_type,title,original_embedding_chars,fallback_embedding_chars |
    Format-Table -AutoSize
} else {
  Write-Host "没有 8192 截断重试记录：logs\embedding_input_limit_report.jsonl 不存在"
}
```

## Ubuntu / WSL

### 1. 进入项目目录

WSL 访问 Windows 盘符时通常使用 `/mnt/e/...`：

```bash
cd /mnt/e/.codex/worktrees/3f3c/ai_agent_competition
```

如果项目在 Ubuntu 原生目录里，改成你的实际路径：

```bash
cd /path/to/ai_agent_competition
```

### 2. 删除当前 Milvus 里的 `biz` collection

```bash
./.venv/bin/python - <<'PY'
from pymilvus import connections, utility
from app.config import config

connections.connect(
    alias="default",
    host=config.milvus_host,
    port=str(config.milvus_port),
    timeout=config.milvus_timeout / 1000,
)

if utility.has_collection("biz"):
    utility.drop_collection("biz")
    print("dropped: biz")
else:
    print("collection not exists: biz")

connections.disconnect("default")
PY
```

### 3. 重新入库，并输出最终汇总

```bash
./.venv/bin/python scripts/index_manual_chunks.py \
  --directory data/manuals/chunks \
  --embedding-limit-report logs/embedding_input_limit_report.jsonl \
  --skipped-chunk-report logs/index_skipped_chunks_report.jsonl
```

跑完后看输出里的 `final_error_summary`。

### 4. 查看实际入库总条数和入库的 `.jsonl` 文件数

```bash
./.venv/bin/python - <<'PY'
from collections import Counter
from app.services.vector_search_service import vector_search_service

rows = vector_search_service.query_all_documents(batch_size=1000)

sources = Counter()
doc_ids = Counter()

for row in rows:
    m = row.metadata or {}
    source = m.get("_source") or m.get("source_file") or "UNKNOWN_SOURCE"
    doc_id = m.get("doc_id") or "UNKNOWN_DOC"
    sources[source] += 1
    doc_ids[doc_id] += 1

print("total_chunks:", len(rows))
print("jsonl_file_count:", len(sources))
print("doc_id_count:", len(doc_ids))

print("\nFILES:")
for source, count in sorted(sources.items()):
    print(f"{count}\t{source}")
PY
```

### 5. 查看被跳过的单条 chunk

```bash
./.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("logs/index_skipped_chunks_report.jsonl")
if not path.exists():
    print("没有单条 chunk 被跳过：logs/index_skipped_chunks_report.jsonl 不存在")
    raise SystemExit

fields = ["doc_id", "chunk_id", "retrieval_tier", "chunk_type", "stage", "error"]
print("\t".join(fields))
for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    print("\t".join(str(row.get(field, "")) for field in fields))
PY
```

### 6. 查看触发 8192 后截断重试的 chunk

```bash
./.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("logs/embedding_input_limit_report.jsonl")
if not path.exists():
    print("没有 8192 截断重试记录：logs/embedding_input_limit_report.jsonl 不存在")
    raise SystemExit

fields = [
    "doc_id",
    "chunk_id",
    "retrieval_tier",
    "chunk_type",
    "title",
    "original_embedding_chars",
    "fallback_embedding_chars",
]
print("\t".join(fields))
for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    print("\t".join(str(row.get(field, "")) for field in fields))
PY
```
