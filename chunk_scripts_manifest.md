# Chunk 脚本清单

这份清单只整理和 `manual -> parsed -> chunks -> 检查 -> 入库` 这条链相关的脚本。

建议默认都在仓库根目录执行命令：

```powershell
cd E:\.codex\worktrees\3f3c\ai_agent_competition
```

如果需要使用项目依赖环境，优先用：

```powershell
.\.venv\Scripts\python.exe <script>
```

## 一、执行顺序

推荐顺序如下：

1. `scripts/organize_manual_raw_assets.py`
2. `scripts/convert_manual_dataset.py`
3. `scripts/batch_convert_manual_datasets.py`
4. `scripts/build_structured_chunks.py`
5. `scripts/verify_chunk_integrity.py`
6. `scripts/check_manual_indexing_readiness.py`
7. `scripts/index_manual_chunks.py`

说明：

- 第 `2` 步是“单本转换”，主要用于调试或抽查。
- 第 `3` 步是“批量转换”，通常正式跑它。
- 第 `6`、`7` 步属于 chunk 后续使用，不是切分本身，但和 chunk 流程直接相关。

## 二、脚本明细

### 1. 原始资料整理

文件：
`scripts/organize_manual_raw_assets.py`

作用：
- 把堆在 `data/manuals/raw/手册` 里的原始 `txt + 插图` 资料，整理成“每本手册一个目录”的结构。
- 尽量使用硬链接，避免重复拷贝大文件。
- 保留原始导入目录不动。

默认输入：
- `data/manuals/raw/手册/*.txt`
- `data/manuals/raw/手册/插图/*`

默认输出：
- `data/manuals/raw/<手册名>/<手册名>.txt`
- `data/manuals/raw/<手册名>/images/*`
- `data/manuals/raw/organization_report.json`

运行命令：

```powershell
python scripts/organize_manual_raw_assets.py
```

适用场景：
- 你刚把比赛组给的原始资料一股脑放进 `data/manuals/raw` 后，先跑这一步。

---

### 2. 单本手册转结构化 parsed JSONL

文件：
`scripts/convert_manual_dataset.py`

作用：
- 把单本手册的 `txt` 记录从：
  - `["正文文本", ["图片ID1", "图片ID2"]]`
- 转成结构化中间格式：
  - `doc_id`
  - `doc_name`
  - `raw_text`
  - `clean_text`
  - `pic_refs`
  - `alignment_ok`

默认输入：
- 任意一份单本手册 txt
- 对应图片目录

默认输出：
- 你指定的 `parsed/*.jsonl`

运行命令：

```powershell
python scripts/convert_manual_dataset.py `
  --input "data/manuals/raw/冰箱手册/冰箱手册.txt" `
  --image-dir "data/manuals/raw/冰箱手册/images" `
  --output "data/manuals/parsed/冰箱手册.jsonl"
```

可选参数：
- `--doc-id`

适用场景：
- 想先单独调试某一本手册时。

---

### 3. 批量转结构化 parsed JSONL

文件：
`scripts/batch_convert_manual_datasets.py`

作用：
- 读取 `organization_report.json`
- 批量处理所有 `alignment_ok=true` 的手册
- 跳过图文数量不一致的异常手册

默认输入：
- `data/manuals/raw/organization_report.json`
- `data/manuals/raw/<手册名>/<手册名>.txt`
- `data/manuals/raw/<手册名>/images/*`

默认输出：
- `data/manuals/parsed/*.jsonl`
- `data/manuals/parsed/conversion_report.json`

运行命令：

```powershell
python scripts/batch_convert_manual_datasets.py
```

适用场景：
- 正式把正常手册批量转成 `parsed` 阶段数据。

注意：
- 当前异常手册会被跳过，不会进入后续 chunk。

---

### 4. 结构切分生成 chunk

文件：
`scripts/build_structured_chunks.py`

作用：
- 从 `parsed/*.jsonl` 生成第一版结构化 chunk。
- 当前策略是：
  - 先按 `# 标题` 切
  - 超长块再按行和句号拆
  - 过小块自动并回相邻块
  - `<PIC>` 按 chunk 内顺序绑定到 `pic_refs`

默认输入：
- `data/manuals/parsed/*.jsonl`

默认输出：
- `data/manuals/chunks/*.jsonl`
- `data/manuals/chunks/chunking_report.json`

运行命令：

```powershell
python scripts/build_structured_chunks.py
```

每个 chunk 里主要字段包括：
- `chunk_id`
- `doc_id`
- `doc_name`
- `chunk_index`
- `section_title`
- `text`
- `raw_text`
- `pic_refs`
- `image_paths`

适用场景：
- 当前第一版 chunk 生产主脚本。

---

### 5. chunk 完整性校验

文件：
`scripts/verify_chunk_integrity.py`

作用：
- 对比切分前后的内容，确认有没有少文本、少 `<PIC>`、少图片引用，或者图片顺序乱掉。

默认输入：
- `data/manuals/parsed/*.jsonl`
- `data/manuals/chunks/*.jsonl`

默认输出：
- `data/manuals/chunks/chunk_integrity_report.json`

运行命令：

```powershell
python scripts/verify_chunk_integrity.py
```

主要检查项：
- `raw_text_match`
- `clean_text_match`
- `pic_count_match`
- `pic_sequence_match`
- `content_preserved`

适用场景：
- 每次修改切分规则后，都建议重跑。

---

### 6. 入库前预检查

文件：
`scripts/check_manual_indexing_readiness.py`

作用：
- 在真正把 chunk 入 Milvus 之前，先检查外部条件是否就绪。

默认检查内容：
- `DASHSCOPE_API_KEY` 是否还是占位值
- `Milvus` 是否可连
- `data/manuals/chunks` 是否存在
- 是否有可入库的 `jsonl`

默认输入：
- `.env`
- `data/manuals/chunks/*.jsonl`

默认输出：
- 标准输出 JSON 报告

运行命令：

```powershell
.\.venv\Scripts\python.exe scripts/check_manual_indexing_readiness.py
```

适用场景：
- 每次正式入库前先跑一次，避免白等。

---

### 7. chunk 入 Milvus

文件：
`scripts/index_manual_chunks.py`

作用：
- 把 `data/manuals/chunks/*.jsonl` 接进当前项目的向量建库流程。
- 现在走的是：
  - 文本 embedding
  - metadata 带上 `doc_id / chunk_id / image_paths / pic_refs`

默认输入：
- `data/manuals/chunks/*.jsonl`

默认输出：
- 写入 Milvus

运行命令：

```powershell
.\.venv\Scripts\python.exe scripts/index_manual_chunks.py
```

可选参数：

```powershell
.\.venv\Scripts\python.exe scripts/index_manual_chunks.py --directory "./data/manuals/chunks"
```

适用场景：
- preflight 通过后，正式开始索引。

前置条件：
- `.env` 里 `DASHSCOPE_API_KEY` 不是占位值
- `Milvus` 已启动并可连接

## 三、目录流转关系

```text
data/manuals/raw
  -> scripts/organize_manual_raw_assets.py
  -> data/manuals/raw/<手册名>/

data/manuals/raw/<手册名>/
  -> scripts/convert_manual_dataset.py
  -> data/manuals/parsed/<手册名>.jsonl

data/manuals/parsed/*.jsonl
  -> scripts/build_structured_chunks.py
  -> data/manuals/chunks/*.jsonl

data/manuals/chunks/*.jsonl
  -> scripts/verify_chunk_integrity.py
  -> scripts/check_manual_indexing_readiness.py
  -> scripts/index_manual_chunks.py
  -> Milvus
```

## 四、常用最短流程

如果原始资料已经整理好了，最常用的是这几步：

```powershell
python scripts/batch_convert_manual_datasets.py
python scripts/build_structured_chunks.py
python scripts/verify_chunk_integrity.py
.\.venv\Scripts\python.exe scripts/check_manual_indexing_readiness.py
.\.venv\Scripts\python.exe scripts/index_manual_chunks.py
```

## 五、当前实际产物位置

当前已经生成的中间结果在：

- `data/manuals/raw/organization_report.json`
- `data/manuals/parsed/conversion_report.json`
- `data/manuals/chunks/chunking_report.json`
- `data/manuals/chunks/chunk_integrity_report.json`

如果后面要继续改 chunk 策略，优先改的主脚本是：

- `scripts/build_structured_chunks.py`

