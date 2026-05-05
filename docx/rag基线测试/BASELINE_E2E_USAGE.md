# 基线 E2E 测试使用说明

这套基线测试用于评估“普通按标题/字数切分 + 纯向量检索”的效果，和主线结构化 chunk RAG 做对照。

它的关键设计是隔离：基线使用独立 Milvus collection `biz_baseline`，不会写入主线 `biz` collection。

## 相关文件

- `scripts/build_langchain_baseline.py`：从审阅后的 `.md` 生成 LangChain 风格 baseline chunks。
- `scripts/baseline_e2e_test.py`：基线端到端测试，负责把 baseline chunks 入库到 `biz_baseline`、检索、调用 LLM 回答。
- `scripts/eval_baseline_recall.py`：离线召回评估，不用 Milvus，不用 LLM，只用 TF-IDF 做快速粗评。

## 数据位置

基线 chunks 固定放在：

```text
data/manuals/langchain_baseline/
```

目录结构通常是：

```text
data/manuals/langchain_baseline/
  功能键盘手册_langchain/chunks.jsonl
  VR头显手册_langchain/chunks.jsonl
  电钻手册_langchain/chunks.jsonl
```

`baseline_e2e_test.py` 会递归扫描这个目录下所有 `chunks.jsonl`。

## 前置条件

运行前需要：

- `.env` 里配置 `DASHSCOPE_API_KEY`。
- Docker/Milvus 已启动。
- `data/manuals/langchain_baseline/` 下已经有 baseline `chunks.jsonl`。
- 在项目根目录运行命令，也就是 `E:\.codex\worktrees\3f3c\ai_agent_competition`。

推荐使用项目虚拟环境：

```powershell
.\.venv\Scripts\python.exe
```

如果终端中文显示乱码，可以先设置：

```powershell
$env:PYTHONIOENCODING='utf-8'
```

## 生成基线 chunks

如果 `data/manuals/langchain_baseline/` 还没有数据，先从审阅后的 `.md` 生成：

```powershell
.\.venv\Scripts\python.exe scripts\build_langchain_baseline.py
```

默认输入目录：

```text
data/manuals/reviewed_md/
```

默认输出目录：

```text
data/manuals/langchain_baseline/
```

可以指定单个文件：

```powershell
.\.venv\Scripts\python.exe scripts\build_langchain_baseline.py --source-md data\manuals\reviewed_md\功能键盘手册.md
```

注意：为了保持基线 E2E 完全隔离，不建议使用：

```powershell
.\.venv\Scripts\python.exe scripts\build_langchain_baseline.py --index
```

这个 `--index` 会调用主线 `vector_index_service`，不属于隔离的 `biz_baseline` 测试链路。隔离入库请用下一节的 `baseline_e2e_test.py --index`。

## 标准 E2E 流程

完整流程分三步。

第 1 步：灌入基线 chunks 到独立集合 `biz_baseline`。

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\baseline_e2e_test.py --index
```

第 2 步：运行内置 60 问端到端测试，包含 LLM 回答。

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\baseline_e2e_test.py --test
```

第 3 步：测试结束后删除基线集合，避免污染本地 Milvus。

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\baseline_e2e_test.py --drop
```

`--drop` 只删除 `biz_baseline`，不会删除主线 `biz`。

## 省 Token 检索测试

如果只想看召回，不想调用 LLM，用：

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\baseline_e2e_test.py --test-no-llm
```

这个模式仍然需要先执行过：

```powershell
.\.venv\Scripts\python.exe scripts\baseline_e2e_test.py --index
```

## 单条问题测试

可以只测一个问题：

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\baseline_e2e_test.py --query "键盘USB-C接口在哪"
```

这个命令会：

- 查询 `biz_baseline`。
- 打印 Top 5 召回片段。
- 用 Top 3 片段直接调用 DashScope LLM 生成回答。

同样需要先执行过 `--index`。

## 输出结果

`baseline_e2e_test.py --test` 会生成：

```text
data/manuals/langchain_baseline/e2e_results.json
```

这个 JSON 记录每个问题的：

- `query`
- `manual`
- `must_contain`
- `hit_at_1`
- `top_title`
- `top_score`
- `answer`

目前脚本本身不会自动生成 Markdown 报告。之前生成的人工整理报告在：

```text
reports/baseline_e2e/
```

如果需要长期固定报告格式，建议后续把报告生成逻辑单独做成一个脚本，而不是混在 `baseline_e2e_test.py` 里。

## 离线快速召回评估

如果只是想快速粗看 baseline chunk 的召回，不想启动 Milvus、不想调用 DashScope，可以跑：

```powershell
.\.venv\Scripts\python.exe scripts\eval_baseline_recall.py
```

它默认读取：

```text
data/manuals/langchain_baseline/
```

并使用内置 60 问做 TF-IDF 召回评估。

写出详细 JSON：

```powershell
.\.venv\Scripts\python.exe scripts\eval_baseline_recall.py --output reports\baseline_e2e\offline_recall.json
```

也可以拿主线结构化 chunks 做同一套离线粗评：

```powershell
.\.venv\Scripts\python.exe scripts\eval_baseline_recall.py --chunks-dir data\manuals\chunks
```

注意：这个离线评估只是粗基准，不等价于真实向量检索效果。

## 如何判断结果

基线测试主要看三个指标：

- `Recall@1`：Top 1 结果是否包含期望关键词。
- `Recall@3`：Top 3 结果中是否有正确片段。
- `Recall@5`：Top 5 结果中是否有正确片段。

`baseline_e2e_test.py` 里的正确性判断是关键词命中，不是人工语义判断。因此它适合做相对比较，不适合当最终比赛评分。

## 常见问题

如果 `--test` 结果全空，通常是没有先执行 `--index`，或者 `biz_baseline` 被 `--drop` 删除了。

如果中文在 PowerShell 里显示乱码，但脚本能运行，通常是终端编码显示问题，不一定是文件内容损坏。优先设置 `$env:PYTHONIOENCODING='utf-8'`。

如果 `--index` 变慢，主要耗时在 DashScope embedding。基线会把 `data/manuals/langchain_baseline/` 下的文本 chunk 批量 embedding 后写入 `biz_baseline`。

如果要重新做一次干净测试，推荐流程是：

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\baseline_e2e_test.py --drop
.\.venv\Scripts\python.exe scripts\baseline_e2e_test.py --index
.\.venv\Scripts\python.exe scripts\baseline_e2e_test.py --test
.\.venv\Scripts\python.exe scripts\baseline_e2e_test.py --drop
```

