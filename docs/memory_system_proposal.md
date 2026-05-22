# Memory System

这份文档说明当前项目的记忆系统是怎么工作的。

核心目标：

```text
当前会话能连续追问。
服务重启后能恢复当前会话最近上下文。
长期稳定记忆可直接给 Agent 参考。
阶段性总结可通过向量检索查询。
原始聊天记录只归档，不默认进入检索。
```

## 1. 总体结构

当前记忆文件都放在：

```text
data/memory/
```

目录结构：

```text
data/memory/
  sessions/
    <session_id>.jsonl
  daily/
    YYYY-MM-DD.md
  MEMORY_PENDING.md
  MEMORY.md
```

整体分工：

```text
sessions/*.jsonl
原始聊天记录。实时写入，只用于归档和当前 session 最近 N 条恢复。

daily/*.md
每日/阶段性总结。由 sessions 自动总结生成，进入 Milvus memory 向量库。

MEMORY_PENDING.md
长期记忆候选。由总结脚本生成，等待人工确认。

MEMORY.md
正式长期记忆。人工确认后维护，每次对话直接进入系统提示词。
```

## 2. 一次对话的记忆流程

用户发起对话后，流程大致是：

```text
前端发送 question + session_id
        ↓
app/api/chat.py
        ↓
写入 sessions/<session_id>.jsonl
        ↓
app/services/rag_agent_service.py
        ↓
构造 Agent messages
        ↓
模型回答
        ↓
回答再次写入 sessions/<session_id>.jsonl
```

Agent 实际看到的上下文来自三部分：

```text
1. 系统提示词
2. MEMORY.md 正式长期记忆
3. 当前 session 的上下文
```

当前 session 上下文优先来自：

```text
MemorySaver
```

如果服务重启后 `MemorySaver` 没有当前会话状态，就从：

```text
data/memory/sessions/<session_id>.jsonl
```

读取最近 N 条历史。

N 的配置在：

```python
memory_recent_limit = 10
```

## 3. 每个文件怎么来的

### sessions/*.jsonl

来源：

```text
用户每次提问时自动写入。
助手每次回答完成后自动写入。
```

对应代码：

```text
app/api/chat.py
app/services/memory_service.py
```

作用：

```text
保存原始聊天流水。
服务重启后恢复当前 session 最近 N 条。
作为 daily 自动总结的数据来源。
```

不会做的事：

```text
不进入 memory_search。
不进入 Milvus memory 向量库。
不默认混入其他会话。
```

### daily/*.md

来源：

```text
运行总结脚本，从所有 sessions/*.jsonl 中筛选同一天的记录并总结生成。
```

比如：

```bash
python scripts/summarize_memory.py --date 2026-05-22
```

这个命令会扫描所有 session 文件，只取本地日期为 `2026-05-22` 的记录，生成：

```text
data/memory/daily/2026-05-22.md
```

如果同一个 session 跨了多天，内容会按日期分散到不同 daily：

```text
session_A.jsonl 的 2026-05-21 内容 → daily/2026-05-21.md
session_A.jsonl 的 2026-05-22 内容 → daily/2026-05-22.md
```

总结脚本会过滤低价值内容，例如：

```text
普通问候
今天星期几
简单数学计算
重复闲聊
临时调试噪声
```

如果当天没有高价值内容，脚本会成功返回，但不写 daily：

```text
success = true
daily_chars = 0
```

作用：

```text
保存阶段性总结。
进入 Milvus memory 向量库。
供 memory_search 检索。
```

### MEMORY_PENDING.md

来源：

```text
总结脚本从 daily 总结结果中提取长期记忆候选。
```

作用：

```text
作为人工审核区。
模型可以提候选，但不直接改正式 MEMORY.md。
```

示例：

```text
# MEMORY_PENDING

## 2026-05-22
- [ ] 用户希望 Python/FastAPI 解释更慢、更具体。
- [ ] 记忆系统决策：sessions JSONL 只做归档和当前 session 恢复。
```

人工确认后，再手动移动到：

```text
data/memory/MEMORY.md
```

### MEMORY.md

来源：

```text
人工维护。
从 MEMORY_PENDING.md 中确认后合并。
也可以手动直接编辑。
```

作用：

```text
保存正式长期记忆。
每次对话都会进入系统提示词。
```

适合放：

```text
用户长期偏好
项目背景
已确认技术方案
关键配置
重要决策
```

不会做的事：

```text
不进入 Milvus memory 向量库。
不通过 memory_search 检索。
```

原因：

```text
MEMORY.md 是核心长期记忆，当前内容较短，直接进系统提示词更稳定。
daily 已经负责从 sessions 中提炼阶段性内容并进入向量库。
```

## 4. memory_search 查什么

`memory_search` 是 Agent 可调用的记忆工具。

对应代码：

```text
app/tools/memory_tool.py
```

当前只查：

```text
data/memory/daily/*.md
```

不查：

```text
data/memory/MEMORY.md
data/memory/sessions/*.jsonl
```

原因：

```text
MEMORY.md 已经直接进入系统提示词。
sessions 是原始流水，不适合直接检索给 Agent。
daily 是整理后的高价值阶段性记忆，适合向量检索。
```

## 5. 向量库怎么更新

daily 不会自动进入 Milvus。

当 daily 更新后，需要手动索引：

```bash
python scripts/index_memory.py --rebuild
```

这里的“重建 memory 向量库”指的是：

```text
清空 Milvus 里的 memory collection
        ↓
重新读取 data/memory/daily/*.md
        ↓
重新生成向量索引
```

它不会删除本地文件：

```text
不会删除 data/memory/daily/*.md
不会删除 data/memory/MEMORY.md
不会删除 data/memory/sessions/*.jsonl
```

重建的目的：

```text
清理旧索引
按当前规则重新生成索引
避免以前索引过的 MEMORY.md 或 sessions 残留在 Milvus 中
```

或者在总结时顺便索引：

```bash
python scripts/summarize_memory.py --date 2026-05-22 --index
```

如果想清理旧索引并重建：

```bash
python scripts/summarize_memory.py --date 2026-05-22 --index --rebuild-index
```

索引脚本只会把下面内容写入 Milvus memory collection：

```text
data/memory/daily/*.md
```

不会索引：

```text
data/memory/MEMORY.md
data/memory/sessions/*.jsonl
```

## 6. 常用命令

生成某一天的 daily：

```bash
python scripts/summarize_memory.py --date 2026-05-22
```

生成 daily 并更新 memory 向量库：

```bash
python scripts/summarize_memory.py --date 2026-05-22 --index
```

生成 daily 并重建 memory 向量库：

```bash
python scripts/summarize_memory.py --date 2026-05-22 --index --rebuild-index
```

单独重建 memory 向量库：

```bash
python scripts/index_memory.py --rebuild
```

如果使用虚拟环境：

```bash
.venv/bin/python scripts/summarize_memory.py --date 2026-05-22
.venv/bin/python scripts/index_memory.py --rebuild
```

## 7. 相关代码

```text
app/api/chat.py
负责接收聊天请求，并在 Agent 前后写入原始聊天记录。

app/services/memory_service.py
负责 sessions JSONL 读写、最近 N 条恢复、MEMORY.md 读取、daily 关键词兜底搜索。

app/services/rag_agent_service.py
负责构造 Agent 上下文：系统提示词 + MEMORY.md + 当前 session 历史 + 当前问题。

app/services/memory_summary_service.py
负责把 sessions 原始记录总结成 daily，并生成 MEMORY_PENDING.md 候选。

app/services/memory_index_service.py
负责把 daily 转成向量 chunk 并写入 Milvus memory collection。

app/services/memory_vector_store.py
负责 Milvus memory collection 的创建、写入和检索。

app/tools/memory_tool.py
负责把 daily 记忆检索包装成 Agent 工具 memory_search。

scripts/summarize_memory.py
手动触发 daily 总结。

scripts/index_memory.py
手动触发 daily 向量索引。
```

## 8. 当前设计原则

```text
当前 session 的连续对话，靠 MemorySaver / 最近 N 条 JSONL。
跨会话的原始聊天，不直接查。
跨会话沉淀，先总结到 daily。
长期稳定内容，人工确认后进入 MEMORY.md。
MEMORY.md 直接进系统提示词。
daily 进入向量库，由 memory_search 按需检索。
```

## 9. 当前完成情况

当前已经完成比赛项目可用版本：

```text
原始聊天记录保存：已完成。
当前 session 最近 N 条恢复：已完成。
MEMORY.md 正式长期记忆：已完成。
daily 自动总结脚本：已完成。
低价值内容过滤：已完成。
MEMORY_PENDING.md 长期记忆候选：已完成。
memory_search 只查 daily：已完成。
daily 向量索引：已完成。
文档说明：已完成。
```

当前版本可以支持：

```text
同一会话连续追问。
服务重启后恢复当前会话最近上下文。
长期稳定记忆直接进入 Agent 上下文。
原始聊天按日期总结成 daily。
daily 进入向量库后供 Agent 检索。
```

## 10. 后续修改建议

这些是增强项，不影响当前主干使用。

### 增量索引

当前索引方式：

```text
python scripts/index_memory.py --rebuild
```

会重建整个 Milvus memory collection。

后续可以改成：

```text
只更新变化过的 daily 文件。
用 index_state.json 记录每个 daily 的更新时间、文件大小、内容哈希。
没有变化的 daily 跳过。
```

### 自动定时总结

当前 daily 总结需要手动运行：

```text
python scripts/summarize_memory.py --date YYYY-MM-DD
```

后续可以增加：

```text
每天固定时间自动总结。
服务关闭前自动总结。
每 N 轮对话后后台总结。
```

### MEMORY_PENDING 合并工具

当前流程是人工打开：

```text
data/memory/MEMORY_PENDING.md
```

然后手动复制确认项到：

```text
data/memory/MEMORY.md
```

后续可以做一个小脚本：

```text
scripts/promote_memory.py
```

作用：

```text
读取 MEMORY_PENDING.md 中已勾选的项。
自动追加到 MEMORY.md。
把已处理项从 pending 中移除或标记完成。
```

### 前端或接口触发

当前总结和索引都通过脚本运行。

后续可以加：

```text
后台管理接口
前端按钮
定时任务状态查看
```

例如：

```text
POST /memory/summarize
POST /memory/index
GET /memory/status
```

### 大规模历史优化

当前数据量小，按日期扫描所有 session 文件可以接受。

如果后面聊天记录很多，可以优化：

```text
按日期建立 session 记录索引。
只读取目标日期相关文件。
按 session 文件增量扫描。
把已总结位置记录下来，避免重复处理。
```
