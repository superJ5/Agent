# SuperBizAgent

> 企业级智能对话与运维辅助系统，支持 RAG 知识库问答、AIOps 只读诊断和多层记忆。

[![Python](https://img.shields.io/badge/Python-3.11--3.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-latest-orange.svg)](https://www.langchain.com/)
[![Milvus](https://img.shields.io/badge/Milvus-vector_db-purple.svg)](https://milvus.io/)

当前主线同时覆盖手册问答和运维诊断：手册 chunks 写入 Milvus `biz` collection，运维知识写入独立的 `aiops_knowledge` collection；Agent 可结合知识检索、日志和主机监控工具生成诊断建议。

## ✨ 核心能力

- RAG 知识问答：向量召回、BM25 召回与重排，手册和运维知识分 collection 管理。
- AIOps 辅助诊断：Plan-Execute-Replan 流程，支持腾讯云 CLS 或 mock 日志，以及本机 CPU/内存快照。
- 多轮记忆：原始会话、短期摘要、结构化 Session State 和 Milvus 长期记忆。
- 多模态输入：接口支持文本与图片。
- 评测工具：比赛 CSV 批量问答、Cloud-OpsBench 离线回放和 LongMemEval 记忆回放。

## 🛠️ 技术栈

- Web/API：FastAPI
- Agent：LangChain / LangGraph
- 模型：DashScope Qwen
- 向量库：Milvus
- 向量模型：DashScope `text-embedding-v4`
- 工具协议：MCP

## 🚀 环境准备

要求：

- Python 3.11、3.12 或 3.13
- Docker / Docker Compose
- DashScope API Key

安装依赖：

```bash
git clone <repository_url>
cd super_biz_agent_py

python3 -m pip install --user uv  # 如果系统已安装 uv，可跳过
uv venv
uv pip install -e .
```

创建仅供本机使用的 `.env.local`：

```bash
touch .env.local
vim .env.local
```

配置按 `.env`、`.env.local` 的顺序加载。建议把个人密钥和本机覆盖项放在 `.env.local`，不要提交这两个文件中的敏感信息。

至少需要配置：

```env
DASHSCOPE_API_KEY=你的 DashScope Key
API_BEARER_TOKEN=你的接口 Bearer Token
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

## 启动与入库

### 第一次部署

第一次部署，或本地 Milvus 还没有 `biz` collection 时：

```bash
make init
```

`make init` 会执行：

```text
1. make up              启动 Milvus 容器
2. make start           启动 FastAPI/MCP 服务
3. make wait            等待服务就绪
4. make index-manuals   创建/加载 biz，并入库手册知识库
```

第一次入库时，如果 `biz` 不存在，代码会创建 `biz` collection，并按当前代码里的索引配置建立向量索引。当前主线索引配置是 `COSINE + HNSW`。

注意：如果 `biz` 已经存在，`make init` 只会重新执行手册入库，不会删除整个 `biz` collection，也不会改变已有索引结构。

`make init` 不会构建 AIOps 知识库。需要运维诊断知识检索时，还要单独执行 `make index-aiops`。

### 日常启动

如果 Milvus 容器和知识库已经准备好，日常只需要：

```bash
make start
```

停止服务：

```bash
make stop
```

`make stop` 停止 FastAPI/MCP 服务，不停止 Milvus 容器。

### 手册知识库入库

当前主线知识库来源：

```text
data/manuals/chunks/*.jsonl
```

先保证容器服务是开启状态：

```bash
make up
```

入库命令：

```bash
make index-manuals
```

等价于：

```bash
.venv/bin/python scripts/index_manual_chunks.py --directory data/manuals/chunks
```

什么时候需要重新入库：

- 第一次部署。
- `data/manuals/chunks` 里的 chunk 文件变了。
- 手动删除了 Milvus 的 `biz` collection 后，需要把手册数据重新写回去。

普通重新入库只会更新手册数据，不会删除整个 Milvus 容器，也不会自动改变已有 collection 的索引方式。

什么时候需要重新创建 `biz`：

- 切换了向量索引方式，例如从 `L2 + IVF_FLAT` 改成 `COSINE + HNSW`。
- `biz` 的向量维度、字段结构或索引结构和代码不一致。
- 想彻底清空旧知识库数据并从 `data/manuals/chunks` 全量重建。

### 重建 `biz` 向量索引并重新入库

这里说的“重建向量索引”不是只改索引参数，它包含删除旧 `biz` 和重新入库两步。删除 `biz` 后，原来的向量数据也会一起清掉，所以必须重新把手册 chunks 写回 Milvus：

```text
1. 删除旧的 biz collection，让代码按当前配置重新创建索引结构。
2. 重新执行手册入库，把 data/manuals/chunks 写回新的 biz。
```

如果要把已有 `biz` 从旧索引切换到新索引，比如从 `L2 + IVF_FLAT` 切到 `COSINE + HNSW`，按下面步骤执行。

先停止服务：

```bash
make stop
```

删除旧的 `biz`：

```bash
.venv/bin/python - <<'PY'
from pymilvus import connections, utility
from app.config import config

connections.connect(alias="default", host=config.milvus_host, port=config.milvus_port)

if utility.has_collection("biz"):
    utility.drop_collection("biz")
    print("已删除旧 biz")

connections.disconnect("default")
PY
```

重新入库并启动服务：

```bash
make index-manuals
make start
```

其中 `make index-manuals` 就是重新入库；如果少了这一步，新的 `biz` collection 虽然有新索引结构，但里面没有手册知识库数据。

检查索引类型：

```bash
.venv/bin/python - <<'PY'
from pymilvus import Collection, connections
from app.config import config

connections.connect(alias="default", host=config.milvus_host, port=config.milvus_port)

collection = Collection("biz")
collection.load()

for index in collection.indexes:
    print(index.to_dict())

connections.disconnect("default")
PY
```

期望看到：

```text
metric_type: COSINE
index_type: HNSW
```

## AIOps 运维诊断

### 运维知识库

运维知识源文件位于 `aiops-docs/*.md`，切分产物位于 `data/aiops/chunks/*.jsonl`，入库到独立的 Milvus `aiops_knowledge` collection：

```bash
make up
make index-aiops
```

`make index-aiops` 会重新切分 Markdown，并重建 `aiops_knowledge`；它不会修改手册使用的 `biz` collection。

### 日志、监控与安全边界

- `CLS_DATA_SOURCE=mock`：使用内置模拟日志，适合本地联调。
- `CLS_DATA_SOURCE=tencent`：从腾讯云 CLS 查询真实日志，还需配置 `TENCENTCLOUD_SECRET_ID`、`TENCENTCLOUD_SECRET_KEY`、`TENCENT_CLS_REGION` 和 `TENCENT_CLS_TOPIC_ID` 等参数。
- Monitor MCP 通过 `psutil` 获取运行服务所在主机的当前 CPU、内存等快照；它不等同于历史监控平台。
- 当前运维工具只负责查询、分析和给出建议，不执行重启服务、修改配置或变更集群等写操作。

调用诊断接口：

```bash
curl -X POST "http://localhost:9900/api/aiops" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"aiops-demo"}' \
  --no-buffer
```

## 比赛评测

评测脚本：

```bash
.venv/bin/python scripts/competition_eval.py --help
```

常用命令：

```bash
# Milvus 和知识库已经准备好，只跑测试
.venv/bin/python scripts/competition_eval.py --test \
  --input data/question_public.csv \
  --workers 1

# 从指定题目开始跑，或只跑一段题目
.venv/bin/python scripts/competition_eval.py --test \
  --input data/question_public.csv \
  --start-id 50 \
  --end-id 100 \
  --workers 1

# 确保 FastAPI/MCP 服务启动后再跑测试
# 注意：不会启动 Milvus 容器，也不会重新入库
.venv/bin/python scripts/competition_eval.py --run \
  --input data/question_public.csv

# 第一次部署或需要重新入库时
.venv/bin/python scripts/competition_eval.py --init
```

命令区别：

```text
--init      安装依赖、启动 Milvus 容器、入库手册知识库。
--start     启动 FastAPI/MCP 服务，不启动 Milvus 容器，不入库。
--test      跑评测；服务未运行会尝试 make start，不启动 Milvus 容器，不入库。
--run       确保服务启动后跑评测，不启动 Milvus 容器，不入库。
--pipeline  执行 init + start + test，会重新入库，耗时较长。
--stop      停止 FastAPI/MCP 服务，不停止 Milvus 容器。
```

输出文件参数：

```text
默认不传 --output：写入 output/submission_YYYYMMDD_HHMMSS.csv。
--output output：写入 output/submission_YYYYMMDD_HHMMSS.csv。
--output data/my_submission.csv：按指定文件名写入，已有文件会被覆盖。
```

题目范围参数：

```text
--start-id  从指定题目 id 开始读取，包含该 id。
--end-id    读取到指定题目 id 结束，包含该 id。
--limit     最多读取多少条题目。
```

这些参数按 CSV 里的 `id` 字段过滤，不按文件行号过滤。常见用法：

```bash
# 从第 50 题开始跑到文件末尾
.venv/bin/python scripts/competition_eval.py --test \
  --input data/question_public.csv \
  --start-id 50

# 只跑 50 到 100 题
.venv/bin/python scripts/competition_eval.py --test \
  --input data/question_public.csv \
  --start-id 50 \
  --end-id 100

# 从第 50 题开始，只跑 20 条
.venv/bin/python scripts/competition_eval.py --test \
  --input data/question_public.csv \
  --start-id 50 \
  --limit 20
```

并发建议：

```text
--workers 1 最稳。
--workers 2/4/8 更快，但更容易造成模型接口拥堵或超时。
```

## Cloud-OpsBench 运维评测

仓库内的 `data/aiops_eval/cloud_ops_bench_30/` 是 Cloud-OpsBench 的 30 个用例子集。评测使用缓存的 Kubernetes 可观测数据进行确定性回放，不连接线上集群，也不代表生产环境实测。

准备标准化输入，并先各跑一条验证环境：

```bash
uv run python scripts/prepare_cloud_ops_bench_eval.py
uv run python scripts/evaluate_cloud_ops_bench.py --limit 1
uv run python scripts/evaluate_cloud_ops_bench_agent.py --limit 1
```

两种评测的边界不同：

- `evaluate_cloud_ops_bench.py` 是“证据到诊断”的 LLM 基线，会跳过 Planner、Executor、Replanner 和工具选择；`results.jsonl`、`results.summary.json` 属于这条链路。
- `evaluate_cloud_ops_bench_agent.py` 回放真实 Plan-Execute-Replan Agent，并记录计划、步骤和工具调用；`agent_results.jsonl` 属于这条链路。

仓库中的 Agent 结果当前不是完整 30 条，因此不能把它描述为完整 Agent 分数。数据来源、筛选方式和文件边界见 `data/aiops_eval/cloud_ops_bench_30/README.md`。

## API

服务地址：

- Web：`http://localhost:9900`
- API 文档：`http://localhost:9900/docs`

核心接口：

| 功能 | 方法 | 路径 | 说明 |
| --- | --- | --- | --- |
| 比赛标准对话 | POST | `/chat` | 比赛评测接口 |
| 普通对话 | POST | `/api/chat` | 旧版对话接口 |
| 流式对话 | POST | `/api/chat_stream` | SSE 流式输出 |
| 原始会话历史 | GET | `/api/chat/session/{session_id}` | 查看 session JSONL 最近记录 |
| 短期语义记忆 | GET | `/api/chat/session/{session_id}/short-term-memory` | 查看当前 session 的短期记忆 |
| 当前会话状态 | GET | `/api/chat/session/{session_id}/session-state` | 查看当前 session 的结构化任务状态 |
| 长期记忆 | GET | `/api/chat/memory/long-term` | 查看 Milvus 中的长期记忆 |
| AIOps 诊断 | POST | `/api/aiops` | 自动故障诊断 |
| 手册入库 | POST | `/api/index_manual_chunks` | 手册 chunks 入库 |
| 健康检查 | GET | `/health` | 服务状态 |

比赛接口示例：

```bash
curl -X POST "http://localhost:9900/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_BEARER_TOKEN" \
  -d '{"question":"空调滤网怎么清洁？","session_id":"test-session"}'
```

### 记忆系统接口

当前已接入四层上下文记忆：

```text
原始会话日志：data/memory/sessions/...
短期语义记忆：data/memory/short_term/...
当前会话状态：data/memory/session_state/...
长期记忆：Milvus collection long_term_memory
```

默认用户仍使用 `<session_id>` 平铺文件；非默认用户写入各目录下的 `users/<user_id哈希>/` 子目录。消息带单调递增的 `message_seq`，短期摘要用 `summary_through_message_seq` 标记已经覆盖到的位置，避免压缩后重复注入历史内容。长期记忆的读写也按可信用户身份隔离。

Milvus 里通常会同时存在多个 collection。当前项目里，`biz` 用于手册知识检索，`aiops_knowledge` 用于运维知识检索，`long_term_memory` 用于长期记忆；它们是同一个 Milvus 服务里的不同 collection。

带 `session_id` 调用 `/chat` 时，系统会写入原始会话日志，并在回答前组合 Session State、短期摘要、未被摘要覆盖的近期原始对话和检索到的长期记忆。短期摘要不是每轮都生成：只有上下文达到预算阈值时才会压缩较早的消息，并保留最近 3 轮用户对话原文；压缩后系统会重新核算硬预算。

记忆开关：

```env
MEMORY_ENABLED=true
MEMORY_WRITE_ENABLED=true
MEMORY_CONTEXT_WORKING_WINDOW_TOKENS=32768
MEMORY_CONTEXT_OUTPUT_RESERVE_TOKENS=4096
MEMORY_CONTEXT_RETRIEVAL_RESERVE_TOKENS=4096
MEMORY_CONTEXT_SAFETY_MARGIN_TOKENS=2048
MEMORY_CONTEXT_COMPACT_RATIO=0.8
```

`MEMORY_ENABLED=false` 会关闭整个业务记忆系统的读取、上下文注入和写入，包括原始会话日志、短期语义记忆、Session State 和长期记忆。`MEMORY_WRITE_ENABLED=false` 只关闭写入/更新，已有记忆仍可能被读取并注入上下文。

查看原始会话历史：

```bash
curl "http://localhost:9900/api/chat/session/test-session"
```

查看短期语义记忆：

```bash
curl "http://localhost:9900/api/chat/session/test-session/short-term-memory"
```

返回示例：

```json
{
  "session_id": "test-session",
  "exists": true,
  "content": "目标：...\n关键事实：..."
}
```

查看当前会话状态：

```bash
curl "http://localhost:9900/api/chat/session/test-session/session-state"
```

返回示例：

```json
{
  "session_id": "test-session",
  "exists": true,
  "state": {
    "session_id": "test-session",
    "goal": "定位支付接口变慢原因",
    "confirmed_facts": ["22:10 后 /api/pay P95 从 300ms 升到 3.8s"],
    "current_hypothesis": ["第三方支付回调超时可能导致接口变慢"],
    "rejected_hypotheses": ["数据库慢查询导致接口变慢"],
    "next_actions": ["检查 v2.3.1 是否改动回调重试逻辑"],
    "user_constraints": [],
    "updated_at": "2026-06-12T00:00:00+00:00"
  }
}
```

查看长期记忆：

```bash
curl "http://localhost:9900/api/chat/memory/long-term"
```

返回示例：

```json
{
  "user_id": "default",
  "count": 1,
  "memories": [
    {
      "memory_id": "ltm_xxx",
      "type": "preference",
      "content": "用户喜欢用大白话解释技术问题",
      "confidence": 0.95,
      "status": "active",
      "expires_at": null
    }
  ]
}
```

清空长期记忆：

```bash
.venv/bin/python - <<'PY'
from pymilvus import utility
from app.core.milvus_client import milvus_manager
from app.services.long_term_memory_service import long_term_memory_service

milvus_manager.connect()
name = long_term_memory_service.collection_name
if utility.has_collection(name):
    utility.drop_collection(name)
    print(f"已删除 collection: {name}")
else:
    print(f"collection 不存在: {name}")
milvus_manager.close()
PY
```

删除后不用手动重建。下次正常对话触发长期记忆检索或写入时，系统会自动重新创建 `long_term_memory` collection。

更完整的上下文记忆设计说明见：

```text
docs/context_memory_system.md
```

## LongMemEval 记忆回放评测

这个脚本测试记忆系统，不测试比赛手册 RAG。它将数据集里的历史问答原样写入隔离的会话，逐轮更新结构化状态和长期记忆；短期摘要只在达到上下文预算阈值时生成。历史助手回答不会重新调用模型生成，只有最后的测试问题会由模型回答，数据集的标准答案不会放进提示词。

运行前确保 Milvus 服务可连接、环境配置中已有 `DASHSCOPE_API_KEY`，并且 `MEMORY_WRITE_ENABLED=true`。无需启动 FastAPI，也无需给数据集新增 `user_id`：程序会为每道题、每次运行自动生成隔离的内部身份。默认路径要求 `LongMemEval` 仓库与本项目处于同级目录；数据集放在其他位置时用 `--data` 指定。

在项目根目录先检查准备运行的题目：

```bash
uv run python -m scripts.run_longmemeval --limit 1
```

默认只检查数据，不调用模型、不创建评测结果。确认题目和历史会话数量后，再真实回放一题：

```bash
uv run python -m scripts.run_longmemeval --limit 1 --execute
```

回放使用项目环境中的模型配置：`RAG_MODEL` 用于最终答题，`SHORT_TERM_MEMORY_MODEL`、`SESSION_STATE_MODEL`、`LONG_TERM_MEMORY_MODEL` 分别控制摘要、结构化状态和长期记忆更新；这些子模型配置为空时会沿回退链使用通用模型配置。Embedding 由 `DASHSCOPE_EMBEDDING_MODEL` 控制。README 不固定声明某台机器当前使用的模型，以实际环境配置为准。

也可以指定题号或数据文件：

```bash
uv run python -m scripts.run_longmemeval --case-id e47becba --execute
uv run python -m scripts.run_longmemeval --data /path/to/longmemeval_s_cleaned.json --limit 5 --execute
```

如果使用 `data/memory_eval/longmemeval/datasets/first10_evidence_plus_2distractors.json` 精简诊断集，并且第一题已单独跑完，可用一条命令跳过它、连续运行剩下 9 题：

```bash
uv run python -m scripts.run_longmemeval \
  --data data/memory_eval/longmemeval/datasets/first10_evidence_plus_2distractors.json \
  --skip 1 --limit 9 --execute
```

`--skip` 是按数据文件顺序跳过题目，不会读取之前的运行状态；剩下 9 题会写入新的结果文件。精简集使用了答案位置标签来选证据会话，只适合排障和回归，不代表官方 LongMemEval-S 分数。

`--limit` 限制题目数，不限制每题的历史对话量；即使只跑一题，也可能触发多次模型、Embedding 调用并产生费用。建议先用默认检查模式看会话数量，再决定是否加 `--execute`。`--run-id` 可指定本次运行标识；不指定时自动生成。已有同名结果不会被覆盖。

真实回放会自动检查或创建 Milvus `long_term_memory_eval` collection，并将原始会话、摘要和状态保存在 `data/memory_eval/longmemeval/`，不写入日常记忆目录或 `long_term_memory` collection。结果文件在 `data/memory_eval/longmemeval/results/`：`<run-id>.jsonl` 保存 `question_id` 和模型回答 `hypothesis`，`<run-id>.diagnostics.jsonl` 保存成功/失败及回放轮次等记录。脚本目前只生成预测结果，不自动计算评测分数。

已有答案可单独评分，不会重新回放对话。下面的示例把已跑完的三份结果合并评为 10 题；评分器从 `.env` 读取 `DASHSCOPE_API_KEY` 和 API 地址，使用 `qwen3.7-flash`，输出逐题 yes/no 和总正确率。输出文件不能与已有文件重名：

```bash
uv run python -m scripts.score_longmemeval \
  data/memory_eval/longmemeval/results/20260919_151206_7db659.jsonl \
  data/memory_eval/longmemeval/results/20260920_005303_6764c1.jsonl \
  data/memory_eval/longmemeval/results/20260920_010836_20689e.jsonl \
  --model qwen3.7-flash \
  --output data/memory_eval/longmemeval/results/first10_qwen37flash.eval.jsonl
```

评分提示词遵循 LongMemEval 的问答判定规则，但裁判模型已换成 Qwen，因此这是“Qwen 裁判的精简集分数”，不是官方 GPT-4o 裁判或完整 LongMemEval-S 分数。评分也会调用模型并产生费用；需要评分其他数据集时用 `--references` 指定相应的标准答案 JSON。

如果评测中的 Session State 或长期记忆结构化更新解析失败，未解析的模型消息与解析错误会追加到 `data/memory_eval/longmemeval/traces/<run-id>.jsonl`；没有这类失败就不会创建该文件。此日志可能含对话事实，仅供本地排障，不要公开上传。已启动的旧进程不会自动加载后来修改的诊断代码，需在新进程中重新运行才能捕获后续失败。

## 常用 Make 命令

```bash
make up              # 启动 Milvus 容器
make down            # 停止 Milvus 容器
make start           # 启动 FastAPI/MCP 服务
make stop            # 停止 FastAPI/MCP 服务
make restart         # 重启 FastAPI/MCP 服务
make index-manuals   # 入库手册 chunks
make index-aiops     # 切分并重建 AIOps 运维知识库
make status-mcp      # 查看 MCP 服务状态
make start-cls       # 单独启动 CLS MCP
make start-monitor   # 单独启动 Monitor MCP
make logs            # 查看日志
make clean           # 清理临时文件
```

## 项目结构

```text
app/
  api/                         FastAPI 路由
  services/                    RAG、向量库、记忆、AIOps 服务
  tools/                       Agent 工具
  core/                        Milvus、LLM 等核心组件
  agent/                       LangGraph Agent 逻辑
  evaluation/                  LongMemEval 等评测适配

aiops-docs/                    AIOps 运维知识源文档

data/
  manuals/chunks/              手册结构化 chunks
  aiops/chunks/                AIOps 知识切分产物
  aiops_eval/                  Cloud-OpsBench 数据和结果
  memory/                      日常记忆数据（运行时生成）
  memory_eval/                 隔离的记忆评测数据（运行时生成）

mcp_servers/                   MCP 服务
scripts/
  index_manual_chunks.py       手册 chunks 入库
  index_aiops_knowledge.py     AIOps 文档切分与入库
  competition_eval.py          比赛批量评测脚本
  evaluate_cloud_ops_bench.py  运维诊断基线评测
  run_longmemeval.py           记忆回放评测

vector-database.yml            Milvus Docker Compose
Makefile                       常用任务命令
```

## 常见问题

### 1. `python` 找不到依赖

优先使用项目虚拟环境：

```bash
.venv/bin/python scripts/competition_eval.py --help
```

不要直接用系统 `python3` 跑项目脚本，否则可能缺少依赖。

### 2. Milvus 连接失败

检查容器：

```bash
docker ps | grep milvus
```

启动容器：

```bash
make up
```

### 3. 向量 metric 不匹配

如果日志里出现类似：

```text
metric type not match
```

说明代码搜索参数和 Milvus 里已有 collection 的索引方式不一致。需要删除 `biz` 后重新入库，见“重建向量索引”。

### 4. `.env` 修改后不生效

重启服务：

```bash
make stop
make start
```

## 许可证

MIT License

author: jianghan
