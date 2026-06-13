# SuperBizAgent

> 企业级智能对话和运维助手，支持 RAG 知识库问答、比赛手册问答和 AIOps 智能诊断。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-latest-orange.svg)](https://www.langchain.com/)
[![Milvus](https://img.shields.io/badge/Milvus-vector_db-purple.svg)](https://milvus.io/)

当前主线面向比赛手册问答：读取结构化手册 chunks，写入 Milvus `biz` collection，并通过 RAG Agent 回答用户问题。

## ✨ 核心能力

- RAG 手册问答：基于 `data/manuals/chunks/*.jsonl` 入库检索。
- 多轮对话：LangGraph Agent + 会话上下文。
- 多模态输入：接口支持文本与图片。
- AIOps 辅助诊断：保留日志、监控 MCP 工具链。
- 比赛评测脚本：批量读取问题 CSV，默认生成 `output/submission_时间戳.csv`。

## 🛠️ 技术栈

- Web/API：FastAPI
- Agent：LangChain / LangGraph
- 模型：DashScope Qwen
- 向量库：Milvus
- 向量模型：DashScope `text-embedding-v4`
- 工具协议：MCP

## 🚀 环境准备

要求：

- Python 3.10+
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

配置 `.env`：

```bash
cp .env.template .env  # 如果仓库里没有模板，就直接编辑 .env
vim .env
```

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

# 答辩演示：只跑少量题目，并打印 Agent/RAG 运行链路摘要
.venv/bin/python scripts/competition_eval.py --run \
  --input data/question_public.csv \
  --limit 3 \
  --workers 1 \
  --show-chain

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

演示参数：

```text
--show-chain  每题成功后打印 Agent/RAG 摘要链路，适合答辩或截图演示。
```

`--show-chain` 会自动使用单并发，避免并发请求导致链路摘要和题目输出交错。

超时参数与服务端兜底：

```text
--timeout 30  客户端等待单题 `/chat` 响应的最长时间，默认 30 秒。
```

`--timeout` 是评测脚本这一侧的外层保护，防止接口卡死时脚本一直等待。正常情况下，服务端会先于它主动收口：比赛 `/chat` 接口默认最多等待 Agent 完整回答 26 秒；如果还没有完整答案，会再等待最多 2.5 秒读取本次检索已经缓存好的证据答案，并返回“根据已检索到的资料……”形式的兜底回答。这样总耗时通常控制在 30 秒以内。

时间层级：

```text
服务端 Agent 完整回答窗口：26s
服务端证据兜底窗口：2.5s
评测脚本客户端超时：30s
```

因此，`--timeout 30` 不会和服务端 26 秒机制冲突；它只是最后一层保险。只有服务端进程异常、网络异常或兜底也未能按时返回时，脚本才会写出 `ERROR: 请求超时`。

如需临时压测兜底逻辑，可用环境变量启动一个测试服务，例如：

```bash
COMPETITION_AGENT_TIMEOUT_SECONDS=10 \
COMPETITION_FALLBACK_TIMEOUT_SECONDS=2.5 \
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 9901
```

再让评测脚本请求测试端口：

```bash
.venv/bin/python scripts/competition_eval.py --test \
  --api-url http://localhost:9901/chat \
  --input data/question_public.csv \
  --output output \
  --workers 1 \
  --timeout 30 \
  --limit 3
```

示例输出：

```text
✅ [ID=64] 12.4s | “使用吹风机时，人员需要佩戴哪些防护装备？”
🔁 链路: CSV → /chat → Agent → RAG(vector+bm25+scan) → Qwen3-Rerank → Evidence → Answer
📌 诊断: intent=general | stage=hybrid_search | reranker=Qwen3-Rerank
📄 证据: manual_84d80d19: 0005(vector+bm25+scan), 0029(vector+bm25+scan), 0008(vector+bm25+scan)
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
原始会话日志：data/memory/sessions/<session_id>.jsonl
短期语义记忆：data/memory/short_term/<session_id>.md
当前会话状态：data/memory/session_state/<session_id>.json
长期记忆：Milvus collection long_term_memory
```

Milvus 里通常会同时存在多个 collection。当前项目里，`biz` 用于手册/知识库检索，`long_term_memory` 用于长期记忆；它们不是两个数据库，而是同一个 Milvus 服务里的两张向量表。

带 `session_id` 调用 `/chat` 时，系统会自动写入原始会话日志。回答前会检索长期记忆，并读取结构化 Session State、短期语义记忆和最近 1-3 轮原始对话作为上下文；回答结束后会后台更新短期语义记忆、Session State 和长期记忆。

记忆开关：

```env
MEMORY_ENABLED=true
MEMORY_WRITE_ENABLED=true
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

## 常用 Make 命令

```bash
make up              # 启动 Milvus 容器
make down            # 停止 Milvus 容器
make start           # 启动 FastAPI/MCP 服务
make stop            # 停止 FastAPI/MCP 服务
make restart         # 重启 FastAPI/MCP 服务
make index-manuals   # 入库手册 chunks
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

data/
  manuals/chunks/              手册结构化 chunks，当前主知识库来源
  memory/                      记忆系统数据

mcp_servers/                   MCP 服务
scripts/
  index_manual_chunks.py       手册 chunks 入库
  competition_eval.py          比赛批量评测脚本

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
