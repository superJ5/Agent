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
- 比赛评测脚本：批量读取问题 CSV，生成 `submission.csv`。

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
  --output data/submission.csv \
  --workers 1

# 从指定题目开始跑，或只跑一段题目
.venv/bin/python scripts/competition_eval.py --test \
  --input data/question_public.csv \
  --output data/submission_from_50.csv \
  --start-id 50 \
  --end-id 100 \
  --workers 1

# 确保 FastAPI/MCP 服务启动后再跑测试
# 注意：不会启动 Milvus 容器，也不会重新入库
.venv/bin/python scripts/competition_eval.py --run \
  --input data/question_public.csv \
  --output data/submission.csv

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
  --output data/submission_from_50.csv \
  --start-id 50

# 只跑 50 到 100 题
.venv/bin/python scripts/competition_eval.py --test \
  --input data/question_public.csv \
  --output data/submission_50_100.csv \
  --start-id 50 \
  --end-id 100

# 从第 50 题开始，只跑 20 条
.venv/bin/python scripts/competition_eval.py --test \
  --input data/question_public.csv \
  --output data/submission_50_next20.csv \
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
