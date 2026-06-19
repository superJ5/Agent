# SuperBizAgent 智能体 API 接口说明文档

## 1. 文档信息

| 项目 | 内容 |
| --- | --- |
| 系统名称 | SuperBizAgent |
| 服务类型 | 标准 RESTful API 服务 |
| API 框架 | FastAPI |
| 当前版本 | 1.0.0 |
| 线上服务地址 | `http://123.56.142.146:9900` |
| 在线接口文档 | `http://123.56.142.146:9900/docs` |
| OpenAPI 描述 | `http://123.56.142.146:9900/openapi.json` |

本文档说明 SuperBizAgent 智能体的 RESTful API 调用方式，覆盖比赛标准对话接口、非流式输出、流式输出、多模态输入、上下文记忆系统、健康检查和调试接口。

当前配置状态：

- `/chat` 非流式文本问答：已在线验证通过。
- `/chat` 流式文本问答：已在线验证通过。
- 多模态图片字段：API 层已支持；可通过请求参数 `model` 为单次请求指定视觉模型，不传时使用 `.env` 默认模型。
- 记忆系统：代码能力已实现；当前本地 `.env` 为 `MEMORY_ENABLED=false`，业务记忆读取、注入和写入处于关闭状态。如需演示记忆能力，需要改为 `MEMORY_ENABLED=true` 并重启服务。

## 2. RESTful 服务说明

本项目已将智能体能力封装为 HTTP API 服务。调用方不需要直接运行 Python 脚本或了解内部 Agent 实现，只需要通过 HTTP 请求访问固定 URL，并使用 JSON 或 SSE 接收结构化响应。

核心能力通过以下方式对外提供：

- `POST /chat`：比赛标准智能体对话接口，支持非流式、流式、文本输入和图片输入字段。
- `GET /health`：服务健康检查。
- `GET /api/chat/session/{session_id}`：查看会话历史。
- `GET /api/chat/session/{session_id}/short-term-memory`：查看短期语义记忆。
- `GET /api/chat/session/{session_id}/session-state`：查看结构化会话状态。
- `GET /api/chat/memory/long-term`：查看长期记忆。

## 3. 服务认证

比赛标准接口 `/chat` 使用 Bearer Token 认证。

请求头：

```http
Authorization: Bearer <API_BEARER_TOKEN>
Content-Type: application/json
```

`API_BEARER_TOKEN` 由服务端 `.env` 配置：

```env
API_BEARER_TOKEN=你的接口访问令牌
```

未携带认证信息、认证格式错误或 token 不匹配时，接口返回 HTTP `401`。

示例：

```bash
TOKEN="替换为实际 API_BEARER_TOKEN"

curl -X POST "http://123.56.142.146:9900/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{
    "question": "空调滤网怎么清洁？",
    "session_id": "demo-session-001",
    "stream": false
  }'
```

## 4. 接口总览

| 功能 | 方法 | 路径 | 是否认证 | 说明 |
| --- | --- | --- | --- | --- |
| 比赛标准对话 | POST | `/chat` | 是 | 推荐提交和评测使用 |
| 健康检查 | GET | `/health` | 否 | 检查服务和 Milvus 状态 |
| Swagger 文档 | GET | `/docs` | 否 | FastAPI 自动接口文档 |
| OpenAPI JSON | GET | `/openapi.json` | 否 | 标准 OpenAPI 描述 |
| 旧版非流式对话 | POST | `/api/chat` | 否 | 兼容旧调用方式 |
| 旧版流式对话 | POST | `/api/chat_stream` | 否 | 兼容旧 SSE 调用方式 |
| 清空会话 | POST | `/api/chat/clear` | 否 | 清空 LangGraph 会话状态 |
| 原始会话历史 | GET | `/api/chat/session/{session_id}` | 否 | 查看最近会话消息 |
| 短期语义记忆 | GET | `/api/chat/session/{session_id}/short-term-memory` | 否 | 查看短期记忆 Markdown |
| 结构化会话状态 | GET | `/api/chat/session/{session_id}/session-state` | 否 | 查看当前任务状态 |
| 长期记忆 | GET | `/api/chat/memory/long-term` | 否 | 查看 Milvus 长期记忆 |
| 手册图片读取 | GET | `/api/manual-images/{image_id}` | 否 | 读取手册图片 |
| 文件上传 | POST | `/api/upload` | 否 | 上传 txt/md 并创建索引 |
| 手册 chunks 入库 | POST | `/api/index_manual_chunks` | 否 | 将结构化手册 chunks 写入 Milvus |
| AIOps 诊断 | POST | `/api/aiops` | 否 | SSE 流式运维诊断 |

比赛提交建议重点说明和演示 `/chat`、`/health`、记忆查询接口。

## 5. 比赛标准对话接口

### 5.1 基本信息

| 项目 | 内容 |
| --- | --- |
| URL | `/chat` |
| 完整地址 | `http://123.56.142.146:9900/chat` |
| 方法 | `POST` |
| Content-Type | `application/json` |
| 认证 | `Authorization: Bearer <API_BEARER_TOKEN>` |
| 主要能力 | 文本问答、RAG 手册检索、多轮会话、流式输出、多模态输入字段、上下文记忆 |

### 5.2 请求参数

```json
{
  "question": "用户问题",
  "session_id": "demo-session-001",
  "stream": false,
  "model": null,
  "images": []
}
```

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `question` | string | 是 | 无 | 用户问题，不能为空字符串 |
| `session_id` | string | 否 | 自动生成 | 会话 ID；同一轮多轮对话应复用同一个 ID |
| `stream` | boolean | 否 | `false` | 是否使用 SSE 流式输出 |
| `model` | string | 否 | `.env` 默认模型 | 本次请求临时使用的模型名，例如 `qwen-vl-plus` |
| `images` | string[] | 否 | `[]` | Base64 图片列表，最多 3 张；支持 Data URL 或裸 Base64 |

兼容旧字段：

| 新字段 | 兼容旧字段 |
| --- | --- |
| `question` | `Question` |
| `session_id` | `Id` |

图片格式限制：

- 最多 3 张图片。
- 每张图片最大 5 MB。
- 推荐使用 Data URL Base64 格式。
- 也兼容裸 Base64；后端会根据图片文件头自动识别 PNG、JPEG、WEBP 并补齐 Data URL 前缀。
- Data URL 支持前缀：
  - `data:image/png;base64,`
  - `data:image/jpg;base64,`
  - `data:image/jpeg;base64,`
  - `data:image/webp;base64,`

### 5.3 非流式请求示例

```bash
TOKEN="替换为实际 API_BEARER_TOKEN"

curl -X POST "http://123.56.142.146:9900/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{
    "question": "空调滤网怎么清洁？",
    "session_id": "docs-api-test-001",
    "stream": false,
    "model": null
  }'
```

成功响应：

```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "answer": "建议每两周清洁一次空气滤网...",
    "session_id": "docs-api-test-001",
    "timestamp": 1781876412
  }
}
```

响应字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `code` | integer | 业务状态码；`0` 表示成功 |
| `msg` | string | 状态消息；成功为 `success` |
| `data.answer` | string | 智能体回答 |
| `data.session_id` | string | 本次会话 ID |
| `data.timestamp` | integer | Unix 秒级时间戳 |

### 5.4 流式请求示例

当 `stream=true` 时，`/chat` 返回 `text/event-stream`，采用 SSE 方式逐段输出。

```bash
TOKEN="替换为实际 API_BEARER_TOKEN"

curl -N -X POST "http://123.56.142.146:9900/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{
    "question": "电钻充电灯闪烁是什么意思？",
    "session_id": "docs-api-test-stream-001",
    "stream": true,
    "model": null
  }'
```

流式响应示例：

```text
event: message
data: {"type": "content", "data": "电钻充电器指示灯闪烁"}

event: message
data: {"type": "content", "data": "的含义取决于充电器型号..."}

event: message
data: {"type": "done", "data": null}
```

SSE 消息类型：

| `type` | 说明 |
| --- | --- |
| `content` | 回答文本片段，需要客户端按顺序拼接 |
| `tool_call` | Agent 工具调用事件 |
| `search_results` | 检索结果事件 |
| `debug` | 调试事件，包含节点和消息类型 |
| `done` | 输出完成 |
| `error` | 输出过程中发生错误 |

客户端处理建议：

- 只展示 `type=content` 的 `data` 字段即可得到完整回答。
- 收到 `type=done` 后结束本轮展示。
- 收到 `type=error` 时提示失败信息。

### 5.5 多模态输入示例

接口请求模型支持图片数组 `images`。图片推荐以 Data URL Base64 字符串传入，也可以直接传裸 Base64。

```bash
TOKEN="替换为实际 API_BEARER_TOKEN"
IMAGE_BASE64="data:image/png;base64,替换为图片Base64内容"

curl -X POST "http://123.56.142.146:9900/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d "{
    \"question\": \"请识别图片中的设备，并说明如何操作。\",
    \"session_id\": \"demo-image-session-001\",
    \"stream\": false,
    \"model\": \"qwen-vl-plus\",
    \"images\": [\"${IMAGE_BASE64}\"]
  }"
```

裸 Base64 也可以直接传入，后端会自动补齐前缀：

```json
{
  "question": "请识别图片中的设备，并说明如何操作。",
  "session_id": "demo-image-session-raw-001",
  "stream": false,
  "model": "qwen-vl-plus",
  "images": ["这里填写不带 data:image 前缀的纯 Base64 字符串"]
}
```

当前实现说明：

- API 层已经支持 `images` 字段校验和透传。
- 有图片时，系统会优先进入手册 RAG 路由。
- 请求可以携带 `model` 临时指定视觉模型，例如 `qwen-vl-plus`、`qwen-vl-max` 或当前 DashScope 可用的 Qwen-VL 系列模型。
- 不传 `model` 时使用 `.env` 中的默认 `RAG_MODEL`，当前为 `qwen3.7-max`。
- 如视觉模型仍返回 DashScope 多模态格式错误，需要继续核对 `app/services/multimodal_message_builder.py` 与 DashScope 当前 OpenAI-compatible 图片消息格式。

## 6. 错误响应

### 6.1 认证错误

未提供认证头：

```bash
curl -X POST "http://123.56.142.146:9900/chat" \
  -H "Content-Type: application/json" \
  -d '{"question":"hello"}'
```

响应：

```json
{
  "detail": "Missing or invalid Authorization header"
}
```

HTTP 状态码：`401`

token 为空：

```json
{
  "detail": "Empty bearer token"
}
```

token 错误：

```json
{
  "detail": "Invalid bearer token"
}
```

### 6.2 参数校验错误

当 `question` 为空、图片数量超过 3 张、图片不是合法 Base64、图片超过 5 MB 时，FastAPI 返回 HTTP `422`。

示例：

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "question"],
      "msg": "Value error, question 不能为空"
    }
  ]
}
```

### 6.3 业务处理错误

当智能体内部调用失败时，`/chat` 可能以 HTTP `200` 返回业务错误：

```json
{
  "code": 500,
  "msg": "错误信息",
  "data": {
    "answer": "",
    "session_id": "demo-session-001",
    "timestamp": 1781876401
  }
}
```

调用方应同时检查 HTTP 状态码和响应体中的 `code`。

## 7. 健康检查接口

### 7.1 请求

```bash
curl -X GET "http://123.56.142.146:9900/health"
```

### 7.2 成功响应

```json
{
  "code": 200,
  "message": "服务运行正常",
  "data": {
    "service": "SuperBizAgent",
    "version": "1.0.0",
    "status": "healthy",
    "milvus": {
      "status": "connected",
      "message": "Milvus 连接正常"
    }
  }
}
```

### 7.3 字段说明

| 字段 | 说明 |
| --- | --- |
| `data.status` | 整体服务状态，`healthy` 表示可用 |
| `data.milvus.status` | Milvus 连接状态，`connected` 表示向量库可用 |
| `code` | HTTP 语义状态码，健康时为 `200` |

## 8. 记忆系统使用说明

### 8.1 记忆系统结构

当前项目代码已接入四层上下文记忆：

| 层级 | 存储位置 | 作用 |
| --- | --- | --- |
| 原始会话日志 | `data/memory/sessions/<session_id>.jsonl` | 记录用户和助手原始消息 |
| 短期语义记忆 | `data/memory/short_term/<session_id>.md` | 总结当前会话目标、关键事实、假设和进展 |
| 结构化会话状态 | `data/memory/session_state/<session_id>.json` | 保存程序可读的当前任务状态 |
| 长期记忆 | Milvus collection `long_term_memory` | 跨会话保存稳定事实、用户偏好和长期经验 |

### 8.2 当前配置状态和启用方式

当前本地 `.env` 记忆相关配置为：

```env
MEMORY_ENABLED=false
MEMORY_WRITE_ENABLED=true
MEMORY_ROOT=data/memory
MEMORY_RECENT_LIMIT=10
LONG_TERM_MEMORY_COLLECTION_NAME=long_term_memory
LONG_TERM_MEMORY_TOP_K=5
LONG_TERM_MEMORY_MIN_CONFIDENCE=0.7
LONG_TERM_MEMORY_MIN_RELEVANCE=0.2
```

这表示记忆能力当前处于关闭状态。`MEMORY_ENABLED=false` 会关闭业务记忆系统的读取、上下文注入和写入，即使 `MEMORY_WRITE_ENABLED=true`，也不会生成或更新短期记忆、Session State 和长期记忆。

如需在比赛演示中展示记忆系统，需要将服务端 `.env` 改为：

```env
MEMORY_ENABLED=true
MEMORY_WRITE_ENABLED=true
MEMORY_ROOT=data/memory
MEMORY_RECENT_LIMIT=10
LONG_TERM_MEMORY_COLLECTION_NAME=long_term_memory
LONG_TERM_MEMORY_TOP_K=5
LONG_TERM_MEMORY_MIN_CONFIDENCE=0.7
LONG_TERM_MEMORY_MIN_RELEVANCE=0.2
```

修改后重启服务：

```bash
make stop
make start
```

配置含义：

| 配置项 | 说明 |
| --- | --- |
| `MEMORY_ENABLED` | 是否启用记忆读取、注入和写入；当前本地配置为 `false` |
| `MEMORY_WRITE_ENABLED` | 是否允许写入和更新记忆 |
| `MEMORY_ROOT` | 本地记忆文件根目录 |
| `MEMORY_RECENT_LIMIT` | 最近原始对话读取条数 |
| `LONG_TERM_MEMORY_COLLECTION_NAME` | 长期记忆 Milvus collection 名称 |
| `LONG_TERM_MEMORY_TOP_K` | 每次检索长期记忆条数 |
| `LONG_TERM_MEMORY_MIN_CONFIDENCE` | 长期记忆最低置信度 |
| `LONG_TERM_MEMORY_MIN_RELEVANCE` | 长期记忆最低相关度 |

### 8.3 调用时如何使用记忆

记忆启用后，调用方只需要在多轮对话中保持同一个 `session_id`：

```bash
TOKEN="替换为实际 API_BEARER_TOKEN"

curl -X POST "http://123.56.142.146:9900/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{
    "question": "我在排查支付接口变慢，22:10 后 P95 从 300ms 升到 3.8s。你先帮我整理排查方向。",
    "session_id": "memory-demo-001",
    "stream": false
  }'
```

第二轮继续使用同一个 `session_id`：

```bash
curl -X POST "http://123.56.142.146:9900/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{
    "question": "数据库慢查询和连接池都查过了，没有异常。22:05 发布了 payment-service v2.3.1，下一步怎么查？",
    "session_id": "memory-demo-001",
    "stream": false
  }'
```

记忆启用后，系统会自动执行：

1. 写入本轮用户原始消息。
2. 根据 `session_id` 读取短期语义记忆。
3. 读取结构化 Session State。
4. 从 Milvus 检索相关长期记忆。
5. 读取最近若干条原始对话。
6. 将这些上下文作为额外 `SystemMessage` 注入 Agent。
7. 回答结束后，后台更新短期记忆、Session State 和长期记忆。

### 8.4 查看原始会话历史

```bash
curl "http://123.56.142.146:9900/api/chat/session/memory-demo-001"
```

响应：

```json
{
  "session_id": "memory-demo-001",
  "message_count": 2,
  "history": [
    {
      "role": "user",
      "content": "我在排查支付接口变慢...",
      "timestamp": "2026-06-19T13:40:00+00:00"
    },
    {
      "role": "assistant",
      "content": "可以从调用链、数据库、第三方依赖等方向排查...",
      "timestamp": "2026-06-19T13:40:15+00:00"
    }
  ]
}
```

### 8.5 查看短期语义记忆

```bash
curl "http://123.56.142.146:9900/api/chat/session/memory-demo-001/short-term-memory"
```

响应：

```json
{
  "session_id": "memory-demo-001",
  "exists": true,
  "content": "目标：排查支付接口 P95 耗时突增问题\n\n关键事实：..."
}
```

### 8.6 查看结构化会话状态

```bash
curl "http://123.56.142.146:9900/api/chat/session/memory-demo-001/session-state"
```

响应：

```json
{
  "session_id": "memory-demo-001",
  "exists": true,
  "state": {
    "session_id": "memory-demo-001",
    "goal": "定位支付接口变慢原因",
    "confirmed_facts": ["22:10 后 P95 从 300ms 升到 3.8s"],
    "current_hypothesis": ["第三方支付回调超时可能导致变慢"],
    "rejected_hypotheses": ["数据库慢查询导致变慢"],
    "next_actions": ["检查 v2.3.1 回调重试逻辑"],
    "user_constraints": [],
    "updated_at": "2026-06-19T13:40:00+00:00"
  }
}
```

### 8.7 查看长期记忆

```bash
curl "http://123.56.142.146:9900/api/chat/memory/long-term?user_id=default&include_inactive=false&limit=100"
```

响应：

```json
{
  "user_id": "default",
  "count": 1,
  "memories": [
    {
      "memory_id": "ltm_xxx",
      "user_id": "default",
      "type": "preference",
      "content": "用户喜欢直接给出排查步骤",
      "evidence": "用户多次要求给下一步排查动作",
      "confidence": 0.95,
      "status": "active",
      "source_session_id": "memory-demo-001",
      "created_at": "2026-06-19T13:40:00+00:00",
      "updated_at": "2026-06-19T13:40:00+00:00",
      "expires_at": null
    }
  ]
}
```

查询参数：

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `user_id` | string | `default` | 用户 ID |
| `include_inactive` | boolean | `false` | 是否包含已失效记忆 |
| `limit` | integer | `100` | 最大返回条数 |

## 9. 旧版兼容接口

### 9.1 旧版非流式对话

```bash
curl -X POST "http://123.56.142.146:9900/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "空调滤网怎么清洁？",
    "session_id": "legacy-session-001"
  }'
```

响应：

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "success": true,
    "answer": "智能体回答",
    "errorMessage": null
  }
}
```

### 9.2 旧版流式对话

```bash
curl -N -X POST "http://123.56.142.146:9900/api/chat_stream" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "电钻充电灯闪烁是什么意思？",
    "session_id": "legacy-stream-session-001"
  }'
```

返回格式同 `/chat` 的 `stream=true`。

比赛提交建议优先使用 `/chat`，旧版接口仅作为兼容说明。

## 10. 手册图片接口

当回答中包含手册图片 ID 时，可通过以下接口读取原始图片。

```bash
curl -X GET "http://123.56.142.146:9900/api/manual-images/Manual27_10" \
  --output Manual27_10.jpg
```

| 项目 | 内容 |
| --- | --- |
| 方法 | `GET` |
| 路径 | `/api/manual-images/{image_id}` |
| 参数 | `image_id` 为手册图片 ID，不带文件后缀 |
| 成功响应 | 图片文件 |
| 失败响应 | HTTP `404` |

## 11. 知识库入库接口

### 11.1 手册 chunks 入库

```bash
curl -X POST "http://123.56.142.146:9900/api/index_manual_chunks"
```

可选参数：

```bash
curl -X POST "http://123.56.142.146:9900/api/index_manual_chunks?directory_path=data/manuals/chunks"
```

说明：

- 默认读取 `data/manuals/chunks/*.jsonl`。
- 写入 Milvus `biz` collection。
- 当前主线索引配置为 `COSINE + HNSW`。

响应：

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "success": true,
    "indexed_count": 100,
    "failed_count": 0
  }
}
```

### 11.2 文件上传并索引

```bash
curl -X POST "http://123.56.142.146:9900/api/upload" \
  -F "file=@example.md"
```

限制：

- 仅支持 `.txt`、`.md`。
- 单文件最大 10 MB。
- 上传后保存到 `uploads/` 并尝试创建向量索引。

## 12. AIOps 流式诊断接口

```bash
curl -N -X POST "http://123.56.142.146:9900/api/aiops" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "aiops-session-001"
  }'
```

返回 SSE 事件：

| `type` | 说明 |
| --- | --- |
| `status` | 状态更新 |
| `plan` | 诊断计划 |
| `step_complete` | 单步执行完成 |
| `report` | 最终诊断报告 |
| `complete` | 诊断完成 |
| `error` | 错误信息 |

## 13. 部署和环境配置

核心 `.env` 配置项：

```env
APP_NAME=SuperBizAgent
APP_VERSION=1.0.0
HOST=0.0.0.0
PORT=9900

DASHSCOPE_API_KEY=你的 DashScope Key
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen3.7-max
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v4

API_BEARER_TOKEN=你的接口 Bearer Token

MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_TIMEOUT=10000

RAG_TOP_K=2
RAG_MODEL=qwen3.7-max
RAG_ENABLE_VECTOR_RECALL=true
RAG_ENABLE_BM25_RECALL=true
RAG_VECTOR_WEIGHT=0.6
RAG_BM25_WEIGHT=0.4
RAG_RERANKER_PROVIDER=dashscope
RAG_RERANKER_MODEL=qwen3-rerank

MEMORY_ENABLED=false
MEMORY_WRITE_ENABLED=true
MEMORY_ROOT=data/memory
```

启动命令：

```bash
make start
```

初始化 Milvus 和手册知识库：

```bash
make init
```

重新入库手册 chunks：

```bash
make index-manuals
```

停止服务：

```bash
make stop
```

## 14. 线上实测结果

实测日期：2026-06-19

| 测试项 | 结果 | 说明 |
| --- | --- | --- |
| `GET /health` | 通过 | HTTP 200，服务 healthy，Milvus connected |
| `GET /docs` | 通过 | HTTP 200，Swagger 页面可访问 |
| `POST /chat` 非流式 | 通过 | 成功返回 `code=0` 和答案 |
| `POST /chat` 流式 | 通过 | SSE 持续返回 `content`，最终返回 `done` |
| `POST /chat` 图片输入 | 部分通过 | API 参数校验通过，模型侧返回 DashScope 400，多模态模型/消息格式需调整 |
| 记忆查询接口 | 可访问但当前为空 | 当前配置 `MEMORY_ENABLED=false`，需开启后重启服务再演示 |

健康检查实测响应：

```json
{
  "code": 200,
  "message": "服务运行正常",
  "data": {
    "service": "SuperBizAgent",
    "version": "1.0.0",
    "status": "healthy",
    "milvus": {
      "status": "connected",
      "message": "Milvus 连接正常"
    }
  }
}
```

非流式对话实测请求：

```bash
curl -X POST "http://123.56.142.146:9900/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{
    "question": "空调滤网怎么清洁？",
    "session_id": "docs-api-test-001",
    "stream": false
  }'
```

流式对话实测请求：

```bash
curl -N -X POST "http://123.56.142.146:9900/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{
    "question": "电钻充电灯闪烁是什么意思？",
    "session_id": "docs-api-test-stream-001",
    "stream": true
  }'
```

## 15. 调用方接入建议

1. 评测或正式调用优先使用 `POST /chat`。
2. 每个用户或任务使用稳定的 `session_id`，这样才能启用多轮上下文和记忆能力。
3. 非流式场景读取 `data.answer`。
4. 流式场景拼接所有 `type=content` 的 `data` 字段，收到 `type=done` 后结束。
5. 调用方应同时判断 HTTP 状态码和响应体 `code`。
6. 图片输入正式演示前，需要先修复当前线上多模态模型/消息格式兼容问题。
7. 记忆系统正式演示前，需要将 `MEMORY_ENABLED` 改为 `true` 并重启服务。
