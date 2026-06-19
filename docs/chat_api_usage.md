# `/chat` 接口使用文档

## 1. 接口概述

`/chat` 是智能体多模态对话交互的唯一核心入口。

接口通过标准 RESTful API 提供服务，支持：

- JSON 格式文本输入。
- Base64 格式图片输入。
- 非流式 JSON 响应。
- 流式 SSE 响应。
- 多轮对话会话标识。

## 2. 服务地址

```text
POST http://123.56.142.146:9900/chat
```

健康检查：

```text
GET http://123.56.142.146:9900/health
```

## 3. 认证与模型调用参数

接口调用需要同时提供两类凭据。

### 3.1 服务访问 Token

服务访问 Token 放在 HTTP Header 中，用于访问本智能体 API 服务。

```http
Authorization: Bearer <SERVICE_BEARER_TOKEN>
```

`<SERVICE_BEARER_TOKEN>` 由参赛方提供给评委。该 Token 只用于访问本智能体 API 服务，可以由参赛方自行设置，不能与 DashScope API Key 相同。

评审用服务访问 Token：

```text
superbiz-agent-judge-token-2026
```

服务端 `.env` 中需要配置为同一个值：

```env
API_BEARER_TOKEN=superbiz-agent-judge-token-2026
```

### 3.2 模型 API Key

模型 API Key 放在 JSON 请求体中，用于本次模型调用。

```json
{
  "dashscope_api_key": "<DASHSCOPE_API_KEY>"
}
```

`<DASHSCOPE_API_KEY>` 由调用方提供。

## 4. 请求格式

请求方法：

```text
POST /chat
```

请求头：

```http
Content-Type: application/json
Authorization: Bearer <SERVICE_BEARER_TOKEN>
```

请求体：

```json
{
  "question": "用户问题",
  "session_id": "session-001",
  "stream": false,
  "model": "qwen-vl-plus",
  "dashscope_api_key": "<DASHSCOPE_API_KEY>",
  "images": []
}
```

## 5. 请求参数

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `question` | string | 是 | 用户问题，不能为空 |
| `dashscope_api_key` | string | 是 | 调用方提供的 DashScope API Key |
| `session_id` | string | 否 | 会话 ID；多轮对话复用同一个 ID |
| `stream` | boolean | 否 | 是否启用流式输出，默认 `false` |
| `model` | string | 否 | 本次请求使用的模型；图片理解建议使用视觉模型 |
| `images` | string[] | 否 | Base64 图片数组，最多 3 张 |

兼容字段：

| 标准字段 | 兼容字段 |
| --- | --- |
| `question` | `Question` |
| `session_id` | `Id` |

## 6. 图片格式

`images` 支持两种 Base64 图片格式。

### 6.1 Data URL Base64

```json
{
  "images": [
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
  ]
}
```

支持前缀：

```text
data:image/png;base64,
data:image/jpg;base64,
data:image/jpeg;base64,
data:image/webp;base64,
```

### 6.2 裸 Base64

```json
{
  "images": [
    "/9j/4AAQSkZJRgABAQ..."
  ]
}
```

服务端会自动识别 PNG、JPEG、WEBP 图片并补齐 Data URL 前缀。

图片限制：

- 最多 3 张。
- 单张图片最大 5 MB。
- 图片宽高不能过小。

## 7. 非流式文本调用

```bash
SERVICE_BEARER_TOKEN="参赛方提供的服务访问 Token"
DASHSCOPE_API_KEY="调用方自己的 DashScope API Key"

curl -X POST "http://123.56.142.146:9900/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${SERVICE_BEARER_TOKEN}" \
  -d "{
    \"question\": \"空调滤网怎么清洁？\",
    \"session_id\": \"text-session-001\",
    \"stream\": false,
    \"model\": \"qwen3.7-max\",
    \"dashscope_api_key\": \"${DASHSCOPE_API_KEY}\"
  }"
```

响应示例：

```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "answer": "建议每两周清洁一次空气滤网...",
    "session_id": "text-session-001",
    "timestamp": 1781876412
  }
}
```

## 8. 流式文本调用

```bash
SERVICE_BEARER_TOKEN="参赛方提供的服务访问 Token"
DASHSCOPE_API_KEY="调用方自己的 DashScope API Key"

curl -N -X POST "http://123.56.142.146:9900/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${SERVICE_BEARER_TOKEN}" \
  -d "{
    \"question\": \"电钻充电灯闪烁是什么意思？\",
    \"session_id\": \"stream-session-001\",
    \"stream\": true,
    \"model\": \"qwen3.7-max\",
    \"dashscope_api_key\": \"${DASHSCOPE_API_KEY}\"
  }"
```

SSE 响应示例：

```text
event: message
data: {"type": "content", "data": "电钻充电灯闪烁"}

event: message
data: {"type": "content", "data": "通常表示正在充电或进入保护状态..."}

event: message
data: {"type": "done", "data": null}
```

客户端处理方式：

- 按顺序拼接所有 `type=content` 的 `data` 字段。
- 收到 `type=done` 表示本轮回答结束。
- 收到 `type=error` 表示本轮调用失败。

## 9. 图片调用

图片理解请求建议使用视觉模型，例如：

```text
qwen-vl-plus
qwen-vl-max
```

### 9.1 Data URL Base64 图片

```bash
SERVICE_BEARER_TOKEN="参赛方提供的服务访问 Token"
DASHSCOPE_API_KEY="调用方自己的 DashScope API Key"
IMAGE_BASE64="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."

curl -X POST "http://123.56.142.146:9900/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${SERVICE_BEARER_TOKEN}" \
  -d "{
    \"question\": \"这张图里是什么？请简要说明。\",
    \"session_id\": \"image-session-001\",
    \"stream\": false,
    \"model\": \"qwen-vl-plus\",
    \"dashscope_api_key\": \"${DASHSCOPE_API_KEY}\",
    \"images\": [\"${IMAGE_BASE64}\"]
  }"
```

### 9.2 裸 Base64 图片

当图片 Base64 很长时，建议先写入 JSON 文件再发送。

请求体示例：

```json
{
  "question": "这张图里是什么？请简要说明。",
  "session_id": "image-session-002",
  "stream": false,
  "model": "qwen-vl-plus",
  "dashscope_api_key": "<DASHSCOPE_API_KEY>",
  "images": [
    "/9j/4AAQSkZJRgABAQ..."
  ]
}
```

调用方式：

```bash
SERVICE_BEARER_TOKEN="参赛方提供的服务访问 Token"

curl -X POST "http://123.56.142.146:9900/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${SERVICE_BEARER_TOKEN}" \
  --data-binary @request.json
```

## 10. 响应字段

非流式成功响应：

```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "answer": "智能体回答",
    "session_id": "session-001",
    "timestamp": 1781876412
  }
}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `code` | integer | 业务状态码，`0` 表示成功 |
| `msg` | string | 状态消息 |
| `data.answer` | string | 智能体回答 |
| `data.session_id` | string | 会话 ID |
| `data.timestamp` | integer | Unix 秒级时间戳 |

## 11. 错误响应

### 11.1 服务访问 Token 缺失或格式错误

```json
{
  "detail": "Missing or invalid Authorization header"
}
```

HTTP 状态码：`401`

### 11.2 服务访问 Token 错误

```json
{
  "detail": "Invalid bearer token"
}
```

HTTP 状态码：`401`

### 11.3 请求参数错误

例如 `question` 为空、图片 Base64 非法、图片数量超过限制、图片超过大小限制：

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "images"],
      "msg": "Value error, images 中包含非法 Base64 内容"
    }
  ]
}
```

HTTP 状态码：`422`

### 11.4 模型调用错误

例如 `dashscope_api_key` 无效、模型名不可用、账号未开通对应模型：

```json
{
  "code": 500,
  "msg": "模型服务返回的错误信息",
  "data": {
    "answer": "",
    "session_id": "session-001",
    "timestamp": 1781876401
  }
}
```

## 12. 接入 checklist

- 使用 `POST http://123.56.142.146:9900/chat`。
- Header 中设置 `Content-Type: application/json`。
- Header 中设置 `Authorization: Bearer <SERVICE_BEARER_TOKEN>`。
- Body 中填写 `question`。
- Body 中填写 `dashscope_api_key`。
- 图片输入放在 `images` 中，支持 Data URL Base64 或裸 Base64。
- 图片理解请求建议设置视觉模型，例如 `qwen-vl-plus`。
- 非流式调用读取 `data.answer`。
- 流式调用拼接 SSE 中所有 `type=content` 的 `data`。
