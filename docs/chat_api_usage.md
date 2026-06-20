# /chat 接口使用说明（评审版）

## 一、支持功能

核心接口只有一个：

```text
POST http://123.56.142.146:9900/chat
```

该接口支持：
1. 文本问答：JSON 请求体中传 question。
2. 图片问答：JSON 请求体中传 images，图片使用带完整前缀的 Base64。
3. 非流式输出：stream=false，返回完整 JSON。
4. 流式输出：stream=true，返回 SSE 流式数据。
5. 多轮会话：传入相同 session_id，可作为同一轮会话标识。
6. 指定模型：传 model；不传时使用服务端默认模型qwen3.7-max，该模型没有图片处理能力。
7. 调用方模型 Key：传 dashscope_api_key，评委使用自己的阿里云 DashScope API Key。
8. 请求级记忆开关：默认不启用记忆；需要演示记忆时传 memory_enabled=true。

测评比赛题目时默认不开启记忆系统，避免历史会话影响 RAG 检索和回答结果。如需检测记忆系统效果，可以在 curl 命令中显式传入 `"memory_enabled": true`。

## 二、最完整 curl 测试命令

```bash
SERVICE_BEARER_TOKEN="123456"
DASHSCOPE_API_KEY="<评委自己的阿里云 DashScope API Key>"
IMAGE_BASE64="<图片 Base64，格式为 data:image/{png/jpg/jpeg/webp};base64,{编码内容}；不测图片时可删除 images 字段，或传空数组 images: []>"

curl -X POST "http://123.56.142.146:9900/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${SERVICE_BEARER_TOKEN}" \
  -d "{
    \"question\": \"这张图里是什么？请简要说明。\",
    \"session_id\": \"judge-session-001\",
    \"stream\": false,
    \"model\": \"qwen-vl-plus\",
    \"dashscope_api_key\": \"${DASHSCOPE_API_KEY}\",
    \"memory_enabled\": false,
    \"images\": [\"${IMAGE_BASE64}\"]
  }"
```

示例：

```bash
SERVICE_BEARER_TOKEN="123456"
DASHSCOPE_API_KEY="<评委自己的阿里云 DashScope API Key>"
IMAGE_BASE64="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."

curl -X POST "http://123.56.142.146:9900/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${SERVICE_BEARER_TOKEN}" \
  -d "{
    \"question\": \"请判断这张凭证图片里的商品或问题是什么？\",
    \"session_id\": \"judge-image-demo-001\",
    \"stream\": false,
    \"model\": \"qwen-vl-plus\",
    \"dashscope_api_key\": \"${DASHSCOPE_API_KEY}\",
    \"memory_enabled\": false,
    \"images\": [\"${IMAGE_BASE64}\"]
  }"
```

## 三、上面命令每个部分是什么意思

固定不需要改的部分：
1. 请求地址固定为：http://123.56.142.146:9900/chat
2. 请求方法固定为：POST
3. Content-Type 固定为：application/json
4. Authorization 固定格式为：Bearer 123456
5. SERVICE_BEARER_TOKEN 当前评审值固定为：123456

评委需要按测试需求填写的部分：
1. DASHSCOPE_API_KEY：评委自己的阿里云 DashScope API Key。
2. question：要问智能体的问题。
3. session_id：会话 ID，可自定义；同一会话连续提问时保持相同。
4. stream：false 表示一次性返回完整答案；true 表示流式返回。
5. model：本次请求使用的模型。文本可用 qwen3.7-max；图片理解要用多模态模型 qwen-vl-plus 或 qwen-vl-max。
6. images：Base64 图片列表，必须携带完整前缀，格式为 data:image/{png/jpg/jpeg/webp};base64,{编码内容}。支持 0-3 张，每张解码后不超过 5MB。只测文本时可以删除整个 images 字段，也可以传空数组 "images": []；不要传 [""] 或未替换的占位字符串。
7. memory_enabled：普通测评不传即可；如需演示记忆系统，传 true 并连续使用同一个 session_id。

## 四、请求字段说明

question：必填，字符串，用户问题。
dashscope_api_key：必填，字符串，评委自己的阿里云 DashScope API Key。
session_id：选填，字符串，会话 ID。
stream：选填，布尔值，默认 false。
model：选填，字符串，不传时使用服务端 .env 默认模型。
images：选填，字符串数组，对应用户上传的凭证图片。Base64 图片必须携带完整前缀，格式为 data:image/{png/jpg/jpeg/webp};base64,{编码内容}。支持 0-3 张，每张解码后不超过 5MB。不传该字段时默认为空数组；文本请求也可以显式传 "images": []。
memory_enabled：选填，布尔值。不传时使用服务端默认配置；本项目默认 false。普通测评无需传该字段；演示记忆系统时传 true。

兼容字段：
question 也兼容 Question。
session_id 也兼容 Id。

## 五、图片 Base64 格式

images 字段必须使用带完整前缀的 Data URL Base64：

```text
data:image/png;base64,iVBORw0KGgo...
data:image/jpg;base64,/9j/4AAQSkZJRgABAQ...
data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...
data:image/webp;base64,UklGR...
```

支持 0-3 张图片，每张图片解码后不超过 5MB。

不传图片时可使用：

```json
"images": []
```

不要使用：

```json
"images": [""]
```

## 六、返回格式

非流式成功响应示例：

```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "answer": "智能体回答内容",
    "session_id": "judge-session-001",
    "timestamp": 1781876412
  }
}
```

字段含义：
code：业务状态码，0 表示成功。
msg：状态信息。
data.answer：智能体回答。
data.session_id：本次会话 ID。
data.timestamp：秒级时间戳。

流式响应格式：

```text
event: message
data: {"type":"content","data":"您好，我"}

event: message
data: {"type":"content","data":"是一名专业的 AI 客服助手。"}

event: message
data: {"type":"done","data":null}
```

客户端处理方式：
1. 顺序拼接所有 type=content 的 data。
2. 收到 type=done 表示本轮回答结束。
3. 收到 type=error 表示本轮调用失败。

## 七、各功能测试示例

1. 文本问答，非流式

SERVICE_BEARER_TOKEN="123456"
DASHSCOPE_API_KEY="<评委自己的阿里云 DashScope API Key>"

```bash
curl -X POST "http://123.56.142.146:9900/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${SERVICE_BEARER_TOKEN}" \
  -d "{
    \"question\": \"空调滤网怎么清洁？\",
    \"session_id\": \"judge-text-001\",
    \"stream\": false,
    \"model\": \"qwen3.7-max\",
    \"dashscope_api_key\": \"${DASHSCOPE_API_KEY}\"
  }"
```

2. 文本问答，流式

SERVICE_BEARER_TOKEN="123456"
DASHSCOPE_API_KEY="<评委自己的阿里云 DashScope API Key>"

```bash
curl -N -X POST "http://123.56.142.146:9900/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${SERVICE_BEARER_TOKEN}" \
  -d "{
    \"question\": \"你好，请用一句话介绍你自己。\",
    \"session_id\": \"judge-stream-001\",
    \"stream\": true,
    \"model\": \"qwen3.7-max\",
    \"dashscope_api_key\": \"${DASHSCOPE_API_KEY}\"
  }"
```

3. 图片问答

SERVICE_BEARER_TOKEN="123456"
DASHSCOPE_API_KEY="<评委自己的阿里云 DashScope API Key>"
IMAGE_BASE64="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."

```bash
curl -X POST "http://123.56.142.146:9900/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${SERVICE_BEARER_TOKEN}" \
  -d "{
    \"question\": \"这张图里是什么？请简要说明。\",
    \"session_id\": \"judge-image-001\",
    \"stream\": false,
    \"model\": \"qwen-vl-plus\",
    \"dashscope_api_key\": \"${DASHSCOPE_API_KEY}\",
    \"memory_enabled\": false,
    \"images\": [\"${IMAGE_BASE64}\"]
  }"
```

4. 图片问答，使用 request.json 发送

request.json 示例：

```json
{
  "question": "这张图里是什么？请简要说明。",
  "session_id": "judge-image-002",
  "stream": false,
  "model": "qwen-vl-plus",
  "dashscope_api_key": "<评委自己的阿里云 DashScope API Key>",
  "memory_enabled": false,
  "images": [
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
  ]
}
```

发送命令：

SERVICE_BEARER_TOKEN="123456"

```bash
curl -X POST "http://123.56.142.146:9900/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${SERVICE_BEARER_TOKEN}" \
  --data-binary @request.json
```

5. 记忆系统演示

普通测评不需要传 memory_enabled，默认不会读取或写入记忆。
如需演示记忆能力，连续请求使用同一个 session_id，并设置 memory_enabled=true。

第一轮：

SERVICE_BEARER_TOKEN="123456"
DASHSCOPE_API_KEY="<评委自己的阿里云 DashScope API Key>"

```bash
curl -X POST "http://123.56.142.146:9900/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${SERVICE_BEARER_TOKEN}" \
  -d "{
    \"question\": \"请记住，我希望回答尽量简洁。\",
    \"session_id\": \"judge-memory-001\",
    \"stream\": false,
    \"model\": \"qwen3.7-max\",
    \"dashscope_api_key\": \"${DASHSCOPE_API_KEY}\",
    \"memory_enabled\": true
  }"
```

第二轮：

```bash
curl -X POST "http://123.56.142.146:9900/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${SERVICE_BEARER_TOKEN}" \
  -d "{
    \"question\": \"你还记得我刚才对回答风格的要求吗？\",
    \"session_id\": \"judge-memory-001\",
    \"stream\": false,
    \"model\": \"qwen3.7-max\",
    \"dashscope_api_key\": \"${DASHSCOPE_API_KEY}\",
    \"memory_enabled\": true
  }"
```

## 八、其他接口说明

1. 健康检查

作用：检查服务是否可访问。

```bash
curl -X GET "http://123.56.142.146:9900/health"
```

2. 在线接口文档

作用：查看 FastAPI 自动生成的接口页面。

```bash
curl -I "http://123.56.142.146:9900/docs"
```

3. 查看会话历史

作用：按 session_id 查看最近会话记录。若服务端关闭记忆功能，可能返回空列表。

```bash
curl -X GET "http://123.56.142.146:9900/api/chat/session/judge-text-001"
```

4. 查看短期记忆

作用：查看指定 session_id 的短期语义记忆。若服务端关闭记忆功能，可能返回不存在或空内容。

```bash
curl -X GET "http://123.56.142.146:9900/api/chat/session/judge-text-001/short-term-memory"
```

5. 查看会话状态

作用：查看指定 session_id 的结构化会话状态。若服务端关闭记忆功能，可能返回不存在或空内容。

```bash
curl -X GET "http://123.56.142.146:9900/api/chat/session/judge-text-001/session-state"
```

6. 查看长期记忆

作用：查看长期记忆列表，limit 控制返回数量。若服务端关闭记忆功能，可能返回空列表。

```bash
curl -X GET "http://123.56.142.146:9900/api/chat/memory/long-term?limit=3"
```

7. 获取手册图片

作用：按 image_id 获取系统中已索引的手册图片。

```bash
curl -I "http://123.56.142.146:9900/api/manual-images/Manual28_12"
```

## 九、常见错误

1. 401 Unauthorized

原因：Authorization 缺失或 Bearer Token 不正确。

正确写法：
Authorization: Bearer 123456

2. 422 Unprocessable Entity

原因：请求体字段不合法，例如 question 为空、图片 Base64 非法、图片超过大小限制、images 超过 3 张。

3. code=500

原因：模型调用失败，例如 dashscope_api_key 无效、模型名不可用、账号未开通对应模型。

4. 请求超时

原因：可能是网络波动、评委本地网络到服务器链路不稳定，或阿里云 DashScope 云端模型临时响应较慢。该问题通常不是接口格式错误，稍后重试或多次测试一般可恢复。

## 十、评审建议

1. 文本测试可使用 model=qwen3.7-max。
2. 图片测试建议使用 model=qwen-vl-plus 或 qwen-vl-max。
3. 不测试图片时删除 images 字段，或传空数组 "images": []。
4. 要测试流式输出时设置 stream=true，并使用 curl -N。
5. 要测试多轮会话时，多次请求使用同一个 session_id。
6. 普通测评不传 memory_enabled；只有演示记忆系统时才传 memory_enabled=true。
