# 01 请求进入 Agent 前

这一篇只讲：

```text
用户请求如何从 /chat 进入 rag_agent_service.query()
```

## 1. 总入口

对应文件：

```text
app/main.py
```

`app/main.py` 负责把各个接口模块挂到 FastAPI 应用上。

和聊天主流程最相关的是：

```python
app.include_router(chat.competition_router, tags=["比赛标准接口"])
app.include_router(chat.router, prefix="/api", tags=["对话"])
```

含义：

```text
chat.competition_router 里的接口直接挂到根路径
chat.router 里的接口统一加 /api 前缀
```

所以当前主比赛接口是：

```text
POST /chat
```

旧版接口还有：

```text
POST /api/chat
POST /api/chat_stream
```

## 2. 请求体

调用方发 `POST /chat` 时，会带 JSON 请求体。

典型格式：

```json
{
  "question": "电钻充电灯闪烁是什么意思？",
  "images": ["data:image/png;base64,..."],
  "session_id": "kf_session_abc",
  "stream": false
}
```

字段含义：

```text
question
用户文字问题。

images
图片列表。图片已经被前端或测试脚本转成 base64 data URL。

session_id
会话 ID。可以理解成一个聊天窗口的编号。

stream
是否流式返回。
false 表示一次性返回完整答案。
true 表示一段一段返回内容。
```

兼容旧字段：

```text
question 或 Question
session_id 或 Id
```

## 3. 请求校验

对应文件：

```text
app/models/request.py
```

核心类：

```python
ChatRequest
```

它负责检查请求体。

主要检查：

```text
question 不能为空
images 最多 3 张
images 每一项必须是字符串
images 必须是 data:image/png;base64,... 等格式
images 必须是合法 base64
单张图片不能超过 5MB
```

校验通过后，接口函数里才能使用：

```python
request.question
request.images
request.session_id
request.stream
```

## 4. /chat 接口函数

对应文件：

```text
app/api/chat.py
```

核心函数：

```python
@competition_router.post("/chat")
async def competition_chat(...)
```

意思是：

```text
如果有人 POST /chat，就执行 competition_chat()
```

这个函数做几件事。

第一，检查 Authorization token：

```python
_require_bearer_token(authorization)
```

第二，处理会话 ID：

```python
session_id = _resolve_session_id(request.session_id)
```

如果前端传了 `session_id`，就用前端传来的。

如果没有传，就生成一个新的：

```text
kf_session_ + 随机字符串
```

第三，判断是否流式：

```python
if request.stream:
    return _build_stream_response(request.question, session_id, request.images)
```

第四，非流式时调用 RAG Agent 服务：

```python
answer = await rag_agent_service.query(
    request.question,
    session_id=session_id,
    images=request.images,
)
```

这句话是接口层进入 Agent 层的关键。

输入：

```text
request.question
用户问题

session_id
当前会话编号

request.images
用户上传的图片列表
```

输出：

```text
answer
最终回答文本
```

## 5. 这一段的总流程

```text
用户 / 前端 / 测评程序
   ↓
POST /chat + JSON 请求体
   ↓
FastAPI 找到 competition_chat()
   ↓
ChatRequest 校验 question/images/session_id/stream
   ↓
competition_chat() 检查 token
   ↓
_resolve_session_id() 得到会话 ID
   ↓
非流式调用 rag_agent_service.query(question, session_id, images)
```

到这里，接口层的工作基本结束。

下一步进入：

```text
RAG Agent 服务层
```
