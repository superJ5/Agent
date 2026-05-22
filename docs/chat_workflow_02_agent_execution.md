# 02 Agent 内部执行

这一篇只讲：

```text
进入 rag_agent_service.query() 后，如何调用 LangChain/LangGraph Agent。
```

## 1. query() 是入口函数

对应文件：

```text
app/services/rag_agent_service.py
```

核心函数：

```python
async def query(
    self,
    question: str,
    session_id: str,
    images: list[str] | None = None,
) -> str:
```

含义：

```text
输入：用户问题 + 会话 ID + 图片列表
输出：最终答案字符串
```

英文解释：

```text
async
异步函数。里面可能要等待大模型、工具、数据库等网络请求。

query
查询。这里表示处理一次非流式问答。

self
当前 RagAgentService 对象本身。

question: str
用户问题，类型是字符串。

session_id: str
会话 ID，类型是字符串。

images: list[str] | None = None
图片字符串列表；也可以没有图片。

-> str
函数最终返回字符串。
```

## 2. 初始化 Agent

`query()` 里第一步是：

```python
await self._initialize_agent()
```

意思是：

```text
确保 Agent 已经创建好。
```

真正创建 Agent 的地方是：

```python
self.agent = create_agent(
    self.model,
    tools=all_tools,
    checkpointer=self.checkpointer,
)
```

这里传给 LangChain/LangGraph 三类东西：

```text
self.model
大模型，例如 qwen3.5-plus。

tools=all_tools
工具列表，例如 retrieve_knowledge、get_current_time、MCP tools。

checkpointer=self.checkpointer
会话状态保存器。
```

可以理解成：

```text
请用这个模型、这些工具、这个会话保存器，帮我创建一个 Agent。
```

## 3. 构建 messages

接着是：

```python
messages = [
    SystemMessage(content=self.system_prompt),
    build_user_message(question, images),
]
```

`messages` 是要交给 Agent 的消息列表。

第一条：

```python
SystemMessage(content=self.system_prompt)
```

这是系统规则。

里面会告诉模型：

```text
你是专业助手
涉及手册、图片、OCR、操作步骤时必须先调用 retrieve_knowledge
不要编造
基于证据回答
```

第二条：

```python
build_user_message(question, images)
```

这是用户消息。

如果没有图片，它返回：

```python
HumanMessage(content=question)
```

如果有图片，它返回类似：

```python
HumanMessage(
    content=[
        {"type": "text", "text": question},
        {"type": "image_url", "image_url": {"url": image_base64}},
    ]
)
```

注意：

```text
build_user_message() 只负责打包文字和图片。
真正理解图片的是后面的多模态模型。
```

## 4. 构建 Agent 输入

代码：

```python
agent_input = {"messages": messages}
```

含义：

```text
把 messages 包装成 LangChain/LangGraph Agent 需要的输入格式。
```

也就是：

```python
{
    "messages": [
        SystemMessage(...),
        HumanMessage(...)
    ]
}
```

## 5. 构建 config

代码：

```python
config_dict = {
    "configurable": {
        "thread_id": session_id
    }
}
```

含义：

```text
把 session_id 传给 LangGraph，当作 thread_id。
```

关系：

```text
session_id → thread_id
```

作用：

```text
同一个 session_id 的请求，会被当作同一段会话。
```

## 6. 真正启动 Agent

核心代码：

```python
result = await self.agent.ainvoke(
    input=agent_input,
    config=config_dict,
)
```

英文解释：

```text
ainvoke
async invoke，异步调用 / 异步执行一次。

input
传给 Agent 的输入。

config
运行配置，例如 thread_id。

result
Agent 执行结束后的结果。
```

这句话是分界线：

```text
这句之前：项目自己的代码在准备输入。
这句里面：LangChain/LangGraph Agent 在调用模型和工具。
这句之后：项目自己的代码从结果里取最终答案。
```

## 7. Agent 内部做什么

`self.agent.ainvoke(...)` 内部大概会这样跑：

```text
读取 thread_id 对应的会话状态
   ↓
把历史消息、系统规则、当前用户消息合起来
   ↓
把可用工具说明告诉模型
   ↓
调用 qwen3.5-plus
   ↓
模型判断：直接回答，还是调用工具
```

如果模型不需要工具：

```text
模型直接生成最终答案
```

如果模型需要工具：

```text
模型输出 tool_call
   ↓
LangGraph 执行对应工具
   ↓
工具结果放回 messages
   ↓
再次调用模型
   ↓
模型生成最终答案
```

这里的“循环问模型要不要调用工具”不是你手写的。

它是 LangChain/LangGraph Agent 框架内部完成的。

## 8. RAG 工具在这里怎么被调用

如果模型认为需要查手册，会调用：

```python
retrieve_knowledge(query)
```

例如用户问：

```text
这张图里的冰箱图标是什么意思？
```

多模态模型先理解图片和文字，可能生成工具调用：

```text
retrieve_knowledge("冰箱 图标 含义")
```

然后 RAG 检索工具去手册知识库里找证据。

这里要分清：

```text
用户图片
先给多模态模型看。

Milvus RAG
主要查提前索引好的手册文本、图片路径和 metadata。
```

## 9. 取最终答案

`ainvoke()` 跑完后，回到 `query()`：

```python
messages_result = result.get("messages", [])
```

含义：

```text
从 Agent 返回结果里取 messages。
```

然后：

```python
last_message = messages_result[-1]
answer = last_message.content
```

含义：

```text
取最后一条消息作为最终答案。
```

最后：

```python
return answer
```

答案回到：

```text
app/api/chat.py 的 competition_chat()
```

再被包装成 JSON 返回给前端或测评程序。

## 10. 这一段的总流程

```text
rag_agent_service.query(question, session_id, images)
   ↓
_initialize_agent() 确保 Agent 已创建
   ↓
SystemMessage 放系统规则
   ↓
build_user_message() 打包用户文字和图片
   ↓
agent_input = {"messages": messages}
   ↓
config_dict 放 thread_id
   ↓
self.agent.ainvoke(...)
   ↓
LangGraph 内部调用模型
   ↓
模型按需调用 retrieve_knowledge 等工具
   ↓
工具结果返回给模型
   ↓
模型生成最终答案
   ↓
query() 取最后一条消息 content
   ↓
return answer
```
