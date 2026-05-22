# Chat Supplementary Notes

这份文档放主工作流之外的补充说明。

主工作流文档只讲“请求怎么走到答案”。这里讲一些容易混淆、但理解项目有帮助的概念。

## 1. FastAPI 是什么

FastAPI 是 Python 的 Web 后端框架。

可以理解成：

```text
FastAPI = 把 Python 函数变成 HTTP 接口的工具
```

它做的事情：

```text
HTTP 请求
   ↓
根据 URL 找到对应 Python 函数
   ↓
执行函数
   ↓
把返回值变成 HTTP 响应
```

## 2. endpoint 是什么

endpoint 是一个具体接口地址。

例如：

```text
GET /health
POST /chat
POST /api/chat
```

注意：

```text
URL 不是文件。
URL 会被 FastAPI 映射到某个 Python 函数。
```

## 3. prefix 是什么

在 `app/main.py` 里：

```python
app.include_router(chat.router, prefix="/api")
```

意思是：

```text
给 chat.router 里的所有接口统一加 /api 前缀。
```

如果 router 里有：

```text
/chat
```

最终地址就是：

```text
/api/chat
```

也可以自己加：

```python
prefix="/jhh"
```

这样路径会变成：

```text
/jhh/chat
```

但比赛如果要求 `POST /chat`，就必须保留 `POST /chat`。

## 4. Agent 的两种含义

Agent 有广义和狭义两种说法。

广义 Agent：

```text
能理解任务
能决定下一步
能调用工具
能根据工具结果继续行动
```

都可以叫 Agent。

狭义 Agent：

```text
LangChain create_agent() 创建出来的现成 Agent。
```

当前项目里主要有两个 Agent 系统。

## 5. RAG 对话 Agent

对应文件：

```text
app/services/rag_agent_service.py
```

入口接口：

```text
POST /chat
POST /api/chat
POST /api/chat_stream
```

它的 Agent 创建方式：

```python
self.agent = create_agent(...)
```

适合：

```text
普通聊天
说明书问答
多模态图片问答
RAG 检索回答
```

## 6. AIOps Agent

对应文件：

```text
app/services/aiops_service.py
app/agent/aiops/planner.py
app/agent/aiops/executor.py
app/agent/aiops/replanner.py
```

入口接口：

```text
POST /api/aiops
```

它不是用 `create_agent()`，而是用 LangGraph 自己搭流程图：

```python
workflow = StateGraph(PlanExecuteState)
workflow.add_node("planner", planner)
workflow.add_node("executor", executor)
workflow.add_node("replanner", replanner)
workflow.compile(...)
```

节点含义：

```text
planner
规划者。把复杂任务拆成步骤。

executor
执行者。执行当前第一个步骤，可以调用工具。

replanner
重新规划者。判断是否继续、是否补充步骤、是否生成最终报告。
```

流程：

```text
planner
→ executor
→ replanner
→ executor 或 END
```

## 7. create_agent 和自己搭 LangGraph 的区别

`create_agent()`：

```text
框架帮你搭好通用 Agent。
你给模型和工具，它自己控制“模型-工具-模型”的循环。
代码少，适合普通问答。
```

自己搭 LangGraph：

```text
你自己定义节点和边。
每一步怎么走更清楚。
代码更多，适合复杂任务流程。
```

所以：

```text
RAG Agent
= 框架帮你搭好的 Agent。

AIOps Agent
= 自己用 LangGraph 拼出来的 Agent 工作流。
```

## 8. stream 流式返回

非流式：

```text
一个请求
等待完整答案
一次性返回 JSON
```

流式：

```text
一个请求
一个持续连接
多个内容片段
前端拼成完整回答
```

流式不是多个 HTTP 响应。

更准确是：

```text
一个 HTTP 连接里不断发送片段。
```

## 9. MemorySaver 的边界和风险

当前 RAG 对话 Agent 使用：

```python
self.checkpointer = MemorySaver()
```

并且调用 Agent 时传：

```python
config_dict = {
    "configurable": {
        "thread_id": session_id
    }
}
```

关系：

```text
session_id → thread_id → MemorySaver 中的会话状态
```

边界：

```text
MemorySaver 是进程内内存保存器。
服务不重启，历史还在。
服务重启，历史会丢。
```

风险：

```text
同一个 thread_id 的上下文可能持续增长。
长对话可能导致请求变慢、token 超限、成本升高、旧信息干扰回答。
```

项目里有：

```python
trim_messages_middleware(state)
```

它的目的：

```text
保留系统消息
只保留最近几轮对话
避免上下文过长
```

但当前 `create_agent(...)` 里没有看到它被接入，所以主流程里暂时不能认为它已经生效。
