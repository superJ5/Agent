# Chat Workflow Notes

这份文档是当前聊天系统工作流的入口目录。

主工作流只讲一件事：

```text
用户请求进来以后，代码如何一步一步处理，直到返回答案。
```

不属于主链路的解释，例如 `MemorySaver` 风险、Agent 概念对比、OpenClaw 记忆方案等，已经拆到补充文档里。

## 主工作流拆分

建议按这个顺序阅读。

1. [01 请求进入 Agent 前](./chat_workflow_01_request_to_agent.md)

   讲：

   ```text
   POST /chat
   → ChatRequest 校验
   → competition_chat()
   → rag_agent_service.query(...)
   ```

2. [02 Agent 内部执行](./chat_workflow_02_agent_execution.md)

   讲：

   ```text
   query()
   → build_user_message()
   → self.agent.ainvoke()
   → 大模型判断是否调用工具
   → retrieve_knowledge
   → 最终 answer
   ```

3. [03 RAG 与数据资产](./chat_workflow_03_rag_and_data.md)

   讲：

   ```text
   retrieve_knowledge()
   → 向量 embedding
   → Milvus 检索
   → data/manuals/chunks
   → 证据返回给模型
   ```

## 补充文档

1. [补充概念说明](./chat_supplementary_notes.md)

   放不属于主工作流、但理解项目有帮助的内容，例如：

   ```text
   FastAPI 是什么
   endpoint 是什么
   Agent 两种实现方式
   MemorySaver 的边界
   stream 流式返回
   ```

2. [记忆系统改造方案](./memory_system_proposal.md)

   放后续如果想把当前项目改成类似 OpenClaw 记忆系统的设计方案。

## 当前系统一句话

当前 `/chat` 主流程可以理解为：

```text
接口接收并校验请求
→ RAG Agent 服务包装文字和图片
→ LangChain/LangGraph Agent 调用大模型
→ 大模型按需调用 retrieve_knowledge 查手册知识库
→ 工具结果回到模型
→ 模型生成最终答案
→ 接口包装成 JSON 或 SSE 返回
```
