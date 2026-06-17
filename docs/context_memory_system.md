# 上下文记忆系统：原始 Session、短期语义记忆、Session State 与长期记忆

本文说明当前项目已经接入的四层上下文记忆：

```text
第一层：Raw Session Log，原始会话日志
第二层：Short-term Semantic Memory，短期语义记忆
第三层：Session State，结构化当前任务状态
第四层：Long-term Memory，跨 session 稳定记忆
```

旧版 `daily/*.md`、`MEMORY.md`、`memory_search` 记忆检索已经移除。现在 `session_id` 不再表示“把整段历史原文塞给模型”，而是表示“当前请求属于哪个会话桶”，用于写原始日志、读取 Session State、读取短期记忆、检索长期记忆，并在回答后更新上下文记忆。

## 1. 总体思路

一次请求的主流程如下：

```text
用户请求：question + session_id
        ↓
读取旧短期语义记忆
data/memory/short_term/<session_id>.md
        ↓
读取当前会话状态
data/memory/session_state/<session_id>.json
        ↓
检索相关长期记忆
Milvus collection: long_term_memory
        ↓
读取最近 1-3 轮原始对话
data/memory/sessions/<session_id>.jsonl
        ↓
构造回答模型上下文：
system prompt
+ 长期记忆
+ Session State
+ 短期语义记忆
+ 最近 1-3 轮原始对话
+ 当前用户问题
        ↓
模型生成回答
        ↓
写入原始会话日志：
本轮 user + assistant
        ↓
后台更新短期语义记忆：
记忆更新器 system prompt
+ 旧短期记忆
+ 最近 1-3 轮原始对话
+ 本轮问答
        ↓
写回 data/memory/short_term/<session_id>.md
        ↓
后台更新 Session State：
Session State 更新器 system prompt
+ 旧 Session State
+ 旧短期语义记忆
+ 最近 1-3 轮原始对话
+ 本轮问答
        ↓
写回 data/memory/session_state/<session_id>.json
        ↓
后台更新长期记忆：
长期记忆更新器 system prompt
+ 本轮问答
+ Session State
+ 短期语义记忆
+ 最近 1-3 轮原始对话
+ 本轮检索到的长期记忆
        ↓
写入/更新 Milvus long_term_memory collection
```

关键点：

```text
原始 session 日志完整保留，但不再全量直接进入回答上下文。
短期语义记忆是给模型阅读的滚动案情摘要，负责过滤重复、无关、已证伪和过期信息。
Session State 是程序可读的当前任务状态，负责保留目标、事实、假设、排除方向、下一步动作和用户约束。
长期记忆是跨 session 的稳定信息，写入前必须满足稳定、有用、确认、非敏感四个标准。
最近 1-3 轮原始对话仍会作为辅助上下文进入回答模型，用来保留刚刚发生的细节和指代关系。
短期记忆、Session State 和长期记忆更新都是后台任务，不阻塞用户回答。
```

## 2. 第一层：原始 Session 日志

原始日志位置：

```text
data/memory/sessions/<session_id>.jsonl
```

每一行是一个 JSON 对象，例如：

```json
{"timestamp":"2026-06-12T04:08:44.123+00:00","session_id":"docs_short_memory_demo_003","role":"user","content":"数据库慢查询和连接池都查过了，没有异常。","metadata":{"source":"competition_chat","stream":false,"images_count":0}}
```

这一层只负责完整记录，不负责总结、筛选或判断。

它的作用：

```text
1. 完整留痕：保留每一轮 user/assistant 原文。
2. 调试回放：回答异常时能追溯原始输入和模型输出。
3. 短期记忆输入：给短期记忆更新器提供最近 1-3 轮原始对话。
4. 后续记忆输入：Session State、短期记忆和长期记忆更新都可以从这里取原始材料。
```

相关代码：

```text
app/services/memory_service.py
- append_message()
  写入一条 user/assistant 消息到 sessions/<session_id>.jsonl。

- load_recent_messages()
  从 sessions/<session_id>.jsonl 读取最近若干条 user/assistant 消息。
```

接口写入位置：

```text
app/api/chat.py
- competition_chat()
  收到请求后先写入 user 消息。
  生成回答后再写入 assistant 消息。
```

### 原始 Session 怎么读取

原始 session 记录可以通过接口读取：

```text
GET /api/chat/session/{session_id}
```

例如：

```bash
curl "http://127.0.0.1:9900/api/chat/session/docs_short_memory_demo_003"
```

返回结构类似：

```json
{
  "session_id": "docs_short_memory_demo_003",
  "message_count": 4,
  "history": [
    {
      "role": "user",
      "content": "我在排查支付接口变慢，22:10 后 /api/pay 的 P95 从 300ms 升到 3.8s。你先帮我整理排查方向。",
      "timestamp": "2026-06-12T04:07:50.000000+00:00"
    },
    {
      "role": "assistant",
      "content": "支付接口 P95 突增通常从以下几个维度排查：...",
      "timestamp": "2026-06-12T04:08:03.000000+00:00"
    }
  ]
}
```

对应代码链路：

```text
app/api/chat.py
- get_session_info(session_id)
  处理 GET /api/chat/session/{session_id}
        ↓
app/services/rag_agent_service.py
- get_session_history(session_id)
  调用 memory_service.load_recent_messages()
        ↓
app/services/memory_service.py
- load_recent_messages(session_id)
  读取 data/memory/sessions/<session_id>.jsonl 最近记录
```

注意：

```text
这个接口只是给人或调试工具查看原始历史。
它不会影响 Agent 回答时的上下文。
Agent 回答时只自动使用 Session State、短期语义记忆和最近 1-3 轮原始对话。
```

## 3. 第二层：短期语义记忆

短期记忆位置：

```text
data/memory/short_term/<session_id>.md
```

短期记忆可以通过接口读取：

```text
GET /api/chat/session/{session_id}/short-term-memory
```

例如：

```bash
curl "http://127.0.0.1:9900/api/chat/session/docs_short_memory_demo_003/short-term-memory"
```

返回结构类似：

```json
{
  "session_id": "docs_short_memory_demo_003",
  "exists": true,
  "content": "目标：...\n关键事实：..."
}
```

对应代码链路：

```text
app/api/chat.py
- get_short_term_memory(session_id)
  处理 GET /api/chat/session/{session_id}/short-term-memory
        ↓
app/services/short_term_memory_service.py
- load_memory(session_id)
  读取 data/memory/short_term/<session_id>.md
```

短期记忆不是原文记录，也不是结构化 `Session State`。

它更像“给模型看的滚动案情摘要”：用自然语言保留当前会话里对后续回答有帮助的主线、事实、假设、排除项和下一步，丢弃噪声。

第三层 `Session State` 偏程序可读的结构化状态，例如 JSON 里的 `goal`、`confirmed_facts`、`rejected_hypotheses`、`next_actions`。两者关系可以理解为：

```text
短期语义记忆：给模型读，偏自然语言，帮助模型快速接上案情。
Session State：给系统维护，偏结构化字段，方便覆盖、合并、校验和程序化更新。
```

固定栏目：

```text
目标：
关键事实：
当前假设：
已排除方向：
下一步动作：
用户约束：
```

例如：

```md
目标：
排查 `/api/pay` 接口 P95 耗时突增问题，重点定位 v2.3.1 版本发布与第三方回调超时的关联

关键事实：
- 接口：`/api/pay`
- 指标：P95 耗时 300ms → 3.8s
- 起始时间：22:10 后
- 发布记录：22:05 发布 `payment-service v2.3.1`
- 现象补充：22:10 后第三方支付回调超时明显增加
- 已排查：数据库慢查询、连接池状态均无异常

当前假设：
- 问题根因高度疑似 v2.3.1 版本变更导致
- 可能涉及：HTTP 客户端配置、超时参数、重试逻辑、回调处理代码或网络配置变更

已排除方向：
- 数据库慢查询
- 数据库连接池满/等待

下一步动作：
1. 对比 v2.3.1 与上一版本代码 diff，重点关注回调处理、HTTP 配置、超时参数
2. 通过调用链追踪确认耗时分布在“等待回调”还是“回调处理”阶段
3. 检查支付渠道侧是否有接口变更或限流
4. 必要时回滚至 v2.3.0 验证是否恢复

用户约束：
无
```

这一层保留：

```text
当前目标
已确认事实
仍在验证的假设
已经排除的方向
下一步动作
用户明确约束
```

这一层丢弃：

```text
寒暄
重复描述
无关细节
模型绕路过程
已过期假设
未确认但被写成事实的猜测
回答中的模板话术
```

### 短期记忆怎么更新

短期记忆更新器输入：

```text
记忆更新器 system prompt
旧短期语义记忆
最近 1-3 轮原始对话
本轮用户问题
本轮助手回答
```

输出：

```text
新的 data/memory/short_term/<session_id>.md
```

更新器 prompt 在：

```text
app/services/short_term_memory_service.py
- _update_system_prompt()
```

核心规则：

```text
只保留对当前 session 后续回答有帮助的信息。
不要记录寒暄、重复内容、模板话术、无关细节。
不要编造输入中没有的信息。
用户或证据确认过的信息才能写入“关键事实”。
模型猜测但尚未确认的信息只能写入“当前假设”。
被证伪但对后续有用的信息写入“已排除方向”。
如果新信息纠正旧信息，以新信息为准。
输出整体不超过 1200 个中文字符。
```

相关代码：

```text
app/services/short_term_memory_service.py
- load_memory()
  读取 short_term/<session_id>.md。

- load_recent_dialogue()
  读取最近 1-3 轮原始对话作为辅助上下文。

- update_after_turn()
  回答后更新短期记忆。

- _generate_updated_memory()
  调用短期记忆模型。

- _write_memory()
  写回 short_term/<session_id>.md。
```

## 4. 第三层：Session State

Session State 位置：

```text
data/memory/session_state/<session_id>.json
```

这一层是结构化 JSON，给系统维护和校验，也会作为一条独立的 `SystemMessage` 注入给回答模型。

固定字段：

```json
{
  "session_id": "test-session",
  "goal": "定位支付接口变慢原因",
  "confirmed_facts": ["22:10 后 /api/pay P95 从 300ms 升到 3.8s"],
  "current_hypothesis": ["第三方支付回调超时可能导致接口变慢"],
  "rejected_hypotheses": ["数据库慢查询导致接口变慢"],
  "next_actions": ["检查 v2.3.1 是否改动回调重试逻辑"],
  "user_constraints": [],
  "updated_at": "2026-06-12T00:00:00+00:00"
}
```

读取接口：

```text
GET /api/chat/session/{session_id}/session-state
```

示例：

```bash
curl "http://127.0.0.1:9900/api/chat/session/test-session/session-state"
```

更新规则：

```text
每轮回答结束后都会后台调用 Session State 更新器。
更新器输入：旧 Session State + 旧短期语义记忆 + 最近 1-3 轮原始对话 + 本轮问答。
更新器输出：should_update + reason + 完整新版 state。

步骤 1：模型判断
- 如果 should_update=false，说明本轮没有值得更新的任务状态变化，不写文件，保留旧 state。
- 如果 should_update=true，说明模型认为本轮值得更新，继续进入代码校验。

步骤 2：代码兜底
- 代码会清洗字段、去重、限制长度。
- 代码会移除和 rejected_hypotheses 冲突的 current_hypothesis。
- 如果清洗后的 state 没有 goal、confirmed_facts、rejected_hypotheses、next_actions、user_constraints，就认为没有形成有效任务状态，不写文件。

步骤 3：落盘保存
- 只有通过模型判断和代码兜底后，才写入 data/memory/session_state/<session_id>.json。
- 真正写入文件的是 state 本身，不写 should_update 和 reason。
current_hypothesis 不能和 rejected_hypotheses 冲突。
```

这意味着：如果第二轮已经形成了状态，第三轮用户只说“你好”，更新器应该返回 `should_update=false`，旧状态继续保留；第四轮如果用户又继续问“那下一步查什么”，回答模型仍会读到第二轮留下来的 Session State。

### Session State 怎么更新

Session State 更新器输入：

```text
Session State 更新器 system prompt
旧 Session State
旧短期语义记忆
最近 1-3 轮原始对话
本轮用户问题
本轮助手回答
```

输出：

```text
SessionStateUpdate:
- should_update
- reason
- state
```

更新器 prompt 在：

```text
app/services/session_state_service.py
- _update_system_prompt()
```

核心规则：

```text
如果只是寒暄、感谢、简单闲聊、无明确任务状态变化，should_update=false。
如果新增了目标、确认事实、排除方向、当前假设、下一步动作或用户约束，should_update=true。
state 必须是完整的新状态，不是只包含本轮新增内容。
不要编造输入中没有的信息。
confirmed_facts 只放用户明确表达、工具/日志证据、或已被本轮确认的信息。
current_hypothesis 只放仍可能成立、需要继续验证的判断。
rejected_hypotheses 放已经被用户、证据或排查结果排除的方向。
next_actions 只保留当前仍需要执行的动作，移除已完成动作。
```

相关代码：

```text
app/services/session_state_service.py
- load_state(session_id)
- build_context_block(session_id)
- update_after_turn(...)

app/services/rag_agent_service.py
- _build_session_state_context_messages(session_id)
- _schedule_session_state_update(...)

app/api/chat.py
- get_session_state(session_id)
```

## 5. 第四层：长期记忆

长期记忆存储在 Milvus：

```text
collection: long_term_memory
```

这一层只保存跨 session 仍然稳定、有用、被确认、非敏感的信息。长期记忆不是越多越好；不确定就不写。

写入前必须同时通过四个标准：

```text
1. 是否长期稳定？
2. 以后是否有用？
3. 是否被用户明确确认？
4. 是否不涉及隐私或敏感信息？
```

长期记忆字段：

```json
{
  "memory_id": "ltm_xxx",
  "user_id": "default",
  "type": "profile",
  "content": "用户主要做 Java 后端",
  "evidence": "用户明确说：我主要做 Java 后端",
  "confidence": 0.95,
  "status": "active",
  "source_session_id": "test-session",
  "created_at": "2026-06-12T10:00:00+08:00",
  "updated_at": "2026-06-12T10:00:00+08:00",
  "expires_at": null
}
```

`status` 只有两种：

```text
active：有效，可以被检索后注入上下文。
inactive：无效，不注入上下文。
```

过期机制：

```text
expires_at 默认 null。
只有用户明确给了时间范围，才写 expires_at。
检索长期记忆时，代码自动过滤 expires_at 已经过期的记忆。
如果 active 记忆已过期，检索时会顺手把 status 改成 inactive。
```

用户删除机制：

```text
如果用户明确要求“删除这条记忆”“忘掉某件事”“不要再记这个”，代码会物理删除目标长期记忆。
```

读取接口：

```text
GET /api/chat/memory/long-term
```

示例：

```bash
curl "http://127.0.0.1:9900/api/chat/memory/long-term"
```

### 长期记忆怎么使用

每轮回答前：

```text
当前问题
+ Session State
+ 短期语义记忆
        ↓
拼成长期记忆检索文本
        ↓
转 embedding
        ↓
Milvus 检索 long_term_memory
        ↓
代码过滤：
- status == active
- confidence >= LONG_TERM_MEMORY_MIN_CONFIDENCE
- expires_at 为空或未过期
        ↓
只取最关键的 3-5 条
        ↓
作为 Long-term Memory SystemMessage 注入上下文
```

### 长期记忆怎么更新

回答结束后后台调用长期记忆更新器。

输入：

```text
本轮用户问题
本轮助手回答
Session State
短期语义记忆
最近 1-3 轮原始对话
本轮检索到的长期记忆
```

输出动作：

```text
create：新增长期记忆
update：更新已有长期记忆
deactivate：旧记忆被替代，改为 inactive
delete：用户明确要求删除，物理删除
noop：不做任何事
```

相关代码：

```text
app/services/long_term_memory_service.py
- retrieve_relevant_memories()
- build_context_block()
- update_after_turn()

app/services/rag_agent_service.py
- _build_long_term_context_messages()
- _schedule_long_term_memory_update()

app/api/chat.py
- list_long_term_memory()
```

## 6. 回答模型实际看到什么

一次 `/chat` 请求里，回答模型不会直接看到完整 session 历史。它实际看到的是一组 messages，顺序大致是：

```text
第 1 条：主 system prompt
  说明 Agent 的角色、RAG 使用规则、回答格式、安全边界等。

第 2 条：Long-term Memory SystemMessage
  包含：
  - Milvus long_term_memory collection 里与当前问题相关的长期记忆

第 3 条：Session State SystemMessage
  包含：
  - data/memory/session_state/<session_id>.json 里的结构化当前任务状态

第 4 条：短期上下文 SystemMessage
  包含：
  - data/memory/short_term/<session_id>.md 里的短期语义记忆
  - 最近 1-3 轮原始对话

第 5 条：当前用户问题 HumanMessage
  也就是本次请求里的 question。
```

所以这里说的“拼上下文”，不是把字符串随便拼到用户问题里，而是按层级把状态和摘要作为额外的 `SystemMessage` 放到 Agent messages 里。

这一节只说明最终装配顺序：

```text
主 system prompt
        ↓
Long-term Memory SystemMessage
        ↓
Session State SystemMessage
        ↓
短期上下文 SystemMessage
        ↓
当前用户问题 HumanMessage
```

第二层短期上下文的具体格式、例子和更新方式已经在第 3 节说明，这里不再重复展开。

第三层 Session State 的具体字段、写入规则和 `should_update` 兜底逻辑已经在第 4 节说明。

第四层长期记忆的写入标准、检索过滤和更新动作已经在第 5 节说明。

相关代码：

```text
app/services/rag_agent_service.py
- _build_long_term_context_messages()
  检索长期记忆，构造成 SystemMessage 注入 Agent。

- _build_session_state_context_messages()
  读取 Session State，构造成 SystemMessage 注入 Agent。

- _build_short_term_context_messages()
  读取短期记忆和最近原始对话，构造成 SystemMessage 注入 Agent。

- query()
  非流式回答入口，回答前注入长期记忆、Session State 和短期上下文，回答后调度记忆更新。

- query_stream()
  流式回答入口，同样注入长期记忆、Session State 和短期上下文，并在流式完成后调度记忆更新。
```

注意：

```text
原始 JSONL 不再通过 MemorySaver 或 load_recent_messages 全量进入 Agent。
LangGraph thread_id 每次请求都会使用新 ID，避免历史原文自动回放。
```

## 7. 配置项和参数

请求参数：

```json
{
  "question": "用户问题",
  "session_id": "同一个会话使用同一个 ID",
  "images": [],
  "stream": false
}
```

`session_id` 的推荐格式：

```text
只使用字母、数字、下划线、短横线、英文点。
例如：test_session_001、docs_short_memory_demo_003
```

短期记忆模型配置：

```env
SHORT_TERM_MEMORY_MODEL=qwen3.5-plus
SESSION_STATE_MODEL=qwen3.5-plus
LONG_TERM_MEMORY_MODEL=qwen3.5-plus
LONG_TERM_MEMORY_COLLECTION_NAME=long_term_memory
LONG_TERM_MEMORY_TOP_K=5
LONG_TERM_MEMORY_MIN_CONFIDENCE=0.7
LONG_TERM_MEMORY_MIN_RELEVANCE=0.2
```

读取位置：

```text
app/config.py
- short_term_memory_model
- session_state_model
- long_term_memory_model
- long_term_memory_collection_name
- long_term_memory_top_k
- long_term_memory_min_confidence
- long_term_memory_min_relevance
```

模型选择逻辑：

```python
config.short_term_memory_model or config.rag_model
config.session_state_model or config.short_term_memory_model or config.rag_model
config.long_term_memory_model or config.session_state_model or config.short_term_memory_model or config.rag_model
```

也就是说：

```text
配置 SHORT_TERM_MEMORY_MODEL 时，用它作为短期记忆更新模型。
配置 SESSION_STATE_MODEL 时，用它作为 Session State 更新模型。
配置 LONG_TERM_MEMORY_MODEL 时，用它作为长期记忆更新模型。
不配置时，回退使用 RAG_MODEL。
```

最近原始对话轮数：

```text
app/services/short_term_memory_service.py
RECENT_DIALOGUE_ROUNDS = 3
```

当前是代码常量，表示最多读取最近 3 轮 user/assistant 原始对话。后续如果需要，可以再改成 `.env` 配置。

短期记忆写入开关：

```env
MEMORY_WRITE_ENABLED=true
```

如果设为 `false`，原始日志、短期记忆和 Session State 写入都会跳过。

## 8. curl 示例

先启动服务：

```bash
.venv/bin/python scripts/competition_eval.py --start
```

或者临时启动：

```bash
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 9900
```

读取 token：

```bash
TOKEN=$(grep '^API_BEARER_TOKEN=' .env | cut -d= -f2- | tr -d '"' | tr -d "'")
```

### 第一次请求

```bash
curl -X POST "http://127.0.0.1:9900/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "question": "我在排查支付接口变慢，22:10 后 /api/pay 的 P95 从 300ms 升到 3.8s。你先帮我整理排查方向。",
    "session_id": "docs_short_memory_demo_003"
  }'
```

真实返回节选：

```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "answer": "支付接口 P95 突增通常从以下几个维度排查：\n\n1. 外部依赖检查\n- 第三方支付渠道（微信/支付宝/银联）响应是否变慢，查看调用耗时监控\n...\n建议优先查看调用链追踪（如 SkyWalking/Zipkin），定位耗时主要分布在哪个环节，再针对性深入排查。",
    "session_id": "docs_short_memory_demo_003",
    "timestamp": 1781237283
  }
}
```

第一次请求后，后台生成：

```bash
cat data/memory/short_term/docs_short_memory_demo_003.md
```

内容节选：

```md
目标：
排查 `/api/pay` 接口 P95 耗时突增问题

关键事实：
- 接口：`/api/pay`
- 指标：P95 耗时
- 变化：300ms → 3.8s
- 起始时间：22:10 后
- 当前进展：已提供五大排查维度（外部依赖、数据库、应用层、基础设施、流量特征）

当前假设：
无

已排除方向：
无
```

### 第二次请求

继续使用同一个 `session_id`：

```bash
curl -X POST "http://127.0.0.1:9900/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "question": "数据库慢查询和连接池都查过了，没有异常。补充一下：22:05 发布了 payment-service v2.3.1，22:10 后第三方支付回调超时明显增加。下一步该怎么查？",
    "session_id": "docs_short_memory_demo_003"
  }'
```

真实返回节选：

```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "answer": "根据你补充的信息，问题高度集中在 v2.3.1 版本发布与第三方回调超时的关联上。排查优先级如下：\n\n1. 版本变更对比\n- 立即拉取 v2.3.1 与上一版本的代码 diff，重点关注支付回调处理、HTTP 客户端配置、超时参数、重试逻辑的变更\n...\n核心思路：版本发布与问题出现时间高度吻合，优先从代码和配置变更入手，配合调用链确认耗时环节，必要时回滚验证。",
    "session_id": "docs_short_memory_demo_003",
    "timestamp": 1781237342
  }
}
```

第二次请求后，短期记忆更新为：

```md
目标：
排查 `/api/pay` 接口 P95 耗时突增问题，重点定位 v2.3.1 版本发布与第三方回调超时的关联

关键事实：
- 接口：`/api/pay`
- 指标：P95 耗时 300ms → 3.8s
- 起始时间：22:10 后
- 发布记录：22:05 发布 `payment-service v2.3.1`
- 现象补充：22:10 后第三方支付回调超时明显增加
- 已排查：数据库慢查询、连接池状态均无异常

当前假设：
- 问题根因高度疑似 v2.3.1 版本变更导致
- 可能涉及：HTTP 客户端配置、超时参数、重试逻辑、回调处理代码或网络配置变更

已排除方向：
- 数据库慢查询
- 数据库连接池满/等待

下一步动作：
1. 对比 v2.3.1 与上一版本代码 diff，重点关注回调处理、HTTP 配置、超时参数
2. 通过调用链追踪确认耗时分布在“等待回调”还是“回调处理”阶段
3. 检查支付渠道侧是否有接口变更或限流
4. 必要时回滚至 v2.3.0 验证是否恢复

用户约束：
无
```

## 9. 当前边界

当前已经完成：

```text
原始 session 日志保留。
旧 daily/MEMORY.md 记忆链路已移除。
回答前注入短期语义记忆。
回答前注入最近 1-3 轮原始对话。
回答后后台更新短期语义记忆。
回答前注入 Session State。
回答后后台更新 Session State。
回答前检索并注入长期记忆。
回答后后台更新长期记忆。
```

尚未完成：

```text
长 session 的上下文压缩。
```

后续推荐顺序：

```text
1. 做上下文压缩：
   只在单 session 很长且 short-term/session state 无法覆盖过程信息时启用。
```
