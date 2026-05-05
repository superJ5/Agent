# 主线 RAG 使用报告

## 0. 当前版本说明

当前项目里有两个版本：

  后续考虑改改基线看看能不能满足要求，主线整的有点复杂了
- 基线版本：使用 LangChain 风格的普通标题/字数切分 chunk，走独立 `biz_baseline` collection。当前测试里基线表现反而好一些，主要用于对照实验和验证主线方案是否真正提升。
- 主线版本：使用子父块结构化 chunk。子块用于精确命中，父块用于补充上下文；召回逻辑集中在 `app/tools/knowledge_tool.py`，会先判断问题意图，再按 `doc_id`、`retrieval_tier`、`chunk_type` 等 metadata 做过滤检索，命中 primary 后补 support 父块，并结合词面重排、fallback 遍历和图片/源数据 metadata 组织上下文。

这份文档说明如何使用当前主线 RAG 代码完成比赛接口调用。

## 1. 主线代码定位

当前正式主线是结构化 chunk RAG：

- API 入口：`app/api/chat.py`
- Agent 编排：`app/services/rag_agent_service.py`
- 核心路由检索：`app/tools/knowledge_tool.py`
- 向量入库：`app/services/vector_index_service.py`
- 向量检索：`app/services/vector_search_service.py`
- Milvus 管理：`app/services/vector_store_manager.py`
- chunk 入库脚本：`scripts/index_manual_chunks.py`

比赛标准接口是：

```text
POST /chat
```

注意：旧接口 `/api/chat` 仍然存在，但比赛提交和队友联调应优先使用 `/chat`。

## 2. 数据放置位置

标准结构化 chunks 放在：

```text
data/manuals/chunks/
```

当前目录下的文件形态应为：

```text
data/manuals/chunks/manual_7f829388__功能键盘手册.jsonl
data/manuals/chunks/vr_bfe50169__VR头显手册.jsonl
data/manuals/chunks/manual_60e374ac__电钻手册.jsonl
```

要求：

- 每本手册一个 `.jsonl` 文件。
- 文件直接放在 `data/manuals/chunks/` 根目录。
- 不建议在 `chunks/` 下继续嵌套目录。
- 同一个 `doc_id` 不要同时保留多个版本，否则重新入库时可能互相覆盖。

当前吹风机手册存在多个版本：

```text
manual_84d80d19__吹风机手册__origin.jsonl
manual_84d80d19__吹风机手册__restrict.jsonl
manual_84d80d19__吹风机手册__rich_types.jsonl
```

正式入库前建议只保留一个版本，避免同一 `doc_id` 重复入库。

## 3. 环境配置

项目根目录需要 `.env`。

至少需要：

```env
DASHSCOPE_API_KEY=你的DashScopeKey
API_BEARER_TOKEN=队友调用/chat时使用的token
```

说明：

- `DASHSCOPE_API_KEY` 用于 embedding 和大模型回答。
- `API_BEARER_TOKEN` 用于比赛接口鉴权。
- 如果 `API_BEARER_TOKEN` 为空，代码仍要求请求头带 `Authorization: Bearer xxx`，但不会校验具体 token 值。
- 正式提交或给队友联调时，建议配置一个固定 token。

默认服务端口：

```text
9900
```

默认 Milvus：

```text
localhost:19530
```

## 4. 启动 Milvus

确保 Docker Desktop 已启动，然后在项目根目录运行：

```powershell
docker compose -f vector-database.yml up -d
```

可以等待 10 秒左右：

```powershell
timeout /t 10
```

## 5. 安装依赖

如果 `.venv` 已存在，可以直接用：

```powershell
.\.venv\Scripts\python.exe
```

如果没有 `.venv`，推荐：

```powershell
uv sync
```

或：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

## 6. 入库标准 chunks

在启动 Milvus 后运行：

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\index_manual_chunks.py
```

默认会扫描：

```text
data/manuals/chunks/
```

也可以只入库某一个文件：

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\index_manual_chunks.py --directory data\manuals\chunks\manual_7f829388__功能键盘手册.jsonl
```

入库逻辑：

- 读取标准 chunk JSONL。
- 生成文本 embedding。
- 写入 Milvus 主线 collection。
- 如果同一个 `doc_id` 已存在，会先删除该文档旧数据，再写入新数据。

## 7. 启动主线 API

启动 FastAPI：

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 9900
```

服务地址：

```text
http://localhost:9900
```

接口文档：

```text
http://localhost:9900/docs
```

也可以用项目自带脚本：

```powershell
.\start-windows.bat
```

## 8. 比赛标准接口

接口：

```http
POST /chat
```

请求头：

```http
Content-Type: application/json
Authorization: Bearer {API_BEARER_TOKEN}
```

请求体：

```json
{
  "question": "我想更换健身追踪器的表带，有其他尺寸可选吗？",
  "images": [],
  "session_id": "kf_session_889900",
  "stream": false
}
```

字段说明：

- `question`：必填，用户问题。
- `images`：可选，Base64 图片数组。
- `session_id`：可选，多轮会话 ID。不传会自动生成。
- `stream`：可选，默认 `false`。比赛批量答题建议用 `false`。

成功响应：

```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "answer": "表带尺寸如下所示。注意：单独销售的配件表带可能略有差异。<PIC:Manual16_51>",
    "session_id": "kf_session_889900",
    "timestamp": 1741008000
  }
}
```

失败响应：

```json
{
  "code": 500,
  "msg": "错误信息",
  "data": {
    "answer": "",
    "session_id": "kf_session_889900",
    "timestamp": 1741008000
  }
}
```

## 9. curl 调用示例

PowerShell 示例：

```powershell
$body = @{
  question = "键盘USB-C接口在哪？"
  images = @()
  session_id = "kf_session_test_001"
  stream = $false
} | ConvertTo-Json -Depth 5

curl.exe -X POST "http://localhost:9900/chat" `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer test-token" `
  -d $body
```

如果 `.env` 配置了：

```env
API_BEARER_TOKEN=test-token
```

则请求头必须使用同一个 token。

## 10. Python 调用示例

```python
import requests

resp = requests.post(
    "http://localhost:9900/chat",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer test-token",
    },
    json={
        "question": "电钻电池怎么充电？",
        "images": [],
        "session_id": "kf_session_test_002",
        "stream": False,
    },
    timeout=30,
)

print(resp.status_code)
print(resp.json()["data"]["answer"])
```

## 11. 使用流程

拿到 400 道题，推荐流程是：

1. 启动 Milvus。
2. 运行 `scripts/index_manual_chunks.py` 完成主线知识库入库。
3. 启动 FastAPI。
4. 写一个批量脚本逐条请求 `POST /chat`。
5. 保存每题的 `data.answer` 作为最终回答。

批量请求时建议：

- 每题设置独立 `session_id`，避免不同问题互相污染上下文。
- `stream` 固定为 `false`。
- 请求超时设置 30 秒。
- 如果请求失败，记录原始问题、错误响应和时间戳，方便重试。

## 12. 当前能力边界

当前主线已经具备：

- 标准 RESTful `/chat` 接口。
- Bearer Token 请求头校验。
- 标准 JSON 请求与响应格式。
- 文本 RAG 检索。
- 结构化 chunk metadata 过滤。
- 图片 ID 和图片路径作为检索上下文提供给 LLM。
- 多轮会话 `session_id`。
- 流式响应能力。

当前仍需注意：

- AIOps 相关 MCP 服务和 `aiops-docs` 上传逻辑还没从项目里删除，这部分可能影响数据库或向量库环境；比赛 RAG 联调时需要留意是否额外写入了非手册数据。
- 当前 chunk 的 `text` 字段实际保留了标题层级、换行、列表编号、`<PIC:...>` 和原始图文顺序；入库时也会把 `text` 作为 `page_content` 保留。
- 但当前回答阶段还没有专门解析 `<PIC:...>` 前后文字来约束图片输出；也就是说，图文结构有数据基础，但代码还没有充分利用。
- `/chat` 响应里没有单独结构化返回 `pic_ids` 数组，图片通常需要由答案文本中的 `<PIC:...>` 或上下文引导输出。
- 如果评分系统要求提交“答案文本 + 图片 ID 列表”的离线文件，还需要额外写批量导出脚本。
- 如果同一手册多个 chunk 文件有相同 `doc_id`，正式入库前必须只保留一个版本。

## 13. 最小可用命令清单

```powershell
# 1. 启动 Milvus
docker compose -f vector-database.yml up -d

# 2. 入库主线 chunks
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\index_manual_chunks.py

# 3. 启动 API
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 9900

# 4. 调用 /chat
curl.exe -X POST "http://localhost:9900/chat" `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer test-token" `
  -d "{\"question\":\"键盘USB-C接口在哪？\",\"images\":[],\"session_id\":\"kf_session_demo\",\"stream\":false}"
```
