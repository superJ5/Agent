# .env 配置改动报告

## 对比范围

- 基准版本：`origin/main:.env`
- 当前版本：工作区当前 `.env`
- 生成时间：2026-05-26
- 安全说明：报告不展示任何真实 API Key、Token、Secret，只说明是否为空、是否已设置、是否新增。

## 总览

| 项目 | origin/main | 当前 .env |
| --- | ---: | ---: |
| 配置项数量 | 20 | 40 |
| 新增配置项 | 0 | 20 |
| 删除配置项 | - | 0 |
| 改动配置项 | - | 2 |

结论：当前 `.env` 在 `origin/main` 基础上主要新增了 RAG 召回、Reranker、Memory 相关配置；没有删除 `origin/main` 原有配置。

## 新增配置项

### 应用版本

| 配置项 | 当前值 | 作用 |
| --- | --- | --- |
| `APP_VERSION` | `1.0.0` | 应用版本号，供 FastAPI 元信息和日志使用。 |

### RAG 召回配置

| 配置项 | 当前值 | 作用 |
| --- | --- | --- |
| `RAG_INTENT_STRATEGY` | `none` | 控制意图理解策略；当前关闭额外意图策略。 |
| `RAG_ENABLE_VECTOR_RECALL` | `true` | 开启向量召回。 |
| `RAG_ENABLE_BM25_RECALL` | `true` | 开启 BM25 召回。 |
| `RAG_VECTOR_WEIGHT` | `0.6` | 合并打分时向量召回权重。 |
| `RAG_BM25_WEIGHT` | `0.4` | 合并打分时 BM25 召回权重。 |
| `RAG_ENABLE_SCAN_FALLBACK` | `true` | 开启 scan fallback，向量或 BM25 不足时补充候选。 |
| `RAG_SCAN_CANDIDATE_LIMIT` | `4096` | scan fallback 最大候选数量。 |

### Reranker 配置

| 配置项 | 当前值 | 作用 |
| --- | --- | --- |
| `RAG_RERANKER_PROVIDER` | `dashscope` | 使用 DashScope reranker provider。 |
| `RAG_RERANKER_MODEL` | `qwen3-rerank` | 重排模型。注意这才是 rerank 模型，不是 `RAG_MODEL`。 |
| `RAG_RERANKER_ENDPOINT` | `https://dashscope.aliyuncs.com/compatible-api/v1/reranks` | DashScope rerank API 地址。 |
| `RAG_RERANKER_TIMEOUT_MS` | `3000` | rerank 调用超时时间，单位毫秒。 |
| `RAG_RERANKER_TOP_N` | `32` | rerank 输入/输出候选数量上限。 |

### Memory 配置

| 配置项 | 当前值 | 作用 |
| --- | --- | --- |
| `MEMORY_ROOT` | `data/memory` | 记忆系统文件根目录。 |
| `MEMORY_RECENT_LIMIT` | `10` | 注入当前会话最近消息的数量上限。 |
| `MEMORY_COLLECTION_NAME` | `memory` | 记忆向量库 collection 名称。 |
| `MEMORY_SEARCH_TOP_K` | `5` | `memory_search` 默认返回数量。 |
| `MEMORY_DAILY_MAX_CHARS` | `5000` | 每日记忆文件最大字符数控制。 |
| `MEMORY_SUMMARY_INPUT_MAX_CHARS` | `20000` | 记忆总结输入最大字符数。 |
| `MEMORY_SUMMARY_MODEL` | `qwen3.5-plus` | 记忆总结使用的模型；当前显式指定为主对话模型。 |

## 改动配置项

| 配置项 | origin/main | 当前 .env | 说明 |
| --- | --- | --- | --- |
| `API_BEARER_TOKEN` | 空 | 已设置 | 当前比赛接口 `/chat` 需要使用匹配的 `Authorization: Bearer ...`。 |
| `DASHSCOPE_API_KEY` | 已设置 | 已设置，值不同 | 两边都有 key，但当前工作区的值与 `origin/main` 不同；真实值未展示。 |

## 保持不变的关键配置

以下关键配置在 `origin/main` 与当前 `.env` 中保持一致：

| 配置项 | 当前值 | 说明 |
| --- | --- | --- |
| `DASHSCOPE_API_BASE` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | ChatQwen/OpenAI 兼容模式 base URL。 |
| `DASHSCOPE_MODEL` | `qwen3.5-plus` | 默认 DashScope 聊天模型。 |
| `DASHSCOPE_EMBEDDING_MODEL` | `text-embedding-v4` | 向量模型。 |
| `RAG_MODEL` | `qwen3.5-plus` | RAG 最终回答模型。 |
| `RAG_TOP_K` | `3` | 最终返回给回答链路的主命中数量。 |
| `MILVUS_HOST` | `localhost` | Milvus 主机。 |
| `MILVUS_PORT` | `19530` | Milvus 端口。 |
| `MILVUS_TIMEOUT` | `10000` | Milvus 超时配置。 |
| `MCP_CLS_URL` | `http://localhost:8003/mcp` | CLS MCP 服务地址。 |
| `MCP_MONITOR_URL` | `http://localhost:8004/mcp` | Monitor MCP 服务地址。 |

## 对运行链路的影响

当前 `.env` 支持以下链路：

1. 启动 FastAPI。
2. 连接 Milvus。
3. 初始化 BM25 provider。
4. 使用向量召回和 BM25 召回。
5. 合并召回候选并打分。
6. 调用 `qwen3-rerank` 做 rerank。
7. 如果 rerank 超时或额度失败，降级到 lexical fallback。
8. 最终返回 top hits 和 evidence metadata。
9. 同时启用 memory 相关配置，用于会话记忆、长期记忆和记忆搜索。

## 注意事项

- 当前 `.env` 包含真实密钥状态，提交前必须确认不会把真实 `DASHSCOPE_API_KEY` 或 `API_BEARER_TOKEN` 推到远端。
- `RAG_MODEL=qwen3.5-plus` 是最终回答模型。
- `RAG_RERANKER_MODEL=qwen3-rerank` 是专门的重排模型。
- `MEMORY_SUMMARY_MODEL=qwen3.5-plus` 当前显式指定；如果为空，代码会退回使用 `RAG_MODEL`。
- 最新日志显示 `qwen3-rerank` 可能因为免费额度耗尽或 3000ms 超时降级到 lexical fallback；这不会阻塞问答，但会影响重排质量。

