# RAG 交接说明

这份说明用于把当前可运行的 RAG 主线代码交给队友，避免把正式检索链和 baseline 对照实验混在一起。

## 1. 正式交付分支

- 当前正式交付分支：`codex/rag-delivery`
- baseline 对照实验分支：`codex/baseline-char-chunk`

建议队友后续联调 `/chat`、试跑赛题、继续接入新手册时，都以 `codex/rag-delivery` 为准。

## 2. 正式 RAG 主线文件

这些文件属于正式 RAG 运行主链：

- `app/tools/knowledge_tool.py`
- `app/services/vector_index_service.py`
- `app/services/vector_search_service.py`
- `app/services/vector_store_manager.py`
- `scripts/index_manual_chunks.py`

这几部分负责：

- 读取标准 `chunks.jsonl`
- 写入 Milvus
- 文本 embedding 检索
- `doc_id / retrieval_tier / chunk_type` 过滤
- 命中主块后补父块 `support`
- 返回 `pic_ids / image_paths / OCR 风险字段`

## 3. 标准输入数据

当前主线默认接收“已经清洗完成的结构化 chunk”。

每条 chunk 建议至少包含这些字段：

- `chunk_id`
- `doc_id`
- `doc_name`
- `source_file`
- `source_lines`
- `chunk_type`
- `retrieval_tier`
- `section_path`
- `title`
- `parent_chunk_id`
- `pic_ids`
- `text`
- `index_text`
- `source_quality`
- `source_issue_flags`
- `source_issue_note`

图文规范：

- 单项图片：`<PIC:...>` 紧跟对应正文
- 章节总览图：放在标题下一行
- OCR 风险：用 `source_issue_flags + source_issue_note` 表达

## 4. baseline 对照工具

这些脚本是实验/对照用途，不属于 `/chat` 主链运行必需：

- `scripts/build_baseline_chunks.py`
- `scripts/compare_baseline_retrieval.py`
- `scripts/build_retrieval_evidence_baseline.py`

它们的用途是：

- 把同一份手册做成简单粗切 baseline chunk
- 和结构化 chunk 做 A/B 检索对比
- 帮助判断主线方案是否真的优于粗切

队友如果只是联调正式系统，可以先不碰这些脚本。

## 5. 当前主线能力边界

当前分支已经跑通：

- 标准 chunk 入库
- 检索召回
- 图片跟随返回
- 新文档自动接入第一版
- OCR / 法务问题的基础压噪

但还没有完全做完的部分：

- 更多新 `chunk_type` 的归一化路由
- 故障排查类问题的进一步调优
- 完整比赛评测集和正式验证报告

所以当前状态可以定义为：

**可交接、可联调、可试题，但还处于继续优化阶段。**

## 6. 本地环境说明

以下内容不建议直接提交仓库：

- `.env`
- `data/`

其中：

- `.env` 包含本地 API Key
- `data/` 包含本地入库数据、baseline 产物、测试报告

## 7. 队友最小使用路径

如果队友只想尽快跑通当前 RAG：

1. 准备 `.env`
2. 启动 Milvus
3. 把标准 `chunks.jsonl` 放到指定输入目录
4. 运行 `scripts/index_manual_chunks.py`
5. 从 `/chat` 或检索调用链验证返回结果

## 8. 一句话区分

- `codex/rag-delivery`：正式可交付 RAG 主线
- `codex/baseline-char-chunk`：baseline 对照实验线
