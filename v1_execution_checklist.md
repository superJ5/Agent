# V1 执行清单：结构化说明书 Chunk 与分层检索

## 1. 这份清单的用途

这是一份面向当前可落地路线的执行清单。

它不再以“先做页级 PDF 图文联合”为唯一主线，而是基于我们已经跑通的人工清洗与结构化经验，把第一版路线收敛成这条最小可交付链：

`清洗后的 Markdown -> 结构化 chunk -> tier/type 感知检索 -> support 补全 -> auxiliary 定向查询/保底 -> 全库遍历 fallback -> 返回文本与图片证据`

这份清单默认输入已经不是“完全原始 PDF”，而是至少完成了基本清洗、标题重建、`<PIC>` 位置调整、OCR 痕迹标注的说明书 Markdown。

这份清单只定义“怎么做”，**不自动授权“做多大范围”**。每次执行时，范围、输入来源、是否覆盖旧产物，都必须优先服从用户当回合的明确要求。

---

## 1.1 执行边界与防误操作规则

### 总原则

- [ ] 用户明确点名哪几份文档，就只处理哪几份文档
- [ ] 用户要求“基于现有 chunk 往下做”，就不得回退到清洗前链路
- [ ] 用户未明确要求全量批跑时，不得默认对全库执行
- [ ] 用户未明确允许覆盖旧结果时，不得直接覆盖已有 chunk 产物

### 输入来源优先级

按下面顺序选择输入，不得跳回更早阶段：

1. 现有 chunk 产物
2. 已清洗 Markdown
3. 更早阶段的 parsed / raw / PDF 中间产物

只有在上一级输入不存在、损坏，或用户明确要求重建时，才允许使用下一级输入。

### 输出策略

- [ ] 默认并行输出到新路径，不覆盖旧结果
- [ ] 只有用户明确说“覆盖原结果”时，才允许覆盖已有文件
- [ ] 批量脚本在未加文档过滤参数前，不得直接运行

### Step 3 专项红线

- [ ] 不得把“通用执行清单”误当成“当前就要全量执行”
- [ ] 不得因为脚本支持批处理，就默认全量重跑
- [ ] 不得在未确认目标文档范围时改写全局 chunk 输出目录
- [ ] 不得为了生成结构化 chunk，反向改坏已被用户认可的旧 chunk

### 执行前自检

在运行任何 chunk 生成脚本前，先确认这 4 件事：

1. 当前只处理哪些文档
2. 当前以上哪一级输入作为起点
3. 本次输出是并行新路径还是覆盖旧路径
4. 若已有用户认可结果，本次是否允许改写

---

## 2. V1 范围冻结

### 2.1 V1 必做

- [ ] 输入文档统一为清洗后的 Markdown
- [ ] `<PIC>` 摆放遵守当前 chunk 规则
- [ ] chunk 产物统一带 `tier`、`type`、`section_path`、`pic_ids`、`source_lines`
- [ ] OCR/缺失/图文错位问题能写入 chunk 元数据
- [ ] Milvus 至少能检索 `primary` 与 `support`
- [ ] `auxiliary` 能被定向检索，并可作为保底层使用
- [ ] 检索层能返回 `text + metadata + pics + source trace`
- [ ] 主路由检索失败后，能触发全库遍历 fallback

### 2.2 V1 不做

- [ ] 不做图片向量检索
- [ ] 不做自动化复杂 OCR 修复
- [ ] 不做 raw PDF 到高质量 Markdown 的全自动重建
- [ ] 不做跨文档复杂知识图谱
- [ ] 不做多 collection 复杂融合重排
- [ ] 不做大规模离线 benchmark 平台

### 2.3 V1 完成标准

满足以下条件即可认为第一版完成：

1. 给定一篇或一组**被明确指定**的说明书输入，可以稳定产出结构化 chunk 文件
2. 给定一个问题，可以按意图路由到合适的 `tier/type`
3. 命中 `primary` 时，能自动补出对应 `support`
4. 命中带图 chunk 时，能同时返回 `pic_ids`
5. 主路由检索失败时，能自动进入扩池和全库遍历 fallback
6. 若仍失败，系统能给出“可能存在 OCR / 源数据缺失”的追溯提示
7. 在未获授权时，不覆盖既有人工认可结果

---

## 3. 当前机制总览

### 3.1 两条核心轴

- `retrieval_tier` 决定这个 chunk 在检索里扮演什么角色
- `chunk_type` 决定这个 chunk 本身是什么语义形态

### 3.2 tier 定义

- `primary`：主回答块，优先作为直接答案证据
- `support`：父级概览块，用于补上下文
- `auxiliary`：辅助块，用于目录、图片路径、OCR 核查、源数据追溯等

### 3.3 type 定义

常见 `chunk_type`：

- `subsection`
- `text_image_atomic`
- `feature_group`
- `procedure_overview`
- `procedure_step`
- `section_summary`
- `legal_clause`
- `aux_navigation`
- `metadata_image_path`

### 3.4 当前 chunk 规则要点

- [ ] 先按文档区域分区
- [ ] 再按标题层级建立骨架
- [ ] 单项图片和正文强绑定时切成 `text_image_atomic`
- [ ] 章节整体图片挂标题下方，相关正文切成 `feature_group` 或 `section_summary`
- [ ] 步骤类内容切成 `procedure_overview + procedure_step`
- [ ] 法务类内容优先切成 `support + legal_clause`
- [ ] 目录、图片锚点顺序、图片路径对照降为 `auxiliary`
- [ ] OCR 标记与原因同步进 `source_issue_flags/source_issue_note`

---

## 4. 检索机制伪代码

### 4.1 意图到候选池的映射

```python
ROUTES = {
    "component_lookup": {
        "candidate_types": ["text_image_atomic", "feature_group", "subsection"],
        "tier_order": ["primary", "support", "auxiliary"],
    },
    "procedure": {
        "candidate_types": ["procedure_step", "subsection", "procedure_overview"],
        "tier_order": ["primary", "support", "auxiliary"],
    },
    "legal_policy": {
        "candidate_types": ["legal_clause", "subsection", "section_summary"],
        "tier_order": ["primary", "support", "auxiliary"],
    },
    "overview": {
        "candidate_types": ["section_summary", "procedure_overview", "subsection"],
        "tier_order": ["support", "primary", "auxiliary"],
    },
    "image_trace": {
        "candidate_types": ["aux_navigation", "metadata_image_path", "text_image_atomic", "feature_group"],
        "tier_order": ["auxiliary", "primary", "support"],
    },
    "ocr_audit": {
        "candidate_types": ["aux_navigation", "metadata_image_path", "subsection", "legal_clause"],
        "tier_order": ["auxiliary", "primary", "support"],
    },
}
```

### 4.2 主检索流程

```python
def retrieve(query: str, doc_filter: str | None = None, top_k: int = 8):
    intent = classify_intent(query)
    route = ROUTES[intent]

    hits = search_by_route(
        query=query,
        doc_filter=doc_filter,
        candidate_types=route["candidate_types"],
        tier_order=route["tier_order"],
        top_k=top_k,
    )

    if has_good_primary_hit(hits):
        hits = attach_parent_support(hits)
        return finalize_result(
            query=query,
            intent=intent,
            hits=hits,
            retrieval_stage="primary_route",
        )

    if needs_support_first(intent) or not has_good_hit(hits):
        support_hits = search_support_pool(
            query=query,
            doc_filter=doc_filter,
            candidate_types=route["candidate_types"],
            top_k=top_k,
        )
        if has_good_hit(support_hits):
            return finalize_result(
                query=query,
                intent=intent,
                hits=support_hits,
                retrieval_stage="support_route",
            )

    if intent in {"image_trace", "ocr_audit"}:
        aux_hits = search_auxiliary_pool(
            query=query,
            doc_filter=doc_filter,
            candidate_types=route["candidate_types"],
            top_k=top_k,
        )
        if has_good_hit(aux_hits):
            return finalize_result(
                query=query,
                intent=intent,
                hits=aux_hits,
                retrieval_stage="auxiliary_route",
            )

    expanded_hits = search_with_broader_filters(
        query=query,
        doc_filter=doc_filter,
        original_route=route,
        top_k=top_k,
    )
    if has_good_hit(expanded_hits):
        expanded_hits = attach_parent_support(expanded_hits)
        return finalize_result(
            query=query,
            intent=intent,
            hits=expanded_hits,
            retrieval_stage="expanded_route",
        )

    scan_hits = traverse_all_chunks(
        query=query,
        doc_filter=doc_filter,
        fields=["title", "section_path", "index_text", "text", "pic_ids", "source_issue_note"],
        top_k=top_k,
    )
    if has_good_hit(scan_hits):
        scan_hits = attach_parent_support(scan_hits)
        return finalize_result(
            query=query,
            intent=intent,
            hits=scan_hits,
            retrieval_stage="full_chunk_scan",
        )

    source_hits = traverse_source_markdown(
        query=query,
        doc_filter=doc_filter,
    )
    return finalize_result(
        query=query,
        intent=intent,
        hits=source_hits,
        retrieval_stage="source_markdown_scan",
        warning="可能存在 OCR、漏切或源数据缺失问题",
    )
```

### 4.3 返回结果最少字段

```json
{
  "intent": "component_lookup",
  "retrieval_stage": "primary_route",
  "hits": [
    {
      "chunk_id": "kbd_sec4_item_10",
      "doc_id": "manual_7f829388",
      "retrieval_tier": "primary",
      "chunk_type": "text_image_atomic",
      "section_path": ["4. 功能键盘"],
      "title": "10. 大小写锁定指示灯",
      "parent_chunk_id": "kbd_sec4_overview",
      "pic_ids": ["Manual21_8"],
      "source_file": "...reviewed_reconstructed.md",
      "source_lines": [48, 48],
      "text": "...",
      "index_text": "..."
    }
  ],
  "support_hits": [],
  "warnings": []
}
```

---

## 5. 总执行顺序

按下面顺序推进，不要跳步：

1. 冻结 chunk schema 与检索路由
2. 统一清洗后的 Markdown 输入目录
3. 统一 `<PIC>` 摆放规则与 OCR 标记规则
4. 生成结构化 chunk JSONL
5. 做 chunk 完整性校验
6. 入 Milvus 并保留 tier/type 过滤能力
7. 改检索逻辑为“意图路由 + 分层召回 + fallback 遍历”
8. 改知识工具输出结构
9. 做最小验收

---

## 6. 一步一步执行

## Step 0. 冻结 V1 schema 与路由

### 目标

今天就把第一版的 chunk 字段、tier/type 体系、检索路由定死。

### 要做的事

- [ ] 冻结最少字段：`chunk_id/doc_id/source_file/source_lines/chunk_type/retrieval_tier/section_path/title/parent_chunk_id/pic_ids/text/index_text`
- [ ] 追加问题标记字段：`source_quality/source_issue_flags/source_issue_note`
- [ ] 明确 `primary/support/auxiliary` 的职责
- [ ] 明确意图分类到 `candidate_types/tier_order` 的映射

### 输出物

- [ ] 统一 chunk schema
- [ ] 统一检索路由表

### 完成标准

后续开发中，凡是新字段或新检索流程如果不直接服务于这套机制，一律延后。

---

## Step 1. 统一输入目录与文档来源

### 目标

先把**当前被选中的文档**对应的清洗 Markdown、图片资源、chunk 产物和索引产物放到固定位置。

### 建议目录

```text
data/
  manuals/
    reviewed_markdown/
    source_images/
    chunk_outputs/
    index_ready/
    retrieval_eval/
```

### 建议使用的现有脚本

- `scripts/organize_manual_raw_assets.py`
- `scripts/convert_manual_dataset.py`
- `scripts/batch_convert_manual_datasets.py`

### 要做的事

- [ ] 只收纳本次目标文档的清洗 Markdown
- [ ] 只收纳本次目标文档的 `<PIC>` 对应图片资源
- [ ] 确保 `doc_id` 与目录命名稳定
- [ ] 确保每个文档都能追溯到原始来源
- [ ] 若库内已有旧 chunk，记录其路径并标记“是否允许覆盖”

### 完成标准

任意一份**本次目标说明书**，都能在固定路径下找到：

- 清洗后的 Markdown
- 图片资源
- chunk 输出目录
- 索引准备目录

---

## Step 2. 固化清洗规范

### 目标

在切 chunk 之前，先把对后续检索影响最大的清洗规范固定下来。

### 必须遵守的规则

- [ ] 如果图片描述的是正文某一项内容，`<PIC>` 放在该项正文后，同一行
- [ ] 如果图片描述的是章节整体，标题下单独一行放 `<PIC>`
- [ ] 如果同一视角多图只覆盖部分子项，图片放在对应子区间之前
- [ ] 状态说明挂到真正描述的部件项下，不机械跟随 OCR 顺序
- [ ] `(ocr)` 以及其原因保留在文档里，并同步进入 chunk 元数据

### 对应规则文档

- [05_chunking_rules.md](<E:/.codex/worktrees/d883/ai_agent_competition/outputs/stage3_claude/readable_review/05_chunking_rules.md:58>)

### 完成标准

同一类图文场景在不同文档里的摆放方式一致，不再靠临场判断。

---

## Step 3. 生成结构化 chunk

### 目标

按当前 chunk 机制，把**当前目标文档**产出为稳定的结构化 JSONL。

### 推荐使用的现有脚本

- `scripts/build_structured_chunks.py`

### 执行前判断

先按下面顺序判断，不得跳步：

1. 是否存在用户已认可的旧 chunk
2. 用户这次是要求“基于现有 chunk 往下做”，还是“从清洗 Markdown 重新切”
3. 本次是否允许覆盖旧产物
4. 若未明确允许覆盖，是否已准备并行输出路径

如果第 1 条答案为“是”，且第 2 条答案为“基于现有 chunk 往下做”，则本步骤输入应优先使用现有 chunk，而不是回退到 Markdown / parsed / raw。

### 要做的事

- [ ] 按区域分区
- [ ] 按标题层级建立 chunk 骨架
- [ ] 生成 `support` 父块
- [ ] 生成 `primary` 子块
- [ ] 生成 `auxiliary` 导航与元数据块
- [ ] 写入 `chunk_type`
- [ ] 写入 `retrieval_tier`
- [ ] 写入 `section_path`
- [ ] 写入 `parent_chunk_id`
- [ ] 写入 `pic_ids`
- [ ] 写入 `source_issue_*`
- [ ] 若旧结果已被认可，优先输出到并行新路径
- [ ] 若用户未授权覆盖，不得改写旧 `chunks.jsonl`

### chunk 重点规则

- [ ] `####` 通常切成 `primary`
- [ ] `#####` 法务条款优先切成 `legal_clause`
- [ ] 步骤类切成 `procedure_overview + procedure_step`
- [ ] 并列条款可切成 `subsection` 子块，但不强制升标题
- [ ] 目录、图片锚点顺序、图片路径对照降为 `auxiliary`

### 输出物

- [ ] `chunks.jsonl`
- [ ] `chunk_inventory.md`
- [ ] `chunk_report.md`

### 完成标准

任意一个**本次新产出的** chunk 都能解释清楚：

- 它为什么是这个 `type`
- 它为什么属于这个 `tier`
- 它跟哪个父块相关
- 它绑定了哪些图片

并且满足：

- [ ] 本次只处理用户指定文档
- [ ] 没有误改非目标文档的 chunk
- [ ] 没有在未授权情况下覆盖旧结果

---

## Step 4. 做 chunk 完整性校验

### 目标

不要只产出 chunk，要验证这批 chunk 是否真的适合入库和检索。

### 推荐使用的现有脚本

- `scripts/verify_chunk_integrity.py`
- `scripts/check_manual_indexing_readiness.py`

### 要做的事

- [ ] 检查 `chunk_id` 是否唯一
- [ ] 检查 `parent_chunk_id` 是否都能追溯到父块
- [ ] 检查 `source_lines` 是否可回溯
- [ ] 检查 `pic_ids` 是否对应真实图片
- [ ] 检查 `auxiliary` 是否没有误混进主回答块
- [ ] 检查 OCR 风险是否进入元数据

### 完成标准

抽查时能回答下面这些问题：

1. 这条为什么是 `primary`
2. 这条为什么不是 `support`
3. 这张图为什么绑到这条正文
4. 这段 OCR 风险为什么打这个标签

---

## Step 5. 把 chunk 入索引

### 目标

先把分层 chunk 检索链打通，重点是 metadata 够强，过滤够稳。

### 推荐改动文件

- [app/services/vector_index_service.py](/E:/.codex/worktrees/3f3c/ai_agent_competition/app/services/vector_index_service.py:1)
- [app/services/vector_store_manager.py](/E:/.codex/worktrees/3f3c/ai_agent_competition/app/services/vector_store_manager.py:1)
- [app/services/vector_search_service.py](/E:/.codex/worktrees/3f3c/ai_agent_competition/app/services/vector_search_service.py:1)
- `scripts/index_manual_chunks.py`

### 要做的事

- [ ] 读取 `chunks.jsonl`
- [ ] 为 `index_text` 生成向量
- [ ] 保留 `text` 作为原文展示字段
- [ ] metadata 至少写入 `retrieval_tier/chunk_type/section_path/title/parent_chunk_id/pic_ids/source_lines/source_issue_flags`
- [ ] 支持按 `doc_id`、`retrieval_tier`、`chunk_type` 过滤

### V1 原则

- [ ] 先以文本向量为主，不做图片向量
- [ ] 先支持单 collection + 强 metadata 过滤
- [ ] `auxiliary` 可以入库，但默认不参与主问答召回

### 完成标准

给定一个问题时，至少能按 `tier/type` 过滤后拿到稳定候选集。

---

## Step 6. 改造检索逻辑

### 目标

把现在的普通文本检索，改成“意图路由 + 分层召回 + fallback 遍历”。

### 推荐改动文件

- [app/tools/knowledge_tool.py](/E:/.codex/worktrees/3f3c/ai_agent_competition/app/tools/knowledge_tool.py:1)
- [app/services/rag_agent_service.py](/E:/.codex/worktrees/3f3c/ai_agent_competition/app/services/rag_agent_service.py:1)
- [app/services/vector_search_service.py](/E:/.codex/worktrees/3f3c/ai_agent_competition/app/services/vector_search_service.py:1)

### 要做的事

- [ ] 增加意图分类
- [ ] 增加 `candidate_types` 映射
- [ ] 增加 `tier_order` 映射
- [ ] 命中 `primary` 后补 `support`
- [ ] 对 `image_trace/ocr_audit` 允许直接查 `auxiliary`
- [ ] 主路召回弱时做扩池检索
- [ ] 扩池仍失败时做全库 chunk 遍历
- [ ] 仍失败时回扫源 Markdown

### fallback 顺序

1. 主路由检索
2. `support` 补查
3. `auxiliary` 定向检索
4. 放宽 `type/tier` 的扩池检索
5. 全库 chunk 遍历
6. 源 Markdown 回扫

### 完成标准

检索失败时系统不再直接“空返回”，而是能进入下一层 fallback。

---

## Step 7. 改造返回结构

### 目标

让上层拿到的不是“几段文本”，而是一组可解释的证据对象。

### 建议返回结构

```json
{
  "intent": "procedure",
  "retrieval_stage": "primary_route",
  "hits": [
    {
      "chunk_id": "...",
      "retrieval_tier": "primary",
      "chunk_type": "procedure_step",
      "title": "...",
      "text": "...",
      "pic_ids": [],
      "source_file": "...",
      "source_lines": [55, 55],
      "source_issue_flags": []
    }
  ],
  "support_hits": [
    {
      "chunk_id": "...",
      "retrieval_tier": "support",
      "chunk_type": "procedure_overview"
    }
  ],
  "warnings": []
}
```

### 要做的事

- [ ] 返回 `intent`
- [ ] 返回 `retrieval_stage`
- [ ] 返回 `tier/type`
- [ ] 返回 `pic_ids`
- [ ] 返回 `source_file/source_lines`
- [ ] 返回 `source_issue_*`

### 完成标准

上游看到命中结果时，能知道“为什么命中、命中在哪、是否有图、是否有风险”。

---

## Step 8. 做最小验收

### 目标

用少量高代表性问题验证这套分层检索是否真的比普通分块检索更稳。

### 验收问题至少覆盖这 6 类

- [ ] 部件定位
- [ ] 操作步骤
- [ ] 法务/保修
- [ ] 章节概览
- [ ] 图片追踪
- [ ] OCR 核查/源数据追溯

### 每条用例检查项

- [ ] 是否路由到了对的意图
- [ ] 是否优先命中了合理的 `primary/support/auxiliary`
- [ ] `chunk_type` 是否符合预期
- [ ] 是否补出了正确父块
- [ ] 图片或 OCR 风险信息是否返回正确
- [ ] 主路失败时是否进入了 fallback

### 完成标准

至少 `6` 类问题都能跑通，且 fallback 不再是摆设。

---

## 7. 文件改动优先级

优先按这个顺序动：

1. `scripts/organize_manual_raw_assets.py`
2. `scripts/convert_manual_dataset.py`
3. `scripts/build_structured_chunks.py`
4. `scripts/verify_chunk_integrity.py`
5. `scripts/check_manual_indexing_readiness.py`
6. `scripts/index_manual_chunks.py`
7. [app/services/vector_search_service.py](/E:/.codex/worktrees/3f3c/ai_agent_competition/app/services/vector_search_service.py:1)
8. [app/services/vector_index_service.py](/E:/.codex/worktrees/3f3c/ai_agent_competition/app/services/vector_index_service.py:1)
9. [app/tools/knowledge_tool.py](/E:/.codex/worktrees/3f3c/ai_agent_competition/app/tools/knowledge_tool.py:1)
10. [app/services/rag_agent_service.py](/E:/.codex/worktrees/3f3c/ai_agent_competition/app/services/rag_agent_service.py:1)

---

## 8. 推荐执行节奏

### 第一天

- [ ] 冻结 schema 与路由
- [ ] 统一输入目录
- [ ] 固化 `<PIC>` 与 OCR 标记规范

### 第二天

- [ ] 生成结构化 chunk
- [ ] 做完整性校验
- [ ] 准备 index-ready 数据

### 第三天

- [ ] 入 Milvus
- [ ] 改造检索逻辑
- [ ] 接出统一返回结构

### 第四天

- [ ] 做 6 类样例验收
- [ ] 调整 fallback 阈值
- [ ] 修正明显误路由与误召回

---

## 9. 过程中要坚决避免的事

- [ ] 不要把 `tier` 和 `type` 混成一套概念
- [ ] 不要只看向量分数，不看 `source_issue_*`
- [ ] 不要让 `auxiliary` 默认污染主问答召回
- [ ] 不要忽略 `<PIC>` 摆放规则对后续检索的影响
- [ ] 不要把所有 `a/b/c` 都机械升成标题
- [ ] 不要在 fallback 缺失时直接返回“查不到”

---

## 10. 你现在就可以开始的第一步

如果从这一刻开工，直接按这个顺序：

1. 选一批已经清洗好的说明书 Markdown
2. 检查 `<PIC>` 摆放是否符合现规则
3. 检查 `(ocr)` 及其原因是否写清楚
4. 用 `scripts/build_structured_chunks.py` 产出 `chunks.jsonl`
5. 用 `scripts/verify_chunk_integrity.py` 做校验
6. 再进入索引与检索实现

只要你先把“高质量清洗 Markdown -> 高质量 chunk -> 可解释 metadata”这条链打稳，后面的 RAG 才会真正稳。
