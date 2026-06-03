# Code Change Report - english_chunkrecall_add vs chunk_middle_result4

## Compare Scope

Comparison direction:

```text
chunk_middle_result4 -> english_chunkrecall_add
```

Current branch:

```text
english_chunkrecall_add
```

Branch relationship observed:

```text
english_chunkrecall_add has 3 commits not in chunk_middle_result4
```

Overall branch diff:

```text
22 files changed, 2562 insertions(+), 2506 deletions(-)
```

This report records code/data/test/config changes between the two branches and separately calls out the two `6000` changes.

## File Change Summary

Modified files:

```text
.env
app/retrieval/bm25_provider.py
app/retrieval/evidence.py
app/retrieval/recall.py
app/services/vector_search_service.py
app/services/vector_store_manager.py
app/tools/knowledge_tool.py
scripts/apply_big_support_policy.py
scripts/index_manual_chunks.py
tests/retrieval/test_bm25_provider.py
tests/retrieval/test_evidence.py
tests/retrieval/test_metadata_indexing_filter.py
tests/retrieval/test_recall.py
```

Added files:

```text
app/retrieval/tier_policy.py
data/test_boat_airfryer_241_250.csv
data/submission_boat_airfryer_241_250.csv
doc/code_change_report_2026-06-02.md
scripts/diagnose_doc_recall.py
tests/retrieval/test_tier_policy.py
```

Removed files relative to `chunk_middle_result4`:

```text
doc/env_change_report.md
doc/rag_strategy_and_workflow.md
doc/retrieval_system_report.md
```

## 6000 Changes

### .env

File:

```text
.env
```

Changed value:

```dotenv
RAG_SCAN_CANDIDATE_LIMIT=4096
```

to:

```dotenv
RAG_SCAN_CANDIDATE_LIMIT=6000
```

Meaning:

- This controls scan fallback candidate volume.
- It is used by retrieval config/options.
- It does not control dynamic document profile construction.

Other `.env` differences between the branches include populated API/token values. They are intentionally not reproduced here.

### app/tools/knowledge_tool.py

File:

```text
app/tools/knowledge_tool.py
```

Changed value:

```python
PROFILE_SCAN_LIMIT = 4096
```

to:

```python
PROFILE_SCAN_LIMIT = 6000
```

Usage:

```python
results = vector_search_service.query_documents(limit=PROFILE_SCAN_LIMIT)
```

Meaning:

- This controls how many metadata rows are read while building dynamic document profiles.
- These profiles feed `load_active_doc_ids()`.
- If this limit is too low, a manual can exist in Milvus but be absent from active document filtering.
- This was the relevant limit for the `manual_f769aafb` active-doc issue.

Local verification after this change showed:

```text
query_documents_6000 4988 has_boat True
active_doc_count 40
has_boat True
```

Important distinction:

```text
RAG_SCAN_CANDIDATE_LIMIT=6000
PROFILE_SCAN_LIMIT = 6000
```

- `RAG_SCAN_CANDIDATE_LIMIT` affects scan fallback.
- `PROFILE_SCAN_LIMIT` affects active document profile construction.

## Retrieval Tier Policy

### app/retrieval/tier_policy.py

New file.

Purpose:

- Centralizes retrieval-tier constants and helper functions.
- Separates normal recall tiers from evidence-only tiers.

Defined tiers:

```python
PRIMARY_TIER = "primary"
SUPPORT_TIER = "support"
BIG_SUPPORT_TIER = "big_support"
AUXILIARY_TIER = "auxiliary"
```

Defined policy groups:

```python
RECALL_INCLUDED_TIERS = ("primary", "support")
EVIDENCE_PARENT_TIERS = ("support", "big_support")
DESCENDANT_QUERY_TIERS = ("primary", "support", "big_support")
```

Behavior:

- `primary` and `support` can participate in normal recall.
- `big_support` is excluded from normal recall.
- `big_support` can be used as evidence parent context.
- `auxiliary` is excluded from normal recall and descendant queries.

## Normal Recall Changes

### app/retrieval/recall.py

Main intent:

- Ensure `big_support` does not enter normal vector/BM25/scan recall.
- Keep `primary/support` recall behavior.
- Add diagnostics showing effective route tiers and recall filtering.

Key changes:

- Imports tier policy helpers from `app/retrieval/tier_policy.py`.
- Adds `effective_recall_tiers(route_tiers)`.
- Adds `filter_recallable_results(results)`.
- Applies `effective_recall_tiers()` before vector recall, BM25 recall, and scan recall.
- Applies `filter_recallable_results()` after each recall channel and again during merge.
- Adds trace fields:

```text
effective_recall_routes
filtered_pre_merge
recall_filter
```

Effect:

- Even if route tiers include `big_support`, normal recall only allows recallable tiers.
- Test stubs or old indexes returning `big_support` are filtered again before merge.
- Trace can show how many results were dropped by tier filtering.

## BM25 Provider Changes

### app/retrieval/bm25_provider.py

Main intent:

- Prevent BM25 from returning `big_support` chunks during normal recall.

Key changes:

- Imports `BIG_SUPPORT_TIER`.
- `_matches_filters()` now checks the result tier before other metadata filters.
- If a result has `retrieval_tier == "big_support"`, BM25 excludes it even when requested tiers contain `big_support`.

Effect:

- `big_support` cannot leak into BM25 normal recall.
- Long index text alone is not treated as `big_support`; the check is tier-based.

## Evidence / big_support Expansion

### app/retrieval/evidence.py

Main intent:

- Keep ordinary recall hits in `hits`.
- Use `support_hits` as the unified supplemental evidence list.
- When a hit's parent is `big_support`, expand that parent by selecting the most relevant descendant primary chunks.

New constants:

```python
BIG_SUPPORT_TOP_N = 3
BIG_SUPPORT_MAX_DEPTH = 4
BIG_SUPPORT_MAX_CANDIDATES = 100
```

Parent fetching changes:

- Parent evidence fetch now allows:

```text
support
big_support
```

- Ordinary `support` parents are added directly.
- `big_support` parents are added as context and then expanded.

big_support expansion flow:

1. Add the `big_support` parent itself to `support_hits`.
2. Collect descendants by walking direct children.
3. Keep primary descendants as candidate evidence.
4. Traverse through `support` / `big_support` child nodes for up to 4 levels.
5. Stop descendant collection after 100 primary candidates.
6. Rerank descendants against the query.
7. Add top 3 primary descendants into `support_hits`.
8. For those selected descendants, fetch only their direct ordinary `support` parents.
9. Do not recursively expand again when fetching selected descendant parents.

Important behavior:

- A primary chunk placed in `support_hits` keeps `retrieval_tier="primary"`.
- `support_hits` is a result-position list, not a data-tier rewrite.
- Deduplication is by `chunk_id`.
- If a chunk already appears in `hits`, it should not be duplicated in `support_hits`.

Trace additions:

```text
big_support_parent_ids
big_support_descendant_count
big_support_expanded_hits
big_support_rerank
big_support_selected_parent_ids
big_support_selected_parent_hits
big_support_expansion_skipped
big_support_descendant_error
big_support_selected_parent_error
```

## Vector Metadata Query Changes

### app/services/vector_search_service.py

Main intent:

- Allow metadata queries to filter by `parent_chunk_id`.

Key changes:

- Adds `parent_chunk_ids` support to:

```text
search_similar_documents()
query_documents()
query_all_documents()
```

Effect:

- Evidence expansion can query direct children of a parent chunk.
- big_support descendant collection can walk the hierarchy through `parent_chunk_id`.

### app/services/vector_store_manager.py

Main intent:

- Support parent-child metadata filters.
- Protect embedding and metadata indexing against provider/input-size limits.
- Produce a report for chunks whose raw embedding input exceeds provider limits.

Key additions:

```python
MAX_EMBEDDING_INPUT_CHARS = 8192
MAX_METADATA_BYTES = 60000
EMBEDDING_LIMIT_REPORT_PATH = Path("logs/embedding_input_limit_report.jsonl")
```

Embedding input handling:

- Keeps raw embedding text separate from truncated embedding text.
- Attempts batch embedding with raw text first.
- If provider rejects input length, retries individual documents.
- For rejected documents, truncates embedding input to `8192` characters.
- Records a JSONL report entry for each fallback truncation.

Metadata handling:

- Adds `_prepare_metadata()`.
- Ensures `index_text` stored in metadata matches the actual embedding input used.
- Compacts oversized metadata fields to keep Milvus metadata JSON below configured size.
- Preserves important fields such as:

```text
doc_id
doc_name
chunk_id
retrieval_tier
chunk_type
title
section_path
parent_chunk_id
source_lines
```

Parent metadata filter:

- `build_metadata_filter_expr()` now supports:

```text
parent_chunk_ids
```

This is required for direct-child queries during big_support expansion.

## Indexing Script Changes

### scripts/index_manual_chunks.py

Main intent:

- Expose embedding-limit-report controls through CLI.

New CLI options:

```text
--embedding-limit-report
--append-embedding-limit-report
```

Behavior:

- Default report path:

```text
logs/embedding_input_limit_report.jsonl
```

- Default mode clears the report at startup.
- Append mode keeps previous entries.
- Script output includes `embedding_limit_report` summary.

### scripts/apply_big_support_policy.py

Branch diff shows a small cleanup around the script's imports/config area. The core big_support threshold remains:

```python
BIG_SUPPORT_THRESHOLD = 8000
```

## Diagnostic Script

### scripts/diagnose_doc_recall.py

New file.

Purpose:

- Diagnose whether a specific document appears in recent retrieval traces.
- Compare service trace results against direct BM25 rebuilt from the current Milvus database.
- Default target:

```text
manual_f769aafb
```

Main outputs:

- Current DB row count for the target document.
- Tier/type counts for the target document.
- Recent trace hits in vector/BM25/scan/merged/rerank/evidence.
- Direct BM25 ranking from current DB.

Default report path:

```text
doc/manual_f769aafb_recall_diagnosis.md
```

## Test Changes

### tests/retrieval/test_tier_policy.py

New file.

Covers:

- Normal recall tiers are `primary/support`.
- `big_support` and `auxiliary` are not recallable.
- Evidence parent tiers include `support/big_support`.
- Descendant query tiers include hierarchy tiers but exclude `auxiliary`.
- `is_big_support_tier()` only matches `big_support`.

### tests/retrieval/test_bm25_provider.py

Added coverage:

- BM25 excludes `big_support` even if requested tiers include it.
- Long `index_text` does not automatically imply `big_support`.
- Auxiliary is not made recallable unless filter behavior explicitly allows it.

### tests/retrieval/test_recall.py

Added coverage:

- `effective_recall_tiers()` defaults to recallable tiers and filters route tiers.
- `filter_recallable_results()` keeps only `primary/support`.
- Merge stage filters `big_support` returned by stubs/old indexes.
- Vector recall filters `big_support` from calls/results.
- BM25 recall filters `big_support` from calls/results.
- Scan recall filters `big_support` from calls/results.

### tests/retrieval/test_evidence.py

Added coverage:

- Parent fetch allows `support/big_support`.
- big_support expands into top primary descendants and direct ordinary support parents.
- Descendant collection caps at 100 primary chunks.
- Descendant collection stops after 4 layers.
- big_support parent without descendants records zero count.
- `expand_big_support=False` keeps parent and skips descendant queries.

### tests/retrieval/test_metadata_indexing_filter.py

Added coverage:

- Metadata filters support single `parent_chunk_id`.
- Metadata filters support multiple `parent_chunk_id` values.
- Parent filters combine correctly with doc/language/tier filters.
- Embedding input is capped to provider limit.
- Embedding limit report records hierarchy fields.
- Provider-accepted long text does not create a false report.
- Report append mode works.
- Large metadata fields are compacted before insert.
- Vector search service forwards `parent_chunk_ids`.

## Data / Evaluation Files

Added:

```text
data/test_boat_airfryer_241_250.csv
data/submission_boat_airfryer_241_250.csv
```

Purpose:

- Local 10-question test set and generated submission for Air Fryer / Boat questions.

## Documentation File Changes

Added:

```text
doc/code_change_report_2026-06-02.md
```

Removed relative to `chunk_middle_result4`:

```text
doc/env_change_report.md
doc/rag_strategy_and_workflow.md
doc/retrieval_system_report.md
```

Note:

- The removed files are branch-diff removals relative to `chunk_middle_result4`.
- The current working tree may still contain local untracked documentation files depending on later manual edits.

## Current Practical Interpretation

The branch `english_chunkrecall_add` is not only a `6000` configuration change. It adds a larger retrieval/evidence change set:

- A central retrieval tier policy.
- Normal recall filtering that excludes `big_support`.
- Evidence-stage big_support expansion with descendant collection and reranking.
- Parent-child metadata query support.
- Embedding input fallback truncation and reporting.
- Metadata compaction before Milvus insert.
- Diagnostic tooling for document recall failures.
- Unit tests for tier policy, recall filtering, evidence expansion, metadata filters, and embedding limit handling.

The two `6000` changes are still important, but they are only a small part of the full branch delta.

