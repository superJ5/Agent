# Code Change Report - 2026-06-02

## Scope

This document records the current working-tree changes related to the English manual retrieval / big_support work, with special attention to the two `6000` limit changes.

## Explicit 6000 Changes

### 1. Environment scan fallback limit

File:

```text
.env
```

Current setting:

```dotenv
RAG_SCAN_CANDIDATE_LIMIT=6000
```

Purpose:

- Controls the maximum number of candidates used by the normal scan fallback path.
- This value is read through the RAG retrieval options/config path.
- It is not the document-profile scan limit.

Related code defaults still exist:

```text
app/config.py
app/retrieval/schemas.py
app/retrieval/recall.py
```

Those defaults are fallback values; the `.env` value is intended to override runtime config when loaded.

### 2. Document profile scan limit

File:

```text
app/tools/knowledge_tool.py
```

Current setting:

```python
PROFILE_SCAN_LIMIT = 6000
```

Usage:

```python
results = vector_search_service.query_documents(limit=PROFILE_SCAN_LIMIT)
```

Purpose:

- Controls how many indexed metadata rows are scanned when building dynamic document profiles.
- These profiles feed `load_active_doc_ids()`.
- If this limit is too small, later indexed manuals can be missing from active document filtering.
- The observed issue was that `manual_f769aafb` could be present in Milvus but absent from `active_doc_ids` when the profile scan only read the first `4096` rows.

Local verification after changing this value to `6000` showed:

```text
query_documents_6000 4988 has_boat True
active_doc_count 40
has_boat True
```

## Retrieval / big_support Code Areas

The current working tree also includes big_support-related changes in these areas:

```text
app/retrieval/tier_policy.py
app/retrieval/bm25_provider.py
app/retrieval/recall.py
app/retrieval/evidence.py
app/services/vector_search_service.py
app/services/vector_store_manager.py
scripts/apply_big_support_policy.py
scripts/index_manual_chunks.py
```

High-level intent:

- Keep `big_support` out of normal recall.
- Allow `big_support` to participate only during evidence expansion.
- Keep `hits` as original recall hits.
- Use `support_hits` as the unified supplemental context list.
- Preserve each chunk's own `retrieval_tier` even when it is placed in `support_hits`.

## Test / Diagnostic Files

Current related test and diagnostic files include:

```text
tests/retrieval/test_bm25_provider.py
tests/retrieval/test_evidence.py
tests/retrieval/test_metadata_indexing_filter.py
tests/retrieval/test_recall.py
tests/retrieval/test_tier_policy.py
scripts/diagnose_doc_recall.py
```

Generated local test data / reports currently present:

```text
data/test_boat_airfryer_241_250.csv
data/submission_boat_airfryer_241_250.csv
doc/manual_f769aafb_recall_diagnosis.md
doc/retrieval_trace_intermediate_report.md
```

## Important Distinction

These two limits are different and should not be treated as interchangeable:

```text
RAG_SCAN_CANDIDATE_LIMIT=6000
PROFILE_SCAN_LIMIT = 6000
```

- `RAG_SCAN_CANDIDATE_LIMIT` affects scan fallback candidate volume.
- `PROFILE_SCAN_LIMIT` affects dynamic document-profile construction and active-doc filtering.

For the `manual_f769aafb` missing-from-recall problem, the directly relevant change is:

```python
PROFILE_SCAN_LIMIT = 6000
```

