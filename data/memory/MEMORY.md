# MEMORY

## Project

- The project is a multimodal RAG assistant for product/manual Q&A.
- Main chat endpoint: `POST /chat`; legacy frontend endpoint: `POST /api/chat`.
- Main conversation model: `qwen3.5-plus`.
- Manual knowledge is indexed into Milvus collection `biz`.

## Memory System

- Raw session messages are stored under `data/memory/sessions/*.jsonl`.
- Long-term stable memory is stored in this file.
- Daily notes can be stored under `data/memory/daily/*.md`.
