"""Application configuration."""

from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "SuperBizAgent"
    app_version: str = "1.0.0"
    app_debug: bool = Field(default=False, validation_alias="APP_DEBUG")
    host: str = "0.0.0.0"
    port: int = 9900

    # DashScope
    dashscope_api_key: str = ""
    dashscope_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_model: str = "qwen3.5-plus"
    dashscope_embedding_model: str = "text-embedding-v4"

    # Competition API auth
    api_bearer_token: str = ""

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000

    # RAG
    rag_top_k: int = 3
    rag_model: str = "qwen3.5-plus"
    rag_intent_strategy: str = "none"
    rag_enable_vector_recall: bool = True
    rag_enable_bm25_recall: bool = True
    rag_vector_weight: float = 0.6
    rag_bm25_weight: float = 0.4
    rag_reranker_provider: str = "dashscope"
    rag_reranker_model: str = "qwen3-rerank"
    rag_reranker_endpoint: str = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
    rag_reranker_timeout_ms: int = 3000
    rag_reranker_top_n: int = 32
    rag_enable_scan_fallback: bool = True
    rag_scan_candidate_limit: int = 4096

    # Memory
    memory_root: str = "data/memory"
    memory_recent_limit: int = 10
    memory_collection_name: str = "memory"
    memory_search_top_k: int = 5
    memory_daily_max_chars: int = 5000
    memory_summary_input_max_chars: int = 20000
    memory_summary_model: str = ""

    # Chunking
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # MCP services
    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://localhost:8003/mcp"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:8004/mcp"

    @property
    def debug(self) -> bool:
        """Compatible config.debug property backed by APP_DEBUG."""
        return self.app_debug

    @property
    def mcp_servers(self) -> dict[str, dict[str, Any]]:
        """Return MCP server configuration."""
        return {
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            },
        }


config = Settings()
