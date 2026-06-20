"""Long-term memory backed by Milvus.

This layer stores only stable, useful, confirmed, non-sensitive memories that
can be reused across sessions. It deliberately rejects weak inferences instead
of keeping pending candidates.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import uuid
from textwrap import dedent
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from loguru import logger
from pydantic import BaseModel, Field
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, MilvusException, utility

from app.config import config
from app.core.request_context import is_memory_enabled
from app.core.milvus_client import milvus_manager
from app.services.vector_embedding_service import vector_embedding_service


VECTOR_DIM = 1024
ID_MAX_LENGTH = 100
CONTENT_MAX_LENGTH = 2000
TYPE_MAX_LENGTH = 40
STATUS_ACTIVE = "active"
STATUS_INACTIVE = "inactive"
VALID_STATUSES = {STATUS_ACTIVE, STATUS_INACTIVE}
VALID_ACTIONS = {"create", "update", "deactivate", "delete", "noop"}
VALID_TYPES = {"profile", "preference", "project", "constraint", "stable_fact"}
MAX_CONTEXT_MEMORIES = 5
MAX_EVIDENCE_CHARS = 400


class LongTermMemory(BaseModel):
    """Stored long-term memory item."""

    memory_id: str = ""
    user_id: str = "default"
    type: str = "stable_fact"
    content: str = ""
    evidence: str = ""
    confidence: float = 0.0
    status: str = STATUS_ACTIVE
    source_session_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    expires_at: str | None = None


class LongTermMemoryAction(BaseModel):
    """One update action proposed by the long-term memory updater."""

    action: str = "noop"
    target_memory_id: str | None = None
    memory: LongTermMemory = Field(default_factory=LongTermMemory)
    reason: str = ""


class LongTermMemoryUpdate(BaseModel):
    """Structured output from the long-term memory updater."""

    should_update: bool = False
    reason: str = ""
    actions: list[LongTermMemoryAction] = Field(default_factory=list)


class LongTermMemoryService:
    """Milvus-backed long-term memory service."""

    def __init__(self) -> None:
        self.collection_name = config.long_term_memory_collection_name
        self._collection: Collection | None = None

    def build_context_block(
        self,
        *,
        question: str,
        session_id: str,
        session_state_context: str = "",
        short_term_context: str = "",
        user_id: str = "default",
    ) -> tuple[str, list[dict[str, Any]]]:
        """Retrieve and render relevant long-term memories for one answer."""
        if not is_memory_enabled():
            return "", []

        memories = self.retrieve_relevant_memories(
            question=question,
            session_state_context=session_state_context,
            short_term_context=short_term_context,
            user_id=user_id,
        )
        if not memories:
            return "", []

        lines = [
            "以下是与当前问题相关的长期记忆，只作为稳定背景参考；",
            "如与当前用户问题冲突，以当前用户问题为准。",
            "",
            "【长期记忆】",
        ]
        for memory in memories[:MAX_CONTEXT_MEMORIES]:
            lines.append(f"- {memory['content']}")
        return "\n".join(lines).strip(), memories

    def retrieve_relevant_memories(
        self,
        *,
        question: str,
        session_state_context: str = "",
        short_term_context: str = "",
        user_id: str = "default",
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search Milvus, filter inactive/expired/low-confidence memories."""
        if not is_memory_enabled():
            return []

        query = self._build_retrieval_query(
            question=question,
            session_state_context=session_state_context,
            short_term_context=short_term_context,
        )
        if not query:
            return []

        try:
            collection = self.ensure_collection()
            query_vector = vector_embedding_service.embed_query(query)
            limit = max(top_k or config.long_term_memory_top_k, 1)
            raw_limit = max(limit * 4, 12)
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=raw_limit,
                output_fields=["memory_id", "content", "type", "metadata"],
            )
        except Exception as exc:
            logger.warning("长期记忆检索失败，跳过注入: {}", exc)
            return []

        now = datetime.now(timezone.utc)
        hits: list[dict[str, Any]] = []
        for hit_group in results:
            for hit in hit_group:
                entity = hit.entity
                metadata = dict(entity.get("metadata") or {})
                memory_id = str(entity.get("memory_id") or metadata.get("memory_id") or "")
                memory = {
                    **metadata,
                    "memory_id": memory_id,
                    "content": str(entity.get("content") or metadata.get("content") or ""),
                    "type": str(entity.get("type") or metadata.get("type") or "stable_fact"),
                    "score": float(hit.distance),
                }
                if not self._is_same_user(memory, user_id):
                    continue
                if self._is_expired(memory, now):
                    self._deactivate_memory(memory_id, reason="expires_at 已过期")
                    continue
                if not self._passes_runtime_filter(memory):
                    continue
                hits.append(memory)

        hits.sort(key=lambda item: float(item.get("score") or 0) * float(item.get("confidence") or 0), reverse=True)
        return hits[:limit]

    async def update_after_turn(
        self,
        *,
        session_id: str,
        user_message: str,
        assistant_message: str,
        session_state_context: str = "",
        short_term_memory: str = "",
        prior_dialogue: list[dict[str, Any]] | None = None,
        retrieved_memories: list[dict[str, Any]] | None = None,
        user_id: str = "default",
    ) -> None:
        """Update long-term memory after one completed turn."""
        if not is_memory_enabled():
            logger.debug("跳过长期记忆写入: MEMORY_ENABLED=false")
            return
        if not config.memory_write_enabled:
            logger.debug("跳过长期记忆写入: MEMORY_WRITE_ENABLED=false")
            return

        user_text = str(user_message or "").strip()
        assistant_text = str(assistant_message or "").strip()
        if not user_text and not assistant_text:
            return

        try:
            update = await self._generate_update(
                session_id=session_id,
                user_id=user_id,
                user_message=user_text,
                assistant_message=assistant_text,
                session_state_context=session_state_context,
                short_term_memory=short_term_memory,
                prior_dialogue=self._format_dialogue(prior_dialogue or []),
                retrieved_memories=retrieved_memories or [],
            )
        except Exception as exc:
            logger.warning("长期记忆更新失败: session_id={}, error={}", session_id, exc)
            return

        if not update.should_update:
            logger.debug("长期记忆无需更新: session_id={}, reason={}", session_id, update.reason)
            return

        for action in update.actions:
            self._apply_action(action, session_id=session_id, user_id=user_id)

    def list_memories(
        self,
        *,
        user_id: str = "default",
        include_inactive: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List long-term memories for debugging."""
        if not is_memory_enabled():
            return []

        try:
            collection = self.ensure_collection()
            rows = collection.query(
                expr=f'user_id == {json.dumps(user_id, ensure_ascii=False)}',
                output_fields=["memory_id", "content", "type", "metadata"],
                limit=max(limit, 1),
            )
        except Exception as exc:
            logger.warning("读取长期记忆列表失败: {}", exc)
            return []

        memories: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc)
        for row in rows:
            metadata = dict(row.get("metadata") or {})
            memory = {
                **metadata,
                "memory_id": str(row.get("memory_id") or metadata.get("memory_id") or ""),
                "content": str(row.get("content") or metadata.get("content") or ""),
                "type": str(row.get("type") or metadata.get("type") or "stable_fact"),
            }
            if self._is_expired(memory, now):
                self._deactivate_memory(memory["memory_id"], reason="expires_at 已过期")
                memory["status"] = STATUS_INACTIVE
            if not include_inactive and memory.get("status") != STATUS_ACTIVE:
                continue
            memories.append(memory)
        return memories

    def ensure_collection(self) -> Collection:
        """Connect to Milvus and create/load the long-term memory collection."""
        if self._collection is not None:
            return self._collection

        _ = milvus_manager.connect()
        if not utility.has_collection(self.collection_name):
            logger.info("长期记忆 collection '{}' 不存在，正在创建...", self.collection_name)
            self._create_collection()
        else:
            self._collection = Collection(self.collection_name)
            self._ensure_vector_dim()

        self._load_collection()
        return self._collection

    async def _generate_update(
        self,
        *,
        session_id: str,
        user_id: str,
        user_message: str,
        assistant_message: str,
        session_state_context: str,
        short_term_memory: str,
        prior_dialogue: str,
        retrieved_memories: list[dict[str, Any]],
    ) -> LongTermMemoryUpdate:
        llm = self._build_llm()
        structured_llm = llm.with_structured_output(LongTermMemoryUpdate)
        messages = [
            SystemMessage(content=self._update_system_prompt()),
            HumanMessage(
                content=dedent(f"""
                    user_id：
                    {user_id}

                    source_session_id：
                    {session_id}

                    当前 Session State：
                    {session_state_context or "（空）"}

                    当前短期语义记忆：
                    {short_term_memory or "（空）"}

                    最近 1-3 轮原始对话：
                    {prior_dialogue or "（空）"}

                    本轮检索到的长期记忆：
                    {self._json(retrieved_memories) if retrieved_memories else "（空）"}

                    本轮用户问题：
                    {user_message}

                    本轮助手回答：
                    {assistant_message}

                    请输出 LongTermMemoryUpdate。
                """).strip()
            ),
        ]
        result = await structured_llm.ainvoke(messages)
        if isinstance(result, LongTermMemoryUpdate):
            return result
        return LongTermMemoryUpdate.model_validate(result)

    @staticmethod
    def _update_system_prompt() -> str:
        return dedent("""
            你是长期记忆更新器。你的任务是判断本轮对话是否产生了值得跨 session 保留的长期记忆。

            长期记忆写入前必须同时满足四个标准：
            1. 长期稳定：不是临时任务、一次性问题、短期状态或偶然行为。
            2. 以后有用：未来回答用户问题时大概率能帮助个性化或延续长期背景。
            3. 被确认过：必须来自用户明确表达、明确纠正或可验证事实；不要写模型猜测。
            4. 非隐私敏感：不要写身份证、手机号、住址、密钥、私人财务、健康隐私等敏感信息。

            重要规则：
            - 不确定就 noop，不要写 pending。
            - status 只能是 active 或 inactive；新建记忆通常是 active。
            - expires_at 默认 null；只有用户明确给出“这周/这个月/到某日期之前”等时间范围，才设置 expires_at。
            - 用户明确要求删除或忘记某条记忆时，输出 delete。
            - 新信息替代旧记忆时，对旧记忆输出 deactivate，对新信息输出 create 或 update。
            - update 必须提供 target_memory_id。
            - deactivate/delete 必须提供 target_memory_id。
            - type 只能是 profile、preference、project、constraint、stable_fact。
            - confidence 范围是 0 到 1；用户明确表达通常应 >= 0.9。
            - evidence 必须说明这条记忆的证据来源。
            - should_update=false 时 actions 为空。
        """).strip()

    def _apply_action(
        self,
        action: LongTermMemoryAction,
        *,
        session_id: str,
        user_id: str,
    ) -> None:
        action_name = str(action.action or "noop").strip().lower()
        if action_name not in VALID_ACTIONS or action_name == "noop":
            return

        target_id = str(action.target_memory_id or "").strip()
        if action_name == "delete":
            if target_id:
                self._delete_memory(target_id)
            return

        if action_name == "deactivate":
            if target_id:
                self._deactivate_memory(target_id, reason=action.reason)
            return

        if action_name == "update":
            if not target_id:
                return
            memory = self._clean_memory(
                action.memory,
                session_id=session_id,
                user_id=user_id,
                existing_memory_id=target_id,
            )
            if memory is None:
                return
            old_memory = self._load_memory_by_id(target_id)
            if old_memory:
                memory.created_at = str(old_memory.get("created_at") or memory.created_at)
            self._delete_memory(target_id)
            self._insert_memory(memory)
            return

        if action_name == "create":
            memory = self._clean_memory(
                action.memory,
                session_id=session_id,
                user_id=user_id,
            )
            if memory is None:
                return
            if self._looks_duplicate(memory):
                return
            self._insert_memory(memory)

    def _clean_memory(
        self,
        memory: LongTermMemory,
        *,
        session_id: str,
        user_id: str,
        existing_memory_id: str | None = None,
    ) -> LongTermMemory | None:
        now = datetime.now(timezone.utc).isoformat()
        content = self._clean_text(memory.content, max_chars=CONTENT_MAX_LENGTH)
        evidence = self._clean_text(memory.evidence, max_chars=MAX_EVIDENCE_CHARS)
        memory_type = str(memory.type or "stable_fact").strip()
        if memory_type not in VALID_TYPES:
            memory_type = "stable_fact"

        status = str(memory.status or STATUS_ACTIVE).strip().lower()
        if status not in VALID_STATUSES:
            status = STATUS_ACTIVE

        confidence = max(0.0, min(float(memory.confidence or 0.0), 1.0))
        expires_at = self._clean_expires_at(memory.expires_at)
        if not content or not evidence:
            return None
        if status != STATUS_ACTIVE:
            return None
        if confidence < config.long_term_memory_min_confidence:
            return None
        if self._looks_sensitive(content) or self._looks_sensitive(evidence):
            return None

        return LongTermMemory(
            memory_id=existing_memory_id or self._new_memory_id(),
            user_id=user_id,
            type=memory_type,
            content=content,
            evidence=evidence,
            confidence=confidence,
            status=status,
            source_session_id=session_id,
            created_at=self._clean_text(memory.created_at, max_chars=80) or now,
            updated_at=now,
            expires_at=expires_at,
        )

    def _insert_memory(self, memory: LongTermMemory) -> None:
        try:
            collection = self.ensure_collection()
            embedding = vector_embedding_service.embed_query(memory.content)
            metadata = memory.model_dump()
            collection.insert(
                [
                    {
                        "memory_id": memory.memory_id,
                        "user_id": memory.user_id,
                        "type": memory.type,
                        "vector": embedding,
                        "content": self._truncate_varchar_bytes(memory.content),
                        "metadata": self._sanitize_metadata(metadata),
                    }
                ]
            )
            collection.flush()
            logger.info("写入长期记忆: memory_id={}, type={}", memory.memory_id, memory.type)
        except Exception as exc:
            logger.warning("写入长期记忆失败: memory_id={}, error={}", memory.memory_id, exc)

    def _load_memory_by_id(self, memory_id: str) -> dict[str, Any] | None:
        if not memory_id:
            return None
        try:
            collection = self.ensure_collection()
            rows = collection.query(
                expr=f'memory_id == {json.dumps(memory_id, ensure_ascii=False)}',
                output_fields=["memory_id", "content", "type", "metadata"],
                limit=1,
            )
        except Exception as exc:
            logger.warning("读取长期记忆失败: memory_id={}, error={}", memory_id, exc)
            return None
        if not rows:
            return None
        row = rows[0]
        metadata = dict(row.get("metadata") or {})
        return {
            **metadata,
            "memory_id": str(row.get("memory_id") or metadata.get("memory_id") or ""),
            "content": str(row.get("content") or metadata.get("content") or ""),
            "type": str(row.get("type") or metadata.get("type") or "stable_fact"),
        }

    def _deactivate_memory(self, memory_id: str, *, reason: str = "") -> None:
        old = self._load_memory_by_id(memory_id)
        if not old or old.get("status") == STATUS_INACTIVE:
            return
        try:
            memory = LongTermMemory.model_validate(old)
        except Exception:
            memory = LongTermMemory(
                memory_id=memory_id,
                user_id=str(old.get("user_id") or "default"),
                type=str(old.get("type") or "stable_fact"),
                content=str(old.get("content") or ""),
                evidence=str(old.get("evidence") or reason or "记忆被废弃"),
                confidence=float(old.get("confidence") or 0.0),
                source_session_id=str(old.get("source_session_id") or ""),
                created_at=str(old.get("created_at") or ""),
                expires_at=old.get("expires_at"),
            )
        memory.status = STATUS_INACTIVE
        memory.updated_at = datetime.now(timezone.utc).isoformat()
        if reason:
            memory.evidence = self._clean_text(f"{memory.evidence}；废弃原因：{reason}", max_chars=MAX_EVIDENCE_CHARS)
        self._delete_memory(memory_id)
        self._insert_memory_allow_inactive(memory)

    def _insert_memory_allow_inactive(self, memory: LongTermMemory) -> None:
        try:
            collection = self.ensure_collection()
            embedding = vector_embedding_service.embed_query(memory.content)
            collection.insert(
                [
                    {
                        "memory_id": memory.memory_id,
                        "user_id": memory.user_id,
                        "type": memory.type,
                        "vector": embedding,
                        "content": self._truncate_varchar_bytes(memory.content),
                        "metadata": self._sanitize_metadata(memory.model_dump()),
                    }
                ]
            )
            collection.flush()
        except Exception as exc:
            logger.warning("写入 inactive 长期记忆失败: memory_id={}, error={}", memory.memory_id, exc)

    def _delete_memory(self, memory_id: str) -> None:
        try:
            collection = self.ensure_collection()
            _ = collection.delete(expr=f'memory_id == {json.dumps(memory_id, ensure_ascii=False)}')
            collection.flush()
        except Exception as exc:
            logger.warning("删除长期记忆失败: memory_id={}, error={}", memory_id, exc)

    def _looks_duplicate(self, memory: LongTermMemory) -> bool:
        existing = self.retrieve_relevant_memories(
            question=memory.content,
            user_id=memory.user_id,
            top_k=3,
        )
        normalized = memory.content.lower()
        return any(str(item.get("content") or "").lower() == normalized for item in existing)

    def _create_collection(self) -> None:
        fields = [
            FieldSchema("memory_id", DataType.VARCHAR, max_length=ID_MAX_LENGTH, is_primary=True),
            FieldSchema("user_id", DataType.VARCHAR, max_length=ID_MAX_LENGTH),
            FieldSchema("type", DataType.VARCHAR, max_length=TYPE_MAX_LENGTH),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
            FieldSchema("content", DataType.VARCHAR, max_length=CONTENT_MAX_LENGTH),
            FieldSchema("metadata", DataType.JSON),
        ]
        schema = CollectionSchema(
            fields=fields,
            description="Long-term user memory collection",
            enable_dynamic_field=False,
        )
        self._collection = Collection(
            name=self.collection_name,
            schema=schema,
            num_shards=2,
        )
        self._collection.create_index(
            field_name="vector",
            index_params={
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 256},
            },
        )

    def _ensure_vector_dim(self) -> None:
        collection = self._collection
        if collection is None:
            return
        for field in collection.schema.fields:
            if field.name == "vector" and getattr(field, "params", {}).get("dim") != VECTOR_DIM:
                logger.warning("long-term memory collection vector dim mismatch; rebuilding collection")
                utility.drop_collection(self.collection_name)
                self._create_collection()
                return

    def _load_collection(self) -> None:
        if self._collection is None:
            self._collection = Collection(self.collection_name)
        try:
            self._collection.load()
        except MilvusException as exc:
            if "loaded" not in str(exc).lower():
                raise

    @staticmethod
    def _build_retrieval_query(
        *,
        question: str,
        session_state_context: str,
        short_term_context: str,
    ) -> str:
        parts = [
            "当前问题：" + str(question or "").strip(),
            "当前会话状态：" + str(session_state_context or "").strip(),
            "短期上下文：" + str(short_term_context or "").strip(),
        ]
        return "\n".join(part for part in parts if part.split("：", 1)[-1].strip())[:3000]

    @staticmethod
    def _format_dialogue(records: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for record in records:
            role = str(record.get("role") or "").strip()
            content = " ".join(str(record.get("content") or "").split()).strip()
            if not content or role not in {"user", "assistant"}:
                continue
            label = "用户" if role == "user" else "助手"
            lines.append(f"{label}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _is_same_user(memory: dict[str, Any], user_id: str) -> bool:
        return str(memory.get("user_id") or "default") == user_id

    @staticmethod
    def _is_expired(memory: dict[str, Any], now: datetime) -> bool:
        expires_at = str(memory.get("expires_at") or "").strip()
        if not expires_at:
            return False
        try:
            expires = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            return expires <= now
        except ValueError:
            return False

    @staticmethod
    def _passes_runtime_filter(memory: dict[str, Any]) -> bool:
        if str(memory.get("status") or "") != STATUS_ACTIVE:
            return False
        if float(memory.get("confidence") or 0.0) < config.long_term_memory_min_confidence:
            return False
        if float(memory.get("score") or 0.0) < config.long_term_memory_min_relevance:
            return False
        return bool(str(memory.get("content") or "").strip())

    @staticmethod
    def _clean_expires_at(value: str | None) -> str | None:
        text = str(value or "").strip()
        if not text or text.lower() == "null":
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.isoformat()
        except ValueError:
            return None

    @staticmethod
    def _clean_text(value: str, *, max_chars: int) -> str:
        text = " ".join(str(value or "").split()).strip()
        if len(text) > max_chars:
            text = text[:max_chars].rstrip()
        return text

    @staticmethod
    def _looks_sensitive(text: str) -> bool:
        lowered = text.lower()
        sensitive_words = [
            "身份证",
            "手机号",
            "手机号码",
            "住址",
            "家庭住址",
            "银行卡",
            "密码",
            "密钥",
            "api key",
            "apikey",
            "token",
            "病历",
            "诊断",
        ]
        return any(word in lowered for word in sensitive_words)

    @staticmethod
    def _truncate_varchar_bytes(value: str, max_bytes: int = CONTENT_MAX_LENGTH) -> str:
        encoded = value.encode("utf-8")
        if len(encoded) <= max_bytes:
            return value
        suffix = "..."
        budget = max_bytes - len(suffix.encode("utf-8"))
        truncated = encoded[:budget]
        while truncated:
            try:
                return truncated.decode("utf-8") + suffix
            except UnicodeDecodeError:
                truncated = truncated[:-1]
        return suffix

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        return json.loads(json.dumps(metadata, ensure_ascii=False, default=str))

    @staticmethod
    def _new_memory_id() -> str:
        return f"ltm_{uuid.uuid4().hex}"

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def _build_llm() -> ChatQwen:
        model_name = (
            config.long_term_memory_model
            or config.session_state_model
            or config.short_term_memory_model
            or config.rag_model
        )
        return ChatQwen(
            model=model_name,
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_api_base,
            temperature=0,
            timeout=60,
        )


long_term_memory_service = LongTermMemoryService()
