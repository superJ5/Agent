"""RAG Agent 服务 - 基于 LangGraph 的智能代理

使用 langchain_qwq 的 ChatQwen 原生集成，
支持真正的流式输出和更好的模型适配。
"""

import re
from collections.abc import AsyncGenerator, Sequence
from typing import Annotated, Any, cast

from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_qwq import ChatQwen
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from loguru import logger
from typing_extensions import TypedDict

from app.agent.mcp_client import get_mcp_client_with_retry
from app.config import config
from app.models.response import sanitize_summary_metadata
from app.services.memory_service import memory_service
from app.services.multimodal_message_builder import build_user_message
from app.tools import get_current_time, memory_search, retrieve_knowledge

EMPTY_IMAGE_ALT_RE = re.compile(r"!\[\]\(([^)]+)\)")
PIC_ID_IN_PATH_RE = re.compile(r"([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)")

# 阿里千问大模型和langchain集成参考： https://docs.langchain.com/oss/python/integrations/chat/qwen
# 注意：需要配置环境变量 DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1 否则默认访问的是新加坡站点
# 同时也需要配置环境变量 DASHSCOPE_API_KEY=your_api_key


class AgentState(TypedDict):
    """Agent 状态"""
    messages: Annotated[Sequence[BaseMessage], add_messages]


def trim_messages_middleware(state: AgentState) -> dict[str, Any] | None:
    """
    修剪消息历史，只保留最近的几条消息以适应上下文窗口

    策略：
    - 保留第一条系统消息（System Message）
    - 保留最近的 6 条消息（3 轮对话）
    - 当消息少于等于 7 条时，不做修剪

    Args:
        state: Agent 状态

    Returns:
        包含修剪后消息的字典，如果无需修剪则返回 None
    """
    messages = state["messages"]

    # 如果消息数量较少，无需修剪
    if len(messages) <= 7:
        return None

    # 提取第一条系统消息
    first_msg = messages[0]

    # 保留最近的 6 条消息（确保包含完整的对话轮次）
    recent_messages = messages[-6:] if len(messages) % 2 == 0 else messages[-7:]

    # 构建新的消息列表
    new_messages = [first_msg] + list(recent_messages)

    logger.debug(f"修剪消息历史: {len(messages)} -> {len(new_messages)} 条")

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


class RagAgentService:
    """RAG Agent 服务 - 使用 LangGraph + ChatQwen 原生集成"""

    def __init__(self, streaming: bool = True):
        """初始化 RAG Agent 服务

        Args:
            streaming: 是否启用流式输出，默认为 True
        """
        self.model_name = config.rag_model
        self.streaming = streaming
        self.system_prompt = self._build_system_prompt()


        self.model = ChatQwen(
            model=self.model_name,
            api_key=cast(Any, config.dashscope_api_key),
            base_url=config.dashscope_api_base,
            temperature=0.7,
            streaming=streaming,
        )

        # 定义基础工具
        self.tools = [retrieve_knowledge, memory_search, get_current_time]

        # MCP 客户端（延迟初始化，使用全局管理）
        self.mcp_tools: list = []
        self._last_retrieval_metadata_by_session: dict[str, dict[str, Any] | None] = {}

        # 创建内存检查点（用于会话管理）
        self.checkpointer = MemorySaver()

        # Agent 初始化（会在异步方法中完成）
        self.agent = None
        self._agent_initialized = False

        logger.info(f"RAG Agent 服务初始化完成 (ChatQwen), model={self.model_name}, streaming={streaming}")

    async def _initialize_agent(self):
        """异步初始化 Agent（包括 MCP 工具）"""
        if self._agent_initialized:
            return

        # 尝试加载 MCP 工具；如果 MCP 服务不可用，则降级为仅使用本地工具。
        try:
            mcp_client = await get_mcp_client_with_retry()

            # 获取 MCP 工具
            mcp_tools = await mcp_client.get_tools()
            logger.info(f"成功加载 {len(mcp_tools)} 个 MCP 工具")

            # 将 MCP 工具添加到实例变量中
            self.mcp_tools = mcp_tools
        except Exception as exc:
            self.mcp_tools = []
            logger.warning(f"MCP 工具加载失败，降级为仅使用本地工具继续运行: {exc}")

        # 合并所有工具
        all_tools = self.tools + self.mcp_tools

        self.agent = create_agent(
            self.model,
            tools=all_tools,
            checkpointer=self.checkpointer,
        )

        self._agent_initialized = True


        if all_tools:
            tool_names = [tool.name if hasattr(tool, "name") else str(tool) for tool in all_tools]
            logger.info(f"可用工具列表: {', '.join(tool_names)}")

    def _build_system_prompt(self) -> str:
        """
        构建系统提示词

        注意：LangChain 框架会自动将工具信息传递给 LLM，
        因此系统提示词中无需列举具体的工具列表。

        Returns:
            str: 系统提示词
        """
        from textwrap import dedent

        prompt = dedent("""
            你是一个专业的AI助手，能够使用多种工具来帮助用户解决问题。

            工作原则:
            1. 理解用户需求，选择合适的工具来完成任务
            2. 当需要获取实时信息或专业知识时，主动使用相关工具
            3. 基于工具返回的结果提供准确、专业的回答
            4. 如果工具无法提供足够信息，请诚实地告知用户

            RAG 使用规则:
            1. 只要问题涉及说明书、手册、部件、操作步骤、保修、图片、OCR 或原文追溯，必须先调用 retrieve_knowledge 再回答。
            2. 优先依据 retrieve_knowledge 返回的证据回答，不要在没有检索证据时直接猜测手册内容。
            3. 如果 retrieve_knowledge 没有找到可靠内容，要明确说明“当前检索到的信息不足”，而不是编造答案。
            4. 如果检索结果里带有图片标识（PIC）或配图信息，回答时要优先结合这些证据。
            5. For manual-related English questions, call retrieve_knowledge first.
            6. When calling retrieve_knowledge for English manual questions, pass a concise English search query.
            7. Do not translate English questions into Chinese unless the user asks.

            记忆使用规则:
            1. 当用户询问之前说过什么、历史偏好、项目长期背景、已讨论方案时，可以调用 memory_search 查询历史记忆。
            2. MEMORY.md 会直接进入系统提示词，memory_search 只查询整理后的每日记忆 daily，不查询 MEMORY.md 或其他会话原始聊天记录。
            3. 当前会话上下文由 MemorySaver 或当前 session 的最近 N 条 JSONL 历史自动提供，不需要用 memory_search 查询。
            4. 历史记忆只作为上下文参考；如果记忆不足，要明确说明未找到足够历史信息。

            回答要求:
            - 保持友好、专业的语气
            - 回答简洁明了，重点突出
            - 基于事实，不编造信息
            - 如有不确定的地方，明确说明

            请根据用户的问题，灵活使用可用工具，提供高质量的帮助。
        """).strip()
        return (
            prompt
            + "\n\nImage output rules:\n"
            + "- If an answer cites an image, use markdown with the picture id as the alt text: "
            + "![Manual01_5](data/manuals/raw/.../Manual01_5.jpg).\n"
            + "- Never use an empty image placeholder like ![](path)."
            +
        )

    def _build_effective_system_prompt(self) -> str:
        """Build system prompt with optional long-term memory context."""
        long_term_memory = memory_service.load_long_term_memory()
        if not long_term_memory:
            return self.system_prompt
        return (
            f"{self.system_prompt}\n\n"
            "长期记忆上下文（来自 data/memory/MEMORY.md，仅作为稳定背景参考）:\n"
            f"{long_term_memory}"
        )

    def _build_persistent_history_messages(
        self,
        session_id: str,
        current_question: str,
    ) -> list[BaseMessage]:
        """Load recent JSONL history when LangGraph has no in-memory checkpoint."""
        if self._has_session_checkpoint(session_id):
            return []

        recent_records = memory_service.load_recent_messages(
            session_id,
            limit=config.memory_recent_limit,
            exclude_latest_user_content=current_question,
        )
        messages: list[BaseMessage] = []
        for record in recent_records:
            role = record.get("role")
            content = str(record.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        if messages:
            logger.info(
                "[会话 {}] 从持久记忆恢复最近 {} 条历史消息",
                session_id,
                len(messages),
            )
        return messages

    def _has_session_checkpoint(self, session_id: str) -> bool:
        """Return whether MemorySaver already has state for this session."""
        try:
            checkpoint_tuple = cast(Any, self.checkpointer).get(
                cast(Any, {"configurable": {"thread_id": session_id}})
            )
            return bool(checkpoint_tuple)
        except Exception:
            return False

    async def query(
        self,
        question: str,
        session_id: str,
        images: list[str] | None = None,
    ) -> str:
        """
        非流式处理用户问题（一次性返回完整答案）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）
            images: Base64 图片列表

        Returns:
            str: 完整答案
        """
        try:
            await self._initialize_agent()
            self._last_retrieval_metadata_by_session[session_id] = None
            self._clear_last_retrieval_metadata()

            image_count = len(images or [])
            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（非流式）: {question}, images={image_count}")

            # 构建消息列表（系统提示 + 用户问题）
            messages = [
                SystemMessage(content=self._build_effective_system_prompt()),
                *self._build_persistent_history_messages(session_id, question),
                build_user_message(question, images),
            ]

            # 构建 Agent 输入
            agent_input = {"messages": messages}

            # 配置 thread_id（用于会话持久化）
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            if self.agent is None:
                raise RuntimeError("Agent 未初始化")

            result = await self.agent.ainvoke(
                input=agent_input,
                config=config_dict,
            )
            self._last_retrieval_metadata_by_session[session_id] = (
                self._read_last_retrieval_metadata()
            )

            # 提取最终答案
            messages_result = result.get("messages", [])
            if messages_result:
                last_message = messages_result[-1]
                answer = last_message.content if hasattr(last_message, 'content') else str(last_message)

                # 记录工具调用
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    tool_names = [tc.get("name", "unknown") for tc in last_message.tool_calls]
                    logger.info(f"[会话 {session_id}] Agent 调用了工具: {tool_names}")

                logger.info(f"[会话 {session_id}] RAG Agent 查询完成（非流式）")
                return self._ensure_image_placeholders(str(answer))

            logger.warning(f"[会话 {session_id}] Agent 返回结果为空")
            return ""

        except Exception as e:
            logger.error(f"[会话 {session_id}] RAG Agent 查询失败（非流式）: {e}")
            raise

    def get_last_retrieval_metadata(self, session_id: str) -> dict[str, Any] | None:
        """Return the latest Summary-level retrieval diagnostics for one session."""
        metadata = self._last_retrieval_metadata_by_session.get(session_id)
        return dict(metadata) if metadata else None

    @staticmethod
    def _read_last_retrieval_metadata() -> dict[str, Any] | None:
        try:
            from app.tools.knowledge_tool import get_last_retrieval_metadata

            metadata = get_last_retrieval_metadata()
        except Exception as exc:
            logger.debug(f"读取检索诊断摘要失败，忽略 metadata 扩展: {exc}")
            return None

        return sanitize_summary_metadata(metadata)

    @staticmethod
    def _clear_last_retrieval_metadata() -> None:
        try:
            from app.tools.knowledge_tool import clear_last_retrieval_metadata

            clear_last_retrieval_metadata()
        except Exception as exc:
            logger.debug(f"清理检索诊断摘要失败，忽略 metadata 扩展: {exc}")

    @staticmethod
    def _ensure_image_placeholders(answer: str) -> str:
        """Fill empty markdown image alt text with the picture id from the path."""

        def replace_empty_alt(match: re.Match[str]) -> str:
            image_path = match.group(1).strip()
            pic_id = RagAgentService._pic_id_from_image_path(image_path)
            if not pic_id:
                return match.group(0)
            return f"![{pic_id}]({image_path})"

        return EMPTY_IMAGE_ALT_RE.sub(replace_empty_alt, answer)

    @staticmethod
    def _pic_id_from_image_path(image_path: str) -> str:
        file_name = image_path.replace("\\", "/").split("/")[-1]
        file_name = file_name.split("?", 1)[0].split("#", 1)[0]
        stem = file_name.rsplit(".", 1)[0]
        match = PIC_ID_IN_PATH_RE.search(stem)
        return match.group(1) if match else stem

    async def query_stream(
        self,
        question: str,
        session_id: str,
        images: list[str] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        流式处理用户问题（逐步返回答案片段）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）
            images: Base64 图片列表

        Yields:
            Dict[str, Any]: 包含流式数据的字典
                - type: "content" | "tool_call" | "complete" | "error"
                - data: 具体内容
        """
        try:
            await self._initialize_agent()
            self._clear_last_retrieval_metadata()

            image_count = len(images or [])
            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（流式）: {question}, images={image_count}")

            # 构建消息列表（系统提示 + 用户问题）
            messages = [
                SystemMessage(content=self._build_effective_system_prompt()),
                *self._build_persistent_history_messages(session_id, question),
                build_user_message(question, images),
            ]

            # 构建 Agent 输入
            agent_input = {"messages": messages}

            # 配置 thread_id（用于会话持久化）
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            if self.agent is None:
                raise RuntimeError("Agent 未初始化")

            async for token, metadata in self.agent.astream(
                input=agent_input,
                config=config_dict,
                stream_mode="messages",
            ):
                node_name = metadata.get('langgraph_node', 'unknown') if isinstance(metadata, dict) else 'unknown'
                message_type = type(token).__name__

                if message_type in ("AIMessage", "AIMessageChunk"):
                    content_blocks = getattr(token, 'content_blocks', None)

                    if content_blocks and isinstance(content_blocks, list):
                        for block in content_blocks:
                            if isinstance(block, dict) and block.get('type') == 'text':
                                text_content = block.get('text', '')
                                if text_content:
                                    yield {
                                        "type": "content",
                                        "data": text_content,
                                        "node": node_name
                                    }

            logger.info(f"[会话 {session_id}] RAG Agent 查询完成（流式）")
            yield {"type": "complete"}

        except Exception as e:
            logger.error(f"[会话 {session_id}] RAG Agent 查询失败（流式）: {e}")
            yield {
                "type": "error",
                "data": str(e)
            }
            raise

    def get_session_history(self, session_id: str) -> list:
        """
        获取会话历史（从 MemorySaver checkpointer 中读取）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            list: 消息历史列表 [{"role": "user|assistant", "content": "...", "timestamp": "..."}]
        """
        try:
            # 使用 checkpointer 的 get 方法获取最新的检查点
            config = {"configurable": {"thread_id": session_id}}

            # 获取该 thread 的最新检查点
            checkpoint_tuple = cast(Any, self.checkpointer).get(cast(Any, config))

            if not checkpoint_tuple:
                logger.info(f"获取会话历史: {session_id}, 消息数量: 0")
                return []

            # checkpoint_tuple 可能是命名元组或普通元组，安全地提取 checkpoint
            # 通常第一个元素是 checkpoint 数据
            if hasattr(checkpoint_tuple, "checkpoint"):
                checkpoint_data = cast(Any, checkpoint_tuple).checkpoint
            else:
                # 如果是普通元组，第一个元素是 checkpoint
                checkpoint_data = checkpoint_tuple[0] if checkpoint_tuple else {}

            # 从检查点中提取消息
            messages = cast(Any, checkpoint_data).get("channel_values", {}).get(
                "messages",
                [],
            )

            # 转换为前端需要的格式
            history = []
            for msg in messages:
                # 跳过系统消息
                if isinstance(msg, SystemMessage):
                    continue

                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                content = msg.content if hasattr(msg, 'content') else str(msg)

                # 提取时间戳（如果有的话）
                timestamp = getattr(msg, 'timestamp', None)
                if timestamp:
                    history.append({
                        "role": role,
                        "content": content,
                        "timestamp": timestamp
                    })
                else:
                    from datetime import datetime
                    history.append({
                        "role": role,
                        "content": content,
                        "timestamp": datetime.now().isoformat()
                    })

            logger.info(f"获取会话历史: {session_id}, 消息数量: {len(history)}")
            return history

        except Exception as e:
            logger.error(f"获取会话历史失败: {session_id}, 错误: {e}")
            return []

    def clear_session(self, session_id: str) -> bool:
        """
        清空会话历史（从 MemorySaver checkpointer 中删除）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            bool: 是否成功
        """
        try:
            # 使用 checkpointer 的 delete_thread 方法删除该 thread 的所有检查点
            self.checkpointer.delete_thread(session_id)

            logger.info(f"已清除会话历史: {session_id}")
            return True

        except Exception as e:
            logger.error(f"清空会话历史失败: {session_id}, 错误: {e}")
            return False

    async def cleanup(self):
        """清理资源"""
        try:
            logger.info("清理 RAG Agent 服务资源...")
            # MCP 客户端由全局管理器统一管理，无需手动清理
            logger.info("RAG Agent 服务资源已清理")
        except Exception as e:
            logger.error(f"清理资源失败: {e}")


# 全局单例 - 启用流式输出
rag_agent_service = RagAgentService(streaming=True)
