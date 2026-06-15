"""RAG Agent 服务 - 基于 LangGraph 的智能代理

使用 langchain_qwq 的 ChatQwen 原生集成，
支持真正的流式输出和更好的模型适配。
"""

import asyncio
import re
import uuid
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
from app.retrieval.diagnostics import reset_trace_chat_context, set_trace_chat_context
from app.services.memory_service import memory_service
from app.services.multimodal_message_builder import build_user_message
from app.services.query_router import should_use_manual_rag
from app.tools import get_current_time, retrieve_knowledge

try:
    from app.services.long_term_memory_service import (
        long_term_memory_service as _long_term_memory_service,
    )
except Exception:  # pragma: no cover - keeps lightweight tests importable with stubs.
    class _NoopLongTermMemoryService:
        def build_context_block(self, **kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
            return "", []

        async def update_after_turn(self, **kwargs: Any) -> None:
            return None

    _long_term_memory_service = _NoopLongTermMemoryService()

long_term_memory_service = _long_term_memory_service

try:
    from app.services.session_state_service import (
        session_state_service as _session_state_service,
    )
except Exception:  # pragma: no cover - keeps lightweight tests importable with stubs.
    class _NoopSessionStateService:
        def build_context_block(self, session_id: str) -> str:
            return ""

        async def update_after_turn(self, **kwargs: Any) -> None:
            return None

    _session_state_service = _NoopSessionStateService()

session_state_service = _session_state_service

try:
    from app.services.short_term_memory_service import (
        short_term_memory_service as _short_term_memory_service,
    )
except Exception:  # pragma: no cover - keeps lightweight tests importable with stubs.
    class _NoopShortTermMemoryService:
        def load_memory(self, session_id: str) -> str:
            return ""

        def load_recent_dialogue(
            self,
            session_id: str,
            *,
            current_question: str | None = None,
        ) -> list[dict[str, Any]]:
            return []

        def format_dialogue(self, records: list[dict[str, Any]]) -> str:
            return ""

        async def update_after_turn(self, **kwargs: Any) -> None:
            return None

    _short_term_memory_service = _NoopShortTermMemoryService()

short_term_memory_service = _short_term_memory_service

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
            temperature=0.2,
            streaming=streaming,
        )

        # 定义基础工具
        self.tools = [retrieve_knowledge, get_current_time]

        # MCP 客户端（延迟初始化，使用全局管理）
        self.mcp_tools: list = []
        self._last_retrieval_metadata_by_session: dict[str, dict[str, Any] | None] = {}

        # 创建内存检查点（用于会话管理）
        self.checkpointer = MemorySaver()

        # Agent 初始化（会在异步方法中完成）
        self.agent = None
        self.customer_service_agent = None
        self._agent_initialized = False
        self._memory_update_tasks: set[asyncio.Task[Any]] = set()
        self._session_state_update_tasks: set[asyncio.Task[Any]] = set()
        self._long_term_memory_update_tasks: set[asyncio.Task[Any]] = set()

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
        self.customer_service_agent = create_agent(
            self.model,
            tools=[get_current_time],
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
            1. 当前问题允许使用产品手册检索时，只要涉及说明书、手册、部件、操作步骤、保修、图片、OCR 或原文追溯，必须先调用 retrieve_knowledge 再回答。
            2. 优先依据 retrieve_knowledge 返回的证据回答，不要在没有检索证据时直接猜测手册内容。
            3. 如果 retrieve_knowledge 没有找到可靠内容，要明确说明“当前检索到的信息不足”，而不是编造答案。
            4. 如果检索结果里带有图片标识（PIC）或配图信息，回答时要优先结合这些证据。
            5. 调用 retrieve_knowledge 时，检索词必须与用户当前问题使用相同语言。
            6. 用户问题包含中文字符时，必须使用中文检索词，禁止将问题或产品名称翻译成英文。
            7. 用户问题不包含中文字符时，使用英文检索词，禁止翻译成中文。
            8. 可以将用户问题适当拆分为同语言关键词，但只能使用用户原问题中已经出现的词、短语、型号、产品名和操作对象。
            9. 禁止扩写、联想、同义词替换、概念泛化、补充隐含条件或添加用户原问题中没有出现的检索词。
            10. 示例：中文问题“如何为蓝牙激光鼠标安装电池？”可检索“蓝牙激光鼠标 安装 电池”，不得加入原问题中没有出现的额外词，也不得检索“bluetooth laser mouse battery installation”。
            11. 示例：英文问题“How do I install the mouse battery?”可检索“How install mouse battery”或“mouse install battery”，不得加入原问题中没有出现的额外词，也不得检索“鼠标 安装电池”。
            12. retrieve_knowledge 返回的证据包含【主命中】和【补充上下文】两部分。
                优先参考【主命中】，但【补充上下文】同样可能包含问题的正确答案或
                必要的补充信息，不可忽略。
            13. 逐条阅读所有证据内容，将每条证据的实际内容与用户问题进行比对，
                找出真正与问题对应的段落；单条证据中不相关的部分不纳入回答。
            14. 比对完所有证据后，判断对用户问题的覆盖程度：
                - 完全覆盖：正常回答
                - 部分覆盖：回答有据可依的部分，对未覆盖的部分明确说明信息不足
                - 未覆盖：直接说明未检索到相关信息，不得用不相关内容拼凑回答
            15. 引用证据内容时，严格保留原文中的设备名称、型号、单位、限定条件等
                事实性表述，不得为了贴合用户问题而替换或改写

            记忆使用规则:
            1. 当前旧版 daily/MEMORY.md 记忆工具已移除，不要使用工具获取历史偏好或项目背景。
            2. 原始 session JSONL 只做日志和后续记忆更新原材料，不会直接进入当前回答上下文。
            3. 历史信息不足时，要明确说明未找到足够历史信息。

            回答要求:
            - 保持友好、专业的语气
            - 回答简洁明了，重点突出
            - 基于事实，不编造信息
            - 如有不确定的地方，明确说明
            - 只回答用户明确询问的内容，不主动扩展其他方案或无关信息
            - 除图片引用外，必须使用纯文本，不使用 Markdown 标题、粗体、斜体、引用、表格或分隔线
            - 可以使用普通数字序号或短横线列表，但不要添加 Markdown 装饰
            - 需要表达对比信息时逐行描述，不要使用表格
            - 直接给出结论或操作步骤，省略“根据手册”“为您详细解答”等开场套话
            - 不复述用户问题，不重复已经说明的内容，不在结尾主动提供额外帮助
            - 简单问答通常控制在100至300字；操作步骤类回答通常控制在300至600字
            - 长度限制是目标而非硬性截断；必要的安全警告、关键条件和图片引用必须保留

            请根据用户的问题，灵活使用可用工具，提供高质量的帮助。
        """).strip()
        return (
            prompt
            + "\n\n图片输出规则:\n"
            + "- 图片引用是唯一允许使用的 Markdown 格式。\n"
            + "- 引用图片时，必须使用图片 ID 作为 Markdown 图片名称，例如："
            + "![Manual01_5](data/manuals/raw/.../Manual01_5.jpg)。\n"
            + "- 禁止使用空图片名称，例如 ![](path)。\n"
            + "- 每张图片必须紧跟在它直接说明的步骤、部件或操作内容之后。\n"
            + "- 不要将多张图片统一堆放在答案末尾。\n"
            + "- 仅引用能够直接帮助理解当前问题的图片，不引用无关或作用重复的图片。\n"
            + "- 图片顺序必须与回答内容和检索证据中的顺序一致。"
        )

    def _build_effective_system_prompt(self, *, manual_rag_enabled: bool = True) -> str:
        """Build system prompt without legacy daily/MEMORY.md injection."""
        prompt = self.system_prompt
        if not manual_rag_enabled:
            prompt += (
                "\n\n通用客服回答规则:\n"
                "- 当前问题没有明确的产品手册依据，不使用产品手册检索。\n"
                "- 不要编造具体平台政策、处理时限或赔偿标准；信息不足时说明需要联系平台客服确认。\n"
                "- 先回应用户当前诉求，再给出简洁、可执行的处理步骤。\n"
                "- 涉及退换货、投诉、物流或售后时，提醒用户保留订单、照片、聊天记录等必要凭证。\n"
                "- 需要平台核实时，建议用户通过订单售后入口或人工客服提交，不承诺具体处理结果。\n"
                "- 使用自然、简洁的客服语气，不使用“根据手册”等表述，不过度道歉，不重复用户问题。"
            )
        return prompt

    def _build_persistent_history_messages(
        self,
        session_id: str,
        current_question: str,
    ) -> list[BaseMessage]:
        """Legacy raw-message context injection is disabled.

        Raw JSONL remains the source of truth for auditing and future memory
        updaters, but it should not be inserted into Agent context directly.
        """
        _ = (session_id, current_question)
        return []

    def _build_short_term_context_messages(
        self,
        session_id: str,
        current_question: str,
    ) -> tuple[list[BaseMessage], list[dict[str, Any]]]:
        """Build short-term semantic memory context for the current request."""
        if not config.memory_enabled:
            return [], []

        try:
            memory = short_term_memory_service.load_memory(session_id)
            recent_dialogue = short_term_memory_service.load_recent_dialogue(
                session_id,
                current_question=current_question,
            )
            recent_text = short_term_memory_service.format_dialogue(recent_dialogue)
        except Exception as exc:
            logger.warning("[会话 {}] 构造短期记忆上下文失败: {}", session_id, exc)
            return [], []

        sections: list[str] = []
        if memory:
            sections.append("【短期语义记忆】\n" + memory)
        if recent_text:
            sections.append("【最近 1-3 轮原始对话】\n" + recent_text)
        if not sections:
            return [], recent_dialogue

        content = (
            "以下是当前 session 的短期上下文，只作为续接当前任务的参考；"
            "如与当前用户问题冲突，以当前用户问题为准。\n\n"
            + "\n\n".join(sections)
        )
        return [SystemMessage(content=content)], recent_dialogue

    def _build_session_state_context_messages(self, session_id: str) -> list[BaseMessage]:
        """Build structured session-state context for the current request."""
        if not config.memory_enabled:
            return []

        try:
            context = session_state_service.build_context_block(session_id)
        except Exception as exc:
            logger.warning("[会话 {}] 构造 Session State 上下文失败: {}", session_id, exc)
            return []
        if not context:
            return []
        return [SystemMessage(content=context)]

    def _build_long_term_context_messages(
        self,
        *,
        session_id: str,
        question: str,
        session_state_messages: list[BaseMessage],
        short_term_messages: list[BaseMessage],
    ) -> tuple[list[BaseMessage], list[dict[str, Any]]]:
        """Build relevant long-term memory context for the current request."""
        if not config.memory_enabled:
            return [], []

        try:
            session_state_context = "\n\n".join(str(message.content) for message in session_state_messages)
            short_term_context = "\n\n".join(str(message.content) for message in short_term_messages)
            context, memories = long_term_memory_service.build_context_block(
                question=question,
                session_id=session_id,
                session_state_context=session_state_context,
                short_term_context=short_term_context,
            )
        except Exception as exc:
            logger.warning("[会话 {}] 构造长期记忆上下文失败: {}", session_id, exc)
            return [], []
        if not context:
            return [], memories
        return [SystemMessage(content=context)], memories

    async def _update_short_term_memory_after_turn(
        self,
        *,
        session_id: str,
        question: str,
        answer: str,
        prior_dialogue: list[dict[str, Any]],
    ) -> None:
        try:
            await short_term_memory_service.update_after_turn(
                session_id=session_id,
                user_message=question,
                assistant_message=answer,
                prior_dialogue=prior_dialogue,
            )
        except Exception as exc:
            logger.warning("[会话 {}] 短期记忆更新异常，已忽略: {}", session_id, exc)

    def _schedule_short_term_memory_update(
        self,
        *,
        session_id: str,
        question: str,
        answer: str,
        prior_dialogue: list[dict[str, Any]],
    ) -> None:
        """Schedule memory update without delaying the user-facing answer."""
        if not str(answer or "").strip():
            return
        try:
            task = asyncio.create_task(
                self._update_short_term_memory_after_turn(
                    session_id=session_id,
                    question=question,
                    answer=answer,
                    prior_dialogue=prior_dialogue,
                )
            )
            self._memory_update_tasks.add(task)
            task.add_done_callback(self._memory_update_tasks.discard)
        except RuntimeError:
            logger.warning("[会话 {}] 无可用事件循环，跳过短期记忆后台更新", session_id)

    async def _update_session_state_after_turn(
        self,
        *,
        session_id: str,
        question: str,
        answer: str,
        prior_dialogue: list[dict[str, Any]],
    ) -> None:
        try:
            await session_state_service.update_after_turn(
                session_id=session_id,
                user_message=question,
                assistant_message=answer,
                prior_dialogue=prior_dialogue,
            )
        except Exception as exc:
            logger.warning("[会话 {}] Session State 更新异常，已忽略: {}", session_id, exc)

    def _schedule_session_state_update(
        self,
        *,
        session_id: str,
        question: str,
        answer: str,
        prior_dialogue: list[dict[str, Any]],
    ) -> None:
        """Schedule session-state update without delaying the answer."""
        if not str(answer or "").strip():
            return
        try:
            task = asyncio.create_task(
                self._update_session_state_after_turn(
                    session_id=session_id,
                    question=question,
                    answer=answer,
                    prior_dialogue=prior_dialogue,
                )
            )
            self._session_state_update_tasks.add(task)
            task.add_done_callback(self._session_state_update_tasks.discard)
        except RuntimeError:
            logger.warning("[会话 {}] 无可用事件循环，跳过 Session State 后台更新", session_id)

    async def _update_long_term_memory_after_turn(
        self,
        *,
        session_id: str,
        question: str,
        answer: str,
        prior_dialogue: list[dict[str, Any]],
        retrieved_memories: list[dict[str, Any]],
        session_state_messages: list[BaseMessage],
        short_term_messages: list[BaseMessage],
    ) -> None:
        try:
            await long_term_memory_service.update_after_turn(
                session_id=session_id,
                user_message=question,
                assistant_message=answer,
                session_state_context="\n\n".join(str(message.content) for message in session_state_messages),
                short_term_memory="\n\n".join(str(message.content) for message in short_term_messages),
                prior_dialogue=prior_dialogue,
                retrieved_memories=retrieved_memories,
            )
        except Exception as exc:
            logger.warning("[会话 {}] 长期记忆更新异常，已忽略: {}", session_id, exc)

    def _schedule_long_term_memory_update(
        self,
        *,
        session_id: str,
        question: str,
        answer: str,
        prior_dialogue: list[dict[str, Any]],
        retrieved_memories: list[dict[str, Any]],
        session_state_messages: list[BaseMessage],
        short_term_messages: list[BaseMessage],
    ) -> None:
        """Schedule long-term memory update without delaying the answer."""
        if not str(answer or "").strip():
            return
        try:
            task = asyncio.create_task(
                self._update_long_term_memory_after_turn(
                    session_id=session_id,
                    question=question,
                    answer=answer,
                    prior_dialogue=prior_dialogue,
                    retrieved_memories=retrieved_memories,
                    session_state_messages=session_state_messages,
                    short_term_messages=short_term_messages,
                )
            )
            self._long_term_memory_update_tasks.add(task)
            task.add_done_callback(self._long_term_memory_update_tasks.discard)
        except RuntimeError:
            logger.warning("[会话 {}] 无可用事件循环，跳过长期记忆后台更新", session_id)

    def _schedule_context_memory_updates(
        self,
        *,
        session_id: str,
        question: str,
        answer: str,
        prior_dialogue: list[dict[str, Any]],
        retrieved_memories: list[dict[str, Any]],
        session_state_messages: list[BaseMessage],
        short_term_messages: list[BaseMessage],
    ) -> None:
        """Schedule all context-memory updates for one completed turn."""
        if not config.memory_enabled:
            logger.debug("[会话 {}] 跳过上下文记忆更新: MEMORY_ENABLED=false", session_id)
            return

        self._schedule_short_term_memory_update(
            session_id=session_id,
            question=question,
            answer=answer,
            prior_dialogue=prior_dialogue,
        )
        self._schedule_long_term_memory_update(
            session_id=session_id,
            question=question,
            answer=answer,
            prior_dialogue=prior_dialogue,
            retrieved_memories=retrieved_memories,
            session_state_messages=session_state_messages,
            short_term_messages=short_term_messages,
        )
        self._schedule_session_state_update(
            session_id=session_id,
            question=question,
            answer=answer,
            prior_dialogue=prior_dialogue,
        )

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
        trace_context_token = set_trace_chat_context(
            question=question,
            session_id=session_id,
        )
        try:
            await self._initialize_agent()
            self._last_retrieval_metadata_by_session[session_id] = None
            self._clear_last_retrieval_metadata(session_id)

            image_count = len(images or [])
            manual_rag_enabled = should_use_manual_rag(question, has_images=image_count > 0)
            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（非流式）: {question}, images={image_count}")
            logger.info(
                f"[会话 {session_id}] 查询分流: "
                f"{'manual_rag' if manual_rag_enabled else 'customer_service'}"
            )
            short_term_messages, prior_dialogue = self._build_short_term_context_messages(
                session_id,
                question,
            )
            session_state_messages = self._build_session_state_context_messages(session_id)
            long_term_messages, retrieved_long_term_memories = self._build_long_term_context_messages(
                session_id=session_id,
                question=question,
                session_state_messages=session_state_messages,
                short_term_messages=short_term_messages,
            )

            # 构建消息列表（系统提示 + 用户问题）
            messages = [
                SystemMessage(
                    content=self._build_effective_system_prompt(
                        manual_rag_enabled=manual_rag_enabled
                    )
                ),
                *long_term_messages,
                *session_state_messages,
                *short_term_messages,
                *self._build_persistent_history_messages(session_id, question),
                build_user_message(question, images),
            ]

            # 构建 Agent 输入
            agent_input = {"messages": messages}

            # 配置 thread_id（用于会话持久化）
            config_dict = {
                "configurable": {
                    "thread_id": self._request_thread_id(session_id)
                }
            }

            selected_agent = self.agent if manual_rag_enabled else self.customer_service_agent
            if selected_agent is None:
                raise RuntimeError("Agent 未初始化")

            result = await selected_agent.ainvoke(
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
                answer_text = self._ensure_image_placeholders(str(answer))
                self._schedule_context_memory_updates(
                    session_id=session_id,
                    question=question,
                    answer=answer_text,
                    prior_dialogue=prior_dialogue,
                    retrieved_memories=retrieved_long_term_memories,
                    session_state_messages=session_state_messages,
                    short_term_messages=short_term_messages,
                )
                return answer_text

            logger.warning(f"[会话 {session_id}] Agent 返回结果为空")
            return ""

        except Exception as e:
            logger.error(f"[会话 {session_id}] RAG Agent 查询失败（非流式）: {e}")
            raise
        finally:
            reset_trace_chat_context(trace_context_token)

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
    def _clear_last_retrieval_metadata(session_id: str) -> None:
        try:
            from app.tools.knowledge_tool import clear_last_retrieval_metadata

            clear_last_retrieval_metadata(session_id=session_id)
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
        trace_context_token = set_trace_chat_context(
            question=question,
            session_id=session_id,
        )
        try:
            await self._initialize_agent()
            self._clear_last_retrieval_metadata(session_id)

            image_count = len(images or [])
            manual_rag_enabled = should_use_manual_rag(question, has_images=image_count > 0)
            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（流式）: {question}, images={image_count}")
            logger.info(
                f"[会话 {session_id}] 查询分流: "
                f"{'manual_rag' if manual_rag_enabled else 'customer_service'}"
            )
            short_term_messages, prior_dialogue = self._build_short_term_context_messages(
                session_id,
                question,
            )
            session_state_messages = self._build_session_state_context_messages(session_id)
            long_term_messages, retrieved_long_term_memories = self._build_long_term_context_messages(
                session_id=session_id,
                question=question,
                session_state_messages=session_state_messages,
                short_term_messages=short_term_messages,
            )

            # 构建消息列表（系统提示 + 用户问题）
            messages = [
                SystemMessage(
                    content=self._build_effective_system_prompt(
                        manual_rag_enabled=manual_rag_enabled
                    )
                ),
                *long_term_messages,
                *session_state_messages,
                *short_term_messages,
                *self._build_persistent_history_messages(session_id, question),
                build_user_message(question, images),
            ]

            # 构建 Agent 输入
            agent_input = {"messages": messages}

            # 配置 thread_id（用于会话持久化）
            config_dict = {
                "configurable": {
                    "thread_id": self._request_thread_id(session_id)
                }
            }

            selected_agent = self.agent if manual_rag_enabled else self.customer_service_agent
            if selected_agent is None:
                raise RuntimeError("Agent 未初始化")

            answer_parts: list[str] = []
            async for token, metadata in selected_agent.astream(
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
                                    answer_parts.append(text_content)
                                    yield {
                                        "type": "content",
                                        "data": text_content,
                                        "node": node_name
                                    }

            logger.info(f"[会话 {session_id}] RAG Agent 查询完成（流式）")
            self._schedule_context_memory_updates(
                session_id=session_id,
                question=question,
                answer="".join(answer_parts),
                prior_dialogue=prior_dialogue,
                retrieved_memories=retrieved_long_term_memories,
                session_state_messages=session_state_messages,
                short_term_messages=short_term_messages,
            )
            yield {"type": "complete"}

        except Exception as e:
            logger.error(f"[会话 {session_id}] RAG Agent 查询失败（流式）: {e}")
            yield {
                "type": "error",
                "data": str(e)
            }
            raise
        finally:
            reset_trace_chat_context(trace_context_token)

    def get_session_history(self, session_id: str) -> list:
        """
        获取会话历史（从 MemorySaver checkpointer 中读取）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            list: 消息历史列表 [{"role": "user|assistant", "content": "...", "timestamp": "..."}]
        """
        try:
            history = []
            for record in memory_service.load_recent_messages(
                session_id,
                limit=config.memory_recent_limit,
            ):
                history.append(
                    {
                        "role": str(record.get("role") or ""),
                        "content": str(record.get("content") or ""),
                        "timestamp": str(record.get("timestamp") or ""),
                    }
                )

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

    @staticmethod
    def _request_thread_id(session_id: str) -> str:
        """Use a fresh LangGraph thread so raw prior messages are not replayed."""
        return f"{session_id}__request_{uuid.uuid4().hex}"

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
