from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


class FakeLogger:
    def info(self, *args, **kwargs) -> None:
        return None

    def warning(self, *args, **kwargs) -> None:
        return None

    def debug(self, *args, **kwargs) -> None:
        return None

    def error(self, *args, **kwargs) -> None:
        return None


class FakeChatQwen:
    def __init__(self, *args, **kwargs) -> None:
        self.kwargs = kwargs


class FakeMemorySaver:
    def get(self, *args, **kwargs):
        return None

    def delete_thread(self, *args, **kwargs) -> None:
        return None


class FakeMessage:
    def __init__(self, content: str = "", id: str | None = None) -> None:
        self.content = content
        self.id = id


def make_module(name: str, **attrs) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def load_rag_agent_service(monkeypatch):
    app_module = make_module("app")
    app_module.__path__ = [str(ROOT / "app")]
    services_module = make_module("app.services")
    services_module.__path__ = [str(ROOT / "app" / "services")]
    agent_module = make_module("app.agent")
    agent_module.__path__ = []
    models_module = make_module("app.models")
    models_module.__path__ = []

    monkeypatch.setitem(sys.modules, "app", app_module)
    monkeypatch.setitem(sys.modules, "app.services", services_module)
    monkeypatch.setitem(sys.modules, "app.agent", agent_module)
    monkeypatch.setitem(sys.modules, "app.models", models_module)
    monkeypatch.setitem(
        sys.modules,
        "app.agent.mcp_client",
        make_module("app.agent.mcp_client", get_mcp_client_with_retry=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "app.config",
        make_module(
            "app.config",
            config=SimpleNamespace(
                rag_model="fake-model",
                dashscope_api_key="fake-key",
                dashscope_api_base="https://example.test",
                memory_recent_limit=3,
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "app.models.response",
        make_module("app.models.response", sanitize_summary_metadata=lambda metadata: metadata),
    )
    monkeypatch.setitem(
        sys.modules,
        "app.services.memory_service",
        make_module(
            "app.services.memory_service",
            memory_service=SimpleNamespace(
                load_recent_messages=lambda *args, **kwargs: [],
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "app.services.multimodal_message_builder",
        make_module(
            "app.services.multimodal_message_builder",
            build_user_message=lambda question, images=None: FakeMessage(question),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "app.tools",
        make_module(
            "app.tools",
            get_current_time=lambda: None,
            retrieve_knowledge=lambda query: ("", []),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain.agents",
        make_module("langchain.agents", create_agent=lambda *args, **kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_core.messages",
        make_module(
            "langchain_core.messages",
            AIMessage=FakeMessage,
            BaseMessage=FakeMessage,
            HumanMessage=FakeMessage,
            RemoveMessage=FakeMessage,
            SystemMessage=FakeMessage,
        ),
    )
    monkeypatch.setitem(sys.modules, "langchain_qwq", make_module("langchain_qwq", ChatQwen=FakeChatQwen))
    monkeypatch.setitem(
        sys.modules,
        "langgraph.checkpoint.memory",
        make_module("langgraph.checkpoint.memory", MemorySaver=FakeMemorySaver),
    )
    monkeypatch.setitem(
        sys.modules,
        "langgraph.graph.message",
        make_module(
            "langgraph.graph.message",
            REMOVE_ALL_MESSAGES="__remove_all__",
            add_messages=lambda *args, **kwargs: None,
        ),
    )
    monkeypatch.setitem(sys.modules, "loguru", make_module("loguru", logger=FakeLogger()))

    spec = importlib.util.spec_from_file_location(
        "rag_agent_service_under_test",
        ROOT / "app" / "services" / "rag_agent_service.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "rag_agent_service_under_test", module)
    spec.loader.exec_module(module)
    return module


def test_system_prompt_preserves_query_language_for_manual_retrieval(monkeypatch):
    module = load_rag_agent_service(monkeypatch)
    prompt = module.rag_agent_service.system_prompt

    assert "检索词必须与用户当前问题使用相同语言" in prompt
    assert "用户问题包含中文字符时，必须使用中文检索词" in prompt
    assert "用户问题不包含中文字符时，使用英文检索词" in prompt
    assert "不得通过翻译改变检索语言" in prompt
    assert "蓝牙激光鼠标 安装电池 电池仓" in prompt
    assert "mouse battery installation battery compartment" in prompt
    assert "禁止使用空图片名称，例如 ![](path)。" in prompt
    assert "每张图片必须紧跟在它直接说明的步骤、部件或操作内容之后。" in prompt
    assert "不要将多张图片统一堆放在答案末尾。" in prompt
    assert "仅引用能够直接帮助理解当前问题的图片" in prompt


def test_model_uses_low_temperature_for_stable_answers(monkeypatch):
    module = load_rag_agent_service(monkeypatch)

    assert module.rag_agent_service.model.kwargs["temperature"] == 0.2


def test_customer_service_prompt_uses_conservative_service_style(monkeypatch):
    module = load_rag_agent_service(monkeypatch)
    prompt = module.rag_agent_service._build_effective_system_prompt(manual_rag_enabled=False)

    assert "保留订单、照片、聊天记录等必要凭证" in prompt
    assert "不承诺具体处理结果" in prompt
    assert "不过度道歉，不重复用户问题" in prompt


def test_initializes_separate_manual_and_customer_service_agents(monkeypatch):
    module = load_rag_agent_service(monkeypatch)
    created_tool_sets = []

    def fake_create_agent(model, *, tools, checkpointer):
        created_tool_sets.append(tools)
        return SimpleNamespace()

    monkeypatch.setattr(module, "create_agent", fake_create_agent)
    service = module.RagAgentService()

    asyncio.run(service._initialize_agent())

    assert len(created_tool_sets) == 2
    assert module.retrieve_knowledge in created_tool_sets[0]
    assert module.retrieve_knowledge not in created_tool_sets[1]


def test_query_selects_agent_using_local_router(monkeypatch):
    module = load_rag_agent_service(monkeypatch)

    class FakeAgent:
        def __init__(self, answer: str) -> None:
            self.answer = answer

        async def ainvoke(self, **kwargs):
            return {"messages": [FakeMessage(self.answer)]}

    service = module.RagAgentService()
    service.agent = FakeAgent("manual")
    service.customer_service_agent = FakeAgent("customer")
    service._agent_initialized = True

    manual_answer = asyncio.run(service.query("如何给蓝牙激光鼠标安装电池？", "manual-session"))
    customer_answer = asyncio.run(service.query("我的快递丢失了，怎么办？", "customer-session"))

    assert manual_answer == "manual"
    assert customer_answer == "customer"


def test_query_injects_and_updates_short_term_memory(monkeypatch):
    module = load_rag_agent_service(monkeypatch)

    class FakeShortTermMemoryService:
        def __init__(self) -> None:
            self.updated_payload = None

        def load_memory(self, session_id: str) -> str:
            assert session_id == "memory-session"
            return "目标：定位支付接口变慢原因。"

        def load_recent_dialogue(self, session_id: str, *, current_question: str | None = None):
            assert session_id == "memory-session"
            assert current_question == "下一步查什么？"
            return [
                {"role": "user", "content": "接口 22:10 后变慢"},
                {"role": "assistant", "content": "先排查数据库慢查询"},
            ]

        def format_dialogue(self, records):
            return "\n".join(
                f"{item['role']}: {item['content']}"
                for item in records
            )

        async def update_after_turn(self, **kwargs):
            self.updated_payload = kwargs

    class FakeSessionStateService:
        def build_context_block(self, session_id: str) -> str:
            assert session_id == "memory-session"
            return (
                "【当前会话状态】\n"
                "目标：定位支付接口变慢原因\n"
                "当前假设：\n"
                "- 第三方回调超时可能导致接口变慢"
            )

    class FakeLongTermMemoryService:
        def build_context_block(self, **kwargs):
            assert kwargs["session_id"] == "memory-session"
            assert kwargs["question"] == "下一步查什么？"
            return (
                "【长期记忆】\n- 用户喜欢直接给排查步骤",
                [{"memory_id": "ltm-test", "content": "用户喜欢直接给排查步骤"}],
            )

    class FakeAgent:
        def __init__(self) -> None:
            self.messages = []

        async def ainvoke(self, **kwargs):
            self.messages = kwargs["input"]["messages"]
            return {"messages": [FakeMessage("继续检查回调超时。")]}

    fake_memory = FakeShortTermMemoryService()
    fake_agent = FakeAgent()
    monkeypatch.setattr(module, "short_term_memory_service", fake_memory)
    monkeypatch.setattr(module, "session_state_service", FakeSessionStateService())
    monkeypatch.setattr(module, "long_term_memory_service", FakeLongTermMemoryService())

    service = module.RagAgentService()
    service.agent = fake_agent
    service.customer_service_agent = fake_agent
    service._agent_initialized = True
    scheduled_payload = {}
    monkeypatch.setattr(
        service,
        "_schedule_context_memory_updates",
        lambda **kwargs: scheduled_payload.update(kwargs),
    )

    answer = asyncio.run(service.query("下一步查什么？", "memory-session"))

    assert answer == "继续检查回调超时。"
    context_text = "\n".join(str(message.content) for message in fake_agent.messages)
    assert "【长期记忆】" in context_text
    assert "用户喜欢直接给排查步骤" in context_text
    assert "【当前会话状态】" in context_text
    assert "第三方回调超时可能导致接口变慢" in context_text
    assert "【短期语义记忆】" in context_text
    assert "目标：定位支付接口变慢原因。" in context_text
    assert "【最近 1-3 轮原始对话】" in context_text
    assert "接口 22:10 后变慢" in context_text
    assert context_text.index("【长期记忆】") < context_text.index("【当前会话状态】")
    assert context_text.index("【当前会话状态】") < context_text.index("【短期语义记忆】")
    assert scheduled_payload["session_id"] == "memory-session"
    assert scheduled_payload["question"] == "下一步查什么？"
    assert scheduled_payload["answer"] == "继续检查回调超时。"
    assert scheduled_payload["prior_dialogue"][0]["content"] == "接口 22:10 后变慢"
    assert scheduled_payload["retrieved_memories"][0]["memory_id"] == "ltm-test"
