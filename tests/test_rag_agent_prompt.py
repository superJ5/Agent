from __future__ import annotations

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
        return None


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
                load_long_term_memory=lambda: "",
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
            memory_search=lambda query: "",
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


def test_system_prompt_instructs_english_manual_retrieval(monkeypatch):
    module = load_rag_agent_service(monkeypatch)
    prompt = module.rag_agent_service.system_prompt

    assert "For manual-related English questions, call retrieve_knowledge first." in prompt
    assert "pass a concise English search query" in prompt
    assert "Do not translate English questions into Chinese unless the user asks." in prompt
    assert "禁止使用空图片名称，例如 ![](path)。" in prompt
    assert "每张图片必须紧跟在它直接说明的步骤、部件或操作内容之后。" in prompt
    assert "不要将多张图片统一堆放在答案末尾。" in prompt
    assert "仅引用能够直接帮助理解当前问题的图片" in prompt
