import os

from app.agent.mcp_client import _ensure_local_mcp_bypasses_proxy


def test_local_mcp_hosts_are_added_to_no_proxy(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "example.com")
    monkeypatch.delenv("no_proxy", raising=False)

    _ensure_local_mcp_bypasses_proxy()

    assert set(os.environ["NO_PROXY"].split(",")) == {
        "127.0.0.1",
        "example.com",
        "localhost",
    }
    assert set(os.environ["no_proxy"].split(",")) == {
        "127.0.0.1",
        "localhost",
    }
