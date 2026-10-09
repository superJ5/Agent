from app.config import config
from app.services.tencent_cls_service import matches_default_topic


def test_generic_system_aliases_use_default_topic():
    assert matches_default_topic("system")
    assert matches_default_topic("monitor")
    assert matches_default_topic("当前系统")


def test_configured_service_name_uses_default_topic(monkeypatch):
    monkeypatch.setattr(config, "tencent_cls_service_name", "superj-vm")

    assert matches_default_topic("superj-vm", fuzzy=False)
    assert matches_default_topic("superj", fuzzy=True)
    assert not matches_default_topic("data-sync-service", fuzzy=True)
