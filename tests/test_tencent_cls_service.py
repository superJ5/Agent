from types import SimpleNamespace

import pytest

from app.config import config
from app.services import tencent_cls_service


def _configure_credentials(monkeypatch):
    monkeypatch.setattr(config, "tencentcloud_secret_id", "test-id")
    monkeypatch.setattr(config, "tencentcloud_secret_key", "test-key")
    monkeypatch.setattr(config, "tencent_cls_region", "ap-guangzhou")


def test_search_logs_normalizes_real_cls_response(monkeypatch):
    _configure_credentials(monkeypatch)
    captured = {}

    class FakeClient:
        def SearchLog(self, request):
            captured["request"] = request
            log = SimpleNamespace(
                Time=1725450000000,
                TopicId="topic-id",
                TopicName="superj-vm-system",
                Source="192.168.147.130",
                HostName="superj-virtual-machine",
                FileName="/var/log/syslog",
                LogJson='{"__CONTENT__":"AIOPS_CLS_TEST"}',
                RawLog=None,
            )
            return SimpleNamespace(
                Results=[log],
                ListOver=True,
                Context="context-token",
                RequestId="request-id",
            )

    monkeypatch.setattr(tencent_cls_service, "build_cls_client", lambda *_: FakeClient())

    result = tencent_cls_service.search_logs(
        topic_id="topic-id",
        start_time=1000,
        end_time=2000,
        query="error",
        limit=20,
    )

    request = captured["request"]
    assert request.TopicId == "topic-id"
    assert request.QueryString == "error"
    assert request.QuerySyntax == 1
    assert result["data_source"] == "tencent_cls"
    assert result["logs"][0]["message"] == "AIOPS_CLS_TEST"
    assert result["logs"][0]["file_name"] == "/var/log/syslog"


def test_search_logs_rejects_invalid_range():
    with pytest.raises(ValueError, match="start_time"):
        tencent_cls_service.search_logs(
            topic_id="topic-id",
            start_time=2000,
            end_time=1000,
            query=None,
            limit=20,
        )


def test_missing_credentials_are_reported(monkeypatch):
    monkeypatch.setattr(config, "tencentcloud_secret_id", "")
    monkeypatch.setattr(config, "tencentcloud_secret_key", "")

    with pytest.raises(tencent_cls_service.TencentCLSConfigError) as exc_info:
        tencent_cls_service.build_cls_client()

    assert "TENCENTCLOUD_SECRET_ID" in str(exc_info.value)
    assert "TENCENTCLOUD_SECRET_KEY" in str(exc_info.value)
