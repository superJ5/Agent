from types import SimpleNamespace

from app.services import local_monitor_service


def test_cpu_snapshot_is_labeled_as_live_local_data(monkeypatch):
    monkeypatch.setattr(local_monitor_service.psutil, "cpu_percent", lambda interval: 23.4)
    monkeypatch.setattr(local_monitor_service.psutil, "cpu_count", lambda logical: 4)
    monkeypatch.setattr(local_monitor_service.psutil, "getloadavg", lambda: (0.2, 0.3, 0.4))
    monkeypatch.setattr(local_monitor_service.socket, "gethostname", lambda: "lab-vm")

    result = local_monitor_service.read_cpu_snapshot("current-system")

    assert result["data_source"] == "local_system"
    assert result["collection_mode"] == "live_snapshot"
    assert result["historical_available"] is False
    assert result["observed_host"] == "lab-vm"
    assert result["statistics"]["current"] == 23.4
    assert result["alert_info"]["triggered"] is False


def test_memory_snapshot_uses_real_counters(monkeypatch):
    gib = 1024**3
    memory = SimpleNamespace(percent=75.2, used=6 * gib, available=2 * gib, total=8 * gib)
    monkeypatch.setattr(local_monitor_service.psutil, "virtual_memory", lambda: memory)
    monkeypatch.setattr(local_monitor_service.socket, "gethostname", lambda: "lab-vm")

    result = local_monitor_service.read_memory_snapshot("current-system")

    point = result["data_points"][0]
    assert point["used_gb"] == 6.0
    assert point["available_gb"] == 2.0
    assert point["total_gb"] == 8.0
    assert result["statistics"]["current"] == 75.2
    assert result["alert_info"]["triggered"] is True
