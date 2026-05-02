from app.runtime.hardware import detect_hardware


def test_detect_hardware_falls_back_to_cpu_when_torch_is_unavailable(monkeypatch):
    def fake_import(name):
        if name == "torch":
            raise ImportError("torch is not installed")
        return __import__(name)

    monkeypatch.setattr("importlib.import_module", fake_import)

    hardware = detect_hardware()

    assert hardware["active_backend"] == "cpu"
    assert hardware["torch_available"] is False
    assert hardware["backends"]["cpu"]["available"] is True
    assert hardware["backends"]["cuda"]["available"] is False
    assert hardware["backends"]["mps"]["available"] is False
