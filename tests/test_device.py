from types import SimpleNamespace

import pytest

from utils import device


@pytest.fixture(autouse=True)
def reset_device_preference():
    device.set_device_preference("auto")
    yield
    device.set_device_preference("auto")


def test_auto_prefers_xpu_over_cpu(monkeypatch):
    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        device.torch, "xpu", SimpleNamespace(is_available=lambda: True)
    )
    monkeypatch.setattr(
        device.torch.backends, "mps", SimpleNamespace(is_available=lambda: False)
    )

    assert device.initialize_device().type == "xpu"


def test_explicit_cpu_overrides_available_accelerators(monkeypatch):
    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: True)
    device.set_device_preference("cpu")

    assert device.initialize_device().type == "cpu"


def test_unavailable_explicit_xpu_fails_clearly(monkeypatch):
    monkeypatch.setattr(
        device.torch, "xpu", SimpleNamespace(is_available=lambda: False)
    )

    with pytest.raises(RuntimeError, match="Requested device 'xpu' is unavailable"):
        device.initialize_device("xpu")
