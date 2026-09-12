import torch

SUPPORTED_DEVICES = ("auto", "cpu", "cuda", "xpu", "mps")
_device_preference = "auto"


def _backend_is_available(device_type):
    if device_type == "cpu":
        return True
    if device_type == "cuda":
        return torch.cuda.is_available()
    if device_type == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if device_type == "mps":
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    return False


def set_device_preference(device):
    global _device_preference
    if device not in SUPPORTED_DEVICES:
        supported = ", ".join(SUPPORTED_DEVICES)
        raise ValueError(f"Unsupported device '{device}'. Expected one of: {supported}")
    _device_preference = device


def initialize_device(device=None):
    requested = device or _device_preference
    if requested != "auto":
        if not _backend_is_available(requested):
            raise RuntimeError(
                f"Requested device '{requested}' is unavailable in this PyTorch environment"
            )
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device('xpu')
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def synchronize_device(device):
    """Wait for queued accelerator work before recording benchmark timings."""
    device_type = torch.device(device).type
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "xpu":
        torch.xpu.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()
