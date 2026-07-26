import torch


def cuda_capability_major(device=None):
    if not torch.cuda.is_available():
        return -1
    if device is not None and torch.device(device).type != "cuda":
        return -1
    return torch.cuda.get_device_capability(device)[0]


def supports_bf16(device=None):
    return cuda_capability_major(device) >= 8


def supports_amp(device=None):
    return cuda_capability_major(device) >= 7


def supports_torch_compile(device=None):
    return cuda_capability_major(device) >= 7
