import functools
import os
import warnings

# Try importing PyTorch and check version
try:
    import torch
except ModuleNotFoundError:
    raise ImportError(
        "Unsloth: Pytorch is not installed. Go to https://pytorch.org/.\n"\
        "We have some installation instructions on our Github page."
    )
except Exception as exception:
    raise exception
pass

@functools.cache
def is_hip():
    return bool(getattr(getattr(torch, "version", None), "hip", None))
pass

@functools.cache
def get_device_type():
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        if is_hip():
            return "hip"
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    raise NotImplementedError("Unsloth currently only works on NVIDIA GPUs and Intel GPUs.")
pass
DEVICE_TYPE : str = get_device_type()

@functools.cache
def get_device_count():
    if DEVICE_TYPE in ("cuda", "hip"):
        return torch.cuda.device_count()
    elif DEVICE_TYPE == "xpu":
        return torch.xpu.device_count()
    else:
        return 1
pass

DEVICE_COUNT : int = get_device_count()

# Reduce VRAM usage by reducing fragmentation
# And optimize pinning of memory
# TODO(billishyahao): need to add hip related optimization...
if (DEVICE_TYPE in ("cuda", "hip")) and (os.environ.get("UNSLOTH_VLLM_STANDBY", "0")=="0"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = \
        "expandable_segments:True,"\
        "roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
elif (DEVICE_TYPE in ("cuda", "hip")) and (os.environ.get("UNSLOTH_VLLM_STANDBY", "0")=="1") and \
    ("expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")):
    warnings.warn(
        "Unsloth: `UNSLOTH_VLLM_STANDBY` is on, but requires `expandable_segments` to be off.\n"\
        "We will remove `expandable_segments`.",
        stacklevel = 2,
    )
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = re.sub(
        r"expandable\_segments\:True\,?",
        "",
        os.environ["PYTORCH_CUDA_ALLOC_CONF"],
    )
pass