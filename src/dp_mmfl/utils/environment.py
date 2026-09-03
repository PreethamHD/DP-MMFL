import platform
import sys
import torch

def get_environment_info():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
