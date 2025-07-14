"""Utilities to manage CUDA devices."""

import torch

from a_snlsr.logging import get_logger

logger = get_logger()


def get_gpu_with_most_available_memory() -> int:
    try:
        import pynvml
    except ImportError:
        logger.error(
            "Error in function `get_gpu_with_most_available_memory`: pynvml couldn't be imported. Returning default value 0."
        )
        return 0

    pynvml.nvmlInit()  # Initialize NVML
    device_count = pynvml.nvmlDeviceGetCount()
    max_available_memory = 0
    selected_device = 0

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # Free memory in bytes
        available_memory: int = mem_info.free  # type: ignore

        if available_memory > max_available_memory:
            max_available_memory = available_memory
            selected_device = i

    pynvml.nvmlShutdown()  # Cleanup NVML
    return selected_device


def load_device() -> torch.device:
    """Loads a the available CUDA device with the most available free memory. If not CUDA model is available, returns the CPU device."""
    if torch.cuda.is_available():
        device_nb = get_gpu_with_most_available_memory()
        return torch.device(f"cuda:{device_nb}")
    return torch.device("cpu")
