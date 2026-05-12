"""GPU detection utility (chonkie handles the embedding model)."""
from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)


def detect_device() -> tuple[str, bool]:
    """Detect available device and whether it has CUDA support.

    Returns:
        (device_name, has_cuda) where device_name is 'cuda' or 'cpu'
    """
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"

    if has_cuda:
        gpu_count = torch.cuda.device_count()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        log.info(
            "GPU detected: %s (%d GPU(s), %.1f GB memory)",
            torch.cuda.get_device_name(0),
            gpu_count,
            gpu_memory,
        )
    else:
        log.info("No GPU available, using CPU")

    return device, has_cuda

