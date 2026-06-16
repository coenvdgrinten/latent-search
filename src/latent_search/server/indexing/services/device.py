"""Device detection utilities for ML services.

Detects the best available compute device (CUDA GPU > CPU) and exposes
it as a reusable constant so all services agree on placement.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_DEVICE: str | None = None


def get_device() -> str:
    """
    Return the best available device string (e.g. ``'cuda'`` or ``'cpu'``).

    Result is cached after the first call so all services share the same
    decision.
    """
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE

    if torch.cuda.is_available():
        _DEVICE = "cuda"
        logger.info(f"CUDA available — using GPU ({torch.cuda.get_device_name(0)})")
    else:
        _DEVICE = "cpu"
        logger.info("CUDA not available — falling back to CPU")

    return _DEVICE
