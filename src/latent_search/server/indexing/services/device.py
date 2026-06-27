"""Device detection utilities for ML services.

Detects the best available compute device (CUDA GPU > CPU) and exposes
it as a reusable constant so all services agree on placement.

The ``LS_DEVICE`` environment variable overrides auto-detection. Set it
to ``cpu`` or ``cuda`` to force a specific device — useful for testing
the offload pipeline on CPU, or for forcing CPU when the GPU is busy.
"""

import logging
import os

import torch

# Attempt to import DirectML backend if available. It registers a "privateuseone" device.
try:
    import torch_directml  # noqa: F401

    _has_directml = True
except Exception:  # pragma: no cover – optional dependency
    _has_directml = False

logger = logging.getLogger(__name__)

_DEVICE: str | None = None


def get_device() -> str:
    """
    Return the best available device string (e.g. ``'cuda'`` or ``'cpu'``).

    Result is cached after the first call so all services share the same
    decision. The ``LS_DEVICE`` env var (``cpu`` or ``cuda``) overrides
    auto-detection.
    """
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE

    override = os.getenv("LS_DEVICE", "").lower()
    if override in ("cpu", "cuda"):
        _DEVICE = override
        logger.info(f"Device override via LS_DEVICE: {_DEVICE}")
    elif torch.cuda.is_available():
        _DEVICE = "cuda"
        logger.info(f"CUDA available — using GPU ({torch.cuda.get_device_name(0)})")
    elif _has_directml:
        # DirectML registers a privateuseone device (e.g., "privateuseone:0")
        try:
            dev = torch_directml.device()
            _DEVICE = str(dev)
            logger.info(f"DirectML available — using device {_DEVICE}")
        except Exception as exc:  # pragma: no cover
            logger.warning(
                f"DirectML import succeeded but device creation failed: {exc}"
            )
            _DEVICE = "cpu"
    else:
        _DEVICE = "cpu"
        logger.info("CUDA not available — falling back to CPU")

    return _DEVICE
