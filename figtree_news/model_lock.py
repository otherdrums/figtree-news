"""Shared lock for the local GPU model.

The 3.64 GB GPU only holds one 3.4 GB model. All forward passes (ingestion,
generation, decomposition, summarization) must be serialized to avoid CUDA
stream interleaving and to prevent ingestion hooks from capturing hidden states
from concurrent generation/decode passes.
"""

from __future__ import annotations

import threading

model_lock = threading.RLock()


def locked_forward(func):
    """Decorator that wraps a function in the shared model lock.

    For use with functions that perform model forward/generation passes.
    """

    def wrapper(*args, **kwargs):
        with model_lock:
            return func(*args, **kwargs)

    return wrapper
