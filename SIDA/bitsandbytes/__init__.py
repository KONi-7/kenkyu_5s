"""Local stub for bitsandbytes.

This repo's inference/eval does not require bitsandbytes, but some upstream
libraries (e.g., accelerate) may import it unconditionally when present.
On machines without a working CUDA setup, the real bitsandbytes package can
crash at import time.

By providing this lightweight stub earlier on sys.path (the project root), we
avoid import-time CUDA initialization while keeping optional dependencies inert.
"""

__all__ = []
__version__ = "0.0.0-stub"
