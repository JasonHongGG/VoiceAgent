"""Bootstrapping for the vendored VibeVoice package.

In this project, VibeVoice code lives under `modules/tts/vibevoice/`.
Some upstream files use absolute imports like `vibevoice.modular...`.
We provide a lightweight alias so those imports resolve without needing
to modify sys.path.
"""

from __future__ import annotations

from pathlib import Path
import sys


def ensure_vibevoice_importable() -> Path:
    """Ensure this package is also importable as top-level `vibevoice`.

    Returns:
        Path to the local vendored VibeVoice package directory.
    """
    # When importing a submodule (e.g. modules.tts.vibevoice.modular.*), Python
    # imports this package first. We can safely alias it for upstream absolute imports.
    pkg = sys.modules.get("modules.tts.vibevoice")
    if pkg is not None:
        sys.modules.setdefault("vibevoice", pkg)
    return Path(__file__).resolve().parent
