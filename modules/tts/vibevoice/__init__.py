"""VibeVoice integration helpers.

This subpackage exists to keep all VibeVoice-specific bootstrapping and imports
in one place, so the main `VibeVoiceTTS` engine stays minimal.
"""

from __future__ import annotations

# Provide `import vibevoice` compatibility for vendored code.
import sys as _sys

_sys.modules.setdefault("vibevoice", _sys.modules[__name__])

from .bootstrap import ensure_vibevoice_importable
from .imports import get_vibevoice_classes
from .voice_presets import VoicePresetMapper

__all__ = [
    "ensure_vibevoice_importable",
    "get_vibevoice_classes",
    "VoicePresetMapper",
]
