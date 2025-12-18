"""VibeVoice integration helpers.

This subpackage exists to keep all VibeVoice-specific bootstrapping and imports
in one place, so the main `VibeVoiceTTS` engine stays minimal.
"""

from __future__ import annotations

# Provide `import vibevoice` compatibility for vendored code.
import sys as _sys
_sys.modules.setdefault("vibevoice", _sys.modules[__name__])

from .voice_presets import VoicePresetMapper

__all__ = [
    "VoicePresetMapper",
]
