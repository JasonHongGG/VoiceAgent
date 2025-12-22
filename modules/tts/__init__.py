"""TTS module initialization."""

from .base import TTSEngine, TTSResult
from .coqui_tts import CoquiTTS
from .chatterbox_tts import ChatterboxTTS

# VibeVoice is an optional backend with pinned deps in requirements-vibevoice.txt.
# Avoid importing it eagerly so the rest of the package works without those deps.
try:
	from .vibevoice_tts import VibeVoiceTTS  # type: ignore
except Exception:  # pragma: no cover
	VibeVoiceTTS = None  # type: ignore

__all__ = ["TTSEngine", "TTSResult", "CoquiTTS", "ChatterboxTTS", "VibeVoiceTTS"]

