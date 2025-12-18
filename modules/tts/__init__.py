"""TTS module initialization."""

from .base import TTSEngine, TTSResult
from .coqui_tts import CoquiTTS
from .vibevoice_tts import VibeVoiceTTS

__all__ = ["TTSEngine", "TTSResult", "CoquiTTS", "VibeVoiceTTS"]

