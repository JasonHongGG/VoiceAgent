"""TTS module initialization."""

from .base import TTSEngine, TTSResult
from .coqui_tts import CoquiTTS

__all__ = ["TTSEngine", "TTSResult", "CoquiTTS"]
