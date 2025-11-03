"""Voice Agent Modules - Modular STT, TTS, and LLM components."""

# Base interfaces
from .stt.base import STTEngine, TranscriptionResult
from .tts.base import TTSEngine, TTSResult
from .llm.base import LLMEngine, LLMResponse

# Implementations
from .stt.fasterwhisper_stt import FasterWhisperSTT
from .tts.coqui_tts import CoquiTTS
from .llm.ollama_llm import OllamaLLM

# High-level agent
from .agent import VoiceAgent

# Utilities
from .utils.audio_utils import to_mono_and_normalize

__all__ = [
    # Base interfaces
    "STTEngine",
    "TTSEngine", 
    "LLMEngine",
    "TranscriptionResult",
    "TTSResult",
    "LLMResponse",
    # Implementations
    "FasterWhisperSTT",
    "CoquiTTS",
    "OllamaLLM",
    # Agent
    "VoiceAgent",
    # Utils
    "to_mono_and_normalize",
]
