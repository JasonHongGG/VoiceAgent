"""Voice Agent Modules - Modular STT, TTS, and LLM components."""

# Base interfaces
from .stt.base import STTEngine, TranscriptionResult
from .tts.base import TTSEngine, TTSResult
from .llm.base import LLMEngine, LLMResponse

# Implementations
from .stt.whisper_stt import WhisperSTT
from .tts.coqui_tts import CoquiTTS
from .llm.ollama_llm import OllamaLLM

# High-level agent
from .agent import VoiceAgent

# Tool system
from .tools.base import BaseTool, ToolParameter, ToolResult
from .tools.manager import ToolManager
from .tools.accounting_tool import AccountingAgentWebHook

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
    "WhisperSTT",
    "CoquiTTS",
    "OllamaLLM",
    # Agent
    "VoiceAgent",
    # Tools
    "BaseTool",
    "ToolParameter",
    "ToolResult",
    "ToolManager",
    "AccountingAgentWebHook",
    # Utils
    "to_mono_and_normalize",
]
