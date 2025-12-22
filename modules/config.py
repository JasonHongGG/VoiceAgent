"""Configuration and initialization for Voice Agent components."""

import os
import json
from dotenv import load_dotenv

from modules.stt import WhisperSTT
from modules.tts import CoquiTTS
from modules.llm import OllamaLLM
from modules.agent import VoiceAgent
from modules.tools import ToolManager, AccountingAgentWebHook


# Cached prompt configuration loaded from JSON (prompts.json by default)
_PROMPT_CONFIG: dict | None = None


def load_environment():
    """載入環境變數。"""
    load_dotenv()


def load_prompt_config() -> dict:
    """載入提示配置（system prompt、greeting）。"""
    global _PROMPT_CONFIG
    if _PROMPT_CONFIG is not None:
        return _PROMPT_CONFIG

    path = os.getenv("PROMPTS_CONFIG", "prompts.json")
    config = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
                print(f"[Config] Loaded prompts from {path}")
        except Exception as exc:
            print(f"[Config] Failed to load {path}, using defaults: {exc}")

    # Fallback to environment values if JSON missing fields
    config.setdefault(
        "language",
        "zh-cn",
    )
    config.setdefault(
        "system_prompt",
        "你是一個聊天助手，對於使用者的問題提供資訊和建議，請用簡短的繁體中文句子回覆。",
    )
    config.setdefault(
        "greeting_message",
        "你好！我是你的語音助理，有什麼可以幫助你的嗎？"
    )

    _PROMPT_CONFIG = config
    return _PROMPT_CONFIG


def initialize_stt_engine():
    """初始化 STT (Speech-to-Text) 引擎。"""
    return WhisperSTT(
        model_size=os.getenv("STT_MODEL_SIZE", "medium"),
        device=os.getenv("DEVICE", "cuda").lower(),
        beam_size=5,
        vad_filter=False,
    )


def initialize_llm_engine():
    """初始化 LLM (Language Model) 引擎。"""
    prompt_config = load_prompt_config()
    return OllamaLLM(
        api_url=os.getenv("LLM_API_URL"),
        model=os.getenv("LLM_MODEL"),
        default_system_prompt=prompt_config.get("system_prompt"),
        timeout=int(os.getenv("LLM_TIMEOUT", "60")),
    )


def initialize_tts_engine():
    """初始化 TTS (Text-to-Speech) 引擎。"""
    engine = (os.getenv("TTS_ENGINE", "coqui") or "coqui").strip().lower()

    if engine in {"coqui", "xtts", "coqui-tts"}:
        return CoquiTTS(
            model_name=os.getenv("TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2"),
            device=os.getenv("DEVICE", "cuda").lower(),
        )

    if engine in {"vibevoice", "vibevoice-realtime"}:
        # Lazy import 
        from modules.tts import VibeVoiceTTS

        return VibeVoiceTTS(
            model_path=os.getenv("VIBEVOICE_MODEL_PATH", "microsoft/VibeVoice-Realtime-0.5B"),
            device=os.getenv("DEVICE", "cuda").lower(),
            voice=os.getenv("VIBEVOICE_VOICE"),
            cfg_scale=float(os.getenv("VIBEVOICE_CFG_SCALE", "1.5")),
            ddpm_steps=int(os.getenv("VIBEVOICE_DDPM_STEPS", "5")),
            voices_dir=os.getenv("VIBEVOICE_VOICES_DIR"),
        )

    if engine in {"chatterbox", "chatterbox-tts", "chatterbox_multilingual", "chatterbox-multilingual"}:
        # Lazy import (downloads weights from HF on first init)
        from modules.tts import ChatterboxTTS

        return ChatterboxTTS(
            device=os.getenv("DEVICE", "cuda").lower(),
            audio_prompt_path=os.getenv("CHATTERBOX_AUDIO_PROMPT"),
            repetition_penalty=float(os.getenv("CHATTERBOX_REPETITION_PENALTY", "2.0")),
            min_p=float(os.getenv("CHATTERBOX_MIN_P", "0.05")),
            top_p=float(os.getenv("CHATTERBOX_TOP_P", "1.0")),
            temperature=float(os.getenv("CHATTERBOX_TEMPERATURE", "0.8")),
            exaggeration=float(os.getenv("CHATTERBOX_EXAGGERATION", "0.5")),
            cfg_weight=float(os.getenv("CHATTERBOX_CFG_WEIGHT", "0.5")),
        )

    raise ValueError(
        f"Unknown TTS_ENGINE='{engine}'. Supported: coqui, vibevoice, chatterbox"
    )



def initialize_tool_manager():
    """初始化工具管理器並註冊工具。"""
    tool_manager = ToolManager()
    
    # 註冊記帳工具
    webhook_url = os.getenv("ACCOUNT_TOOL_WEBHOOK")
    if webhook_url:
        tool_manager.register_tool(AccountingAgentWebHook(webhook_url))
    
    print(f"[Tools] Registered tools: {tool_manager.list_tools()}")
    return tool_manager


def initialize_voice_agent(
    stt_engine, 
    llm_engine, 
    tts_engine, 
    tool_manager, 
):
    """初始化 Voice Agent。
    
    Args:
        stt_engine: STT 引擎實例
        llm_engine: LLM 引擎實例
        tts_engine: TTS 引擎實例
        tool_manager: 工具管理器實例
    Returns:
        配置好的 VoiceAgent 實例
    """
    return VoiceAgent(
        stt_engine=stt_engine,
        llm_engine=llm_engine,
        tts_engine=tts_engine,
        tool_manager=tool_manager,
        enable_llm=True,
        sentence_delimiters=r'[。！？\.!?;；\n]',
        min_sentence_length=5,
    )


def setup_voice_agent():
    """完整設置 Voice Agent，包含所有依賴元件。
    
    Returns:
        配置好的 VoiceAgent 實例
    """
    load_environment()
    
    stt_engine = initialize_stt_engine()
    llm_engine = initialize_llm_engine()
    tts_engine = initialize_tts_engine()
    tool_manager = initialize_tool_manager()
    
    return initialize_voice_agent(
        stt_engine, 
        llm_engine, 
        tts_engine, 
        tool_manager,
    )
