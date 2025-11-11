"""Configuration and initialization for Voice Agent components."""

import os
from dotenv import load_dotenv

from modules.stt import WhisperSTT
from modules.tts import CoquiTTS
from modules.llm import OllamaLLM
from modules.agent import VoiceAgent
from modules.tools import ToolManager, AccountingAgentWebHook


def load_environment():
    """載入環境變數。"""
    load_dotenv()


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
    return OllamaLLM(
        api_url=os.getenv("LLM_API_URL"),
        model=os.getenv("LLM_MODEL"),
        default_system_prompt=os.getenv("LLM_SYSTEM_PROMPT"),
        timeout=int(os.getenv("LLM_TIMEOUT", "60")),
    )


def initialize_tts_engine():
    """初始化 TTS (Text-to-Speech) 引擎。"""
    return CoquiTTS(
        model_name=os.getenv("TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2"),
        device=os.getenv("DEVICE", "cuda").lower(),
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


def initialize_voice_agent(stt_engine, llm_engine, tts_engine, tool_manager):
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
    
    return initialize_voice_agent(stt_engine, llm_engine, tts_engine, tool_manager)
