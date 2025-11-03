import os
from typing import Tuple
import numpy as np
import gradio as gr

from fastrtc import Stream, ReplyOnPause
from fastrtc.utils import AdditionalOutputs
from dotenv import load_dotenv

# 導入模組化的元件
from modules.stt import FasterWhisperSTT
from modules.tts import CoquiTTS
from modules.llm import OllamaLLM
from modules.agent import VoiceAgent

load_dotenv()

# ========== 初始化模組化元件 ==========

# 1. 初始化 STT (Speech-to-Text)
stt_engine = FasterWhisperSTT(
    model_size=os.getenv("STT_MODEL_SIZE", "medium"),
    device=os.getenv("DEVICE", "cuda").lower(),
    beam_size=5,
    vad_filter=False,
)

# 2. 初始化 LLM (Language Model)
llm_engine = OllamaLLM(
    api_url=os.getenv("LLM_API_URL"),
    model=os.getenv("LLM_MODEL"),
    default_system_prompt=os.getenv("LLM_SYSTEM_PROMPT"),
    timeout=int(os.getenv("LLM_TIMEOUT", "60")),
)

# 3. 初始化 TTS (Text-to-Speech)
tts_engine = CoquiTTS(
    model_name=os.getenv("TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2"),
    device=os.getenv("DEVICE", "cuda").lower(),
)

# 4. 建立 Voice Agent
voice_agent = VoiceAgent(
    stt_engine=stt_engine,
    llm_engine=llm_engine,
    tts_engine=tts_engine,
)

# ========== FastRTC Handler ==========

def echo(audio: Tuple[int, np.ndarray]):
    """處理音訊輸入並返回語音回應。"""
    print("Received audio chunk for processing.")
    
    # 使用 VoiceAgent 處理音訊
    tts_result, transcription, llm_response = voice_agent.process_audio(
        audio=audio,
        return_transcript=True,
    )
    
    # 準備回應文字
    if transcription and transcription.text:
        response_text = transcription.text
        if llm_response:
            response_text = f"User: {transcription.text}\nAI: {llm_response.content}"
    else:
        response_text = "未偵測到語音內容。"
    
    # 返回 TTS 結果和文字
    if tts_result:
        yield tts_result.as_tuple(), AdditionalOutputs(response_text)
    else:
        yield AdditionalOutputs(response_text)

# ========== Gradio UI 設定 ==========

transcript_box = gr.Textbox(label="Transcript", lines=6)


def update_transcript(current_text: str, new_text: str):
    """更新轉錄文字框。"""
    return new_text


# 建立 FastRTC Stream
stream = Stream(
    handler=ReplyOnPause(echo),
    modality="audio",
    mode="send-receive",
    additional_outputs_handler=update_transcript,
    additional_outputs=[transcript_box],
)

if __name__ == "__main__":
    stream.ui.launch()
