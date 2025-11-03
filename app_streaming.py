import os
from typing import Tuple
import numpy as np
import gradio as gr

from fastrtc import Stream, ReplyOnPause
from fastrtc.utils import AdditionalOutputs
from dotenv import load_dotenv

# 導入模組化的元件
from modules.stt import WhisperSTT
from modules.tts import CoquiTTS
from modules.llm import OllamaLLM
from modules.streaming_agent import StreamingVoiceAgent

load_dotenv()

# ========== 初始化模組化元件 ==========

# 1. 初始化 STT (Speech-to-Text)
stt_engine = WhisperSTT(
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

# 4. 建立流式 Voice Agent（關鍵改變！）
voice_agent = StreamingVoiceAgent(
    stt_engine=stt_engine,
    llm_engine=llm_engine,
    tts_engine=tts_engine,
    enable_llm=True,
    sentence_delimiters=r'[。！？\.!?;；\n]',  # 句子分隔符
    min_sentence_length=5,  # 最小句子長度
)

# ========== FastRTC Handler (流式版本) ==========

def echo(audio: Tuple[int, np.ndarray]):
    """
    流式處理音訊輸入並即時返回語音回應。
    
    與原版差異：
    - 原版：等待完整的 LLM 回應後才開始 TTS
    - 流式版：LLM 每生成一個句子就立即 TTS，大幅減少延遲
    """
    print("Received audio chunk for streaming processing.")
    
    full_response_text = ""
    
    # 使用流式 VoiceAgent 處理音訊
    # 這會即時 yield 每個句子的音訊，而非等待全部完成
    for tts_result, sentence in voice_agent.process_audio_stream(audio):
        full_response_text += sentence
        
        print(f"[Streaming] Yielding sentence: '{sentence[:50]}...'")
        
        # 立即返回這個句子的音訊和目前累積的文字
        yield tts_result.as_tuple(), AdditionalOutputs(full_response_text)
    
    # 如果沒有生成任何內容
    if not full_response_text:
        yield AdditionalOutputs("未偵測到語音內容。")

# ========== Gradio UI 設定 ==========

transcript_box = gr.Textbox(label="Response (Streaming)", lines=6)


def update_transcript(current_text: str, new_text: str):
    """更新轉錄文字框（會持續更新顯示流式生成的內容）。"""
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
    print("\n" + "="*60)
    print("🚀 流式語音助理 - 即時回應模式")
    print("="*60)
    print("特色：")
    print("✅ LLM 流式生成（逐字輸出）")
    print("✅ 智能分句處理")
    print("✅ 即時 TTS 合成")
    print("✅ 邊生成邊播放，大幅減少延遲")
    print("="*60 + "\n")
    
    stream.ui.launch()
