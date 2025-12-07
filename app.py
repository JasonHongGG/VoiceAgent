"""流式語音助理應用 - 支援 WebRTC STUN/TURN 配置。"""

import os
from typing import Tuple
import numpy as np
import gradio as gr

from fastrtc import Stream, ReplyOnPause
from fastrtc.utils import AdditionalOutputs

from modules.config import setup_voice_agent
from modules.utils.rtc_config import get_client_rtc_config, get_server_rtc_config

# ========== WebRTC 配置 ==========
RTC_CLIENT_CONFIG = get_client_rtc_config()
RTC_SERVER_CONFIG = get_server_rtc_config()

# ========== 初始化 Voice Agent ==========
voice_agent = setup_voice_agent()

# 預先合成歡迎語音，避免連線後再等待 TTS 啟動
GREETING_TEXT = os.getenv("GREETING_MESSAGE", "你好！我是你的語音助理，有什麼可以幫助你的嗎？")
PRECOMPUTED_GREETING = None
try:
    PRECOMPUTED_GREETING = voice_agent.synthesize_speech(
        text=GREETING_TEXT,
        language="zh",
    )
    print(f"[Greeting] Warmed up greeting audio: {len(PRECOMPUTED_GREETING.audio)} samples")
except Exception as exc:
    print(f"[Greeting] Warmup failed, will synthesize on demand: {exc}")

# ========== FastRTC Handler ==========

def greet_user():
    """
    啟動時的歡迎函數，會在 WebRTC 連接建立時自動執行。
    直接使用 TTS，不經過 LLM，避免無限循環。
    """
    greeting_text = GREETING_TEXT
    print(f"[Greeting] Sending welcome message: '{greeting_text}'")

    try:
        tts_result = PRECOMPUTED_GREETING 
        print(f"[Greeting] TTS generated {len(tts_result.audio)} samples")
        yield tts_result.as_tuple(), AdditionalOutputs(greeting_text)
    except Exception as e:
        print(f"[Greeting] Error generating greeting: {e}")
        import traceback
        traceback.print_exc()


def echo(audio: Tuple[int, np.ndarray]):
    """
    流式處理音訊輸入並即時返回語音回應。
    LLM 每生成一個句子就立即 TTS，大幅減少延遲。
    """
    print("Received audio chunk for streaming processing.")
    
    full_response_text = ""
    
    # 使用流式 VoiceAgent 處理音訊
    for tts_result, sentence in voice_agent.process_audio(audio):
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


# 建立 FastRTC Stream，使用 startup_fn 實現自動歡迎
stream = Stream(
    handler=ReplyOnPause(
        fn=echo,                    # 主要的音訊處理函數
        startup_fn=greet_user,      # 啟動時自動執行的歡迎函數 🎯
    ),
    modality="audio",
    mode="send-receive",
    rtc_configuration=RTC_CLIENT_CONFIG,
    server_rtc_configuration=RTC_SERVER_CONFIG,
    additional_outputs_handler=update_transcript,
    additional_outputs=[transcript_box],
)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("流式語音助理 - 即時回應模式")
    print("="*60)
    
    # 可用環境變數指定 HOST/PORT 與 SSL 憑證；行動裝置音訊裝置切換在 HTTPS 下更穩定
    server_name = os.getenv("HOST", "0.0.0.0")
    server_port = int(os.getenv("PORT", "5000"))
    ssl_certfile = os.getenv("SSL_CERTFILE")
    ssl_keyfile = os.getenv("SSL_KEYFILE")

    launch_kwargs = {"server_name": server_name, "server_port": server_port, "share": True}
    if ssl_certfile and ssl_keyfile:
        launch_kwargs["ssl_certfile"] = ssl_certfile
        launch_kwargs["ssl_keyfile"] = ssl_keyfile

    stream.ui.launch(**launch_kwargs)
