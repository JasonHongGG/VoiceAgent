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
from modules.agent import VoiceAgent
from modules.tools import ToolManager, AccountingAgentWebHook

load_dotenv()

# ========== WebRTC/Networking 配置（行動/外網穿透穩定性） ==========
# 允許透過環境變數提供 STUN/TURN 設定，並可在瀏覽器端強制走 TURN (relay)
# 以避免 ngrok/企業網無法轉發 UDP 造成的 ICE/DTLS/SRTP 連線問題。
def _build_ice_servers_from_env():
        """從環境變數產生 ICE 伺服器設定。

        支援：
            - RTC_STUN_URLS: 逗號分隔，例如
                "stun:stun.l.google.com:19302,stun:stun1.l.google.com:19302"
            - RTC_TURN_URL:  TURN 入口，例如
                "turns:turn.example.com:443?transport=tcp" 或 "turn:turn.example.com:3478"
            - RTC_TURN_USERNAME / RTC_TURN_PASSWORD: TURN 認證
        """
        ice_servers = []

        stun_urls = os.getenv("RTC_STUN_URLS")
        if stun_urls:
                urls = [u.strip() for u in stun_urls.split(",") if u.strip()]
                if urls:
                        ice_servers.append({"urls": urls})

        turn_url = os.getenv("RTC_TURN_URL")
        turn_user = os.getenv("RTC_TURN_USERNAME")
        turn_pass = os.getenv("RTC_TURN_PASSWORD")
        if turn_url:
                turn_entry = {"urls": [turn_url]}
                if turn_user and turn_pass:
                        turn_entry["username"] = turn_user
                        turn_entry["credential"] = turn_pass
                ice_servers.append(turn_entry)

        # 若未指定，預設提供 Google STUN 提升存活率
        if not ice_servers:
                ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]

        return ice_servers

RTC_CLIENT_CONFIG = {"iceServers": _build_ice_servers_from_env()}
_ice_policy = os.getenv("RTC_ICE_TRANSPORT_POLICY")  # 例："relay" 強制走 TURN
if _ice_policy:
        # 只會作用在瀏覽器端 (RTCPeerConnection)，server 端 aiortc 以 iceServers 為主
        RTC_CLIENT_CONFIG["iceTransportPolicy"] = _ice_policy

# 伺服器端 aiortc 也可以帶相同 iceServers（雖然 iceTransportPolicy 不適用於 server）
RTC_SERVER_CONFIG = {"iceServers": _build_ice_servers_from_env()}

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

# 4. 初始化工具管理器和記帳工具
tool_manager = ToolManager()
tool_manager.register_tool(AccountingAgentWebHook(os.getenv("ACCOUNT_TOOL_WEBHOOK")))
print(f"[Tools] Registered tools: {tool_manager.list_tools()}")

# 5. 建立流式 Voice Agent（關鍵改變！）
voice_agent = VoiceAgent(
    stt_engine=stt_engine,
    llm_engine=llm_engine,
    tts_engine=tts_engine,
    tool_manager=tool_manager,  # 加入工具管理器
    enable_llm=True,
    sentence_delimiters=r'[。！？\.!?;；\n]',  # 句子分隔符
    min_sentence_length=5,  # 最小句子長度
)

# ========== FastRTC Handler (流式版本) ==========

def echo(audio: Tuple[int, np.ndarray]):
    """
    流式處理音訊輸入並即時返回語音回應。
    LLM 每生成一個句子就立即 TTS，大幅減少延遲
    """
    print("Received audio chunk for streaming processing.")
    
    full_response_text = ""
    
    # 使用流式 VoiceAgent 處理音訊
    # 這會即時 yield 每個句子的音訊，而非等待全部完成
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


# 建立 FastRTC Stream
stream = Stream(
    handler=ReplyOnPause(echo),
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
