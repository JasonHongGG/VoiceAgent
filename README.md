# VoiceAgent 🎙️

模組化的語音助理框架，整合 STT（語音辨識）、LLM（語言模型）和 TTS（語音合成）。

## ✨ 特色

- 🎯 **統一 API**：一個 `VoiceAgent` 類別支援批次和串流兩種模式
- ⚡ **即時回應**：串流模式下 2-5 秒內開始回應（預設模式）
- � **自動歡迎**：WebRTC 連接建立時自動打招呼
- �🔧 **模組化設計**：輕鬆替換 STT、LLM、TTS 引擎
- 🛠️ **工具系統**：支援 LLM 調用外部工具（記帳、查詢等）
- 🌐 **WebRTC 支援**：完整的 STUN/TURN 配置，穩定的網路穿透
- 📦 **開箱即用**：完整的範例和文件
- 🔄 **框架無關**：可在 FastRTC 或其他框架中使用

## 🚀 快速開始

### 安裝依賴

```bash
pip install -r requiremenet.txt
```

### 基本使用（推薦）

```python
from modules import VoiceAgent, WhisperSTT, OllamaLLM, CoquiTTS

# 建立 Agent（使用預設串流模式）
agent = VoiceAgent(
    stt=WhisperSTT(),
    llm=OllamaLLM(),
    tts=CoquiTTS(),
)

# 處理文字輸入（串流模式）
for tts_result, sentence in agent.process_text("你好，今天天氣如何？"):
    print(f"回應: {sentence}")
    # 播放 tts_result.audio
```

### 批次模式

```python
# 如需等待完整回應，使用批次模式
agent = VoiceAgent(
    stt=WhisperSTT(),
    llm=OllamaLLM(),
    tts=CoquiTTS(),
    enable_streaming=False,  # 批次模式
)

# 處理文字輸入（批次模式）
tts_result = agent.process_text("你好")
```

## 📊 模式對比

| 特性 | 串流模式（預設） | 批次模式 |
|------|-----------------|----------|
| 首次回應時間 | 2-5 秒 ⚡ | 15-40 秒 |
| 使用者體驗 | 流暢即時 | 需等待完整回應 |
| 適用場景 | 對話、即時互動 | 音訊錄製、分析 |

## 🛠️ 工具系統

支援 LLM 調用外部工具擴展功能：

```python
from modules import VoiceAgent
from modules.tools import ToolManager, AccountingAgentWebHook

# 建立工具管理器
tool_manager = ToolManager()
tool_manager.register_tool(AccountingAgentWebHook())

# 建立帶工具的 Agent
agent = VoiceAgent(
    stt=WhisperSTT(),
    llm=OllamaLLM(),
    tts=CoquiTTS(),
    tool_manager=tool_manager,
)

# 使用（自動調用記帳工具）
for tts_result, sentence in agent.process_text("我今天買了咖啡花了50元，幫我記帳"):
    print(sentence)
```

## 📁 專案結構

```
fastRTC/
├── modules/                   # 核心模組
│   ├── stt/                  # 語音辨識
│   │   ├── base.py          # STT 基礎介面
│   │   └── whisper_stt.py   # Whisper 實作
│   ├── llm/                  # 語言模型
│   │   ├── base.py          # LLM 基礎介面
│   │   └── ollama_llm.py    # Ollama 實作
│   ├── tts/                  # 語音合成
│   │   ├── base.py          # TTS 基礎介面
│   │   └── coqui_tts.py     # Coqui 實作
│   ├── tools/                # 工具系統
│   │   ├── base.py          # 工具基礎介面
│   │   ├── manager.py       # 工具管理器
│   │   └── accounting_tool.py  # 記帳工具
│   ├── agent.py              # 統一的 VoiceAgent
│   └── utils/                # 工具函數
├── app.py                     # FastRTC 應用（批次模式）
├── app_streaming.py           # FastRTC 應用（串流模式）
├── demo_unified_agent.py      # 統一 Agent 展示
└── demo_tools.py              # 工具系統展示
```

## 📚 文件

- **[UNIFIED_AGENT.md](UNIFIED_AGENT.md)** - 統一 Agent 完整說明 ⭐ 必讀
- **[TOOLS_GUIDE.md](TOOLS_GUIDE.md)** - 工具系統使用指南
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - 架構設計說明
- **[QUICKREF.md](QUICKREF.md)** - 快速參考指南

## 🎯 範例

### 1. 基本對話

```python
from modules import VoiceAgent, WhisperSTT, OllamaLLM, CoquiTTS

agent = VoiceAgent(
    stt=WhisperSTT(),
    llm=OllamaLLM(),
    tts=CoquiTTS(),
)

for tts_result, sentence in agent.process_text("介紹一下你自己"):
    print(sentence)
```

### 2. 處理語音輸入

```python
import soundfile as sf

# 讀取音訊
audio_data, sample_rate = sf.read("input.wav")

# 處理
for tts_result, sentence in agent.process_audio((sample_rate, audio_data)):
    print(sentence)
    # 播放回應
```

### 3. 建立自訂工具

```python
from modules.tools import BaseTool, ToolParameter, ToolResult

class WeatherTool(BaseTool):
    @property
    def name(self) -> str:
        return "weather_query"
    
    @property
    def description(self) -> str:
        return "查詢城市天氣"
    
    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="city",
                type="string",
                description="城市名稱",
                required=True,
            )
        ]
    
    def execute(self, **kwargs) -> ToolResult:
        city = kwargs.get("city")
        # 呼叫天氣 API...
        return ToolResult(success=True, data=f"{city}的天氣...")

# 註冊並使用
tool_manager.register_tool(WeatherTool())
```

## 🔧 配置

### STT（語音辨識）

```python
from modules import WhisperSTT

stt = WhisperSTT(
    model_size="medium",  # tiny, base, small, medium, large
    device="cuda",         # cuda 或 cpu
    compute_type="float16",
)
```

### LLM（語言模型）

```python
from modules import OllamaLLM

llm = OllamaLLM(
    api_url="http://localhost:11434",
    model="llama3:8b",
)
```

### TTS（語音合成）

```python
from modules import CoquiTTS

tts = CoquiTTS(
    device="cuda",  # cuda 或 cpu
    reference_speaker="path/to/speaker.wav",  # 可選
)
```

## 🚀 執行範例

### 基本展示

```bash
python demo_unified_agent.py
```

### 工具系統展示

```bash
python demo_tools.py
```

### FastRTC 應用

```bash
# 串流模式（推薦）- 支援 STUN/TURN 和自動歡迎
python app_turn.py

# 批次模式
python app.py
```

### 環境變數配置

複製 `.env.example` 到 `.env` 並根據需求修改：

```bash
cp .env.example .env
```

**重要配置項：**

```bash
# 歡迎語設定（當用戶連接時自動播放）
GREETING_MESSAGE=你好！我是你的語音助理，有什麼可以幫助你的嗎？

# WebRTC 配置（提升連接穩定性）
RTC_STUN_URLS=stun:stun.l.google.com:19302
# RTC_TURN_URL=turns:turn.example.com:443?transport=tcp
# RTC_TURN_USERNAME=username
# RTC_TURN_PASSWORD=password
```

## 💡 何時使用哪種模式？

### 串流模式（預設，推薦）

✅ **適合：**
- 即時對話應用
- 語音助理
- 客服機器人
- 需要快速回應的場景

### 批次模式

✅ **適合：**
- 需要完整音訊檔案
- 音訊後處理或分析
- 錄製完整回應
- 離線處理

## 🤝 貢獻

歡迎貢獻！可以：

1. 新增其他 STT/LLM/TTS 引擎的實作
2. 建立更多工具
3. 改進文件
4. 回報 Bug

## 📝 授權

MIT License

## 🙏 致謝

- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) - STT 引擎
- [Coqui TTS](https://github.com/coqui-ai/TTS) - TTS 引擎  
- [Ollama](https://ollama.ai/) - LLM 服務
- [FastRTC](https://github.com/gptlink/fastrtc) - WebRTC 框架

