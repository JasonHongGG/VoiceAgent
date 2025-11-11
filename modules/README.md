# Modules 目錄說明

本目錄包含語音助理系統的所有模組化元件。

## 目錄結構

```
modules/
├── __init__.py
├── config.py                  # 配置與初始化管理
├── agent.py                   # 主要 Voice Agent 邏輯
├── llm/                       # 語言模型相關
│   ├── base.py
│   └── ollama_llm.py
├── stt/                       # 語音轉文字
│   ├── base.py
│   └── whisper_stt.py
├── tts/                       # 文字轉語音
│   ├── base.py
│   └── coqui_tts.py
├── tools/                     # Agent 工具
│   ├── base.py
│   ├── manager.py
│   └── accounting_tool.py
└── utils/                     # 工具函數
    ├── audio_utils.py         # 音訊處理工具
    └── rtc_config.py          # WebRTC/STUN/TURN 配置
```

## 主要模組說明

### `config.py`
負責系統配置和元件初始化：
- `load_environment()`: 載入環境變數
- `initialize_stt_engine()`: 初始化 STT 引擎
- `initialize_llm_engine()`: 初始化 LLM 引擎
- `initialize_tts_engine()`: 初始化 TTS 引擎
- `initialize_tool_manager()`: 初始化工具管理器
- `setup_voice_agent()`: 一鍵設置完整的 Voice Agent

### `utils/rtc_config.py`
WebRTC 網路配置管理：
- `build_ice_servers_from_env()`: 從環境變數建立 ICE 伺服器配置
- `get_client_rtc_config()`: 取得客戶端 RTC 配置
- `get_server_rtc_config()`: 取得伺服器端 RTC 配置

支援的環境變數：
- `RTC_STUN_URLS`: STUN 伺服器 URL（逗號分隔）
- `RTC_TURN_URL`: TURN 伺服器 URL
- `RTC_TURN_USERNAME`: TURN 認證用戶名
- `RTC_TURN_PASSWORD`: TURN 認證密碼
- `RTC_ICE_TRANSPORT_POLICY`: ICE 傳輸策略（如 "relay"）

### `utils/audio_utils.py`
音訊處理工具函數：
- `to_mono_and_normalize()`: 音訊轉單聲道並正規化

## 使用範例

```python
from modules.config import setup_voice_agent
from modules.utils.rtc_config import get_client_rtc_config

# 初始化 Voice Agent（自動載入所有元件）
voice_agent = setup_voice_agent()

# 取得 WebRTC 配置
rtc_config = get_client_rtc_config()
```

## 設計原則

1. **模組化**：每個功能都有獨立的模組，易於維護和測試
2. **可擴展**：透過基礎類別設計，方便新增不同的實作
3. **配置驅動**：使用環境變數管理配置，提高靈活性
4. **關注點分離**：主應用檔案專注於業務邏輯，配置和初始化邏輯分離
