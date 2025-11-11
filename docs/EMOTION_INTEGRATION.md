# 🎭 情感控制整合指南

## 📋 概述

本指南說明如何在 Voice Agent 中整合情感控制功能，讓每句話都能自動使用正確的參考音訊和 TTS 參數。

## 🏗️ 架構設計

### 核心組件

1. **EmotionManager** - 管理情感音訊和參數
   - 載入情感參考音訊 (speaker_wav)
   - 提供預設的情感參數配置
   - 自動偵測文字中的情感

2. **VoiceAgent** - 整合情感控制
   - 在串流模式下逐句應用情感
   - 自動或手動選擇情感
   - 組合參考音訊和參數

3. **CoquiTTS** - 執行情感化 TTS
   - 接受 speaker_wav 參數
   - 接受 temperature, speed 等參數
   - 合成帶有情感的語音

## 🎯 使用方式

### 方法 1: 啟用自動情感控制（推薦）

在 `app_turn.py` 中啟用情感控制：

```python
from modules.config import setup_voice_agent

# 啟用自動情感控制
voice_agent = setup_voice_agent(enable_emotion_control=True)
```

這樣設定後，Voice Agent 會：
1. 自動從每句話中偵測情感（基於關鍵字）
2. 選擇對應的參考音訊（如果存在）
3. 應用對應的 TTS 參數（temperature, speed 等）

### 方法 2: 手動指定情感

如果您想手動控制某句話的情感：

```python
# 在 VoiceAgent 中直接指定情感
emotion_config = voice_agent.emotion_manager.get_emotion_config(
    emotion="happy",  # 手動指定情感
    text="今天真開心！"
)

tts_result = voice_agent.tts.synthesize(
    text="今天真開心！",
    language="zh-cn",
    **emotion_config  # 展開情感配置
)
```

### 方法 3: 僅使用參數（不用參考音訊）

如果您沒有錄製參考音訊，也可以只使用參數：

```python
tts_result = voice_agent.tts.synthesize(
    text="今天真開心！",
    language="zh-cn",
    temperature=1.0,  # 提高情感表達
    speed=1.1,        # 稍微加快
)
```

## 📁 目錄結構

```
VoiceAgent/
├── modules/
│   ├── agent.py                    # VoiceAgent (整合情感控制)
│   ├── tts/
│   │   └── coqui_tts.py           # CoquiTTS (支援情感參數)
│   ├── utils/
│   │   └── emotion_manager.py     # EmotionManager (情感管理)
│   └── config.py                   # 配置和初始化
├── resource/
│   └── emotions/                   # 情感參考音訊目錄
│       ├── happy.wav              # 開心的參考音訊
│       ├── sad.wav                # 悲傷的參考音訊
│       ├── neutral.wav            # 中性的參考音訊
│       └── ...                    # 其他情感
└── app_turn.py                     # 主應用程式
```

## 🎨 情感配置

### 預設支援的情感

EmotionManager 預設支援以下情感配置：

| 情感 | Temperature | Speed | 說明 |
|------|-------------|-------|------|
| **neutral** | 0.4 | 1.0 | 中性、播報 |
| **happy** | 1.0 | 1.1 | 開心、友善 |
| **excited** | 1.1 | 1.2 | 興奮、熱情 |
| **sad** | 0.7 | 0.85 | 悲傷、同情 |
| **angry** | 0.9 | 1.15 | 生氣、憤怒 |
| **gentle** | 0.65 | 0.9 | 溫柔、安慰 |
| **professional** | 0.5 | 0.95 | 專業、正式 |

### 自動情感偵測規則

基於文字中的關鍵字自動偵測：

```python
emotion_keywords = {
    "happy": ["開心", "快樂", "太好了", "太棒了", "哈哈", "😊", "😄", "🎉"],
    "excited": ["興奮", "激動", "驚喜", "哇", "😍", "🤩"],
    "sad": ["難過", "傷心", "遺憾", "可惜", "😢", "😭"],
    "angry": ["生氣", "憤怒", "可惡", "😠", "😡"],
    "gentle": ["溫柔", "輕聲", "別擔心", "沒關係", "安慰"],
    "professional": ["報告", "數據", "分析", "根據", "顯示"],
}
```

## 🔧 環境變數設定

在 `.env` 中添加：

```bash
# 情感音訊目錄
EMOTION_AUDIO_DIR=resource/emotions

# 如果想為所有語音設定預設參考音訊
TTS_SPEAKER_WAV=resource/emotions/neutral.wav
```

## 💡 實際範例

### 範例 1: 基本使用（自動情感）

```python
# app_turn.py
from modules.config import setup_voice_agent

# 啟用自動情感控制
voice_agent = setup_voice_agent(enable_emotion_control=True)

# 在串流處理中，每句話會自動應用情感
# 例如：
# "太好了！" -> 偵測到 "happy" -> 應用開心的參考音訊和參數
# "很遺憾..." -> 偵測到 "sad" -> 應用悲傷的參考音訊和參數
```

### 範例 2: 進階控制

```python
from modules.config import (
    initialize_stt_engine,
    initialize_llm_engine,
    initialize_tts_engine,
    initialize_tool_manager,
    initialize_emotion_manager,
    initialize_voice_agent,
)

# 分別初始化各個組件
stt_engine = initialize_stt_engine()
llm_engine = initialize_llm_engine()
tts_engine = initialize_tts_engine()
tool_manager = initialize_tool_manager()
emotion_manager = initialize_emotion_manager()

# 自訂情感參數
emotion_manager.DEFAULT_EMOTION_PARAMS["happy"]["temperature"] = 1.2
emotion_manager.DEFAULT_EMOTION_PARAMS["happy"]["speed"] = 1.3

# 添加自訂情感
emotion_manager.add_emotion("cheerful", "resource/emotions/cheerful.wav")
emotion_manager.DEFAULT_EMOTION_PARAMS["cheerful"] = {
    "temperature": 1.15,
    "speed": 1.25,
    "repetition_penalty": 7.0,
    "top_p": 0.95,
}

# 初始化 Voice Agent
voice_agent = initialize_voice_agent(
    stt_engine,
    llm_engine,
    tts_engine,
    tool_manager,
    emotion_manager,
    enable_emotion_control=True
)
```

### 範例 3: 手動控制特定句子的情感

如果您想在程式中動態控制某些特定句子的情感：

```python
# 修改 VoiceAgent 的 _stream_llm_and_tts 方法
# 或在呼叫 TTS 時手動指定

# 例如：問候語總是用友善的語氣
greeting_config = emotion_manager.get_emotion_config(emotion="happy")
greeting_audio = tts_engine.synthesize(
    text="你好！我是你的語音助理！",
    language="zh-cn",
    **greeting_config
)

# 錯誤訊息用溫柔的語氣
error_config = emotion_manager.get_emotion_config(emotion="gentle")
error_audio = tts_engine.synthesize(
    text="抱歉，我沒有聽清楚，可以再說一次嗎？",
    language="zh-cn",
    **error_config
)
```

## 🎬 工作流程

### 串流模式下的情感控制流程

```
用戶說話
  ↓
STT 轉錄
  ↓
LLM 流式生成回應
  ↓
每產生一個完整句子
  ↓
EmotionManager 分析句子
  ├─ 偵測情感（關鍵字匹配）
  ├─ 選擇參考音訊
  └─ 選擇 TTS 參數
  ↓
CoquiTTS 合成（帶情感）
  ↓
串流回傳音訊給用戶
```

## 📝 準備參考音訊

### 錄製建議

1. **錄製環境**
   - 安靜的環境
   - 使用良好的麥克風
   - 避免背景噪音

2. **錄製內容**
   - 每種情感錄製 5-10 秒
   - 說自然的句子（不是單字）
   - 保持情感一致

3. **技術規格**
   - 格式：WAV
   - 採樣率：22050 Hz 或更高
   - 單聲道
   - 位元深度：16-bit

4. **範例句子**
   - **happy**: "今天天氣真好！我們一起出去玩吧！"
   - **sad**: "很遺憾聽到這個消息，我能理解你的感受..."
   - **neutral**: "今日天氣預報，多雲，溫度攝氏二十五度。"
   - **gentle**: "別擔心，一切都會好起來的，慢慢來就好。"

### 使用工具錄製

```bash
# 使用 ffmpeg 錄製（5 秒）
ffmpeg -f alsa -i default -t 5 -ar 22050 -ac 1 resource/emotions/happy.wav

# 或使用 Audacity（圖形介面）
# 1. 開啟 Audacity
# 2. 點擊錄音按鈕
# 3. 說出帶有情感的句子
# 4. 停止錄音
# 5. 檔案 -> 匯出 -> 匯出為 WAV
# 6. 儲存到 resource/emotions/
```

## 🐛 除錯技巧

### 檢查情感是否正確載入

```python
from modules.utils.emotion_manager import EmotionManager

emotion_mgr = EmotionManager()
print("Available emotions:", emotion_mgr.list_emotions())

# 測試情感偵測
text = "今天真開心！"
emotion = emotion_mgr.detect_emotion_from_text(text)
print(f"Detected emotion: {emotion}")

# 測試完整配置
config = emotion_mgr.get_emotion_config(text=text, auto_detect=True)
print(f"Emotion config: {config}")
```

### 檢查 TTS 是否正確接收參數

啟用 VoiceAgent 後，觀察終端輸出：

```
[EmotionManager] Using emotion: happy
[EmotionManager] Selected emotion 'happy': resource/emotions/happy.wav
[VoiceAgent] Emotion config: {'speaker_wav': 'resource/emotions/happy.wav', 'temperature': 1.0, 'speed': 1.1, ...}
[CoquiTTS] Using dynamic speaker_wav: resource/emotions/happy.wav
```

## ⚙️ 效能考量

### 對串流延遲的影響

- **情感偵測**: 非常快（正則表達式匹配），<1ms
- **參考音訊載入**: 首次載入較慢，之後快取
- **TTS 合成**: 參數調整不影響速度，參考音訊可能增加 10-20% 時間

### 優化建議

1. **預載入參考音訊**: EmotionManager 啟動時載入所有音訊
2. **快取情感配置**: 相同文字不重複偵測
3. **批次處理**: 如果不需要即時回應，可批次合成

## 🎓 最佳實踐

1. **不要過度依賴自動偵測**
   - 關鍵句子手動指定情感
   - 使用自動偵測作為備案

2. **參考音訊 + 參數組合使用**
   - 參考音訊控制基礎風格
   - 參數微調細節表達

3. **為不同場景創建配置**
   - 客服場景：溫柔、專業
   - 故事講述：豐富、有表現力
   - 新聞播報：中性、穩定

4. **定期測試和調整**
   - A/B 測試不同配置
   - 收集用戶反饋
   - 持續優化參數

## 📚 相關文檔

- [TTS_PARAMETERS.md](TTS_PARAMETERS.md) - TTS 參數詳細說明
- [EMOTION_CONTROL.md](EMOTION_CONTROL.md) - 情感控制完整指南
- [demo_tts_parameters.py](../demo_tts_parameters.py) - 參數測試腳本

## ❓ 常見問題

### Q: 沒有參考音訊可以使用情感控制嗎？

**A**: 可以！即使沒有參考音訊，仍然可以使用 temperature、speed 等參數控制情感表達。

### Q: 如何關閉自動情感控制？

**A**: 在 `setup_voice_agent()` 時設定 `enable_emotion_control=False`（預設即為 False）。

### Q: 可以同時使用多個參考音訊嗎？

**A**: 不行，每次只能使用一個參考音訊。但您可以創建混合音訊檔案。

### Q: 情感偵測不準確怎麼辦？

**A**: 
1. 擴充關鍵字字典
2. 手動指定重要句子的情感
3. 使用更複雜的情感分析模型（需自行整合）

### Q: 參數調整對所有語言都有效嗎？

**A**: 是的，temperature、speed 等參數對所有 XTTS 支援的語言都有效。
