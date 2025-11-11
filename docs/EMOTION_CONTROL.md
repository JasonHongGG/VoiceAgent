# TTS 情感控制指南

## 📋 概述

由於 Coqui Studio 的 `emotion` 參數已被棄用，我們使用 **Voice Cloning（聲音克隆）** 技術來控制 TTS 的情感和語氣。

## 🎭 工作原理

通過提供帶有特定情感的**參考音訊**，XTTS 模型會模仿該音訊的：
- 🎵 **語調**（tone）
- 🎤 **語速**（speed）
- 💬 **說話風格**（style）
- 😊 **情感**（emotion）

## 📁 準備情感參考音訊

### 1. 創建情感音訊目錄

```bash
mkdir -p resource/emotions
```

### 2. 準備不同情感的參考音訊

每個情感準備一個 5-30 秒的 WAV 音訊檔案：

```
resource/emotions/
├── happy.wav       # 開心/愉快的語氣
├── sad.wav         # 悲傷/低落的語氣
├── angry.wav       # 生氣/激動的語氣
├── neutral.wav     # 中性/平靜的語氣
├── excited.wav     # 興奮/熱情的語氣
├── gentle.wav      # 溫柔/柔和的語氣
└── professional.wav # 專業/正式的語氣
```

### 3. 音訊要求

✅ **格式**: WAV（16-bit PCM）  
✅ **採樣率**: 22050 Hz 或更高  
✅ **時長**: 5-30 秒（10 秒左右最佳）  
✅ **內容**: 清晰的人聲，最好是中文  
✅ **質量**: 無噪音、無背景音樂

## 🔧 使用方法

### 方法 1：使用環境變數（全域設定）

在 `.env` 中設定預設的參考音訊：

```bash
# 預設使用的情感音訊
TTS_SPEAKER_WAV=resource/emotions/neutral.wav

# 情感音訊目錄
EMOTION_AUDIO_DIR=resource/emotions
```

### 方法 2：使用 EmotionManager（動態切換）

```python
from modules.utils.emotion_manager import EmotionManager

# 初始化情感管理器
emotion_mgr = EmotionManager()

# 列出可用情感
print(emotion_mgr.list_emotions())
# 輸出: ['happy', 'sad', 'angry', 'neutral', 'excited', 'gentle', 'professional']

# 取得特定情感的音訊路徑
happy_audio = emotion_mgr.get_emotion_audio("happy")

# 使用帶情感的 TTS
tts_result = tts_engine.synthesize(
    text="今天天氣真好！",
    language="zh-cn",
    speaker_wav=happy_audio  # 使用開心的語氣
)
```

### 方法 3：整合到 VoiceAgent

修改 `modules/agent.py`，讓 Agent 根據對話內容自動選擇情感：

```python
from modules.utils.emotion_manager import get_emotion_manager

class VoiceAgent:
    def __init__(self, ...):
        # ...
        self.emotion_mgr = get_emotion_manager()
    
    def _detect_emotion(self, text: str) -> str:
        """簡單的情感檢測（可以更複雜）。"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["開心", "高興", "哈哈", "😊", "太好了"]):
            return "happy"
        elif any(word in text_lower for word in ["難過", "傷心", "😢", "遺憾"]):
            return "sad"
        elif any(word in text_lower for word in ["生氣", "憤怒", "😠", "可惡"]):
            return "angry"
        else:
            return "neutral"
    
    def synthesize_with_emotion(self, text: str, language: str = None):
        """帶情感的 TTS 合成。"""
        # 檢測情感
        emotion = self._detect_emotion(text)
        
        # 取得對應的參考音訊
        speaker_wav = self.emotion_mgr.get_emotion_audio(emotion)
        
        # 合成
        return self.tts.synthesize(
            text=text,
            language=language,
            speaker_wav=speaker_wav
        )
```

## 🎯 實用範例

### 範例 1：根據場景切換情感

```python
# 歡迎語 - 使用友善的語氣
greeting = tts_engine.synthesize(
    text="您好！歡迎使用語音助理！",
    speaker_wav="resource/emotions/gentle.wav"
)

# 錯誤提示 - 使用正式的語氣
error = tts_engine.synthesize(
    text="抱歉，我無法處理您的請求。",
    speaker_wav="resource/emotions/professional.wav"
)

# 成功回饋 - 使用開心的語氣
success = tts_engine.synthesize(
    text="太好了！已經幫您完成了！",
    speaker_wav="resource/emotions/happy.wav"
)
```

### 範例 2：LLM 控制情感

讓 LLM 在回應中指定情感標記：

```python
# LLM System Prompt
system_prompt = """
你是一個友善的助理。
在回應的開頭用 [EMOTION:xxx] 標記來指定語氣，例如：
- [EMOTION:happy] 表示開心
- [EMOTION:sad] 表示同情
- [EMOTION:neutral] 表示中性
"""

# 解析 LLM 回應
response = "[EMOTION:happy] 真高興能幫到你！今天天氣真好！"

# 提取情感標記
import re
match = re.match(r'\[EMOTION:(\w+)\](.*)', response)
if match:
    emotion = match.group(1)
    text = match.group(2).strip()
    
    # 使用對應情感合成
    speaker_wav = emotion_mgr.get_emotion_audio(emotion)
    tts_result = tts_engine.synthesize(text, speaker_wav=speaker_wav)
```

## 🎬 製作參考音訊的技巧

### 使用 Text-to-Speech 生成

如果沒有真人錄音，可以用高質量 TTS 生成：

```python
from TTS.api import TTS

# 使用高質量模型生成參考音訊
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

# 生成不同情感的音訊（需要人工調整文本以表達情感）
tts.tts_to_file(
    text="今天真是太開心了！一切都很順利！",
    language="zh-cn",
    file_path="resource/emotions/happy.wav"
)

tts.tts_to_file(
    text="這件事讓我感到很難過。希望能夠改善。",
    language="zh-cn", 
    file_path="resource/emotions/sad.wav"
)
```

### 從影片/音訊提取

```bash
# 使用 FFmpeg 提取並轉換格式
ffmpeg -i input_video.mp4 -ss 00:01:30 -t 00:00:10 \
       -ar 22050 -ac 1 resource/emotions/happy.wav
```

## ⚙️ 進階配置

### 調整參考音訊的影響程度

某些 TTS 模型支援調整克隆強度（需查看模型文檔）：

```python
# 部分模型支援的參數（視模型而定）
tts.tts(
    text="測試文本",
    speaker_wav="reference.wav",
    temperature=0.7,  # 控制創造性
    # 其他可能的參數...
)
```

## 🔍 故障排除

### 問題：情感不明顯

**解決方案**：
1. 使用更長的參考音訊（10-15 秒）
2. 確保參考音訊的情感表達明確
3. 使用更高質量的參考音訊

### 問題：參考音訊無效

**解決方案**：
```python
# 檢查音訊檔案
from pathlib import Path
import librosa

audio_path = "resource/emotions/happy.wav"
if Path(audio_path).is_file():
    # 載入並檢查
    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"音訊時長: {duration:.2f} 秒")
    print(f"採樣率: {sr} Hz")
else:
    print(f"檔案不存在: {audio_path}")
```

## 📚 相關資源

- [XTTS 文檔](https://github.com/coqui-ai/TTS)
- [Voice Cloning 最佳實踐](https://docs.coqui.ai/en/latest/tutorial_for_nervous_beginners.html)
