# 🎭 TTS 情感控制快速參考

## ⚡ 快速開始

### 1. 準備情感音訊
```bash
mkdir -p resource/emotions
# 將不同情感的 WAV 檔案放入此目錄
# 例如: happy.wav, sad.wav, neutral.wav
```

### 2. 配置環境變數
```bash
# .env
TTS_SPEAKER_WAV=resource/emotions/neutral.wav
EMOTION_AUDIO_DIR=resource/emotions
```

### 3. 使用情感控制
```python
from modules.tts import CoquiTTS
from modules.utils.emotion_manager import EmotionManager

tts = CoquiTTS()
emotion_mgr = EmotionManager()

# 使用開心的語氣
happy_audio = emotion_mgr.get_emotion_audio("happy")
result = tts.synthesize(
    text="太好了！",
    language="zh-cn",
    speaker_wav=happy_audio
)
```

## 📋 常用情感類型

| 情感 | 檔案名 | 適用場景 |
|------|--------|----------|
| 😊 開心 | happy.wav | 成功訊息、祝賀 |
| 😢 悲傷 | sad.wav | 同情、道歉 |
| 😐 中性 | neutral.wav | 一般資訊 |
| 💼 專業 | professional.wav | 客服、正式場合 |
| 🎉 興奮 | excited.wav | 促銷、活動 |
| 🌸 溫柔 | gentle.wav | 安慰、關懷 |
| 😠 生氣 | angry.wav | 警告、嚴肅 |

## 🎯 實用範例

### 動態切換情感
```python
emotions = {
    "greeting": "gentle",
    "success": "happy", 
    "error": "professional",
    "goodbye": "neutral"
}

for scenario, emotion in emotions.items():
    speaker_wav = emotion_mgr.get_emotion_audio(emotion)
    # 使用對應情感合成...
```

### LLM 控制情感
```python
# System Prompt
"在回應開頭加上 [EMOTION:xxx]，如 [EMOTION:happy]"

# 解析回應
response = "[EMOTION:happy] 很高興幫到你！"
emotion = extract_emotion(response)  # 提取 "happy"
text = remove_emotion_tag(response)   # 提取文本

# 使用情感合成
speaker_wav = emotion_mgr.get_emotion_audio(emotion)
tts.synthesize(text, speaker_wav=speaker_wav)
```

## 🔧 故障排除

**問題**: 情感不明顯  
**解決**: 使用 10-15 秒的清晰參考音訊

**問題**: 找不到情感檔案  
**解決**: 檢查檔案路徑和權限
```bash
ls -la resource/emotions/
```

**問題**: 音訊格式錯誤  
**解決**: 確保是 WAV 格式，22050Hz
```bash
ffmpeg -i input.mp3 -ar 22050 -ac 1 output.wav
```

## 📚 更多資訊

詳細文檔: `docs/EMOTION_CONTROL.md`  
演示程式: `demo_emotion_control.py`
