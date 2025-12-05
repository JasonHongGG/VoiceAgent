# 🚀 快速開始：啟用情感控制

這個指南會帶您在 5 分鐘內啟用 Voice Agent 的情感控制功能。

## 📋 步驟 1: 更新 app_turn.py

找到您的 `app_turn.py` 文件，將：

```python
from modules.config import setup_voice_agent

# 舊版本（無情感控制）
voice_agent = setup_voice_agent()
```

改為：

```python
from modules.config import setup_voice_agent

# 新版本（啟用自動情感控制）
voice_agent = setup_voice_agent()
```

**就這麼簡單！** 現在您的 Voice Agent 已經支援情感控制了。

## 🎯 步驟 2: 測試基本功能

即使沒有參考音訊，情感控制也能使用參數調整：

```bash
python test_emotion_integration.py
```

選擇 `1` 測試 EmotionManager 基本功能，您會看到：

```
✓ 可用的情感: 無（將使用預設參數）
✓ 情感偵測測試:
  '今天真開心！' -> happy
  '很遺憾聽到這個消息...' -> sad
  '請提供完整的報告和數據分析。' -> professional
```

## 🎨 步驟 3: 添加參考音訊（可選但推薦）

### 方法 A: 錄製您自己的聲音

1. 創建目錄：
```bash
mkdir -p resource/emotions
```

2. 使用任何錄音軟體錄製 5-10 秒的音訊：
   - **happy.wav**: 用開心的語氣說 "今天天氣真好！我們一起出去玩吧！"
   - **sad.wav**: 用悲傷的語氣說 "很遺憾聽到這個消息..."
   - **neutral.wav**: 用平淡的語氣說 "今日天氣預報，多雲。"

3. 儲存為 `.wav` 格式到 `resource/emotions/` 目錄

### 方法 B: 使用現有的音訊檔案

如果您有任何中文語音檔案（例如從影片、播客等提取），可以直接使用：

```bash
# 複製現有音訊
cp /path/to/your/happy_voice.wav resource/emotions/happy.wav
```

## 🧪 步驟 4: 測試完整功能

```bash
# 執行完整測試
python test_emotion_integration.py

# 選擇 2 - 情感控制 + TTS 整合
```

您會看到類似這樣的輸出：

```
🎯 Test Case 1:
  文字: '太好了！我們成功了！'
  期望情感: happy
  偵測情感: happy
  配置: {'speaker_wav': 'resource/emotions/happy.wav', 'temperature': 1.0, 'speed': 1.1, ...}
  ✅ 已儲存: output/emotion_test/test_1_happy.wav
```

## 🎬 步驟 5: 啟動您的應用

```bash
python app_turn.py
```

現在當用戶說話時，Voice Agent 會：
1. 自動偵測每句話的情感
2. 選擇對應的參考音訊（如果有）
3. 應用對應的 TTS 參數（temperature, speed 等）
4. 串流回傳帶有情感的語音

## 💡 常見場景

### 場景 1: 沒有參考音訊

**完全沒問題！** 系統會自動使用參數控制：

- "太好了！" → temperature=1.0, speed=1.1（開心）
- "很遺憾..." → temperature=0.7, speed=0.85（悲傷）
- "一般對話" → temperature=0.4, speed=1.0（中性）

### 場景 2: 只有部分參考音訊

例如只有 `happy.wav`：

- "太好了！" → 使用 happy.wav + happy 參數
- "很遺憾..." → 只使用 sad 參數（無參考音訊）
- "一般對話" → 只使用 neutral 參數

### 場景 3: 手動控制特定句子

在程式碼中手動指定：

```python
# 問候語總是用開心的語氣
greeting_config = voice_agent.emotion_manager.get_emotion_config(emotion="happy")
greeting = voice_agent.tts.synthesize(
    text="你好！我是你的語音助理！",
    language="zh-cn",
    **greeting_config
)
```

## 🔧 進階設定

### 調整情感參數

在 `app_turn.py` 啟動前：

```python
from modules.config import setup_voice_agent

voice_agent = setup_voice_agent()

# 自訂 happy 情感的參數
voice_agent.emotion_manager.DEFAULT_EMOTION_PARAMS["happy"]["temperature"] = 1.2
voice_agent.emotion_manager.DEFAULT_EMOTION_PARAMS["happy"]["speed"] = 1.3

# 添加新情感
voice_agent.emotion_manager.DEFAULT_EMOTION_PARAMS["cheerful"] = {
    "temperature": 1.15,
    "speed": 1.25,
    "repetition_penalty": 7.0,
    "top_p": 0.95,
}
```

### 擴充情感偵測關鍵字

修改 `modules/utils/emotion_manager.py` 的 `detect_emotion_from_text` 方法：

```python
emotion_keywords = {
    "happy": ["開心", "快樂", "太好了", "太棒了", "哈哈", "😊", "😄", "🎉", "耶"],
    "excited": ["興奮", "激動", "驚喜", "哇", "😍", "🤩", "amazing"],
    # ... 添加更多關鍵字
}
```

## 📊 效果對比

測試不同配置的效果：

```bash
python demo_tts_parameters.py
```

聆聽並比較：
- `output/temperature_test/` - 不同 temperature 的效果
- `output/speed_test/` - 不同 speed 的效果
- `output/emotion_presets/` - 不同情感的效果

## 🎓 學習資源

- **完整指南**: [docs/EMOTION_INTEGRATION.md](docs/EMOTION_INTEGRATION.md)
- **參數說明**: [docs/TTS_PARAMETERS.md](docs/TTS_PARAMETERS.md)
- **測試腳本**: `test_emotion_integration.py`
- **演示腳本**: `demo_tts_parameters.py`

## ❓ 疑難排解

### 問題 1: 情感偵測不工作

檢查是否啟用：
```python
voice_agent = setup_voice_agent()
```

### 問題 2: 參考音訊未被使用

檢查檔案是否存在：
```bash
ls -la resource/emotions/
```

檢查終端輸出是否顯示：
```
[EmotionManager] Selected emotion 'happy': resource/emotions/happy.wav
[CoquiTTS] Using dynamic speaker_wav: resource/emotions/happy.wav
```

### 問題 3: 效果不明顯

1. 調高 temperature 值（例如從 0.75 改為 1.2）
2. 錄製更有表現力的參考音訊
3. 檢查是否使用了正確的情感

## 🎉 完成！

現在您的 Voice Agent 已經支援情感控制了！每句話都會自動使用合適的情感和參數。

下一步建議：
1. 錄製更多情感的參考音訊
2. 調整參數以符合您的需求
3. 為特定場景創建自訂情感配置
4. 測試並收集用戶反饋

祝您使用愉快！ 🚀
