# 🎭 情感控制功能總覽

## 📚 文檔索引

### 快速開始
- **[QUICKSTART_EMOTION.md](QUICKSTART_EMOTION.md)** - 5 分鐘啟用情感控制 ⭐ **從這裡開始**

### 完整指南
- **[EMOTION_INTEGRATION.md](EMOTION_INTEGRATION.md)** - 架構設計和整合指南
- **[TTS_PARAMETERS.md](TTS_PARAMETERS.md)** - TTS 參數詳細說明
- **[EMOTION_CONTROL.md](EMOTION_CONTROL.md)** - 情感控制完整教程

### 測試腳本
- **[test_emotion_integration.py](../test_emotion_integration.py)** - 功能測試腳本
- **[demo_tts_parameters.py](../demo_tts_parameters.py)** - 參數演示腳本
- **[demo_emotion_control.py](../demo_emotion_control.py)** - 情感控制演示

## 🎯 功能特性

### ✅ 已實現

1. **雙重情感控制機制**
   - 參考音訊 (speaker_wav) - 控制音色和基礎風格
   - 參數調整 (temperature, speed 等) - 微調情感表達

2. **自動情感偵測**
   - 基於關鍵字的情感識別
   - 支援 7 種預設情感：neutral, happy, excited, sad, angry, gentle, professional

3. **完整 TTS 參數支援**
   - temperature (0.1-1.5) - 控制情感豐富度
   - speed (0.5-2.0) - 控制語速
   - repetition_penalty (1.0-20.0) - 減少重複
   - top_p (0.1-1.0) - 控制詞彙多樣性
   - length_penalty, top_k - 其他微調參數

4. **串流模式整合**
   - 逐句應用情感控制
   - 即時調整 TTS 參數
   - 無需等待完整回應

5. **靈活的配置系統**
   - 環境變數配置
   - 程式化 API
   - 預設配置 + 自訂覆蓋

## 🏗️ 架構概覽

```
┌─────────────────────────────────────────────────┐
│                  VoiceAgent                     │
│  ┌───────────────────────────────────────────┐  │
│  │         enable_emotion_control=True       │  │
│  └───────────────────────────────────────────┘  │
│                      │                          │
│         ┌────────────┴────────────┐             │
│         ▼                         ▼             │
│  ┌──────────────┐         ┌──────────────┐     │
│  │   Emotion    │         │   CoquiTTS   │     │
│  │   Manager    │────────▶│              │     │
│  │              │         │  synthesize()│     │
│  └──────────────┘         └──────────────┘     │
│         │                         │             │
│         ▼                         ▼             │
│  ┌──────────────┐         ┌──────────────┐     │
│  │ speaker_wav  │         │  temperature │     │
│  │ (參考音訊)    │         │  speed       │     │
│  │ happy.wav    │         │  top_p, ...  │     │
│  └──────────────┘         └──────────────┘     │
│                                                 │
│         每句話 → 偵測情感 → 應用配置 → 合成      │
└─────────────────────────────────────────────────┘
```

## 📂 核心文件

```
VoiceAgent/
├── modules/
│   ├── agent.py                    # VoiceAgent 主類別
│   │   └─ enable_emotion_control   # 情感控制開關
│   ├── tts/
│   │   └── coqui_tts.py           # 支援情感參數的 TTS
│   ├── utils/
│   │   └── emotion_manager.py     # 情感管理器
│   └── config.py                   # 配置和初始化
├── resource/
│   └── emotions/                   # 情感參考音訊目錄
│       ├── happy.wav
│       ├── sad.wav
│       ├── neutral.wav
│       └── ...
├── docs/
│   ├── QUICKSTART_EMOTION.md      # 快速開始 ⭐
│   ├── EMOTION_INTEGRATION.md     # 完整指南
│   └── TTS_PARAMETERS.md          # 參數說明
└── test_emotion_integration.py    # 測試腳本
```

## 🚀 使用方式

### 最簡單的方式（1 行代碼）

```python
# app_turn.py
from modules.config import setup_voice_agent

voice_agent = setup_voice_agent(enable_emotion_control=True)
```

就這樣！現在每句話都會自動應用情感控制。

### 效果示例

**用戶**: "幫我記錄一筆支出"

**Agent** (自動偵測為 professional):
- 使用 professional 參數 (temperature=0.5, speed=0.95)
- 語氣穩定、專業
- **回應**: "好的，請告訴我金額和用途。"

**用戶**: "太好了！謝謝你！"

**Agent** (自動偵測為 happy):
- 使用 happy 參數 (temperature=1.0, speed=1.1)
- 如果有 happy.wav，還會使用參考音訊
- 語氣開心、熱情
- **回應**: "不客氣！很高興能幫到你！"

## 📊 支援的情感

| 情感 | Temperature | Speed | 適用場景 | 關鍵字範例 |
|------|-------------|-------|----------|------------|
| **neutral** | 0.4 | 1.0 | 新聞播報、一般對話 | - |
| **happy** | 1.0 | 1.1 | 慶祝、鼓勵 | 開心、快樂、太好了 |
| **excited** | 1.1 | 1.2 | 驚喜、激動 | 興奮、哇、太棒了 |
| **sad** | 0.7 | 0.85 | 同情、安慰 | 難過、遺憾、可惜 |
| **angry** | 0.9 | 1.15 | 不滿、憤怒 | 生氣、可惡 |
| **gentle** | 0.65 | 0.9 | 安慰、溫柔 | 別擔心、沒關係 |
| **professional** | 0.5 | 0.95 | 商務、正式 | 報告、數據、分析 |

## 🎯 測試和驗證

### 1. 快速測試
```bash
python test_emotion_integration.py
```

### 2. 參數演示
```bash
python demo_tts_parameters.py
```

### 3. 檢查輸出
```bash
ls -la output/emotion_test/
ls -la output/temperature_test/
```

## 💡 最佳實踐

### ✅ 推薦做法

1. **組合使用參考音訊和參數**
   ```python
   # 參考音訊控制基礎風格，參數微調細節
   config = emotion_manager.get_emotion_config(
       text="太好了！",
       auto_detect=True
   )
   # 結果: {'speaker_wav': 'happy.wav', 'temperature': 1.0, 'speed': 1.1, ...}
   ```

2. **為關鍵句子手動指定情感**
   ```python
   # 問候語總是用友善的語氣
   greeting_config = emotion_manager.get_emotion_config(emotion="happy")
   ```

3. **根據場景創建自訂情感**
   ```python
   emotion_manager.DEFAULT_EMOTION_PARAMS["customer_service"] = {
       "temperature": 0.6,
       "speed": 1.0,
       "repetition_penalty": 12.0,
   }
   ```

### ❌ 避免的做法

1. **不要過度依賴自動偵測**
   - 重要句子應該手動指定情感

2. **不要使用過高的 temperature**
   - 超過 1.5 可能導致不穩定的輸出

3. **不要忽略參考音訊的質量**
   - 低質量音訊會影響整體效果

## 🔧 環境變數

```bash
# .env
EMOTION_AUDIO_DIR=resource/emotions
TTS_SPEAKER_WAV=resource/emotions/neutral.wav  # 預設參考音訊（可選）
```

## 📈 效能影響

- **情感偵測**: <1ms (正則表達式匹配)
- **參數調整**: 0ms (不影響 TTS 速度)
- **參考音訊**: +10-20% TTS 時間（首次載入後快取）

**結論**: 對串流延遲影響極小，完全可以用於即時應用。

## 🎓 學習路徑

1. **初學者**: 閱讀 [QUICKSTART_EMOTION.md](QUICKSTART_EMOTION.md)
2. **進階**: 閱讀 [EMOTION_INTEGRATION.md](EMOTION_INTEGRATION.md)
3. **專家**: 閱讀 [TTS_PARAMETERS.md](TTS_PARAMETERS.md) 並自訂配置

## 🤝 貢獻

歡迎提交：
- 新的情感配置
- 更好的關鍵字字典
- 使用案例和範例
- 文檔改進

## 📞 支援

遇到問題？
1. 查看 [QUICKSTART_EMOTION.md](QUICKSTART_EMOTION.md) 的疑難排解章節
2. 執行 `test_emotion_integration.py` 檢查環境
3. 檢查終端輸出中的 `[EmotionManager]` 和 `[CoquiTTS]` 日誌

## 📜 變更日誌

### v1.0 (2024-11-11)
- ✅ 實現 EmotionManager
- ✅ 整合到 VoiceAgent
- ✅ 支援 7 種預設情感
- ✅ 支援 6 個 TTS 參數
- ✅ 自動情感偵測
- ✅ 完整文檔和測試

---

**開始使用**: [QUICKSTART_EMOTION.md](QUICKSTART_EMOTION.md) 🚀
