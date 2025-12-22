# 🎚️ TTS 設定（精簡版）

本專案的 TTS 介面刻意保持精簡：`synthesize(text, language=None)`。

目前支援的後端：
- **Chatterbox**（預設；支援中文/日文等多語）
- **VibeVoice**（可選；需要額外依賴）

## ✅ Chatterbox

使用環境變數切換：

```bash
export TTS_ENGINE=chatterbox

# （可選）提供 5-10 秒左右的參考音檔做 zero-shot voice cloning
export CHATTERBOX_AUDIO_PROMPT=/path/to/ref.wav
```

## ✅ VibeVoice（可選）

安裝額外依賴：

```bash
pip install -r requirements.txt -r requirements-vibevoice.txt
```

使用環境變數切換：

```bash
export TTS_ENGINE=vibevoice
export VIBEVOICE_MODEL_PATH=microsoft/VibeVoice-Realtime-0.5B
export VIBEVOICE_VOICES_DIR=resources/voices/streaming_model
export VIBEVOICE_VOICE=en-emma_woman
```

## ℹ️ 為什麼沒有一堆生成參數？

為了易維護/易理解，本 repo 的 wrapper 不把大量「生成旋鈕」暴露成 public API。
如果你確實需要更細的控制（如溫度、採樣等），建議直接在對應後端的 wrapper 中調整。
    tts.synthesizer.save_wav(result.audio, f"test_temp_{temp}.wav")
```

### 2. A/B 測試

```python
# 版本 A: 只用參考音訊
result_a = tts.synthesize(
    text="測試",
    speaker_wav="happy.wav"
)

# 版本 B: 參考音訊 + 調整參數
result_b = tts.synthesize(
    text="測試",
    speaker_wav="happy.wav",
    temperature=1.1,
    speed=1.1
)

# 比較效果
```

---

## 🔍 常見問題

### Q: `temperature` 和 `speaker_wav` 哪個影響更大？

**A**: 
- `speaker_wav` 控制**基礎風格**（音色、節奏模式）
- `temperature` 控制**表達強度**（情感豐富程度）
- 建議**同時使用**以獲得最佳效果

### Q: 為什麼高 `temperature` 會導致不穩定？

**A**: 
- 高 temperature 增加隨機性和創造性
- 可能產生不自然的發音或節奏
- 建議不超過 1.2

### Q: 如何找到最佳參數？

**A**: 
1. 從預設值開始（temperature=0.75）
2. 根據需求微調（±0.1-0.2）
3. 使用 A/B 測試比較效果
4. 為不同場景創建配置預設

---

## 📊 參數影響總結

| 參數 | 影響情感 | 影響穩定性 | 建議調整範圍 |
|------|----------|------------|--------------|
| temperature | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 0.5 - 1.0 |
| speed | ⭐⭐⭐ | ⭐⭐⭐⭐ | 0.8 - 1.2 |
| repetition_penalty | ⭐⭐ | ⭐⭐⭐⭐⭐ | 8.0 - 15.0 |
| top_p | ⭐⭐ | ⭐⭐⭐ | 0.75 - 0.9 |

**⭐ 越多表示影響越大**

---

## 🚀 快速開始

最簡單的情感控制方式：

```python
# 平淡
tts.synthesize("新聞播報", temperature=0.4)

# 正常
tts.synthesize("日常對話", temperature=0.75)

# 豐富
tts.synthesize("故事講述", temperature=1.1)
```

完整文檔: `demo_tts_parameters.py`
