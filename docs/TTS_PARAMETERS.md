# 🎚️ TTS 參數調整指南

## 📋 概述

XTTS 模型提供了多個參數來控制語音的**情感強度**、**語速**、**穩定性**等特性。

## 🎯 主要參數

### 1. `temperature` - 情感表達強度 ⭐

**最重要的情感控制參數**

- **作用**: 控制生成的創造性和情感表達的豐富程度
- **範圍**: 0.1 - 1.5（建議 0.5 - 1.0）
- **預設值**: 0.75

```python
# 平淡、穩定的語氣（適合播報新聞）
tts.synthesize(text="今日新聞...", temperature=0.3)

# 正常、自然的語氣
tts.synthesize(text="您好！", temperature=0.75)

# 豐富、有表現力的語氣（適合故事講述）
tts.synthesize(text="真是太棒了！", temperature=1.2)
```

**效果對比**:
- 📉 **0.1-0.4**: 語氣平淡、機械，情感表達少，但很穩定
- 🎯 **0.5-0.8**: 自然的情感表達，推薦日常使用
- 📈 **0.9-1.5**: 情感豐富、有表現力，但可能不穩定

---

### 2. `speed` - 語速控制 🏃

- **作用**: 控制說話速度
- **範圍**: 0.5 - 2.0
- **預設值**: 1.0

```python
# 慢速（適合教學、重要訊息）
tts.synthesize(text="請仔細聆聽...", speed=0.7)

# 正常速度
tts.synthesize(text="一般對話", speed=1.0)

# 快速（適合時間緊迫、興奮的場景）
tts.synthesize(text="快跑！", speed=1.5)
```

---

### 3. `repetition_penalty` - 避免重複 🔄

- **作用**: 懲罰重複的音節和詞語，讓語音更自然
- **範圍**: 1.0 - 20.0
- **預設值**: 10.0

```python
# 較低懲罰（可能重複）
tts.synthesize(text="...", repetition_penalty=5.0)

# 正常懲罰
tts.synthesize(text="...", repetition_penalty=10.0)

# 高懲罰（避免任何重複）
tts.synthesize(text="...", repetition_penalty=15.0)
```

---

### 4. `top_p` - 詞彙多樣性 🎲

- **作用**: 核採樣參數，控制詞彙選擇的多樣性
- **範圍**: 0.1 - 1.0
- **預設值**: 0.85

```python
# 保守、可預測的詞彙選擇
tts.synthesize(text="...", top_p=0.5)

# 平衡的多樣性
tts.synthesize(text="...", top_p=0.85)

# 高度多樣化（可能不穩定）
tts.synthesize(text="...", top_p=0.95)
```

---

### 5. `length_penalty` - 長度控制

- **作用**: 影響生成長度的傾向
- **範圍**: 0.5 - 2.0
- **預設值**: 1.0

---

### 6. `top_k` - 候選詞數量

- **作用**: 限制每次選擇時考慮的候選詞數量
- **範圍**: 1 - 100
- **預設值**: 50

---

## 🎭 情感預設配置

根據不同情感場景，推薦的參數組合：

### 😐 中性/播報
```python
{
    "temperature": 0.4,
    "speed": 1.0,
    "repetition_penalty": 12.0,
    "top_p": 0.75
}
```

### 😊 友善/溫暖
```python
{
    "temperature": 0.8,
    "speed": 1.05,
    "repetition_penalty": 10.0,
    "top_p": 0.85
}
```

### 🎉 興奮/熱情
```python
{
    "temperature": 1.1,
    "speed": 1.2,
    "repetition_penalty": 8.0,
    "top_p": 0.9
}
```

### 😢 悲傷/同情
```python
{
    "temperature": 0.7,
    "speed": 0.85,
    "repetition_penalty": 12.0,
    "top_p": 0.8
}
```

### 💼 專業/正式
```python
{
    "temperature": 0.5,
    "speed": 0.95,
    "repetition_penalty": 15.0,
    "top_p": 0.75
}
```

### 🌸 溫柔/安慰
```python
{
    "temperature": 0.65,
    "speed": 0.9,
    "repetition_penalty": 11.0,
    "top_p": 0.8
}
```

---

## 💡 實用範例

### 範例 1: 組合使用參數和參考音訊

```python
from modules.tts import CoquiTTS

tts = CoquiTTS()

# 使用開心的參考音訊 + 提高情感表達
result = tts.synthesize(
    text="今天天氣真好！我們去公園玩吧！",
    language="zh-cn",
    speaker_wav="resource/emotions/happy.wav",  # 參考音訊
    temperature=1.0,  # 提高情感表達
    speed=1.1,        # 稍微加快語速
    top_p=0.9         # 增加多樣性
)
```

### 範例 2: 動態調整情感強度

```python
# 根據文本長度調整 temperature
def adjust_temperature(text: str) -> float:
    """短文本用高 temperature，長文本用低 temperature"""
    if len(text) < 20:
        return 1.0  # 短句，豐富表達
    elif len(text) < 100:
        return 0.75  # 中等長度
    else:
        return 0.6  # 長文本，保持穩定

text = "這是測試文本"
temp = adjust_temperature(text)
result = tts.synthesize(text, temperature=temp)
```

### 範例 3: 場景化配置

```python
# 定義不同場景的配置
EMOTION_CONFIGS = {
    "neutral": {
        "temperature": 0.4,
        "speed": 1.0,
        "repetition_penalty": 12.0,
    },
    "happy": {
        "temperature": 1.0,
        "speed": 1.1,
        "repetition_penalty": 8.0,
    },
    "sad": {
        "temperature": 0.7,
        "speed": 0.85,
        "repetition_penalty": 12.0,
    },
}

# 使用配置
emotion = "happy"
config = EMOTION_CONFIGS[emotion]

result = tts.synthesize(
    text="太棒了！",
    language="zh-cn",
    **config  # 展開配置參數
)
```

## ⚙️ 調試技巧

### 1. 測試不同參數組合

```python
# 批次測試
test_text = "這是一個測試句子"
temperatures = [0.3, 0.5, 0.75, 1.0, 1.2]

for temp in temperatures:
    print(f"Testing temperature={temp}")
    result = tts.synthesize(
        text=test_text,
        temperature=temp
    )
    # 儲存並比較結果
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
