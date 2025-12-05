# 🎭 情感控制快速參考

## 🚀 一行啟用

```python
voice_agent = setup_voice_agent()
```

## 🎯 7 種預設情感

| 情感 | Temp | Speed | 場景 |
|------|------|-------|------|
| neutral | 0.4 | 1.0 | 播報 |
| happy | 1.0 | 1.1 | 鼓勵 |
| excited | 1.1 | 1.2 | 驚喜 |
| sad | 0.7 | 0.85 | 安慰 |
| angry | 0.9 | 1.15 | 不滿 |
| gentle | 0.65 | 0.9 | 溫柔 |
| professional | 0.5 | 0.95 | 正式 |

## 🎚️ 主要參數

```python
temperature  # 0.1-1.5, 控制情感豐富度 ⭐
speed        # 0.5-2.0, 控制語速
```

## 📁 目錄結構

```
resource/emotions/
├── happy.wav      # 開心的參考音訊
├── sad.wav        # 悲傷的參考音訊
└── neutral.wav    # 中性的參考音訊
```

## 💻 手動控制

```python
# 取得情感配置
config = emotion_manager.get_emotion_config(
    emotion="happy",  # 或使用 auto_detect=True
    text="今天真開心！"
)

# 合成
result = tts.synthesize(
    text="今天真開心！",
    language="zh-cn",
    **config
)
```

## 🧪 測試

```bash
python test_emotion_integration.py
```

## 📖 完整文檔

- **快速開始**: [QUICKSTART_EMOTION.md](QUICKSTART_EMOTION.md)
- **完整指南**: [EMOTION_INTEGRATION.md](EMOTION_INTEGRATION.md)
- **參數說明**: [TTS_PARAMETERS.md](TTS_PARAMETERS.md)
