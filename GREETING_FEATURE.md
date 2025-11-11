# 歡迎語功能說明

## 功能描述

當用戶點擊錄音按鈕並開始 WebRTC 連接時，Voice Agent 會**立即**播放歡迎語，而不需要等待用戶先說話。

## 實現方式

### 自定義 StreamHandler

我們創建了一個 `GreetingReplyOnPause` 類別，繼承自 `ReplyOnPause`：

```python
class GreetingReplyOnPause(ReplyOnPause):
    """擴展 ReplyOnPause，在首次接收音訊時先發送歡迎語。"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_greeted = False  # 追蹤是否已打招呼
    
    def __call__(self, audio: Tuple[int, np.ndarray]):
        # 首次調用時先發送歡迎語
        if not self.has_greeted:
            self.has_greeted = True
            # 生成並 yield 歡迎語音
            ...
        
        # 然後處理用戶的音訊輸入
        ...
```

### 工作流程

```
用戶點擊錄音按鈕
    ↓
WebRTC 連接建立
    ↓
開始接收音訊流 (即使用戶還沒說話)
    ↓
Handler 首次被調用 (has_greeted = False)
    ↓
🎙️ 立即播放歡迎語
    ↓
設置 has_greeted = True
    ↓
等待並處理用戶的語音輸入
    ↓
正常的對話流程...
```

## 配置

在 `.env` 文件中設置歡迎語內容：

```bash
GREETING_MESSAGE=你好！我是你的語音助理，有什麼可以幫助你的嗎？
```

如果不設置，將使用預設歡迎語。

## 特點

✅ **即時觸發**：用戶點擊錄音按鈕後立即播放，無需等待用戶說話  
✅ **只播放一次**：每個連接只在首次接收音訊時播放  
✅ **流式播放**：歡迎語也支援逐句流式播放，體驗流暢  
✅ **可配置**：通過環境變數輕鬆自定義歡迎內容  
✅ **不影響正常對話**：歡迎語播放完後，正常處理用戶輸入

## 技術細節

- **繼承 ReplyOnPause**：保留了原有的暫停檢測功能
- **狀態追蹤**：使用 `has_greeted` 標記確保只歡迎一次
- **Generator 機制**：使用 yield 實現流式輸出
- **與 VoiceAgent 整合**：複用 `voice_agent.process_text()` 的流式 TTS 功能

## 測試

啟動應用後：

1. 打開瀏覽器訪問應用
2. 點擊「開始錄音」按鈕
3. **立即**聽到歡迎語（無需先說話）
4. 歡迎語播放完畢後，可以開始正常對話

## 注意事項

- 每次**新的 WebRTC 連接**都會播放歡迎語
- 如果用戶重新連接（例如刷新頁面），會再次播放歡迎語
- 歡迎語生成可能需要 2-5 秒（取決於 LLM 和 TTS 的速度）
