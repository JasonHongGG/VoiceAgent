"""測試歡迎功能的簡單腳本。"""

import os
from dotenv import load_dotenv
from modules.config import setup_voice_agent

# 載入環境變數
load_dotenv()

# 初始化 Voice Agent
print("正在初始化 Voice Agent...")
voice_agent = setup_voice_agent()

# 測試歡迎語
greeting_text = os.getenv("GREETING_MESSAGE", "你好！我是你的語音助理，有什麼可以幫助你的嗎？")
print(f"\n{'='*60}")
print(f"測試歡迎語: '{greeting_text}'")
print(f"{'='*60}\n")

# 使用流式模式處理歡迎語
print("開始生成歡迎語音...\n")
sentence_count = 0

for tts_result, sentence in voice_agent.process_text(greeting_text):
    sentence_count += 1
    print(f"[句子 {sentence_count}] {sentence}")
    print(f"  ├─ 音訊長度: {len(tts_result.audio)} 樣本")
    print(f"  ├─ 採樣率: {tts_result.sample_rate} Hz")
    print(f"  └─ 時長: {len(tts_result.audio) / tts_result.sample_rate:.2f} 秒\n")

print(f"{'='*60}")
print(f"✓ 歡迎語測試完成！共生成 {sentence_count} 個句子")
print(f"{'='*60}")
