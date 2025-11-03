"""測試工具系統是否正確使用中文回覆。"""

import os
from dotenv import load_dotenv

from modules import VoiceAgent, WhisperSTT, OllamaLLM, CoquiTTS
from modules.tools import ToolManager, AccountingAgentWebHook

load_dotenv()


def test_tool_with_chinese():
    """測試記帳工具是否用中文回覆。"""
    print("\n" + "="*60)
    print("🧪 測試記帳工具 - 中文回覆")
    print("="*60)
    
    # 建立工具管理器
    tool_manager = ToolManager()
    tool_manager.register_tool(AccountingAgentWebHook())
    
    # 建立 Agent（串流模式）
    agent = VoiceAgent(
        stt=WhisperSTT(model_size="tiny", device="cpu"),  # 使用 tiny 模型加快測試
        llm=OllamaLLM(api_url=os.getenv("LLM_API_URL", "http://localhost:11434")),
        tts=CoquiTTS(device="cpu"),  # 使用 CPU 避免 CUDA 衝突
        tool_manager=tool_manager,
        enable_streaming=True,
    )
    
    # 測試文字
    test_text = "幫我記帳，我今天下午2點吃了牛肉麵200元"
    print(f"\n使用者: {test_text}")
    print("處理中...\n")
    
    # 收集回應
    responses = []
    for tts_result, sentence in agent.process_text(test_text):
        print(f"📢 {sentence}")
        responses.append(sentence)
    
    print(f"\n✅ 完成！共 {len(responses)} 個回應")
    
    # 檢查是否有英文
    full_response = " ".join(responses)
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in full_response)
    has_english_words = any(word.lower() in full_response.lower() 
                           for word in ["congratulations", "successfully", "completed", "records"])
    
    print("\n檢查結果:")
    print(f"  包含中文: {'✓' if has_chinese else '✗'}")
    print(f"  包含英文: {'✗ (好)' if not has_english_words else '✓ (需修正)'}")
    
    if has_chinese and not has_english_words:
        print("\n🎉 測試通過！回覆使用中文。")
    else:
        print("\n❌ 測試失敗！回覆包含英文或缺少中文。")


if __name__ == "__main__":
    test_tool_with_chinese()
