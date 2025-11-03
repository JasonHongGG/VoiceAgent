"""
展示流式 Voice Agent 與傳統 Voice Agent 的效能差異。

這個範例會比較：
1. 傳統模式：等待完整 LLM 回應後才開始 TTS
2. 流式模式：LLM 每生成一句就立即 TTS
"""

import os
import time
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

from modules.stt import WhisperSTT
from modules.tts import CoquiTTS
from modules.llm import OllamaLLM
from modules.agent import VoiceAgent
from modules.streaming_agent import StreamingVoiceAgent

load_dotenv()


def demo_traditional_mode():
    """傳統模式：等待完整回應。"""
    print("\n" + "="*60)
    print("📊 傳統模式測試")
    print("="*60)
    
    # 初始化
    stt = WhisperSTT(model_size="medium", device="cuda")
    llm = OllamaLLM(api_url=os.getenv("LLM_API_URL", "http://localhost:11434"))
    tts = CoquiTTS(device="cuda")
    
    agent = VoiceAgent(stt, llm, tts)
    
    # 測試問題（會產生較長回應）
    test_text = "請詳細解釋什麼是深度學習，包括它的歷史、原理和應用。"
    
    print(f"問題：{test_text}")
    print("\n開始處理...")
    start_time = time.time()
    
    # 處理（會等待完整的 LLM + TTS）
    tts_result = agent.process_text(test_text)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\n✅ 完成！")
    print(f"⏱️  總耗時：{elapsed:.2f} 秒")
    print(f"🔊 音訊長度：{len(tts_result.audio) / tts_result.sample_rate:.2f} 秒")
    print(f"⚠️  使用者等待時間：{elapsed:.2f} 秒（才開始聽到第一個字）")
    
    # 儲存結果
    sf.write("traditional_output.wav", tts_result.audio, tts_result.sample_rate)
    print(f"💾 已儲存至 traditional_output.wav")
    
    return elapsed


def demo_streaming_mode():
    """流式模式：即時回應。"""
    print("\n" + "="*60)
    print("⚡ 流式模式測試")
    print("="*60)
    
    # 初始化
    stt = WhisperSTT(model_size="medium", device="cuda")
    llm = OllamaLLM(api_url=os.getenv("LLM_API_URL", "http://localhost:11434"))
    tts = CoquiTTS(device="cuda")
    
    agent = StreamingVoiceAgent(
        stt, llm, tts,
        sentence_delimiters=r'[。！？\.!?;；\n]',
        min_sentence_length=5
    )
    
    # 相同的測試問題
    test_text = "請詳細解釋什麼是深度學習，包括它的歷史、原理和應用。"
    
    print(f"問題：{test_text}")
    print("\n開始處理...")
    start_time = time.time()
    first_audio_time = None
    
    all_audio = []
    sentence_count = 0
    
    # 流式處理
    for tts_result, sentence in agent.process_text_stream(test_text):
        sentence_count += 1
        current_time = time.time()
        
        # 記錄第一個音訊片段的時間
        if first_audio_time is None:
            first_audio_time = current_time
            time_to_first_audio = first_audio_time - start_time
            print(f"\n🎯 第一個句子音訊已生成！")
            print(f"⏱️  時間：{time_to_first_audio:.2f} 秒")
            print(f"📝 句子：{sentence[:50]}...")
        
        all_audio.append(tts_result.audio)
        elapsed = current_time - start_time
        print(f"   [{sentence_count}] +{elapsed:.2f}s: {sentence[:50]}...")
    
    end_time = time.time()
    total_elapsed = end_time - start_time
    
    # 合併所有音訊
    combined_audio = np.concatenate(all_audio)
    
    print(f"\n✅ 完成！")
    print(f"⏱️  總耗時：{total_elapsed:.2f} 秒")
    print(f"🎯 首次回應：{time_to_first_audio:.2f} 秒")
    print(f"📊 句子數量：{sentence_count}")
    print(f"🔊 總音訊長度：{len(combined_audio) / tts_result.sample_rate:.2f} 秒")
    print(f"✨ 使用者體驗：{time_to_first_audio:.2f} 秒後就開始聽到回應")
    
    # 儲存結果
    sf.write("streaming_output.wav", combined_audio, tts_result.sample_rate)
    print(f"💾 已儲存至 streaming_output.wav")
    
    return time_to_first_audio, total_elapsed


def demo_comparison():
    """比較兩種模式的效能。"""
    print("\n" + "🎭 "*20)
    print("Voice Agent 效能比較")
    print("🎭 "*20)
    
    # 測試傳統模式
    traditional_time = demo_traditional_mode()
    
    # 等待一下
    time.sleep(2)
    
    # 測試流式模式
    streaming_first_time, streaming_total_time = demo_streaming_mode()
    
    # 顯示比較結果
    print("\n" + "="*60)
    print("📊 效能比較總結")
    print("="*60)
    print(f"傳統模式 - 使用者等待時間：{traditional_time:.2f} 秒")
    print(f"流式模式 - 首次回應時間：  {streaming_first_time:.2f} 秒")
    print(f"流式模式 - 總處理時間：    {streaming_total_time:.2f} 秒")
    print("")
    improvement = ((traditional_time - streaming_first_time) / traditional_time) * 100
    print(f"💡 流式模式將首次回應時間減少了 {improvement:.1f}%")
    print(f"✨ 使用者感知延遲從 {traditional_time:.2f}s 降至 {streaming_first_time:.2f}s")
    print("="*60)
    
    print("\n🎯 結論：")
    print("傳統模式：使用者必須等待完整的 LLM 生成 + TTS 合成")
    print("流式模式：使用者只需等待第一句話的生成 + TTS，體驗更流暢")
    print("\n這就是 ElevenLabs 等服務能做到即時回應的秘密！")


def demo_streaming_visualization():
    """視覺化流式處理的過程。"""
    print("\n" + "="*60)
    print("🎬 流式處理過程視覺化")
    print("="*60)
    
    # 初始化
    stt = WhisperSTT(model_size="medium", device="cuda")
    llm = OllamaLLM(api_url=os.getenv("LLM_API_URL", "http://localhost:11434"))
    tts = CoquiTTS(device="cuda")
    
    agent = StreamingVoiceAgent(stt, llm, tts)
    
    test_text = "什麼是人工智慧？"
    
    print(f"問題：{test_text}\n")
    print("時間軸：")
    print("-" * 60)
    
    start_time = time.time()
    
    for i, (tts_result, sentence) in enumerate(agent.process_text_stream(test_text), 1):
        elapsed = time.time() - start_time
        bar_length = int(elapsed * 10)  # 視覺化時間
        bar = "█" * bar_length
        
        print(f"[{elapsed:5.2f}s] {bar}")
        print(f"         句子 {i}: {sentence}")
        print(f"         音訊: {len(tts_result.audio)} 樣本")
        print()
    
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"總時間：{total_time:.2f} 秒")


if __name__ == "__main__":
    import sys
    
    print("\n" + "🎙️ "*20)
    print("流式 Voice Agent 效能展示")
    print("🎙️ "*20)
    
    options = [
        ("完整比較測試", demo_comparison),
        ("流式模式測試", demo_streaming_mode),
        ("傳統模式測試", demo_traditional_mode),
        ("流式處理視覺化", demo_streaming_visualization),
    ]
    
    print("\n請選擇測試項目：")
    for i, (name, _) in enumerate(options, 1):
        print(f"{i}. {name}")
    
    print("\n輸入數字 (1-4)，或按 Enter 執行完整比較測試: ", end="")
    choice = input().strip()
    
    if choice == "" or choice == "1":
        demo_comparison()
    elif choice.isdigit() and 1 <= int(choice) <= len(options):
        options[int(choice) - 1][1]()
    else:
        print("無效的選擇。")
