"""演示如何使用情感控制 TTS。"""

import os
from dotenv import load_dotenv

from modules.tts import CoquiTTS
from modules.utils.emotion_manager import EmotionManager

# 載入環境變數
load_dotenv()

def demo_basic_emotion_control():
    """基本的情感控制演示。"""
    print("="*60)
    print("演示 1: 基本情感控制")
    print("="*60)
    
    # 初始化 TTS
    tts = CoquiTTS()
    
    # 初始化情感管理器
    emotion_mgr = EmotionManager()
    
    # 列出可用情感
    print(f"\n可用情感: {emotion_mgr.list_emotions()}")
    
    # 測試不同情感
    test_cases = [
        ("neutral", "您好，我是語音助理。"),
        ("happy", "太好了！今天天氣真棒！"),
        ("sad", "很抱歉聽到這個消息。"),
        ("professional", "請問有什麼可以幫助您的？"),
    ]
    
    for emotion, text in test_cases:
        print(f"\n--- 測試情感: {emotion} ---")
        print(f"文本: {text}")
        
        # 取得情感音訊
        speaker_wav = emotion_mgr.get_emotion_audio(emotion)
        
        if speaker_wav:
            # 合成
            result = tts.synthesize(
                text=text,
                language="zh-cn",
                speaker_wav=speaker_wav
            )
            print(f"✓ 合成成功: {len(result.audio)} 樣本")
            # 這裡可以播放或儲存音訊
        else:
            print(f"✗ 找不到情感音訊檔案")


def demo_emotion_detection():
    """自動情感檢測演示。"""
    print("\n" + "="*60)
    print("演示 2: 自動情感檢測")
    print("="*60)
    
    def detect_emotion(text: str) -> str:
        """簡單的情感檢測。"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["開心", "高興", "哈哈", "太好了", "棒"]):
            return "happy"
        elif any(word in text_lower for word in ["難過", "傷心", "遺憾", "抱歉"]):
            return "sad"
        elif any(word in text_lower for word in ["請問", "您好", "幫助"]):
            return "professional"
        else:
            return "neutral"
    
    # 測試文本
    test_texts = [
        "您好！請問有什麼可以幫助您的？",
        "太好了！您的訂單已經成功提交！",
        "很抱歉，系統目前無法處理您的請求。",
        "今天天氣如何？",
    ]
    
    tts = CoquiTTS()
    emotion_mgr = EmotionManager()
    
    for text in test_texts:
        # 檢測情感
        emotion = detect_emotion(text)
        print(f"\n文本: {text}")
        print(f"檢測到的情感: {emotion}")
        
        # 取得對應音訊
        speaker_wav = emotion_mgr.get_emotion_audio(emotion)
        
        if speaker_wav:
            result = tts.synthesize(
                text=text,
                language="zh-cn",
                speaker_wav=speaker_wav
            )
            print(f"✓ 使用 {emotion} 情感合成成功")


def demo_create_emotion_audio():
    """演示如何創建情感參考音訊。"""
    print("\n" + "="*60)
    print("演示 3: 創建情感參考音訊")
    print("="*60)
    
    from TTS.api import TTS
    from pathlib import Path
    
    # 確保目錄存在
    emotion_dir = Path("resource/emotions")
    emotion_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化 TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    
    # 不同情感的示範文本（需要表達出對應情感）
    emotion_texts = {
        "happy": "今天真是太開心了！一切都很順利，讓人感到非常愉快！",
        "sad": "這件事讓我感到很難過。真希望情況能夠改善。",
        "neutral": "這是一段平靜、中性的敘述。沒有特別的情緒起伏。",
        "professional": "歡迎致電客服中心。請問有什麼可以為您服務的？",
        "excited": "哇！這真是太棒了！我等不及要開始了！",
    }
    
    print("\n開始生成情感參考音訊...")
    
    for emotion, text in emotion_texts.items():
        output_path = emotion_dir / f"{emotion}.wav"
        
        print(f"\n生成 {emotion} 情感音訊...")
        print(f"  文本: {text}")
        
        try:
            # 生成音訊
            wav = tts.tts(text=text, language="zh-cn")
            
            # 儲存
            tts.synthesizer.save_wav(wav, str(output_path))
            print(f"  ✓ 已儲存至: {output_path}")
            
        except Exception as e:
            print(f"  ✗ 生成失敗: {e}")
    
    print("\n完成！請檢查 resource/emotions/ 目錄")
    print("注意: 自動生成的音訊可能需要手動調整或使用真人錄音替換")


if __name__ == "__main__":
    print("\n🎭 TTS 情感控制演示\n")
    
    # 選擇要執行的演示
    print("請選擇演示:")
    print("1. 基本情感控制")
    print("2. 自動情感檢測")
    print("3. 創建情感參考音訊")
    print("4. 全部執行")
    
    choice = input("\n請輸入選項 (1-4): ").strip()
    
    if choice == "1":
        demo_basic_emotion_control()
    elif choice == "2":
        demo_emotion_detection()
    elif choice == "3":
        demo_create_emotion_audio()
    elif choice == "4":
        demo_basic_emotion_control()
        demo_emotion_detection()
        demo_create_emotion_audio()
    else:
        print("無效的選項")
    
    print("\n✓ 演示完成！")
