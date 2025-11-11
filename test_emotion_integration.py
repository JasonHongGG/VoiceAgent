"""
🎭 情感控制快速測試

測試 EmotionManager 和 VoiceAgent 的情感控制功能。
"""

import os
from pathlib import Path
from modules.utils.emotion_manager import EmotionManager
from modules.tts import CoquiTTS


def test_emotion_manager():
    """測試 EmotionManager 基本功能。"""
    print("\n" + "="*60)
    print("📋 Test 1: EmotionManager 基本功能")
    print("="*60)
    
    emotion_mgr = EmotionManager()
    
    # 1. 列出可用的情感
    emotions = emotion_mgr.list_emotions()
    print(f"\n✓ 可用的情感: {emotions if emotions else '無（將使用預設參數）'}")
    
    # 2. 測試情感偵測
    test_texts = [
        "今天真開心！",
        "很遺憾聽到這個消息...",
        "請提供完整的報告和數據分析。",
        "別擔心，一切都會好起來的。",
        "這是一般的對話。",
    ]
    
    print("\n✓ 情感偵測測試:")
    for text in test_texts:
        detected = emotion_mgr.detect_emotion_from_text(text)
        print(f"  '{text}' -> {detected}")
    
    # 3. 測試情感配置
    print("\n✓ 情感配置測試:")
    test_emotion = "happy"
    config = emotion_mgr.get_emotion_config(
        emotion=test_emotion,
        text="測試文字"
    )
    print(f"  情感 '{test_emotion}' 的配置:")
    for key, value in config.items():
        print(f"    {key}: {value}")


def test_emotion_with_tts():
    """測試情感控制和 TTS 整合。"""
    print("\n" + "="*60)
    print("🎤 Test 2: 情感控制 + TTS 整合")
    print("="*60)
    
    tts = CoquiTTS()
    emotion_mgr = EmotionManager()
    
    test_cases = [
        ("太好了！我們成功了！", "happy"),
        ("很遺憾，這次沒有成功...", "sad"),
        ("根據最新報告，數據顯示良好。", "professional"),
        ("別擔心，我會幫助你的。", "gentle"),
    ]
    
    output_dir = Path("output/emotion_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n輸出目錄: {output_dir}\n")
    
    for i, (text, expected_emotion) in enumerate(test_cases, 1):
        print(f"🎯 Test Case {i}:")
        print(f"  文字: '{text}'")
        print(f"  期望情感: {expected_emotion}")
        
        # 取得情感配置
        config = emotion_mgr.get_emotion_config(
            text=text,
            auto_detect=True
        )
        
        detected_emotion = emotion_mgr.detect_emotion_from_text(text)
        print(f"  偵測情感: {detected_emotion}")
        print(f"  配置: {config}")
        
        # 合成語音
        try:
            result = tts.synthesize(
                text=text,
                language="zh-cn",
                **config
            )
            
            # 儲存音訊
            output_file = output_dir / f"test_{i}_{detected_emotion}.wav"
            tts.synthesizer.save_wav(result.audio, str(output_file))
            print(f"  ✅ 已儲存: {output_file}")
        except Exception as e:
            print(f"  ❌ 合成失敗: {e}")
        
        print()


def test_parameter_comparison():
    """測試不同參數配置的效果。"""
    print("\n" + "="*60)
    print("📊 Test 3: 參數對比測試")
    print("="*60)
    
    tts = CoquiTTS()
    test_text = "這是一個測試句子。"
    
    configs = {
        "default": {},
        "high_emotion": {
            "temperature": 1.2,
            "speed": 1.1,
        },
        "low_emotion": {
            "temperature": 0.3,
            "speed": 0.9,
        },
    }
    
    output_dir = Path("output/param_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n測試文字: '{test_text}'")
    print(f"輸出目錄: {output_dir}\n")
    
    for name, config in configs.items():
        print(f"🎚️  配置: {name}")
        print(f"   參數: {config if config else '預設'}")
        
        try:
            result = tts.synthesize(
                text=test_text,
                language="zh-cn",
                **config
            )
            
            output_file = output_dir / f"{name}.wav"
            tts.synthesizer.save_wav(result.audio, str(output_file))
            print(f"   ✅ 已儲存: {output_file}")
        except Exception as e:
            print(f"   ❌ 失敗: {e}")
        
        print()


def test_custom_emotion():
    """測試自訂情感配置。"""
    print("\n" + "="*60)
    print("🎨 Test 4: 自訂情感配置")
    print("="*60)
    
    emotion_mgr = EmotionManager()
    
    # 添加自訂情感參數
    emotion_mgr.DEFAULT_EMOTION_PARAMS["cheerful"] = {
        "temperature": 1.15,
        "speed": 1.25,
        "repetition_penalty": 7.0,
        "top_p": 0.95,
    }
    
    print("\n✓ 已添加自訂情感 'cheerful'")
    print(f"   配置: {emotion_mgr.DEFAULT_EMOTION_PARAMS['cheerful']}")
    
    # 測試自訂情感
    config = emotion_mgr.get_emotion_config(emotion="cheerful")
    print(f"\n✓ 取得配置: {config}")
    
    # 如果有參考音訊，可以手動添加
    cheerful_audio = Path("resource/emotions/cheerful.wav")
    if cheerful_audio.exists():
        emotion_mgr.add_emotion("cheerful", str(cheerful_audio))
        print(f"\n✓ 已添加參考音訊: {cheerful_audio}")


def check_environment():
    """檢查環境設定。"""
    print("\n" + "="*60)
    print("⚙️  環境檢查")
    print("="*60)
    
    # 檢查情感音訊目錄
    emotion_dir = os.getenv("EMOTION_AUDIO_DIR", "resource/emotions")
    emotion_path = Path(emotion_dir)
    
    print(f"\n情感音訊目錄: {emotion_dir}")
    print(f"  存在: {'✓' if emotion_path.exists() else '✗'}")
    
    if emotion_path.exists():
        wav_files = list(emotion_path.glob("*.wav"))
        print(f"  WAV 檔案數量: {len(wav_files)}")
        if wav_files:
            print("  可用的情感:")
            for wav_file in wav_files:
                print(f"    - {wav_file.stem}: {wav_file}")
    else:
        print(f"\n⚠️  目錄不存在，將創建: {emotion_dir}")
        emotion_path.mkdir(parents=True, exist_ok=True)
        print("  ✓ 已創建目錄")
        print("\n💡 提示: 請將情感參考音訊放入此目錄")
        print("  例如: happy.wav, sad.wav, neutral.wav")
    
    # 檢查預設參考音訊
    default_speaker = os.getenv("TTS_SPEAKER_WAV")
    if default_speaker:
        print(f"\n預設參考音訊: {default_speaker}")
        print(f"  存在: {'✓' if Path(default_speaker).exists() else '✗'}")


def main():
    """執行所有測試。"""
    print("\n" + "="*70)
    print(" 🎭 情感控制功能測試")
    print("="*70)
    print("\n這個測試會驗證 EmotionManager 和 TTS 的整合。")
    print("請確保您的環境已正確設定。\n")
    
    # 環境檢查
    check_environment()
    
    # 選單
    tests = [
        ("1", "EmotionManager 基本功能", test_emotion_manager),
        ("2", "情感控制 + TTS 整合", test_emotion_with_tts),
        ("3", "參數對比測試", test_parameter_comparison),
        ("4", "自訂情感配置", test_custom_emotion),
        ("0", "執行所有測試", None),
    ]
    
    print("\n請選擇要執行的測試:")
    for code, name, _ in tests:
        print(f"  [{code}] {name}")
    print()
    
    choice = input("請輸入選項 (0-4，直接按 Enter 執行全部): ").strip()
    
    if not choice or choice == "0":
        # 執行所有測試
        for _, _, func in tests[:-1]:  # 排除 "執行所有測試" 本身
            func()
    else:
        # 執行單個測試
        for code, _, func in tests:
            if code == choice and func:
                func()
                break
        else:
            print("❌ 無效的選項！")
            return
    
    print("\n" + "="*70)
    print("✅ 測試完成！")
    print("="*70)
    print("\n💡 下一步:")
    print("  1. 檢查 output/ 目錄下的音訊檔案")
    print("  2. 在 app_turn.py 中啟用情感控制:")
    print("     voice_agent = setup_voice_agent(enable_emotion_control=True)")
    print("  3. 錄製或生成情感參考音訊放入 resource/emotions/")
    print("\n📖 詳細文檔: docs/EMOTION_INTEGRATION.md")
    print()


if __name__ == "__main__":
    main()
