"""
🎚️ TTS 參數調整演示

這個腳本展示如何使用 XTTS 的各種參數來控制語音的情感和風格。
"""

import os
from pathlib import Path
from modules.tts.coqui_tts import CoquiTTS


def demo_temperature_control():
    """演示 temperature 參數對情感表達的影響。"""
    print("\n" + "="*60)
    print("🌡️  Demo 1: Temperature 參數控制")
    print("="*60)
    
    tts = CoquiTTS()
    test_text = "今天天氣真好！我們一起出去玩吧！"
    
    # 測試不同的 temperature 值
    temperatures = [
        (0.3, "平淡、機械"),
        (0.6, "自然、穩定"),
        (0.9, "有表現力"),
        (1.2, "豐富、熱情"),
    ]
    
    output_dir = Path("output/temperature_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n測試文本: '{test_text}'")
    print(f"輸出目錄: {output_dir}\n")
    
    for temp, desc in temperatures:
        print(f"🎚️  Temperature = {temp} ({desc})")
        
        result = tts.synthesize(
            text=test_text,
            language="zh-cn",
            temperature=temp
        )
        
        # 儲存音訊
        output_file = output_dir / f"temp_{temp}.wav"
        tts.synthesizer.save_wav(result.audio, str(output_file))
        print(f"   ✅ 已儲存: {output_file}")
        print()


def demo_speed_control():
    """演示 speed 參數對語速的影響。"""
    print("\n" + "="*60)
    print("🏃 Demo 2: Speed 參數控制")
    print("="*60)
    
    tts = CoquiTTS()
    test_text = "請仔細聆聽這段重要訊息。"
    
    # 測試不同的 speed 值
    speeds = [
        (0.7, "慢速（教學、重要訊息）"),
        (1.0, "正常速度"),
        (1.3, "快速（時間緊迫）"),
    ]
    
    output_dir = Path("output/speed_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n測試文本: '{test_text}'")
    print(f"輸出目錄: {output_dir}\n")
    
    for speed, desc in speeds:
        print(f"⚡ Speed = {speed} ({desc})")
        
        result = tts.synthesize(
            text=test_text,
            language="zh-cn",
            speed=speed
        )
        
        # 儲存音訊
        output_file = output_dir / f"speed_{speed}.wav"
        tts.synthesizer.save_wav(result.audio, str(output_file))
        print(f"   ✅ 已儲存: {output_file}")
        print()


def demo_emotion_presets():
    """演示不同情感場景的參數預設配置。"""
    print("\n" + "="*60)
    print("🎭 Demo 3: 情感預設配置")
    print("="*60)
    
    tts = CoquiTTS()
    
    # 定義情感預設
    EMOTION_CONFIGS = {
        "neutral": {
            "text": "今日天氣預報：多雲，溫度攝氏二十五度。",
            "params": {
                "temperature": 0.4,
                "speed": 1.0,
                "repetition_penalty": 12.0,
                "top_p": 0.75
            },
            "desc": "中性/播報"
        },
        "happy": {
            "text": "太棒了！我們成功了！",
            "params": {
                "temperature": 1.0,
                "speed": 1.1,
                "repetition_penalty": 8.0,
                "top_p": 0.9
            },
            "desc": "開心/興奮"
        },
        "sad": {
            "text": "很遺憾聽到這個消息...",
            "params": {
                "temperature": 0.7,
                "speed": 0.85,
                "repetition_penalty": 12.0,
                "top_p": 0.8
            },
            "desc": "悲傷/同情"
        },
        "professional": {
            "text": "根據最新報告，我們需要調整策略。",
            "params": {
                "temperature": 0.5,
                "speed": 0.95,
                "repetition_penalty": 15.0,
                "top_p": 0.75
            },
            "desc": "專業/正式"
        },
        "gentle": {
            "text": "別擔心，一切都會好起來的。",
            "params": {
                "temperature": 0.65,
                "speed": 0.9,
                "repetition_penalty": 11.0,
                "top_p": 0.8
            },
            "desc": "溫柔/安慰"
        }
    }
    
    output_dir = Path("output/emotion_presets")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n輸出目錄: {output_dir}\n")
    
    for emotion, config in EMOTION_CONFIGS.items():
        print(f"😊 {emotion.upper()} - {config['desc']}")
        print(f"   文本: '{config['text']}'")
        print(f"   參數: {config['params']}")
        
        result = tts.synthesize(
            text=config['text'],
            language="zh-cn",
            **config['params']
        )
        
        # 儲存音訊
        output_file = output_dir / f"{emotion}.wav"
        tts.synthesizer.save_wav(result.audio, str(output_file))
        print(f"   ✅ 已儲存: {output_file}")
        print()


def demo_combined_control():
    """演示結合參考音訊和參數調整。"""
    print("\n" + "="*60)
    print("🎨 Demo 4: 參考音訊 + 參數調整")
    print("="*60)
    
    tts = CoquiTTS()
    test_text = "真的太棒了！我好開心！"
    
    # 檢查是否有參考音訊
    emotion_dir = Path(os.getenv("EMOTION_AUDIO_DIR", "resource/emotions"))
    happy_audio = emotion_dir / "happy.wav"
    
    output_dir = Path("output/combined_control")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n測試文本: '{test_text}'")
    print(f"輸出目錄: {output_dir}\n")
    
    # 測試 1: 僅使用預設參數
    print("📌 Test 1: 預設參數（無參考音訊）")
    result1 = tts.synthesize(
        text=test_text,
        language="zh-cn"
    )
    output_file1 = output_dir / "default.wav"
    tts.synthesizer.save_wav(result1.audio, str(output_file1))
    print(f"   ✅ 已儲存: {output_file1}\n")
    
    # 測試 2: 調整參數（無參考音訊）
    print("📌 Test 2: 調整參數（temperature=1.1, speed=1.2）")
    result2 = tts.synthesize(
        text=test_text,
        language="zh-cn",
        temperature=1.1,
        speed=1.2
    )
    output_file2 = output_dir / "params_only.wav"
    tts.synthesizer.save_wav(result2.audio, str(output_file2))
    print(f"   ✅ 已儲存: {output_file2}\n")
    
    # 測試 3: 使用參考音訊（如果存在）
    if happy_audio.exists():
        print(f"📌 Test 3: 使用參考音訊 ({happy_audio})")
        result3 = tts.synthesize(
            text=test_text,
            language="zh-cn",
            speaker_wav=str(happy_audio)
        )
        output_file3 = output_dir / "speaker_wav_only.wav"
        tts.synthesizer.save_wav(result3.audio, str(output_file3))
        print(f"   ✅ 已儲存: {output_file3}\n")
        
        # 測試 4: 組合使用
        print("📌 Test 4: 參考音訊 + 參數調整（最佳效果）")
        result4 = tts.synthesize(
            text=test_text,
            language="zh-cn",
            speaker_wav=str(happy_audio),
            temperature=1.1,
            speed=1.2,
            top_p=0.9
        )
        output_file4 = output_dir / "combined.wav"
        tts.synthesizer.save_wav(result4.audio, str(output_file4))
        print(f"   ✅ 已儲存: {output_file4}\n")
    else:
        print(f"⚠️  參考音訊不存在: {happy_audio}")
        print(f"   請先創建情感參考音訊檔案")
        print()


def demo_parameter_comparison():
    """演示所有參數的對比測試。"""
    print("\n" + "="*60)
    print("📊 Demo 5: 參數對比測試")
    print("="*60)
    
    tts = CoquiTTS()
    test_text = "這是一個測試句子"
    
    output_dir = Path("output/parameter_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n測試文本: '{test_text}'")
    print(f"輸出目錄: {output_dir}\n")
    
    # 基準測試（預設值）
    print("📌 Baseline: 預設參數")
    baseline = tts.synthesize(
        text=test_text,
        language="zh-cn"
    )
    baseline_file = output_dir / "baseline.wav"
    tts.synthesizer.save_wav(baseline.audio, str(baseline_file))
    print(f"   ✅ 已儲存: {baseline_file}\n")
    
    # 測試各個參數的影響
    tests = [
        {
            "name": "high_temperature",
            "desc": "高 temperature (1.3)",
            "params": {"temperature": 1.3}
        },
        {
            "name": "low_temperature",
            "desc": "低 temperature (0.3)",
            "params": {"temperature": 0.3}
        },
        {
            "name": "fast_speed",
            "desc": "快速 (1.5x)",
            "params": {"speed": 1.5}
        },
        {
            "name": "slow_speed",
            "desc": "慢速 (0.7x)",
            "params": {"speed": 0.7}
        },
        {
            "name": "high_repetition_penalty",
            "desc": "高重複懲罰 (18.0)",
            "params": {"repetition_penalty": 18.0}
        },
        {
            "name": "low_top_p",
            "desc": "低 top_p (0.5)",
            "params": {"top_p": 0.5}
        },
        {
            "name": "high_top_p",
            "desc": "高 top_p (0.95)",
            "params": {"top_p": 0.95}
        },
    ]
    
    for test in tests:
        print(f"📌 Test: {test['desc']}")
        print(f"   參數: {test['params']}")
        
        result = tts.synthesize(
            text=test_text,
            language="zh-cn",
            **test['params']
        )
        
        output_file = output_dir / f"{test['name']}.wav"
        tts.synthesizer.save_wav(result.audio, str(output_file))
        print(f"   ✅ 已儲存: {output_file}")
        print()


def main():
    """執行所有演示。"""
    print("\n" + "="*70)
    print(" 🎚️  TTS 參數調整演示")
    print("="*70)
    print("\n這個演示會生成多個音訊檔案，展示不同參數的效果。")
    print("請在演示結束後聆聽並比較 output/ 目錄下的音訊檔案。\n")
    
    # 選單
    demos = [
        ("1", "Temperature 參數控制", demo_temperature_control),
        ("2", "Speed 參數控制", demo_speed_control),
        ("3", "情感預設配置", demo_emotion_presets),
        ("4", "參考音訊 + 參數調整", demo_combined_control),
        ("5", "參數對比測試", demo_parameter_comparison),
        ("0", "執行所有演示", None),
    ]
    
    print("請選擇要執行的演示:")
    for code, name, _ in demos:
        print(f"  [{code}] {name}")
    print()
    
    choice = input("請輸入選項 (0-5，直接按 Enter 執行全部): ").strip()
    
    if not choice or choice == "0":
        # 執行所有演示
        for _, _, func in demos[:-1]:  # 排除 "執行所有演示" 本身
            func()
    else:
        # 執行單個演示
        for code, _, func in demos:
            if code == choice and func:
                func()
                break
        else:
            print("❌ 無效的選項！")
            return
    
    print("\n" + "="*70)
    print("✅ 演示完成！")
    print("="*70)
    print("\n請檢查 output/ 目錄下的音訊檔案，比較不同參數的效果。")
    print("\n💡 建議:")
    print("  1. 使用音訊播放器依序播放同一組的檔案")
    print("  2. 注意情感表達、語速、穩定性的差異")
    print("  3. 根據您的需求選擇最合適的參數配置")
    print("\n📖 詳細文檔: docs/TTS_PARAMETERS.md")
    print()


if __name__ == "__main__":
    main()
