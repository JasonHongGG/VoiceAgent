"""情感音訊管理器 - 用於控制 TTS 的情感和語氣。"""

import os
import re
from pathlib import Path
from typing import Optional, Dict, Any


class EmotionManager:
    """
    管理不同情感的參考音訊和 TTS 參數，用於 TTS 情感控制。
    
    通過使用不同情感的參考音訊和參數配置，可以讓 TTS 生成相應情感的語音。
    支援兩種控制方式：
    1. 參考音訊 (speaker_wav) - 控制基礎音色和風格
    2. 參數調整 (temperature, speed 等) - 微調情感表達
    """
    
    # 預設的情感參數配置
    DEFAULT_EMOTION_PARAMS = {
        "neutral": {
            "temperature": 0.4,
            "speed": 1.0,
            "repetition_penalty": 12.0,
            "top_p": 0.75,
        },
        "happy": {
            "temperature": 1.0,
            "speed": 1.1,
            "repetition_penalty": 8.0,
            "top_p": 0.9,
        },
        "excited": {
            "temperature": 1.1,
            "speed": 1.2,
            "repetition_penalty": 8.0,
            "top_p": 0.9,
        },
        "sad": {
            "temperature": 0.7,
            "speed": 0.85,
            "repetition_penalty": 12.0,
            "top_p": 0.8,
        },
        "angry": {
            "temperature": 0.9,
            "speed": 1.15,
            "repetition_penalty": 10.0,
            "top_p": 0.85,
        },
        "gentle": {
            "temperature": 0.65,
            "speed": 0.9,
            "repetition_penalty": 11.0,
            "top_p": 0.8,
        },
        "professional": {
            "temperature": 0.5,
            "speed": 0.95,
            "repetition_penalty": 15.0,
            "top_p": 0.75,
        },
    }
    
    def __init__(self, emotion_audio_dir: Optional[str] = None):
        """
        初始化情感管理器。
        
        Args:
            emotion_audio_dir: 存放情感參考音訊的目錄路徑
        """
        self.emotion_audio_dir = emotion_audio_dir or os.getenv("EMOTION_AUDIO_DIR", "resource/emotions")
        self.emotion_map: Dict[str, str] = {}
        
        # 載入情感音訊映射
        self._load_emotion_map()
    
    def _load_emotion_map(self):
        """從目錄載入情感音訊映射。"""
        emotion_dir = Path(self.emotion_audio_dir)
        
        if not emotion_dir.exists():
            print(f"[EmotionManager] Emotion directory '{self.emotion_audio_dir}' not found, creating...")
            emotion_dir.mkdir(parents=True, exist_ok=True)
            return
        
        # 掃描目錄中的 .wav 檔案
        for audio_file in emotion_dir.glob("*.wav"):
            # 使用檔名（不含副檔名）作為情感名稱
            emotion_name = audio_file.stem.lower()
            self.emotion_map[emotion_name] = str(audio_file)
            print(f"[EmotionManager] Loaded emotion '{emotion_name}': {audio_file}")
        
        if not self.emotion_map:
            print(f"[EmotionManager] No emotion audio files found in '{self.emotion_audio_dir}'")
    
    def detect_emotion_from_text(self, text: str) -> str:
        """
        從文字內容自動偵測情感。
        
        Args:
            text: 輸入文字
            
        Returns:
            偵測到的情感名稱（如果無法判斷則返回 "neutral"）
        """
        text = text.lower()
        
        # 簡單的關鍵字匹配（可以根據需求擴展）
        emotion_keywords = {
            "happy": ["開心", "快樂", "太好了", "太棒了", "哈哈", "😊", "😄", "🎉"],
            "excited": ["興奮", "激動", "驚喜", "哇", "😍", "🤩"],
            "sad": ["難過", "傷心", "遺憾", "可惜", "😢", "😭"],
            "angry": ["生氣", "憤怒", "可惡", "😠", "😡"],
            "gentle": ["溫柔", "輕聲", "別擔心", "沒關係", "安慰"],
            "professional": ["報告", "數據", "分析", "根據", "顯示"],
        }
        
        # 檢查是否匹配任何情感關鍵字
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    print(f"[EmotionManager] Detected emotion '{emotion}' from keyword '{keyword}'")
                    return emotion
        
        # 預設返回中性
        return "neutral"
    
    def get_emotion_config(
        self,
        emotion: Optional[str] = None,
        text: Optional[str] = None,
        auto_detect: bool = True,
    ) -> Dict[str, Any]:
        """
        取得情感的完整配置（參考音訊 + TTS 參數）。
        
        Args:
            emotion: 指定的情感名稱
            text: 文字內容（用於自動偵測情感）
            auto_detect: 是否自動偵測情感
            
        Returns:
            包含 speaker_wav 和 TTS 參數的字典
        """
        # 1. 決定使用哪種情感
        final_emotion = None
        
        if emotion:
            # 優先使用指定的情感
            final_emotion = emotion.lower()
        elif auto_detect and text:
            # 從文字自動偵測
            final_emotion = self.detect_emotion_from_text(text)
        else:
            # 預設使用中性
            final_emotion = "neutral"
        
        print(f"[EmotionManager] Using emotion: {final_emotion}")
        
        # 2. 準備配置
        config = {}
        
        # 2.1 加入參考音訊（如果存在）
        speaker_wav = self.get_emotion_audio(final_emotion)
        if speaker_wav:
            config["speaker_wav"] = speaker_wav
        
        # 2.2 加入 TTS 參數
        emotion_params = self.DEFAULT_EMOTION_PARAMS.get(
            final_emotion,
            self.DEFAULT_EMOTION_PARAMS["neutral"]
        )
        config.update(emotion_params)
        
        return config
    
    def get_emotion_audio(self, emotion: str) -> Optional[str]:
        """
        取得指定情感的參考音訊路徑。
        
        Args:
            emotion: 情感名稱（如 "happy", "sad", "angry", "neutral" 等）
            
        Returns:
            參考音訊的檔案路徑，如果找不到則返回 None
        """
        emotion = emotion.lower()
        audio_path = self.emotion_map.get(emotion)
        
        if audio_path:
            print(f"[EmotionManager] Selected emotion '{emotion}': {audio_path}")
        else:
            print(f"[EmotionManager] Emotion '{emotion}' not found. Available: {list(self.emotion_map.keys())}")
        
        return audio_path
    
    def list_emotions(self) -> list[str]:
        """列出所有可用的情感。"""
        return list(self.emotion_map.keys())
    
    def add_emotion(self, emotion: str, audio_path: str):
        """
        手動添加情感音訊映射。
        
        Args:
            emotion: 情感名稱
            audio_path: 音訊檔案路徑
        """
        if not Path(audio_path).is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        self.emotion_map[emotion.lower()] = audio_path
        print(f"[EmotionManager] Added emotion '{emotion}': {audio_path}")


# 全域情感管理器實例（可選）
_global_emotion_manager = None


def get_emotion_manager() -> EmotionManager:
    """取得全域情感管理器實例。"""
    global _global_emotion_manager
    if _global_emotion_manager is None:
        _global_emotion_manager = EmotionManager()
    return _global_emotion_manager
