"""情感音訊管理器 - 用於控制 TTS 的情感和語氣。"""

import os
from pathlib import Path
from typing import Optional, Dict


class EmotionManager:
    """
    管理不同情感的參考音訊，用於 TTS 情感控制。
    
    通過使用不同情感的參考音訊，可以讓 TTS 生成相應情感的語音。
    """
    
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
