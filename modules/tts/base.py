"""Base interface for Text-to-Speech engines."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np


class TTSResult:
    """TTS 合成結果。"""
    
    def __init__(self, audio: np.ndarray, sample_rate: int):
        self.audio = audio
        self.sample_rate = sample_rate
    
    def as_tuple(self) -> Tuple[int, np.ndarray]:
        """返回 (sample_rate, audio) 元組格式。"""
        return (self.sample_rate, self.audio)


class TTSEngine(ABC):
    """Text-to-Speech 引擎的基礎類別。"""
    
    @abstractmethod
    def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
    ) -> TTSResult:
        """
        將文字合成為語音。
        
        Args:
            text: 要合成的文字
            language: 語言代碼 (例如 "en", "zh-cn")
            speaker: 說話者名稱
            
        Returns:
            TTSResult: 合成的音訊結果
        """
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> list[str]:
        """
        取得支援的語言列表。
        
        Returns:
            語言代碼列表
        """
        pass
    
    @abstractmethod
    def get_supported_speakers(self) -> list[str]:
        """
        取得支援的說話者列表。
        
        Returns:
            說話者名稱列表
        """
        pass
