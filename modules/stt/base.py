"""Base interface for Speech-to-Text engines."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np


class TranscriptionResult:
    """語音辨識結果。"""
    
    def __init__(self, text: str, language: Optional[str] = None, confidence: float = 1.0):
        self.text = text
        self.language = language
        self.confidence = confidence
    
    def __str__(self) -> str:
        return self.text
    
    def __repr__(self) -> str:
        return f"TranscriptionResult(text='{self.text}', language='{self.language}', confidence={self.confidence})"


class STTEngine(ABC):
    """Speech-to-Text 引擎的基礎類別。"""
    
    @abstractmethod
    def transcribe(self, audio: Tuple[int, np.ndarray]) -> TranscriptionResult:
        """
        將音訊轉換為文字。
        
        Args:
            audio: (sample_rate, audio_data) 的元組
            
        Returns:
            TranscriptionResult: 辨識結果
        """
        pass
    
    @abstractmethod
    def transcribe_file(self, file_path: str) -> TranscriptionResult:
        """
        從檔案轉換為文字。
        
        Args:
            file_path: 音訊檔案路徑
            
        Returns:
            TranscriptionResult: 辨識結果
        """
        pass
