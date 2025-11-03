"""Coqui TTS implementation."""
import os
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from TTS.api import TTS

from .base import TTSEngine, TTSResult


class CoquiTTS(TTSEngine):
    """使用 Coqui TTS (XTTS) 的 TTS 引擎。"""
    
    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: Optional[str] = None,
    ):
        """
        初始化 Coqui TTS 引擎。
        
        Args:
            model_name: TTS 模型名稱
            device: 使用的裝置 ("cuda" 或 "cpu")，None 則自動選擇
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model_name = model_name
        self.device = device
        
        print(f"[CoquiTTS] Initializing TTS model '{model_name}' on {device}...")
        try:
            self.tts = TTS(model_name).to(device)
            print(f"[CoquiTTS] Successfully initialized on {device}")
        except Exception as exc:
            print(f"[CoquiTTS] Failed to initialize on {device}: {exc}")
            raise 
        
        # 取得支援的語言和說話者
        self._languages = self.tts.languages or []
        self._speakers = self.tts.speakers or []
        self._default_speaker = self._speakers[0] if self._speakers else None
    
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
            language: 語言代碼
            speaker: 說話者名稱
            
        Returns:
            TTSResult: 合成的音訊結果
        """
        # 準備 TTS 參數
        tts_kwargs = {"text": text}
        
        # 語言選擇
        self._select_language(tts_kwargs, language)
        
        # 說話者選擇
        self._select_speaker(tts_kwargs, speaker)
        
        # 合成音訊
        try:
            synth_audio = self.tts.tts(**tts_kwargs)
        except Exception as exc:
            print(f"[CoquiTTS] Synthesis failed on {self.device}: {exc}")
            raise
        
        # 轉換為 numpy array
        synth_audio = np.asarray(synth_audio, dtype=np.float32)
        
        # 取得採樣率
        sample_rate = self._get_sample_rate()
        
        return TTSResult(audio=synth_audio, sample_rate=sample_rate)
    
    def _select_language(self, tts_kwargs, detected: Optional[str]) -> Optional[str]:
        """選擇適當的語言。"""
        selected_language = None
        detected = detected.lower()
        
        # 特殊處理：中文 (在 coqui-tts 中為 "zh-cn")
        if detected == "zh":
            selected_language = "zh-cn"
        
        # 檢查是否在支援列表中
        elif detected in self._languages:
            selected_language = detected
        
        # 預設返回第一個支援的語言
        else:
            selected_language = self._languages[0]

        tts_kwargs["language"] = selected_language
        print(f"[CoquiTTS] Using language: {selected_language}")
        return selected_language

    def _select_speaker(self, tts_kwargs, speaker: Optional[str]) -> Optional[str]:
        """選擇適當的說話者。"""
        # 驗證參考音訊檔案
        reference_speaker = os.getenv("TTS_SPEAKER_WAV")
        if reference_speaker and not Path(reference_speaker).is_file():
            print(f"[CoquiTTS] Warning: reference_speaker '{reference_speaker}' not found, ignoring.")

        
        # 選擇 speaker_wav、speaker 或預設說話者
        if reference_speaker and Path(reference_speaker).is_file():
            tts_kwargs["speaker_wav"] = reference_speaker
            print(f"[CoquiTTS] Using speaker_wav: {reference_speaker}")
        elif speaker:
            tts_kwargs["speaker"] = speaker
            print(f"[CoquiTTS] Using speaker: {speaker}")
        elif self._default_speaker:
            tts_kwargs["speaker"] = self._default_speaker
            print(f"[CoquiTTS] Using default speaker: {self._default_speaker}")
        
    
    def _get_sample_rate(self) -> int:
        """取得模型的輸出採樣率。"""
        try:
            if hasattr(self.tts, "synthesizer") and self.tts.synthesizer:
                return self.tts.synthesizer.output_sample_rate
        except Exception:
            pass
        
        # 預設值
        return 24000
    
    def get_supported_languages(self) -> list[str]:
        """取得支援的語言列表。"""
        return self._languages.copy()
    
    def get_supported_speakers(self) -> list[str]:
        """取得支援的說話者列表。"""
        return self._speakers.copy()
