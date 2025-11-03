"""Whisper-based Speech-to-Text implementation."""

import tempfile
from typing import Tuple, Optional
import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel

from .base import STTEngine, TranscriptionResult
from ..utils.audio_utils import to_mono_and_normalize


class FasterWhisperSTT(STTEngine):
    """使用 Faster Whisper 的 STT 引擎。"""
    
    def __init__(
        self,
        model_size: str = "medium",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = False,
    ):
        """
        初始化 Whisper STT 引擎。
        
        Args:
            model_size: Whisper 模型大小 (tiny, base, small, medium, large)
            device: 使用的裝置 ("cuda" 或 "cpu")，None 則自動選擇
            compute_type: 計算類型 ("float16", "int8" 等)，None 則自動選擇
            beam_size: Beam search 大小
            vad_filter: 是否使用 VAD 過濾
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"
        
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        
        print(f"Initializing Whisper STT model '{model_size}' on {device} with {compute_type}")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    def transcribe(self, audio: Tuple[int, np.ndarray]) -> TranscriptionResult:
        """
        將音訊轉換為文字。
        
        Args:
            audio: (sample_rate, audio_data) 的元組
            
        Returns:
            TranscriptionResult: 辨識結果
        """
        sr, data = audio
        
        # 轉單聲道並正規化
        data = to_mono_and_normalize(data)
        
        # 記錄偵錯資訊
        if data.size:
            dur = data.shape[-1] / float(sr)
            print(f"[WhisperSTT] Audio stats -> sr:{sr}, shape:{data.shape}, "
                  f"dtype:{data.dtype}, duration:{dur:.2f}s, "
                  f"min:{data.min():.3f}, max:{data.max():.3f}")
        else:
            print(f"[WhisperSTT] Audio stats -> sr:{sr}, EMPTY array")
        
        # 使用臨時 WAV 檔路徑，交由 faster-whisper 解碼/重採樣
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, data, sr)
            
            # 另存一份最近一次的輸入音訊，方便除錯
            try:
                sf.write("/tmp/fastrtc_last.wav", data, sr)
            except Exception:
                pass
            
            result = self._transcribe_file_internal(tmp.name)
        
        return result
    
    def transcribe_file(self, file_path: str) -> TranscriptionResult:
        """
        從檔案轉換為文字。
        
        Args:
            file_path: 音訊檔案路徑
            
        Returns:
            TranscriptionResult: 辨識結果
        """
        return self._transcribe_file_internal(file_path)
    
    def _transcribe_file_internal(self, file_path: str) -> TranscriptionResult:
        """內部方法：從檔案進行轉錄。"""
        segments, info = self.model.transcribe(
            file_path,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )
        
        language = info.language if info else None
        print(f"[WhisperSTT] Detected language: {language}")
        
        # 將 segments 轉成 list 並提取文字
        segments_list = list(segments)
        print(f"[WhisperSTT] Transcription: {len(segments_list)} segments")
        
        transcript = "".join(seg.text for seg in segments_list).strip()
        
        for seg in segments_list:
            print(f"[WhisperSTT] Segment: {seg.text}")
        
        return TranscriptionResult(
            text=transcript if transcript else "",
            language=language,
            confidence=1.0  # faster-whisper 沒有直接提供整體信心分數
        )
