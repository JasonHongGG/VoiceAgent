"""Voice Agent - 整合 STT, LLM, 和 TTS 的高階介面。"""

from typing import Tuple, Optional
import numpy as np

from .stt.base import STTEngine, TranscriptionResult
from .tts.base import TTSEngine, TTSResult
from .llm.base import LLMEngine, LLMResponse


class VoiceAgent:
    """
    語音助理代理，整合語音辨識、語言模型和語音合成。
    
    這個類別提供了一個簡單的介面來處理語音輸入，通過 LLM 生成回應，
    並將回應轉換為語音輸出。可以在 fastRTC 或其他框架中使用。
    """
    
    def __init__(
        self,
        stt_engine: STTEngine,
        llm_engine: LLMEngine,
        tts_engine: TTSEngine,
    ):
        """
        初始化語音助理。
        
        Args:
            stt_engine: 語音辨識引擎
            llm_engine: 語言模型引擎
            tts_engine: 語音合成引擎
            enable_llm: 是否啟用 LLM（若為 False，則直接將 STT 結果轉為語音）
        """
        self.enable_llm = True
        self.stt = stt_engine
        self.llm = llm_engine
        self.tts = tts_engine
        
        print("[VoiceAgent] Initialized successfully")
        print(f"[VoiceAgent] STT: {type(stt_engine).__name__}")
        print(f"[VoiceAgent] LLM: {type(llm_engine).__name__}")
        print(f"[VoiceAgent] TTS: {type(tts_engine).__name__}")
    
    def process_audio(
        self,
        audio: Tuple[int, np.ndarray],
        return_transcript: bool = True,
    ) -> Tuple[Optional[TTSResult], Optional[TranscriptionResult], Optional[LLMResponse]]:
        """
        處理音訊輸入，返回語音回應。
        
        Args:
            audio: (sample_rate, audio_data) 的元組
            return_transcript: 是否返回轉錄結果
            
        Returns:
            (tts_result, transcription, llm_response) 的元組
            - tts_result: TTS 合成的音訊結果
            - transcription: STT 辨識結果（如果 return_transcript=True）
            - llm_response: LLM 回應
        """
        print("[VoiceAgent] Processing audio input...")
        
        # 1. 語音轉文字
        transcription = self.stt.transcribe(audio)
        print(f"[VoiceAgent] Transcription: '{transcription.text}'")
        
        if not transcription.text:
            print("[VoiceAgent] No speech detected, skipping LLM and TTS")
            return None, transcription if return_transcript else None, None
        
        # 2. LLM 生成回應（如果啟用）
        llm_response = None
        response_text = transcription.text
        
        if self.enable_llm:
            llm_response = self.llm.query(transcription.text)
            response_text = llm_response.content
            print(f"[VoiceAgent] LLM response: '{response_text}'")
        
        if not response_text:
            print("[VoiceAgent] No text to synthesize")
            return None, transcription if return_transcript else None, llm_response
        
        # 3. 文字轉語音
        tts_result = self.tts.synthesize(
            text=response_text,
            language=transcription.language,
        )
        print(f"[VoiceAgent] TTS synthesized {len(tts_result.audio)} samples at {tts_result.sample_rate} Hz")
        
        return tts_result, transcription if return_transcript else None, llm_response
    
    def process_text(self, text: str, language: Optional[str] = None) -> TTSResult:
        """
        處理文字輸入，返回語音回應。
        
        Args:
            text: 輸入文字
            language: 語言代碼（可選）
            
        Returns:
            TTSResult: TTS 合成的音訊結果
        """
        print(f"[VoiceAgent] Processing text input: '{text}'")
        
        # 1. LLM 生成回應（如果啟用）
        response_text = text
        if self.enable_llm:
            llm_response = self.llm.query(text)
            response_text = llm_response.content
            print(f"[VoiceAgent] LLM response: '{response_text}'")
        
        # 2. 文字轉語音
        tts_result = self.tts.synthesize(text=response_text, language=language)
        print(f"[VoiceAgent] TTS synthesized {len(tts_result.audio)} samples at {tts_result.sample_rate} Hz")
        
        return tts_result
    
    def transcribe_audio(self, audio: Tuple[int, np.ndarray]) -> TranscriptionResult:
        """
        僅執行 STT。
        
        Args:
            audio: (sample_rate, audio_data) 的元組
            
        Returns:
            TranscriptionResult: 辨識結果
        """
        return self.stt.transcribe(audio)
    
    def synthesize_speech(
        self,
        text: str,
        language: Optional[str] = None,
    ) -> TTSResult:
        """
        僅執行 TTS 。
        
        Args:
            text: 要合成的文字
            language: 語言代碼（可選）
            
        Returns:
            TTSResult: 合成結果
        """
        return self.tts.synthesize(text=text, language=language)
    
    def query_llm(self, prompt: str) -> LLMResponse:
        """
        僅執行 LLM 查詢。
        
        Args:
            prompt: 提示訊息
            
        Returns:
            LLMResponse: LLM 回應
        """
        return self.llm.query(prompt)
