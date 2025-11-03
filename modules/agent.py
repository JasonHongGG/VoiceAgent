"""Voice Agent - 整合 STT, LLM, 和 TTS 的高階介面。"""

import re
from typing import Tuple, Optional, Iterator, Union
import numpy as np

from .stt.base import STTEngine, TranscriptionResult
from .tts.base import TTSEngine, TTSResult
from .llm.base import LLMEngine, LLMResponse
from .tools.manager import ToolManager


class VoiceAgent:
    """
    語音助理代理，整合語音辨識、語言模型和語音合成。
    
    這個類別提供了一個簡單的介面來處理語音輸入，通過 LLM 生成回應，
    並將回應轉換為語音輸出。可以在 fastRTC 或其他框架中使用。
    
    支援兩種模式：
    - 批次模式（enable_streaming=False）：等待完整回應後一次性返回
    - 串流模式（enable_streaming=True, 預設）：即時返回，類似 ElevenLabs
    """
    
    def __init__(
        self,
        stt_engine: STTEngine,
        llm_engine: LLMEngine,
        tts_engine: TTSEngine,
        tool_manager: Optional[ToolManager] = None,
        enable_llm: bool = True,
        enable_streaming: bool = True,
        sentence_delimiters: str = r'[。！？\.!?;；]',
        min_sentence_length: int = 5,
    ):
        """
        初始化語音助理。
        
        Args:
            stt_engine: 語音辨識引擎
            llm_engine: 語言模型引擎
            tts_engine: 語音合成引擎
            tool_manager: 工具管理器（可選）
            enable_llm: 是否啟用 LLM（若為 False，則直接將 STT 結果轉為語音）
            enable_streaming: 是否啟用串流模式（預設為 True）
            sentence_delimiters: 句子分隔符的正則表達式（僅串流模式使用）
            min_sentence_length: 最小句子長度（僅串流模式使用）
        """
        self.stt = stt_engine
        self.llm = llm_engine
        self.tts = tts_engine
        self.tool_manager = tool_manager
        self.enable_llm = enable_llm
        self.enable_streaming = enable_streaming
        self.sentence_delimiters = sentence_delimiters
        self.min_sentence_length = min_sentence_length
        
        mode = "streaming" if enable_streaming else "batch"
        print(f"[VoiceAgent] Initialized successfully (mode: {mode})")
        print(f"[VoiceAgent] STT: {type(stt_engine).__name__}")
        print(f"[VoiceAgent] LLM: {type(llm_engine).__name__} (enabled: {enable_llm})")
        print(f"[VoiceAgent] TTS: {type(tts_engine).__name__}")
        if tool_manager and tool_manager.has_tools():
            print(f"[VoiceAgent] Tools: {tool_manager.list_tools()}")
    
    def process_audio(
        self,
        audio: Tuple[int, np.ndarray],
        return_transcript: bool = True,
    ) -> Union[
        Tuple[Optional[TTSResult], Optional[TranscriptionResult], Optional[LLMResponse]],
        Iterator[Tuple[TTSResult, str]]
    ]:
        """
        處理音訊輸入，返回語音回應。
        
        根據 enable_streaming 設定，此方法會：
        - 批次模式：返回完整的 (tts_result, transcription, llm_response) 元組
        - 串流模式：返回 Iterator，逐句 yield (tts_result, sentence)
        
        Args:
            audio: (sample_rate, audio_data) 的元組
            return_transcript: 是否返回轉錄結果（僅批次模式）
            
        Returns:
            批次模式：(tts_result, transcription, llm_response) 的元組
            串流模式：Iterator[Tuple[TTSResult, str]]，逐句返回 (音訊, 句子)
        """
        if self.enable_streaming:
            return self._process_audio_stream(audio)
        else:
            return self._process_audio_batch(audio, return_transcript)
    
    def _process_audio_batch(
        self,
        audio: Tuple[int, np.ndarray],
        return_transcript: bool = True,
    ) -> Tuple[Optional[TTSResult], Optional[TranscriptionResult], Optional[LLMResponse]]:
        """批次模式：處理音訊輸入（內部方法）。"""
        print("[VoiceAgent] Processing audio input (batch mode)...")
        
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
            # 準備 system prompt（包含工具說明）
            system_prompt = None
            if self.tool_manager and self.tool_manager.has_tools():
                system_prompt = self.tool_manager.get_tools_description()
            
            llm_response = self.llm.query(transcription.text, system_prompt=system_prompt)
            response_text = llm_response.content
            print(f"[VoiceAgent] LLM response: '{response_text}'")
            
            # 檢查是否需要執行工具
            if self.tool_manager:
                tool_call = self.tool_manager.parse_tool_call(response_text)
                if tool_call:
                    tool_name, parameters = tool_call
                    print(f"[VoiceAgent] Executing tool: {tool_name}")
                    
                    # 執行工具
                    tool_result = self.tool_manager.execute_tool(tool_name, **parameters)
                    
                    # 將工具結果告訴 LLM，讓它生成最終回應
                    follow_up_prompt = (
                        f"工具 '{tool_name}' 執行結果：\n"
                        f"{tool_result}\n\n"
                        f"請根據這個結果給使用者一個友善的回應。"
                    )
                    llm_response = self.llm.query(follow_up_prompt)
                    response_text = llm_response.content
                    print(f"[VoiceAgent] Final response after tool: '{response_text}'")
        
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
    
    def _process_audio_stream(
        self,
        audio: Tuple[int, np.ndarray],
    ) -> Iterator[Tuple[TTSResult, str]]:
        """串流模式：處理音訊輸入（內部方法）。"""
        print("[VoiceAgent] Processing audio input (streaming mode)...")
        
        # 1. 語音轉文字（這步驟無法串流，必須完整轉錄）
        transcription = self.stt.transcribe(audio)
        print(f"[VoiceAgent] Transcription: '{transcription.text}'")
        
        if not transcription.text:
            print("[VoiceAgent] No speech detected")
            return
        
        # 2. 如果啟用 LLM，使用流式生成
        if self.enable_llm:
            yield from self._stream_llm_and_tts(
                transcription.text, 
                transcription.language
            )
        else:
            # 不使用 LLM，直接 TTS
            tts_result = self.tts.synthesize(
                text=transcription.text,
                language=transcription.language,
            )
            yield tts_result, transcription.text
    
    def process_text(
        self, 
        text: str, 
        language: Optional[str] = None
    ) -> Union[TTSResult, Iterator[Tuple[TTSResult, str]]]:
        """
        處理文字輸入，返回語音回應。
        
        根據 enable_streaming 設定，此方法會：
        - 批次模式：返回完整的 TTSResult
        - 串流模式：返回 Iterator，逐句 yield (tts_result, sentence)
        
        Args:
            text: 輸入文字
            language: 語言代碼（可選）
            
        Returns:
            批次模式：TTSResult
            串流模式：Iterator[Tuple[TTSResult, str]]
        """
        if self.enable_streaming:
            return self._process_text_stream(text, language)
        else:
            return self._process_text_batch(text, language)
    
    def _process_text_batch(self, text: str, language: Optional[str] = None) -> TTSResult:
        """批次模式：處理文字輸入（內部方法）。"""
        print(f"[VoiceAgent] Processing text input (batch mode): '{text}'")
        
        # 1. LLM 生成回應（如果啟用）
        response_text = text
        if self.enable_llm:
            # 準備 system prompt（包含工具說明）
            system_prompt = None
            if self.tool_manager and self.tool_manager.has_tools():
                system_prompt = self.tool_manager.get_tools_description()
            
            llm_response = self.llm.query(text, system_prompt=system_prompt)
            response_text = llm_response.content
            print(f"[VoiceAgent] LLM response: '{response_text}'")
            
            # 檢查是否需要執行工具
            if self.tool_manager:
                tool_call = self.tool_manager.parse_tool_call(response_text)
                if tool_call:
                    tool_name, parameters = tool_call
                    print(f"[VoiceAgent] Executing tool: {tool_name}")
                    
                    # 執行工具
                    tool_result = self.tool_manager.execute_tool(tool_name, **parameters)
                    
                    # 將工具結果告訴 LLM
                    follow_up_prompt = (
                        f"記帳工具已成功執行。\n\n"
                        f"請用**繁體中文**簡短地告訴使用者記帳已完成。\n"
                        f"例如：「已經幫你記錄了這筆消費！」\n"
                        f"不要使用英文，不要重複工具的技術細節。"
                    )
                    llm_response = self.llm.query(follow_up_prompt)
                    response_text = llm_response.content
                    print(f"[VoiceAgent] Final response after tool: '{response_text}'")
        
        # 2. 文字轉語音
        tts_result = self.tts.synthesize(text=response_text, language=language)
        print(f"[VoiceAgent] TTS synthesized {len(tts_result.audio)} samples at {tts_result.sample_rate} Hz")
        
        return tts_result
    
    def _process_text_stream(
        self, 
        text: str, 
        language: Optional[str] = None
    ) -> Iterator[Tuple[TTSResult, str]]:
        """串流模式：處理文字輸入（內部方法）。"""
        print(f"[VoiceAgent] Processing text input (streaming mode): '{text}'")
        
        if self.enable_llm:
            yield from self._stream_llm_and_tts(text, language)
        else:
            tts_result = self.tts.synthesize(text=text, language=language)
            yield tts_result, text
    
    def _stream_llm_and_tts(
        self, 
        prompt: str, 
        language: Optional[str] = None
    ) -> Iterator[Tuple[TTSResult, str]]:
        """
        內部方法：流式處理 LLM 生成和 TTS 合成。
        
        Args:
            prompt: 提示文字
            language: 語言代碼
            
        Yields:
            (tts_result, sentence): TTS 音訊結果和對應的句子
        """
        # 準備 system prompt（包含工具說明）
        system_prompt = None
        if self.tool_manager and self.tool_manager.has_tools():
            system_prompt = self.tool_manager.get_tools_description()
            print(f"[VoiceAgent] Using system prompt with {len(self.tool_manager)} tools")
        
        buffer = ""  # 累積未完成的句子
        full_response = ""  # 累積完整回應（用於檢查工具調用）
        tool_call_detected = False  # 標記是否偵測到工具調用
        
        # 流式獲取 LLM 回應
        for chunk in self.llm.query_stream(prompt, system_prompt=system_prompt):
            buffer += chunk
            full_response += chunk
            
            # 先檢查是否已經出現工具調用的開始標記
            if "```tool" in full_response and not tool_call_detected:
                tool_call_detected = True
                print("[VoiceAgent] Detected tool call marker, accumulating full response...")
            
            # 如果偵測到工具調用，繼續累積不要 TTS
            if tool_call_detected:
                continue
            
            # 檢查是否有完整的句子（如果沒有工具調用）
            sentences = self._extract_sentences(buffer)
            
            for sentence in sentences:
                if len(sentence.strip()) >= self.min_sentence_length:
                    print(f"[VoiceAgent] Synthesizing sentence: '{sentence}'")
                    
                    # 立即合成這個句子
                    try:
                        tts_result = self.tts.synthesize(
                            text=sentence,
                            language=language,
                        )
                        yield tts_result, sentence
                    except Exception as e:
                        print(f"[VoiceAgent] TTS failed for sentence: {e}")
                        continue
                    
                    # 從 buffer 中移除已處理的句子
                    buffer = buffer.replace(sentence, "", 1).lstrip()
        
        # 檢查是否有工具調用（優先處理）
        if self.tool_manager:
            tool_call = self.tool_manager.parse_tool_call(full_response)
            if tool_call:
                tool_name, parameters = tool_call
                print(f"[VoiceAgent] ✓ 執行工具: {tool_name}")
                
                # 執行工具
                tool_result = self.tool_manager.execute_tool(tool_name, **parameters)
                
                # 將工具結果告訴 LLM，讓它生成最終回應
                follow_up_prompt = (
                    f"記帳工具已成功執行。\n\n"
                    f"請用**繁體中文**簡短地告訴使用者記帳已完成。\n"
                    f"例如：「已經幫你記錄了這筆消費！」\n"
                    f"不要使用英文，不要重複工具的技術細節。"
                )
                
                # 流式生成工具執行後的回應
                buffer = ""
                for chunk in self.llm.query_stream(follow_up_prompt):
                    buffer += chunk
                    sentences = self._extract_sentences(buffer)
                    
                    for sentence in sentences:
                        if len(sentence.strip()) >= self.min_sentence_length:
                            try:
                                tts_result = self.tts.synthesize(
                                    text=sentence,
                                    language=language,
                                )
                                yield tts_result, sentence
                            except Exception as e:
                                print(f"[VoiceAgent] TTS failed: {e}")
                                continue
                            buffer = buffer.replace(sentence, "", 1).lstrip()
                
                # 處理最後的 buffer
                if buffer.strip() and len(buffer.strip()) >= self.min_sentence_length:
                    try:
                        tts_result = self.tts.synthesize(
                            text=buffer.strip(),
                            language=language,
                        )
                        yield tts_result, buffer.strip()
                    except Exception as e:
                        print(f"[VoiceAgent] TTS failed: {e}")
                
                # 工具調用已處理，直接返回
                return
        
        # 處理剩餘的 buffer（最後一句可能沒有標點符號）
        # 只有在沒有工具調用時才處理
        if buffer.strip() and len(buffer.strip()) >= self.min_sentence_length:
            print(f"[VoiceAgent] Synthesizing final buffer: '{buffer.strip()}'")
            try:
                tts_result = self.tts.synthesize(
                    text=buffer.strip(),
                    language=language,
                )
                yield tts_result, buffer.strip()
            except Exception as e:
                print(f"[VoiceAgent] TTS failed for final buffer: {e}")
    
    def _extract_sentences(self, text: str) -> list[str]:
        """
        從文字中提取完整的句子。
        
        Args:
            text: 輸入文字
            
        Returns:
            完整句子的列表
        """
        # 使用正則表達式分割句子
        # 保留分隔符
        parts = re.split(f'({self.sentence_delimiters})', text)
        
        sentences = []
        current_sentence = ""
        
        for i, part in enumerate(parts):
            if re.match(self.sentence_delimiters, part):
                # 這是一個分隔符，加到當前句子並完成
                current_sentence += part
                if current_sentence.strip():
                    sentences.append(current_sentence)
                current_sentence = ""
            else:
                # 這是句子內容
                current_sentence += part
        
        return sentences
    
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
