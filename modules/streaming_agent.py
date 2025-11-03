"""流式 Voice Agent - 實現即時回應的語音助理。"""

import re
from typing import Tuple, Optional, Iterator
import numpy as np

from .stt.base import STTEngine, TranscriptionResult
from .tts.base import TTSEngine, TTSResult
from .llm.base import LLMEngine
from .tools.manager import ToolManager


class StreamingVoiceAgent:
    """
    流式語音助理代理，實現類似 ElevenLabs 的即時回應。
    
    工作原理：
    1. STT 轉錄使用者語音
    2. LLM 流式生成回應（逐字輸出）
    3. 將 LLM 輸出按句子分割
    4. 每個句子立即送入 TTS 合成
    5. 逐句返回音訊，而非等待完整回應
    """
    
    def __init__(
        self,
        stt_engine: STTEngine,
        llm_engine: LLMEngine,
        tts_engine: TTSEngine,
        tool_manager: Optional[ToolManager] = None,
        enable_llm: bool = True,
        sentence_delimiters: str = r'[。！？\.!?;；]',
        min_sentence_length: int = 5,
    ):
        """
        初始化流式語音助理。
        
        Args:
            stt_engine: 語音辨識引擎
            llm_engine: 語言模型引擎
            tts_engine: 語音合成引擎
            tool_manager: 工具管理器（可選）
            enable_llm: 是否啟用 LLM
            sentence_delimiters: 句子分隔符的正則表達式
            min_sentence_length: 最小句子長度（字符數）
        """
        self.stt = stt_engine
        self.llm = llm_engine
        self.tts = tts_engine
        self.tool_manager = tool_manager
        self.enable_llm = enable_llm
        self.sentence_delimiters = sentence_delimiters
        self.min_sentence_length = min_sentence_length
        
        print("[StreamingVoiceAgent] Initialized with streaming support")
        print(f"[StreamingVoiceAgent] STT: {type(stt_engine).__name__}")
        print(f"[StreamingVoiceAgent] LLM: {type(llm_engine).__name__} (streaming enabled)")
        print(f"[StreamingVoiceAgent] TTS: {type(tts_engine).__name__}")
        if tool_manager and tool_manager.has_tools():
            print(f"[StreamingVoiceAgent] Tools: {tool_manager.list_tools()}")
    
    def process_audio_stream(
        self,
        audio: Tuple[int, np.ndarray],
    ) -> Iterator[Tuple[TTSResult, str]]:
        """
        流式處理音訊輸入，即時返回語音回應。
        
        流程：
        1. STT 轉錄音訊
        2. LLM 流式生成回應
        3. 分句並立即 TTS 合成
        4. 逐句 yield 音訊
        
        Args:
            audio: (sample_rate, audio_data) 的元組
            
        Yields:
            (tts_result, sentence): TTS 音訊結果和對應的句子
        """
        print("[StreamingVoiceAgent] Processing audio input with streaming...")
        
        # 1. 語音轉文字（這步驟無法串流，必須完整轉錄）
        transcription = self.stt.transcribe(audio)
        print(f"[StreamingVoiceAgent] Transcription: '{transcription.text}'")
        
        if not transcription.text:
            print("[StreamingVoiceAgent] No speech detected")
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
    
    def process_text_stream(
        self, 
        text: str, 
        language: Optional[str] = None
    ) -> Iterator[Tuple[TTSResult, str]]:
        """
        流式處理文字輸入。
        
        Args:
            text: 輸入文字
            language: 語言代碼（可選）
            
        Yields:
            (tts_result, sentence): TTS 音訊結果和對應的句子
        """
        print(f"[StreamingVoiceAgent] Processing text input with streaming: '{text}'")
        
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
        
        buffer = ""  # 累積未完成的句子
        full_response = ""  # 累積完整回應（用於檢查工具調用）
        
        # 流式獲取 LLM 回應
        for chunk in self.llm.query_stream(prompt, system_prompt=system_prompt):
            buffer += chunk
            full_response += chunk
            
            # 檢查是否有完整的句子
            sentences = self._extract_sentences(buffer)
            
            for sentence in sentences:
                if len(sentence.strip()) >= self.min_sentence_length:
                    print(f"[StreamingVoiceAgent] Synthesizing sentence: '{sentence}'")
                    
                    # 立即合成這個句子
                    try:
                        tts_result = self.tts.synthesize(
                            text=sentence,
                            language=language,
                        )
                        yield tts_result, sentence
                    except Exception as e:
                        print(f"[StreamingVoiceAgent] TTS failed for sentence: {e}")
                        continue
                    
                    # 從 buffer 中移除已處理的句子
                    buffer = buffer.replace(sentence, "", 1).lstrip()
        
        # 處理剩餘的 buffer（最後一句可能沒有標點符號）
        if buffer.strip() and len(buffer.strip()) >= self.min_sentence_length:
            print(f"[StreamingVoiceAgent] Synthesizing final buffer: '{buffer.strip()}'")
            try:
                tts_result = self.tts.synthesize(
                    text=buffer.strip(),
                    language=language,
                )
                yield tts_result, buffer.strip()
            except Exception as e:
                print(f"[StreamingVoiceAgent] TTS failed for final buffer: {e}")
        
        # 檢查是否有工具調用
        if self.tool_manager:
            tool_call = self.tool_manager.parse_tool_call(full_response)
            if tool_call:
                tool_name, parameters = tool_call
                print(f"[StreamingVoiceAgent] Executing tool: {tool_name}")
                
                # 執行工具
                tool_result = self.tool_manager.execute_tool(tool_name, **parameters)
                
                # 將工具結果告訴 LLM，讓它生成最終回應
                follow_up_prompt = (
                    f"工具 '{tool_name}' 執行結果：\n"
                    f"{tool_result}\n\n"
                    f"請根據這個結果給使用者一個友善的回應。"
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
                                print(f"[StreamingVoiceAgent] TTS failed: {e}")
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
                        print(f"[StreamingVoiceAgent] TTS failed: {e}")
    
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
        僅執行語音辨識。
        
        Args:
            audio: (sample_rate, audio_data) 的元組
            
        Returns:
            TranscriptionResult: 辨識結果
        """
        return self.stt.transcribe(audio)
