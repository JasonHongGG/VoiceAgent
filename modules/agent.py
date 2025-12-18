"""Voice Agent - 整合 STT, LLM, 和 TTS 的高階介面。"""

import re
from dataclasses import dataclass
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
    
    （已移除情感控制）
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
        self.conversation_history: list[dict[str, str]] = []
        
        mode = "streaming" if enable_streaming else "batch"
        print(f"[VoiceAgent] Initialized successfully (mode: {mode})")
        print(f"[VoiceAgent] STT: {type(stt_engine).__name__}")
        print(f"[VoiceAgent] LLM: {type(llm_engine).__name__} (enabled: {enable_llm})")
        print(f"[VoiceAgent] TTS: {type(tts_engine).__name__}")
        if tool_manager and tool_manager.has_tools():
            print(f"[VoiceAgent] Tools: {tool_manager.list_tools()}")

    def _synthesize(
        self,
        text: str,
        language: Optional[str] = None,
    ) -> TTSResult:
        """執行語音合成。"""
        tts_kwargs = {"text": text}
        if language is not None:
            tts_kwargs["language"] = language
        return self.tts.synthesize(**tts_kwargs)

    def reset_history(self) -> None:
        """重置對話記憶。"""
        self.conversation_history = []

    def _append_history(self, role: str, content: str) -> None:
        self.conversation_history.append({"role": role, "content": content})

    def _build_system_prompt(self) -> str:
        base = getattr(self.llm, "default_system_prompt", None) or ""
        if self.tool_manager and self.tool_manager.has_tools():
            tool_desc = self.tool_manager.get_tools_description()
            return f"{base}\n\n{tool_desc}" if base else tool_desc
        return base

    def _build_messages(self, user_text: str, system_prompt: Optional[str]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_text})
        return messages

    def _chat_with_history(self, user_text: str, system_prompt: Optional[str]) -> LLMResponse:
        return self.llm.chat(self._build_messages(user_text, system_prompt=system_prompt))

    def _tool_followup_prompt(self) -> str:
        # Keep prompt content consistent with previous behavior.
        return (
            "記帳工具已成功執行。\n\n"
            "請用**繁體中文**簡短地告訴使用者記帳已完成。\n"
            "例如：「已經幫你記錄了這筆消費！」\n"
            "不要使用英文，不要重複工具的技術細節。"
        )

    def _run_tools_if_needed(self, llm_text: str) -> Optional[tuple[str, dict]]:
        if not self.tool_manager:
            return None
        return self.tool_manager.parse_tool_call(llm_text)

    def _execute_tool(self, tool_name: str, parameters: dict) -> str:
        assert self.tool_manager is not None
        return self.tool_manager.execute_tool(tool_name, **parameters)

    def _chat_and_maybe_use_tool_batch(
        self,
        user_text: str,
        system_prompt: Optional[str],
    ) -> tuple[str, Optional[LLMResponse]]:
        """Run LLM for one turn; if tool call present, execute tool and ask LLM for a final response.

        Returns:
            (final_text, llm_response)
        """
        llm_response: Optional[LLMResponse] = None
        response_text = user_text

        if not self.enable_llm:
            return response_text, llm_response

        llm_response = self._chat_with_history(user_text, system_prompt=system_prompt)
        response_text = llm_response.content
        print(f"[VoiceAgent] LLM response: '{response_text}'")

        tool_call = self._run_tools_if_needed(response_text)
        if tool_call:
            tool_name, parameters = tool_call
            print(f"[VoiceAgent] Executing tool: {tool_name}")
            tool_result = self._execute_tool(tool_name, parameters)

            follow_up_prompt = self._tool_followup_prompt()
            llm_response = self._chat_with_history(follow_up_prompt, system_prompt=system_prompt)
            response_text = llm_response.content
            print(f"[VoiceAgent] Final response after tool: '{response_text}'")
            return response_text, llm_response

        # No tool call -> update history
        self._append_history("user", user_text)
        self._append_history("assistant", response_text)
        return response_text, llm_response


    @dataclass
    class _SentenceSplitResult:
        sentences: list[str]
        remainder: str


    def _split_sentences(text: str, sentence_delimiters: str) -> _SentenceSplitResult:
        """Split text into complete sentences based on delimiters, leaving unfinished remainder."""
        parts = re.split(f"({sentence_delimiters})", text)
        sentences: list[str] = []
        current = ""

        for part in parts:
            if not part:
                continue
            if re.match(sentence_delimiters, part):
                current += part
                if current.strip():
                    sentences.append(current)
                current = ""
            else:
                current += part

        return VoiceAgent._SentenceSplitResult(sentences=sentences, remainder=current)

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
        system_prompt = self._build_system_prompt() if self.enable_llm else None
        response_text, llm_response = self._chat_and_maybe_use_tool_batch(
            transcription.text,
            system_prompt=system_prompt,
        )
        
        if not response_text:
            print("[VoiceAgent] No text to synthesize")
            return None, transcription if return_transcript else None, llm_response
        
        # 3. 文字轉語音
        lang_to_use = self._detect_language_from_text(response_text, fallback=None)
        tts_result = self._synthesize(
            text=response_text,
            language=lang_to_use,
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
            tts_result = self._synthesize(
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
        
        system_prompt = self._build_system_prompt() if self.enable_llm else None
        response_text, _ = self._chat_and_maybe_use_tool_batch(
            text,
            system_prompt=system_prompt,
        )
        
        # 2. 文字轉語音
        lang_to_use = self._detect_language_from_text(response_text, fallback=None)
        tts_result = self._synthesize(text=response_text, language=lang_to_use)
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
            tts_result = self._synthesize(text=text, language=language)
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
        use_tools = bool(self.tool_manager and self.tool_manager.has_tools())

        system_prompt = self._build_system_prompt()
        if use_tools:
            print(f"[VoiceAgent] Using system prompt with {len(self.tool_manager)} tools")
        
        buffer = ""  # 累積未完成的句子
        full_response = ""  # 累積完整回應（用於檢查工具調用）
        tool_call_detected = False  # 標記是否偵測到工具調用

        messages = self._build_messages(prompt, system_prompt=system_prompt)
        spoken_response = ""

        for chunk in self.llm.chat_stream(messages, system_prompt=None):
            buffer += chunk
            full_response += chunk

            # 先檢查是否已經出現工具調用的開始標記
            if use_tools and "```tool" in full_response and not tool_call_detected:
                tool_call_detected = True
                print("[VoiceAgent] Detected tool call marker, accumulating full response...")

            # 如果偵測到工具調用，繼續累積不要 TTS
            if tool_call_detected:
                continue

            split_result = VoiceAgent._split_sentences(buffer, self.sentence_delimiters)
            buffer = split_result.remainder

            for sentence in split_result.sentences:
                if len(sentence.strip()) < self.min_sentence_length:
                    continue

                print(f"[VoiceAgent] Synthesizing sentence: '{sentence}'")
                try:
                    lang_to_use = self._detect_language_from_text(sentence, fallback=None)
                    tts_result = self._synthesize(
                        text=sentence,
                        language=lang_to_use,
                    )
                    yield tts_result, sentence
                    spoken_response += sentence
                except Exception as e:
                    print(f"[VoiceAgent] TTS failed for sentence: {e}")
                    continue
        
        # 檢查是否有工具調用（優先處理）
        if use_tools and self.tool_manager:
            tool_call = self.tool_manager.parse_tool_call(full_response)
            if tool_call:
                tool_name, parameters = tool_call
                print(f"[VoiceAgent] ✓ 執行工具: {tool_name}")
                
                # 執行工具
                tool_result = self.tool_manager.execute_tool(tool_name, **parameters)
                
                follow_up_prompt = self._tool_followup_prompt()
                
                follow_up_messages: list[dict[str, str]] = []
                if system_prompt:
                    follow_up_messages.append({"role": "system", "content": system_prompt})
                follow_up_messages.extend(self.conversation_history)
                follow_up_messages.append({"role": "assistant", "content": full_response})
                follow_up_messages.append({"role": "user", "content": follow_up_prompt})

                # 流式生成工具執行後的回應
                buffer = ""
                for chunk in self.llm.chat_stream(follow_up_messages, system_prompt=None):
                    buffer += chunk
                    split_result = VoiceAgent._split_sentences(buffer, self.sentence_delimiters)
                    buffer = split_result.remainder

                    for sentence in split_result.sentences:
                        if len(sentence.strip()) < self.min_sentence_length:
                            continue
                        try:
                            tts_result = self._synthesize(
                                text=sentence,
                                language=language,
                            )
                            yield tts_result, sentence
                            spoken_response += sentence
                        except Exception as e:
                            print(f"[VoiceAgent] TTS failed: {e}")
                            continue
                
                # 處理最後的 buffer
                if buffer.strip() and len(buffer.strip()) >= self.min_sentence_length:
                    try:
                        lang_to_use = self._detect_language_from_text(buffer.strip(), fallback=None)
                        tts_result = self._synthesize(
                            text=buffer.strip(),
                            language=lang_to_use,
                        )
                        yield tts_result, buffer.strip()
                        spoken_response += buffer.strip()
                    except Exception as e:
                        print(f"[VoiceAgent] TTS failed: {e}")
                
                # 工具調用已處理，直接返回
                self._append_history("user", prompt)
                self._append_history("assistant", spoken_response)
                return
        
        # 處理剩餘的 buffer（最後一句可能沒有標點符號）
        # 只有在沒有工具調用時才處理
        if buffer.strip() and len(buffer.strip()) >= self.min_sentence_length:
            print(f"[VoiceAgent] Synthesizing final buffer: '{buffer.strip()}'")
            try:
                lang_to_use = self._detect_language_from_text(buffer.strip(), fallback=None)
                tts_result = self._synthesize(
                    text=buffer.strip(),
                    language=lang_to_use,
                )
                yield tts_result, buffer.strip()
                spoken_response += buffer.strip()
            except Exception as e:
                print(f"[VoiceAgent] TTS failed for final buffer: {e}")

        # 更新對話記憶
        self._append_history("user", prompt)
        if spoken_response.strip():
            self._append_history("assistant", spoken_response)
    
    def _extract_sentences(self, text: str) -> list[str]:
        """從文字中提取完整的句子（保留舊 API，供相容/除錯使用）。"""
        return VoiceAgent._split_sentences(text, self.sentence_delimiters).sentences
    
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
        return self._synthesize(text=text, language=language)
    
    def query_llm(self, prompt: str) -> LLMResponse:
        """
        僅執行 LLM 查詢。
        
        Args:
            prompt: 提示訊息
            
        Returns:
            LLMResponse: LLM 回應
        """
        return self.llm.query(prompt)

    def _detect_language_from_text(self, text: str, fallback: Optional[str] = None) -> Optional[str]:
        """根據 LLM 文字粗略偵測語言，盡量對應 TTS 語言碼。"""
        if not text:
            return fallback

        # 依據 Unicode 範圍判斷主要語言
        if re.search(r"[\u4e00-\u9fff]", text):  # CJK Unified Ideographs -> 中文
            return "zh-cn"
        if re.search(r"[\uac00-\ud7af]", text):  # Hangul
            return "ko"
        if re.search(r"[\u3040-\u30ff]", text):  # Hiragana + Katakana
            return "ja"
        if re.search(r"[\u0400-\u04FF]", text):  # Cyrillic
            return "ru"

        # 檢查含重音的拉丁字母，粗略判斷西語/法語/德語
        if re.search(r"[áéíóúñ¿¡]", text, re.IGNORECASE):
            return "es"
        if re.search(r"[àâçéèêëïîôùûüÿœ]", text, re.IGNORECASE):
            return "fr"
        if re.search(r"[äöüß]", text, re.IGNORECASE):
            return "de"

        # 大多數為 ASCII 字母則推英文
        letters = re.findall(r"[A-Za-z]", text)
        if letters and len(letters) / max(len(text), 1) > 0.4:
            return "en"

        return fallback
