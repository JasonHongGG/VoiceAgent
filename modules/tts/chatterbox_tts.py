"""Chatterbox TTS implementation.

This wraps the `chatterbox-tts` python package (https://github.com/resemble-ai/chatterbox)
into the repo's `TTSEngine` interface.

Design goals:
- Keep behavior close to upstream examples (a single `generate()` per request).
- Only add minimal glue: device selection, language normalization, optional audio prompt,
    and converting output into `TTSResult`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .base import TTSEngine, TTSResult


def _resolve_device(device: Optional[str]) -> str:
    requested = (device or "").strip().lower() or None
    if requested in {"gpu"}:
        requested = "cuda"
    if requested in {"mps", "mpx"}:
        requested = "mps"

    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"

    if requested == "mps":
        # On linux this is typically unavailable; chatterbox will also validate.
        if not torch.backends.mps.is_available():
            return "cpu"
        return "mps"

    if requested in {"cpu", "cuda", "mps"}:
        return requested

    # Default behavior: prefer CUDA when available.
    return "cuda" if torch.cuda.is_available() else "cpu"


def _normalize_language_id(language: Optional[str]) -> str:
    if not language:
        return "en"

    lang = str(language).strip().lower()

    # Common aliases
    if lang in {"zh", "zh-cn", "zh-hans", "zh-hant", "zh-tw", "zh-hk", "zh-sg"}:
        return "zh"
    if lang in {"ja", "jp", "jpn", "japanese"}:
        return "ja"
    if lang in {"en", "en-us", "en-gb", "english"}:
        return "en"

    # Keep first subtag (e.g. fr-ca -> fr)
    if "-" in lang:
        lang = lang.split("-", 1)[0]

    return lang


class ChatterboxTTS(TTSEngine):
    """Use Resemble AI Chatterbox as a TTS engine."""

    def __init__(
        self,
        device: Optional[str] = None,
        audio_prompt_path: Optional[str] = None,
        # Match upstream defaults to minimize behavior differences.
        repetition_penalty: float = 2.0,
        min_p: float = 0.05,
        top_p: float = 1.0,
        temperature: float = 0.8,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
    ):
        self.device = _resolve_device(device)

        # Generation defaults (can be overridden per-call via kwargs).
        self.repetition_penalty = float(repetition_penalty)
        self.min_p = float(min_p)
        self.top_p = float(top_p)
        self.temperature = float(temperature)
        self.exaggeration = float(exaggeration)
        self.cfg_weight = float(cfg_weight)

        # Optional voice reference prompt
        self.audio_prompt_path = audio_prompt_path

        print(f"[ChatterboxTTS] Initializing multilingual model on device='{self.device}'...")

        # Import lazily so that importing this module doesn't force dependency usage.
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS as _UpstreamTTS
        from chatterbox.mtl_tts import SUPPORTED_LANGUAGES as _SUPPORTED

        self._tts = _UpstreamTTS.from_pretrained(device=torch.device(self.device))
        # SUPPORTED_LANGUAGES is a dict; list() yields language ids.
        self._languages = list(_SUPPORTED)

        self._sample_rate = int(getattr(self._tts, "sr", 24000))
        print(f"[ChatterboxTTS] Ready. sample_rate={self._sample_rate}, languages={len(self._languages)}")

    def _resolve_audio_prompt(
        self,
        speaker: Optional[str],
        speaker_wav: Optional[str],
    ) -> Optional[str]:
        # 1) explicit speaker_wav kwarg
        if speaker_wav and Path(speaker_wav).is_file():
            return speaker_wav

        # 2) treat speaker as an audio prompt path if it looks like a file
        if speaker and Path(speaker).is_file():
            return speaker

        # 3) explicit engine setting
        if self.audio_prompt_path and Path(self.audio_prompt_path).is_file():
            return self.audio_prompt_path

        # 4) env var overrides
        env_prompt = os.getenv("CHATTERBOX_AUDIO_PROMPT")
        if env_prompt and Path(env_prompt).is_file():
            return env_prompt

        # 5) reuse existing coqui env var if set
        ref = os.getenv("TTS_SPEAKER_WAV")
        if ref and Path(ref).is_file():
            return ref

        return None

    def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        **kwargs,
    ) -> TTSResult:
        language_id = _normalize_language_id(language)
        if language_id not in set(self._languages):
            raise ValueError(
                f"Chatterbox multilingual does not support language='{language}'. "
                f"Supported: {', '.join(self._languages)}"
            )

        if not text:
            return TTSResult(audio=np.zeros((0,), dtype=np.float32), sample_rate=self._sample_rate)

        speaker_wav = kwargs.get("speaker_wav")
        audio_prompt_path = self._resolve_audio_prompt(speaker=speaker, speaker_wav=speaker_wav)

        repetition_penalty = float(kwargs.get("repetition_penalty", self.repetition_penalty))
        min_p = float(kwargs.get("min_p", self.min_p))
        top_p = float(kwargs.get("top_p", self.top_p))
        temperature = float(kwargs.get("temperature", self.temperature))
        exaggeration = float(kwargs.get("exaggeration", self.exaggeration))
        cfg_weight = float(kwargs.get("cfg_weight", self.cfg_weight))

        with torch.inference_mode():
            wav = self._tts.generate(
                text=text,
                language_id=language_id,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )

        # Upstream returns torch.Tensor shaped (1, n). Convert to 1D float32.
        if torch.is_tensor(wav):
            audio = wav.detach().cpu().numpy().squeeze().astype(np.float32)
        else:
            audio = np.asarray(wav).squeeze().astype(np.float32)

        if audio.ndim != 1:
            audio = np.asarray(audio).reshape(-1).astype(np.float32)

        return TTSResult(audio=audio, sample_rate=self._sample_rate)

    def get_supported_languages(self) -> list[str]:
        return self._languages.copy()

    def get_supported_speakers(self) -> list[str]:
        # Chatterbox doesn't expose a built-in preset list; voice can be conditioned by an audio prompt.
        return []
