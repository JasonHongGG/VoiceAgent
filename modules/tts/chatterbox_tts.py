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
    ):
        self.device = _resolve_device(device)

        # Optional voice reference prompt
        self.audio_prompt_path = audio_prompt_path

        # Generation tuning (env-driven; keep public API minimal)
        # Defaults match upstream examples to minimize behavior differences.
        self.repetition_penalty = float(os.getenv("CHATTERBOX_REPETITION_PENALTY", 2.0))
        self.min_p = float(os.getenv("CHATTERBOX_MIN_P", 0.05))
        self.top_p = float(os.getenv("CHATTERBOX_TOP_P", 1.0))
        self.temperature = float(os.getenv("CHATTERBOX_TEMPERATURE", 0.8))
        self.exaggeration = float(os.getenv("CHATTERBOX_EXAGGERATION", 0.5))
        self.cfg_weight = float(os.getenv("CHATTERBOX_CFG_WEIGHT", 0.5))

        print(f"[ChatterboxTTS] Initializing multilingual model on device='{self.device}'...")

        # Import lazily so that importing this module doesn't force dependency usage.
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS as _UpstreamTTS
        from chatterbox.mtl_tts import SUPPORTED_LANGUAGES as _SUPPORTED

        self._tts = _UpstreamTTS.from_pretrained(device=torch.device(self.device))
        # SUPPORTED_LANGUAGES is a dict; list() yields language ids.
        self._languages = list(_SUPPORTED)

        self._sample_rate = int(getattr(self._tts, "sr", 24000))
        print(f"[ChatterboxTTS] Ready. sample_rate={self._sample_rate}, languages={len(self._languages)}")

    def _resolve_audio_prompt(self) -> Optional[str]:
        # 1) explicit engine setting
        if self.audio_prompt_path and Path(self.audio_prompt_path).is_file():
            return self.audio_prompt_path

        # 2) env var overrides
        env_prompt = os.getenv("CHATTERBOX_AUDIO_PROMPT")
        if env_prompt and Path(env_prompt).is_file():
            return env_prompt

        # 3) reuse legacy env var name if set
        ref = os.getenv("TTS_SPEAKER_WAV")
        if ref and Path(ref).is_file():
            return ref

        return None

    def synthesize(self, text: str, language: Optional[str] = None) -> TTSResult:
        language_id = _normalize_language_id(language)
        if language_id not in set(self._languages):
            raise ValueError(
                f"Chatterbox multilingual does not support language='{language}'. "
                f"Supported: {', '.join(self._languages)}"
            )

        if not text:
            return TTSResult(audio=np.zeros((0,), dtype=np.float32), sample_rate=self._sample_rate)

        audio_prompt_path = self._resolve_audio_prompt()

        with torch.inference_mode():
            wav = self._tts.generate(
                text=text,
                language_id=language_id,
                audio_prompt_path=audio_prompt_path,
                exaggeration=self.exaggeration,
                cfg_weight=self.cfg_weight,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty,
                min_p=self.min_p,
                top_p=self.top_p,
            )

        # Upstream returns torch.Tensor shaped (1, n). Convert to 1D float32.
        if torch.is_tensor(wav):
            audio = wav.detach().cpu().numpy().squeeze().astype(np.float32)
        else:
            audio = np.asarray(wav).squeeze().astype(np.float32)

        if audio.ndim != 1:
            audio = np.asarray(audio).reshape(-1).astype(np.float32)

        return TTSResult(audio=audio, sample_rate=self._sample_rate)
