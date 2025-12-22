"""Chatterbox TTS implementation.

This wraps the `chatterbox-tts` python package (https://github.com/resemble-ai/chatterbox)
into the repo's `TTSEngine` interface.

Notes:
- The upstream model uses a built-in voice conditioning file (`conds.pt`) by default.
- You can optionally provide an audio prompt (voice reference) via:
  - `speaker` argument (if it is a valid file path), or
  - `speaker_wav` kwarg, or
  - env var `CHATTERBOX_AUDIO_PROMPT`, or
  - env var `TTS_SPEAKER_WAV` (shared with Coqui settings).

Language support:
- This engine uses the multilingual model, which supports Chinese (`zh`) and Japanese (`ja`).
"""

from __future__ import annotations

import os
import re
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


def _split_text_into_chunks(text: str, max_chunk_chars: int = 180) -> list[str]:
    """Split long text into smaller chunks to reduce TTS truncation.

    Chatterbox multilingual sets a fixed `max_new_tokens` internally; long or complex
    inputs can still end up truncated. Chunking by sentence boundaries is a simple
    mitigation.
    """
    if not text:
        return []

    cleaned = text.strip()
    if len(cleaned) <= max_chunk_chars:
        return [cleaned]

    # Keep delimiters.
    parts = re.split(r"([。！？.!?;；\n]+)", cleaned)

    sentences: list[str] = []
    buf = ""
    for part in parts:
        if not part:
            continue
        if re.fullmatch(r"[。！？.!?;；\n]+", part):
            buf += part
            if buf.strip():
                sentences.append(buf.strip())
            buf = ""
        else:
            # plain text
            buf += part

    if buf.strip():
        sentences.append(buf.strip())

    # Merge into chunks up to max_chunk_chars
    chunks: list[str] = []
    cur = ""
    for sentence in sentences:
        if not cur:
            cur = sentence
            continue
        if len(cur) + 1 + len(sentence) <= max_chunk_chars:
            cur = f"{cur} {sentence}".strip()
        else:
            chunks.append(cur)
            cur = sentence
    if cur:
        chunks.append(cur)

    # As a last resort, hard-split any very long chunk.
    final: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chunk_chars:
            final.append(chunk)
            continue
        for i in range(0, len(chunk), max_chunk_chars):
            final.append(chunk[i : i + max_chunk_chars].strip())

    return [c for c in final if c]


def _env_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _get_max_chunk_chars(language_id: str) -> int:
    # Allow overriding chunk size without code changes.
    # This is the single most effective mitigation for early-EOS truncation.
    v = _env_int("CHATTERBOX_MAX_CHUNK_CHARS")
    if v is not None and v > 0:
        return v

    # Default: keep behavior close to upstream (single-shot generation).
    # Chunking is only applied for very long texts, and we still have a
    # fallback re-chunking path if output is suspiciously short.
    return 180


def _audio_is_suspiciously_short(audio_samples: int, sample_rate: int, text: str) -> bool:
    """Heuristic: detect when the model likely forced EOS too early.

    Chatterbox multilingual can force EOS on token repetition; when that happens,
    we often see very short audio relative to text length (e.g. only saying the
    first 1-2 words).
    """
    if sample_rate <= 0:
        return False

    t = (text or "").strip()
    if len(t) < 8:
        return False

    secs = audio_samples / float(sample_rate)
    # Very conservative expectation: roughly >= 40ms per char for longer utterances.
    expected_min = min(10.0, max(0.8, 0.04 * len(t)))
    return secs < expected_min


def _hard_split(text: str, max_chars: int) -> list[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    if max_chars <= 0:
        return [cleaned]
    return [cleaned[i : i + max_chars].strip() for i in range(0, len(cleaned), max_chars) if cleaned[i : i + max_chars].strip()]


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

        max_chunk_chars = _get_max_chunk_chars(language_id)
        chunks = _split_text_into_chunks(text, max_chunk_chars=max_chunk_chars)
        if not chunks:
            return TTSResult(audio=np.zeros((0,), dtype=np.float32), sample_rate=self._sample_rate)

        audios: list[np.ndarray] = []
        silence = np.zeros((int(self._sample_rate * 0.12),), dtype=np.float32)

        def _generate_once(gen_text: str, *, t: float, rp: float, tp: float, mp: float) -> np.ndarray:
            with torch.inference_mode():
                wav = self._tts.generate(
                    text=gen_text,
                    language_id=language_id,
                    repetition_penalty=rp,
                    min_p=mp,
                    top_p=tp,
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=t,
                )

            # Upstream returns torch.Tensor shaped (1, n). Convert to 1D float32.
            if torch.is_tensor(wav):
                return wav.detach().cpu().numpy().squeeze().astype(np.float32)
            return np.asarray(wav).squeeze().astype(np.float32)

        for idx, chunk in enumerate(chunks):
            audio = _generate_once(chunk, t=temperature, rp=repetition_penalty, tp=top_p, mp=min_p)

            # If the model likely forced EOS early, retry with smaller chunks.
            if _audio_is_suspiciously_short(audio.size, self._sample_rate, chunk):
                # First fallback: split the chunk much more aggressively.
                sub_max = max(12, min(24, max_chunk_chars // 2))
                sub_chunks = _split_text_into_chunks(chunk, max_chunk_chars=sub_max)
                if len(sub_chunks) <= 1:
                    sub_chunks = _hard_split(chunk, max_chars=sub_max)

                sub_audios: list[np.ndarray] = []
                sub_silence = np.zeros((int(self._sample_rate * 0.08),), dtype=np.float32)
                for j, sub in enumerate(sub_chunks):
                    sub_audio = _generate_once(sub, t=temperature, rp=repetition_penalty, tp=top_p, mp=min_p)
                    # Second fallback: one cheap retry with slightly higher entropy.
                    if _audio_is_suspiciously_short(sub_audio.size, self._sample_rate, sub):
                        sub_audio = _generate_once(
                            sub,
                            t=max(0.95, temperature),
                            rp=max(1.05, min(repetition_penalty, 1.2)),
                            tp=min(0.98, top_p),
                            mp=min_p,
                        )
                    if sub_audio.size:
                        sub_audios.append(sub_audio)
                        if j != len(sub_chunks) - 1:
                            sub_audios.append(sub_silence)

                audio = np.concatenate(sub_audios, axis=0) if sub_audios else audio

            if audio.size:
                audios.append(audio)
                if idx != len(chunks) - 1:
                    audios.append(silence)

        merged = np.concatenate(audios, axis=0) if audios else np.zeros((0,), dtype=np.float32)
        return TTSResult(audio=merged, sample_rate=self._sample_rate)

    def get_supported_languages(self) -> list[str]:
        return self._languages.copy()

    def get_supported_speakers(self) -> list[str]:
        # Chatterbox doesn't expose a built-in preset list; voice can be conditioned by an audio prompt.
        return []
