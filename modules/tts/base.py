"""Base interface for Text-to-Speech engines.

This project only needs a very small TTS surface area:
- Synthesize text -> waveform

Keep this interface minimal so adding/removing backends stays easy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TTSResult:
    """TTS 合成結果。"""

    audio: np.ndarray
    sample_rate: int

    def as_tuple(self) -> Tuple[int, np.ndarray]:
        return (self.sample_rate, self.audio)


class TTSEngine(ABC):
    """Text-to-Speech engine interface."""

    @abstractmethod
    def synthesize(self, text: str, language: Optional[str] = None) -> TTSResult:
        """Synthesize `text` into audio."""
        raise NotImplementedError
