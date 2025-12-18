"""Lazy imports for VibeVoice dependencies.

We keep these imports isolated so importing the VoiceAgent project doesn't
require all VibeVoice dependencies unless the user opts in.
"""

from __future__ import annotations

from typing import Tuple, Type

from .bootstrap import ensure_vibevoice_importable


def get_vibevoice_classes() -> Tuple[Type, Type]:
    """Return (ModelClass, ProcessorClass) for VibeVoice realtime inference."""
    ensure_vibevoice_importable()

    try:
        from .modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference,
        )
        from .processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
    except Exception as exc:
        raise ImportError(
            "Failed to import VibeVoice (vendored under modules/tts/vibevoice). "
            "If you want to use this TTS engine, install its optional dependencies (see requirements-vibevoice.txt)."
        ) from exc

    return VibeVoiceStreamingForConditionalGenerationInference, VibeVoiceStreamingProcessor
