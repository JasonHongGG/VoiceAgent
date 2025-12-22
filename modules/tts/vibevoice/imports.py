"""VibeVoice lazy import helpers.

This repo vendors VibeVoice model code, but it depends on specific `transformers`
versions. Keep imports here so:
- importing `modules` doesn't fail when optional deps are missing/incompatible
- errors are actionable and point to requirements-vibevoice.txt
"""

from __future__ import annotations


def get_vibevoice_classes():
    """Return (model_cls, processor_cls) if importable.

    Raises:
        ImportError: with actionable message when optional deps are missing or incompatible.
    """

    try:
        from ..vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference,
        )
        from ..vibevoice.processor.vibevoice_streaming_processor import (
            VibeVoiceStreamingProcessor,
        )
        return VibeVoiceStreamingForConditionalGenerationInference, VibeVoiceStreamingProcessor
    except Exception as exc:
        raise ImportError(
            "VibeVoice dependencies are missing or incompatible. "
            "Install the optional pinned deps: `pip install -r requirements-vibevoice.txt`. "
            "(Also ensure transformers version matches the pins.)"
        ) from exc