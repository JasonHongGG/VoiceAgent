"""Vendored VibeVoice 'modular' subpackage.

This backend has optional/pinned dependencies (see requirements-vibevoice.txt).
To avoid breaking the whole project when those deps are missing or incompatible,
we keep this package import-safe.
"""

try:
    from .modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
    from .configuration_vibevoice_streaming import VibeVoiceStreamingConfig
    from .modeling_vibevoice_streaming import VibeVoiceStreamingModel, VibeVoiceStreamingPreTrainedModel
    from .streamer import AudioStreamer, AsyncAudioStreamer

    __all__ = [
        "VibeVoiceStreamingForConditionalGenerationInference",
        "VibeVoiceStreamingConfig",
        "VibeVoiceStreamingModel",
        "VibeVoiceStreamingPreTrainedModel",
        "AudioStreamer",
        "AsyncAudioStreamer",
    ]
except Exception:  # pragma: no cover
    # Optional deps may be missing/incompatible; allow import of this package.
    __all__ = []