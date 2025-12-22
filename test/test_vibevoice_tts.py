import os
import unittest
from pathlib import Path


class TestVibeVoiceIntegration(unittest.TestCase):
    def test_voice_presets_discoverable(self):
        """Voice presets should be discoverable under resources/voices/streaming_model."""
        repo_root = Path(__file__).resolve().parents[1]
        voices_dir = repo_root / "resources" / "voices" / "streaming_model"
        self.assertTrue(voices_dir.is_dir(), f"Missing voices dir: {voices_dir}")

        from modules.tts.vibevoice.voice_presets import VoicePresetMapper

        mapper = VoicePresetMapper(voices_dir)
        voices = mapper.list()
        self.assertGreater(len(voices), 0, "No .pt voice presets found")

        # Spot-check a common preset shipped with the repo
        if "en-emma_woman" in voices:
            resolved = mapper.resolve("en-emma_woman")
            self.assertTrue(resolved.is_file())

    def test_vibevoice_vendored_imports(self):
        """Direct vendored imports should either work or raise ImportError.

        VibeVoice is optional and may require pinned deps in requirements-vibevoice.txt.
        """

        try:
            from vibevoice.modular.modeling_vibevoice_streaming_inference import (
                VibeVoiceStreamingForConditionalGenerationInference,
            )
            from vibevoice.processor.vibevoice_streaming_processor import (
                VibeVoiceStreamingProcessor,
            )
        except ImportError:
            return

        self.assertTrue(callable(VibeVoiceStreamingForConditionalGenerationInference))
        self.assertTrue(callable(VibeVoiceStreamingProcessor))



if __name__ == "__main__":
    unittest.main()
