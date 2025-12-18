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

    def test_lazy_imports_are_actionable(self):
        """VibeVoice imports should either work or fail with a helpful ImportError."""
        from modules.tts.vibevoice.imports import get_vibevoice_classes

        try:
            model_cls, processor_cls = get_vibevoice_classes()
        except ImportError as exc:
            # If deps are missing, error should guide user to the optional requirements.
            self.assertIn("requirements-vibevoice.txt", str(exc))
            return

        self.assertTrue(callable(model_cls))
        self.assertTrue(callable(processor_cls))


if __name__ == "__main__":
    unittest.main()
