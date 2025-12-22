import unittest


class TestChatterboxIntegration(unittest.TestCase):
    def test_imports_are_actionable(self):
        """Chatterbox wrapper should be importable."""
        from modules.tts.chatterbox_tts import ChatterboxTTS

        self.assertTrue(callable(ChatterboxTTS))

    def test_language_normalization_is_supported(self):
        """Multilingual variant should advertise zh/ja support without needing synthesis."""
        # Note: constructing the engine will download model weights if missing.
        # We avoid instantiation here and just validate that upstream exposes the language list.
        from chatterbox.mtl_tts import SUPPORTED_LANGUAGES

        self.assertIn("zh", SUPPORTED_LANGUAGES)
        self.assertIn("ja", SUPPORTED_LANGUAGES)


if __name__ == "__main__":
    unittest.main()
