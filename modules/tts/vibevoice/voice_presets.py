"""Voice preset discovery for VibeVoice cached prompts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class VoicePresetMapper:
    def __init__(self, voices_dir: Path):
        self._voices_dir = voices_dir
        self._presets: dict[str, Path] = {}
        self._scan()

    def _scan(self) -> None:
        if not self._voices_dir.is_dir():
            self._presets = {}
            return
        presets: dict[str, Path] = {}
        for pt_path in self._voices_dir.rglob("*.pt"):
            presets[pt_path.stem.lower()] = pt_path.resolve()
        self._presets = dict(sorted(presets.items(), key=lambda kv: kv[0]))

    def list(self) -> list[str]:
        return list(self._presets.keys())

    def resolve(self, speaker: Optional[str]) -> Path:
        if not self._presets:
            raise FileNotFoundError(f"No VibeVoice voice presets found under: {self._voices_dir}")

        if not speaker:
            return next(iter(self._presets.values()))

        key = speaker.strip().lower()
        if key in self._presets:
            return self._presets[key]

        matched: Optional[Path] = None
        for preset_name, preset_path in self._presets.items():
            if preset_name in key or key in preset_name:
                if matched is not None:
                    raise ValueError(
                        f"Multiple VibeVoice presets match speaker='{speaker}'. "
                        f"Please specify one of: {', '.join(self.list()[:20])}{'...' if len(self._presets) > 20 else ''}"
                    )
                matched = preset_path

        if matched is not None:
            return matched

        return next(iter(self._presets.values()))
