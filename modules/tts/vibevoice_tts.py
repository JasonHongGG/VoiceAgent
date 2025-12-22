"""VibeVoice TTS implementation.

This integrates the VibeVoice Realtime streaming model as a TTSEngine option.

Notes:
- The VibeVoice model weights are not vendored in this repo. By default this
  engine loads from Hugging Face: microsoft/VibeVoice-Realtime-0.5B.
- Voice prompts are provided as cached prompt .pt files under
  VibeVoice/demo/voices/streaming_model.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .base import TTSEngine, TTSResult
from .vibevoice.voice_presets import VoicePresetMapper
from .vibevoice.imports import get_vibevoice_classes

@dataclass(frozen=True)
class _DeviceConfig:
    device: str
    torch_dtype: torch.dtype
    attn_implementation: str


def _resolve_device_config(device: str) -> _DeviceConfig:
    device = (device or "cpu").lower()
    if device == "mpx":
        device = "mps"

    if device == "mps":
        if not torch.backends.mps.is_available():
            device = "cpu"
        return _DeviceConfig(device=device, torch_dtype=torch.float32, attn_implementation="sdpa")

    if device == "cuda":
        if not torch.cuda.is_available():
            device = "cpu"
            return _DeviceConfig(device=device, torch_dtype=torch.float32, attn_implementation="sdpa")
        return _DeviceConfig(device=device, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

    return _DeviceConfig(device="cpu", torch_dtype=torch.float32, attn_implementation="sdpa")


class VibeVoiceTTS(TTSEngine):
    """Use VibeVoice Realtime model as a TTS engine."""

    def __init__(
        self,
        model_path: str = "microsoft/VibeVoice-Realtime-0.5B",
        device: str = "cpu",
        voice: Optional[str] = None,
        voices_dir: Optional[str] = None,
    ):
        # ensure_vibevoice_importable()

        self.model_path = model_path
        # Env-driven tuning knobs (keep public API minimal)
        self.cfg_scale = float(os.getenv("VIBEVOICE_CFG_SCALE", 1.5))
        self.ddpm_steps = int(os.getenv("VIBEVOICE_DDPM_STEPS", 5))

        # Device config
        self.device = device
        self._device_cfg = _resolve_device_config(device)

        # Voice preset mapper
        REPO_ROOT = Path(__file__).resolve().parents[2]
        voices_dir = voices_dir or "resources/voices/streaming_model"
        voices_path = (REPO_ROOT / voices_dir).resolve()
        self._voice_mapper = VoicePresetMapper(voices_path)

        self._current_voice_name = (voice or "").strip().lower() or None
        self._cached_voice_preset: Optional[dict] = None

        print(
            "[VibeVoiceTTS] Initializing model='{}' on device='{}' (dtype={}, attn={}, cfg_scale={}, ddpm_steps={})".format(
                self.model_path,
                self._device_cfg.device,
                str(self._device_cfg.torch_dtype).replace("torch.", ""),
                self._device_cfg.attn_implementation,
                self.cfg_scale,
                self.ddpm_steps,
            )
        )

        model_cls, processor_cls = get_vibevoice_classes()

        # Load processor
        self.processor = processor_cls.from_pretrained(self.model_path)

        # Load model with conservative fallbacks (mirrors demo logic)
        attn_impl = self._device_cfg.attn_implementation
        try:
            if self._device_cfg.device == "mps":
                self.model = model_cls.from_pretrained(
                    self.model_path,
                    torch_dtype=self._device_cfg.torch_dtype,
                    attn_implementation=attn_impl,
                    device_map=None,
                )
                self.model.to("mps")
            else:
                self.model = model_cls.from_pretrained(
                    self.model_path,
                    torch_dtype=self._device_cfg.torch_dtype,
                    attn_implementation=attn_impl,
                    device_map=self._device_cfg.device,
                )
        except Exception as exc:
            if attn_impl == "flash_attention_2":
                print(
                    f"[VibeVoiceTTS] Model load failed with flash_attention_2 ({type(exc).__name__}: {exc}). Falling back to SDPA."
                )
                if self._device_cfg.device == "mps":
                    self.model = model_cls.from_pretrained(
                        self.model_path,
                        torch_dtype=self._device_cfg.torch_dtype,
                        attn_implementation="sdpa",
                        device_map=None,
                    )
                    self.model.to("mps")
                else:
                    self.model = model_cls.from_pretrained(
                        self.model_path,
                        torch_dtype=self._device_cfg.torch_dtype,
                        attn_implementation="sdpa",
                        device_map=self._device_cfg.device,
                    )
            else:
                raise

        self.model.eval()
        self.model.set_ddpm_inference_steps(num_steps=self.ddpm_steps)

        # Prime cached prompt
        self._load_voice_preset(self._current_voice_name)

        self._sample_rate = int(getattr(self.processor.audio_processor, "sampling_rate", 24000))
        print(f"[VibeVoiceTTS] Ready. sample_rate={self._sample_rate}, voices={len(self._voice_mapper.list())}")

    def _load_voice_preset(self, voice_name: Optional[str]) -> None:
        voice_pt = self._voice_mapper.resolve(voice_name)
        self._current_voice_name = voice_pt.stem.lower()
        self._cached_voice_preset = torch.load(str(voice_pt), map_location=self._device_cfg.device, weights_only=False)
        print(f"[VibeVoiceTTS] Using voice preset: {self._current_voice_name} ({voice_pt})")

    def synthesize(self, text: str, language: Optional[str] = None) -> TTSResult:
        if not text:
            return TTSResult(audio=np.zeros((0,), dtype=np.float32), sample_rate=self._sample_rate)

        if self._cached_voice_preset is None:
            self._load_voice_preset(self._current_voice_name)

        cleaned = (
            text.replace("’", "'")
            .replace("“", '"')
            .replace("”", '"')
            .strip()
        )

        inputs = self.processor.process_input_with_cached_prompt(
            text=cleaned,
            cached_prompt=self._cached_voice_preset,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        for key, value in list(inputs.items()):
            if torch.is_tensor(value):
                inputs[key] = value.to(self._device_cfg.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=self.cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
                all_prefilled_outputs=copy.deepcopy(self._cached_voice_preset),
            )

        if not getattr(outputs, "speech_outputs", None) or outputs.speech_outputs[0] is None:
            raise RuntimeError("VibeVoice did not produce speech output")

        audio_tensor = outputs.speech_outputs[0]
        audio = audio_tensor.detach().cpu().float().numpy().squeeze().astype(np.float32)
        return TTSResult(audio=audio, sample_rate=self._sample_rate)
