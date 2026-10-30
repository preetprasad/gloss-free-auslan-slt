"""Text-to-speech for the Auslan-to-speech demonstrator (research gap G2).

The translator emits English; this turns it into audible speech and times the operation, so the
deployment claim in Chapter 4 covers the whole chain rather than stopping at text.

DESIGN NOTES, in the terms the 2026-09-09 interface brief set out.

  Utterance-final, not simultaneous. The translator consumes a whole clip and produces a whole
  sentence, so speech begins after a completed utterance. Nothing here streams partial output, and
  the demonstrator must not be described as simultaneous interpreting.

  Speak only when the model is confident. The selective-prediction result (2026-09-09y) is that the
  model's own mean per-token cross-entropy ranks its outputs well enough that answering the more
  confident half raises BLEU-4 from 9.52 to 17.42. A speech interface can act on that in a way a
  text interface cannot: saying a wrong sentence aloud, confidently, is worse than declining. So
  `Speaker.speak` takes a confidence and a threshold and returns what it did.

  Backends. macOS `say` is used here: it is offline, it ships an Australian English voice (Karen),
  and it needs no model download, which keeps the demonstrator reproducible on the machine the rest
  of this project runs on. It is NOT a research contribution and is deliberately swappable -- the
  interface below is the contract, and a Piper or Coqui backend would satisfy it unchanged. Quality
  of the synthetic voice is not evaluated in this thesis; latency is.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

AU_VOICE = "Karen"          # the en_AU voice shipped with macOS
DEFAULT_RATE = 180          # words per minute; macOS default is ~175


@dataclass
class Utterance:
    """What the interface did with one translation, and how long each part took."""
    text: str
    spoken: bool
    reason: str                     # "spoken" | "declined: low confidence" | "declined: empty"
    confidence: Optional[float]     # mean per-token cross-entropy; lower is more confident
    synth_ms: float = 0.0
    audio_path: Optional[str] = None
    audio_seconds: Optional[float] = None


class Speaker:
    """Speak an English sentence, or decline to.

    threshold: the confidence gate, in nats of mean per-token cross-entropy. None disables the gate
    and always speaks, which is the ungated baseline the thesis compares against.
    """

    def __init__(self, voice: str = AU_VOICE, rate: int = DEFAULT_RATE,
                 threshold: Optional[float] = None, backend: str = "auto"):
        self.voice, self.rate, self.threshold = voice, rate, threshold
        self.backend = self._resolve(backend)

    @staticmethod
    def _resolve(backend: str) -> str:
        if backend != "auto":
            return backend
        if shutil.which("say"):
            return "say"
        raise RuntimeError(
            "no speech backend available. On macOS `say` is built in; on Linux install Piper "
            "(https://github.com/rhasspy/piper) and add a backend here -- the Speaker interface "
            "is what the rest of the pipeline depends on, not the synthesiser."
        )

    def available_voice(self) -> bool:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
        return any(line.startswith(self.voice + " ") for line in out.splitlines())

    def synthesise(self, text: str, out_path: Optional[str] = None) -> Utterance:
        """Render text to an audio file and time it. Does not play anything."""
        text = (text or "").strip()
        if not text:
            return Utterance(text="", spoken=False, reason="declined: empty", confidence=None)
        out_path = out_path or os.path.join(tempfile.mkdtemp(prefix="auslan_tts_"), "utt.aiff")
        t0 = time.perf_counter()
        subprocess.run(["say", "-v", self.voice, "-r", str(self.rate), "-o", out_path, text],
                       check=True, capture_output=True)
        synth_ms = (time.perf_counter() - t0) * 1000.0
        return Utterance(text=text, spoken=False, reason="synthesised", confidence=None,
                         synth_ms=synth_ms, audio_path=out_path,
                         audio_seconds=_duration_seconds(out_path))

    def speak(self, text: str, confidence: Optional[float] = None,
              play: bool = False) -> Utterance:
        """Apply the confidence gate, then synthesise (and optionally play).

        A high cross-entropy means the model is unsure of its own output, so the interface declines
        and the caller should ask the signer to sign again rather than voice a guess.
        """
        text = (text or "").strip()
        if not text:
            return Utterance(text="", spoken=False, reason="declined: empty", confidence=confidence)
        if self.threshold is not None and confidence is not None and confidence > self.threshold:
            return Utterance(text=text, spoken=False, reason="declined: low confidence",
                             confidence=confidence)
        u = self.synthesise(text)
        u.confidence, u.spoken, u.reason = confidence, True, "spoken"
        if play:
            subprocess.run(["afplay", u.audio_path], check=False)
        return u


def _duration_seconds(path: str) -> Optional[float]:
    """Length of a rendered file, read back from the file rather than estimated from the text."""
    try:
        out = subprocess.run(["afinfo", path], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if "estimated duration" in line:
                return float(line.split(":")[1].strip().split()[0])
    except Exception:
        pass
    return None
