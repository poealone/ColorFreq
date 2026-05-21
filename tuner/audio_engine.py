"""Real-time stereo audio synthesis for brainwave entrainment.

Three modes: binaural, isochronic, monaural. Phase accumulators are carried
across callbacks so live parameter changes (carrier/beat/mode/volume) never
produce clicks. Master gain handles fade-in/fade-out.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import numpy as np
import sounddevice as sd

from .presets import SessionParams

TWO_PI = 2.0 * math.pi


class AudioEngine:
    def __init__(self, params: SessionParams, samplerate: int = 44100, blocksize: int = 256):
        self.params = params
        self.samplerate = samplerate
        self.blocksize = blocksize
        self._lock = threading.Lock()

        self._phase_L = 0.0
        self._phase_R = 0.0
        self._phase_C = 0.0
        self._phase_C2 = 0.0
        self._phase_beat = 0.0

        self._stream: sd.OutputStream | None = None
        self._t_started: float | None = None
        self._t_stop_request: float | None = None
        self._closed = False

    def start(self) -> None:
        if self._stream is not None:
            return
        self._t_started = time.monotonic()
        self._stream = sd.OutputStream(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            channels=2,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self, fade_out: bool = True) -> None:
        if self._stream is None or self._closed:
            return
        if fade_out:
            with self._lock:
                self._t_stop_request = time.monotonic()
            fade = self.params.fade_out_s + 0.2
            time.sleep(max(0.05, fade))
        self._closed = True
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
        self._stream = None

    def update_params(self, **kwargs: Any) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self.params, k):
                    setattr(self.params, k, v)

    def snapshot(self) -> SessionParams:
        with self._lock:
            return SessionParams(**self.params.__dict__)

    def _envelope_gain(self, now: float) -> float:
        if self._t_started is None:
            return 0.0
        elapsed = now - self._t_started
        fade_in = max(0.001, self.params.fade_in_s)
        gain_in = min(1.0, elapsed / fade_in) if fade_in > 0 else 1.0
        gain_out = 1.0
        if self._t_stop_request is not None:
            since_stop = now - self._t_stop_request
            fade_out = max(0.001, self.params.fade_out_s)
            gain_out = max(0.0, 1.0 - since_stop / fade_out)
        if self.params.duration_s is not None and elapsed >= self.params.duration_s and self._t_stop_request is None:
            with self._lock:
                self._t_stop_request = now
        return max(0.0, min(1.0, gain_in)) * gain_out * self.params.volume

    def _callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            pass  # could log under/overflows

        with self._lock:
            mode = self.params.audio_mode
            f_c = float(self.params.carrier_hz)
            f_b = float(self.params.beat_hz)
            audio_on = self.params.audio_enabled
            iso_env = self.params.isochronic_envelope

        now = time.monotonic()
        block_t = now
        gain_start = self._envelope_gain(block_t)
        gain_end = self._envelope_gain(block_t + frames / self.samplerate)

        if not audio_on or (gain_start == 0.0 and gain_end == 0.0):
            outdata.fill(0.0)
            self._advance_phases(frames, f_c, f_b)
            return

        n = np.arange(frames, dtype=np.float64)
        gain = gain_start + (gain_end - gain_start) * (n / max(1, frames))

        d_L = TWO_PI * f_c / self.samplerate
        d_R = TWO_PI * (f_c + f_b) / self.samplerate
        d_C = TWO_PI * f_c / self.samplerate
        d_C2 = TWO_PI * (f_c + f_b) / self.samplerate
        d_beat = TWO_PI * f_b / self.samplerate

        phase_L_arr = self._phase_L + d_L * n
        phase_R_arr = self._phase_R + d_R * n
        phase_C_arr = self._phase_C + d_C * n
        phase_C2_arr = self._phase_C2 + d_C2 * n
        phase_beat_arr = self._phase_beat + d_beat * n

        if mode == "binaural":
            left = np.sin(phase_L_arr) * gain
            right = np.sin(phase_R_arr) * gain
        elif mode == "isochronic":
            if iso_env == "square":
                gate = (np.sin(phase_beat_arr) >= 0).astype(np.float64)
            else:
                gate = 0.5 * (1.0 + np.sin(phase_beat_arr))
            mono = np.sin(phase_C_arr) * gate * gain
            left = right = mono
        elif mode == "monaural":
            mono = 0.5 * (np.sin(phase_C_arr) + np.sin(phase_C2_arr)) * gain
            left = right = mono
        else:
            left = right = np.zeros(frames, dtype=np.float64)

        np.clip(left, -1.0, 1.0, out=left)
        np.clip(right, -1.0, 1.0, out=right)
        outdata[:, 0] = left.astype(np.float32)
        outdata[:, 1] = right.astype(np.float32)

        self._advance_phases(frames, f_c, f_b)

    def _advance_phases(self, frames: int, f_c: float, f_b: float) -> None:
        self._phase_L = (self._phase_L + TWO_PI * f_c * frames / self.samplerate) % TWO_PI
        self._phase_R = (self._phase_R + TWO_PI * (f_c + f_b) * frames / self.samplerate) % TWO_PI
        self._phase_C = (self._phase_C + TWO_PI * f_c * frames / self.samplerate) % TWO_PI
        self._phase_C2 = (self._phase_C2 + TWO_PI * (f_c + f_b) * frames / self.samplerate) % TWO_PI
        self._phase_beat = (self._phase_beat + TWO_PI * f_b * frames / self.samplerate) % TWO_PI

    def is_finished(self) -> bool:
        if self._t_stop_request is None:
            return False
        return (time.monotonic() - self._t_stop_request) > (self.params.fade_out_s + 0.1)
