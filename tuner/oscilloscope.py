"""Oscilloscope: captures the actual played audio (via microphone or WASAPI
system loopback) and renders it as a Lissajous X-Y figure or a time-domain
sweep. Not a re-render of the formula -- this reads from the soundcard ADC
(mic) or the WASAPI mix tap, so what you see is the signal that's really
coming out.

Design intent: a "mind-sync" exercise. The user watches the trace while the
binaural beat plays from the main tuner session. In X-Y mode, two perfectly
phase-locked tones draw a stable line; a binaural beat (Hz offset on the
right channel) draws a slowly-rotating ellipse. The Lissajous can ONLY
"flatten" if the L/R frequencies converge -- so the honest interpretation
of "flatten with the mind" is that the user mentally tracks the rotation
and, optionally, uses the tuner's hotkeys to drop beat_hz toward zero
(their attention guides the hand). Microphone mode also picks up breath
and room acoustics, so steady, focused breathing visibly stabilizes the
trace -- a real mind/body biofeedback path.

Run standalone:  python -m tuner.oscilloscope
Or via tuner:    python -m tuner.tuner --scope
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pygame
import sounddevice as sd


def _find_wasapi_loopback_device() -> tuple[int, int]:
    """Return (device_index, samplerate) for WASAPI loopback on the default output."""
    wasapi_api_idx = None
    for i, host in enumerate(sd.query_hostapis()):
        if "WASAPI" in host["name"]:
            wasapi_api_idx = i
            break
    if wasapi_api_idx is None:
        raise RuntimeError("WASAPI host API not available on this system.")
    host = sd.query_hostapis(wasapi_api_idx)
    out_idx = host["default_output_device"]
    if out_idx < 0:
        raise RuntimeError("No WASAPI default output device.")
    info = sd.query_devices(out_idx)
    return out_idx, int(info["default_samplerate"])


class Oscilloscope:
    def __init__(self, source: str = "mic", view: str = "xy"):
        self.source = source            # "mic" | "loopback"
        self.view = view                # "xy" | "time"
        self.samplerate = 44100
        self.blocksize = 512
        self.buffer_seconds = 0.5
        self.buffer_n = int(self.samplerate * self.buffer_seconds)
        self.buffer = np.zeros((self.buffer_n, 2), dtype=np.float32)
        self.write_idx = 0

        self.gain = 2.0
        self.persistence_decay = 16     # 0-255; higher = faster fade
        self.window_samples = 2048      # samples per scope frame

        self.stream: sd.InputStream | None = None
        self.screen: pygame.Surface | None = None
        self.persistence: pygame.Surface | None = None
        self.font: pygame.font.Font | None = None
        self.big_font: pygame.font.Font | None = None
        self.status: str = ""

    # ------------------------------------------------------------------
    # Audio capture
    # ------------------------------------------------------------------
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            pass
        # Ensure stereo shape
        if indata.ndim == 1 or indata.shape[1] == 1:
            data = np.column_stack([indata.reshape(-1), indata.reshape(-1)])
        else:
            data = indata[:, :2]
        end = self.write_idx + frames
        if end <= self.buffer_n:
            self.buffer[self.write_idx:end] = data
        else:
            split = self.buffer_n - self.write_idx
            self.buffer[self.write_idx:] = data[:split]
            self.buffer[: end - self.buffer_n] = data[split:]
        self.write_idx = end % self.buffer_n

    def _open_stream(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        try:
            if self.source == "loopback":
                dev_idx, dev_sr = _find_wasapi_loopback_device()
                self.samplerate = dev_sr
                self.buffer_n = int(self.samplerate * self.buffer_seconds)
                self.buffer = np.zeros((self.buffer_n, 2), dtype=np.float32)
                self.write_idx = 0
                self.stream = sd.InputStream(
                    device=dev_idx,
                    channels=2,
                    samplerate=self.samplerate,
                    blocksize=self.blocksize,
                    dtype="float32",
                    callback=self._audio_callback,
                    extra_settings=sd.WasapiSettings(loopback=True),
                )
                self.status = f"Source: WASAPI loopback @ {self.samplerate} Hz (device #{dev_idx})"
            else:
                # Microphone
                self.samplerate = 44100
                self.buffer_n = int(self.samplerate * self.buffer_seconds)
                self.buffer = np.zeros((self.buffer_n, 2), dtype=np.float32)
                self.write_idx = 0
                # Try stereo first; fall back to mono if device is mono
                try:
                    self.stream = sd.InputStream(
                        channels=2,
                        samplerate=self.samplerate,
                        blocksize=self.blocksize,
                        dtype="float32",
                        callback=self._audio_callback,
                    )
                    self.status = f"Source: microphone (stereo) @ {self.samplerate} Hz"
                except Exception:
                    self.stream = sd.InputStream(
                        channels=1,
                        samplerate=self.samplerate,
                        blocksize=self.blocksize,
                        dtype="float32",
                        callback=self._audio_callback,
                    )
                    self.status = f"Source: microphone (mono, duplicated to L/R) @ {self.samplerate} Hz"
            self.stream.start()
        except Exception as e:
            self.status = f"Could not open {self.source}: {e}"
            self.stream = None

    # ------------------------------------------------------------------
    # Pygame loop
    # ------------------------------------------------------------------
    def run(self) -> int:
        pygame.init()
        self.screen = pygame.display.set_mode((900, 720), pygame.RESIZABLE)
        pygame.display.set_caption("Oscilloscope -- mind/scope sync")
        self.font = pygame.font.SysFont("consolas,couriernew,monospace", 13)
        self.big_font = pygame.font.SysFont("consolas,couriernew,monospace", 26, bold=True)
        self.persistence = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        self.persistence.fill((0, 0, 0, 0))
        self._open_stream()

        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key)
                elif event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                    self.persistence = pygame.Surface(event.size, pygame.SRCALPHA)
                    self.persistence.fill((0, 0, 0, 0))

            self._render_frame()
            pygame.display.flip()
            clock.tick(60)

        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        pygame.quit()
        return 0

    def _handle_key(self, key: int) -> bool:
        if key in (pygame.K_ESCAPE, pygame.K_q):
            return False
        elif key == pygame.K_x:
            self.view = "xy"
        elif key == pygame.K_t:
            self.view = "time"
        elif key == pygame.K_m:
            self.source = "mic"
            self._open_stream()
            if self.persistence is not None:
                self.persistence.fill((0, 0, 0, 0))
        elif key == pygame.K_l:
            self.source = "loopback"
            self._open_stream()
            if self.persistence is not None:
                self.persistence.fill((0, 0, 0, 0))
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.gain = min(50.0, self.gain * 1.25)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.gain = max(0.05, self.gain / 1.25)
        elif key == pygame.K_c:
            if self.persistence is not None:
                self.persistence.fill((0, 0, 0, 0))
        return True

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _snapshot(self) -> np.ndarray:
        N = self.window_samples
        idx = self.write_idx
        if idx >= N:
            out = self.buffer[idx - N: idx].copy()
        else:
            out = np.concatenate([self.buffer[self.buffer_n - (N - idx):], self.buffer[:idx]])
        return out * self.gain

    def _render_frame(self):
        assert self.screen is not None and self.persistence is not None
        sw, sh = self.screen.get_size()
        samples = self._snapshot()

        # Phosphor decay on persistence layer
        decay = pygame.Surface((sw, sh), pygame.SRCALPHA)
        decay.fill((0, 0, 0, self.persistence_decay))
        self.persistence.blit(decay, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)

        if self.view == "xy":
            self._render_xy(samples, sw, sh)
        else:
            self._render_time(samples, sw, sh)

        # Compose
        self.screen.fill((10, 10, 18))
        self._draw_graticule(sw, sh)
        self.screen.blit(self.persistence, (0, 0))
        self._draw_hud(samples, sw, sh)

    def _render_xy(self, samples: np.ndarray, sw: int, sh: int):
        if len(samples) < 2:
            return
        cx, cy = sw // 2, sh // 2
        radius = min(cx, cy) - 60
        l = np.clip(samples[:, 0], -1.0, 1.0)
        r = np.clip(samples[:, 1], -1.0, 1.0)
        xs = (cx + l * radius).astype(int)
        ys = (cy - r * radius).astype(int)
        pts = list(zip(xs.tolist(), ys.tolist()))
        try:
            pygame.draw.lines(self.persistence, (60, 240, 130, 220), False, pts, 1)
        except (ValueError, TypeError):
            pass

    def _render_time(self, samples: np.ndarray, sw: int, sh: int):
        if len(samples) < 2:
            return
        N = len(samples)
        # L and R get their own bands
        band_l_center = sh // 3
        band_r_center = (sh * 2) // 3
        amp = sh // 6 - 16

        step = max(1, N // max(1, sw))
        l_pts = []
        r_pts = []
        for x in range(sw):
            i = x * step
            if i >= N:
                break
            l_v = np.clip(samples[i, 0], -1.0, 1.0)
            r_v = np.clip(samples[i, 1], -1.0, 1.0)
            l_pts.append((x, band_l_center - int(l_v * amp)))
            r_pts.append((x, band_r_center - int(r_v * amp)))
        if len(l_pts) >= 2:
            pygame.draw.lines(self.persistence, (60, 240, 130, 220), False, l_pts, 1)
        if len(r_pts) >= 2:
            pygame.draw.lines(self.persistence, (240, 220, 80, 220), False, r_pts, 1)

    def _draw_graticule(self, sw: int, sh: int):
        col_minor = (32, 32, 50)
        col_major = (56, 56, 84)
        for i in range(1, 10):
            pygame.draw.line(self.screen, col_minor, (i * sw // 10, 0), (i * sw // 10, sh), 1)
            pygame.draw.line(self.screen, col_minor, (0, i * sh // 10), (sw, i * sh // 10), 1)
        pygame.draw.line(self.screen, col_major, (sw // 2, 0), (sw // 2, sh), 1)
        pygame.draw.line(self.screen, col_major, (0, sh // 2), (sw, sh // 2), 1)

    def _draw_hud(self, samples: np.ndarray, sw: int, sh: int):
        assert self.font is not None and self.big_font is not None
        l = samples[:, 0]
        r = samples[:, 1]
        rms_l = float(np.sqrt(np.mean(l ** 2))) if len(l) else 0.0
        rms_r = float(np.sqrt(np.mean(r ** 2))) if len(r) else 0.0
        if rms_l > 1e-5 and rms_r > 1e-5:
            corr = float(np.corrcoef(l, r)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        else:
            corr = 0.0

        lines = [
            f"View    : {self.view.upper()}    [X=Lissajous  T=time]",
            f"Source  : {self.source.upper()}    [M=mic  L=loopback]",
            f"Gain    : {self.gain:5.2f}x   [+/-=adjust  C=clear trail]",
            f"L RMS   : {rms_l:.3f}",
            f"R RMS   : {rms_r:.3f}",
            f"L<->R coherence: {corr:+.3f}",
        ]
        if self.status:
            lines.append(f"{self.status}")

        for i, line in enumerate(lines):
            surf = self.font.render(line, True, (210, 210, 230))
            self.screen.blit(surf, (12, 8 + i * 17))

        # Big coherence indicator (top-right)
        if abs(corr) > 0.97:
            msg, color = "LOCKED", (60, 240, 130)
        elif abs(corr) > 0.5:
            msg, color = "SYNCING", (240, 220, 80)
        elif rms_l < 1e-4 and rms_r < 1e-4:
            msg, color = "NO SIGNAL", (140, 80, 80)
        else:
            msg, color = "BEATING", (240, 110, 110)
        big = self.big_font.render(msg, True, color)
        self.screen.blit(big, big.get_rect(topright=(sw - 12, 8)))

        # Footer hint
        hint = self.font.render(
            "Esc/Q quit  |  Coherence near +1 = perfectly in-phase line (the 'flat' Lissajous)",
            True, (140, 140, 170),
        )
        self.screen.blit(hint, hint.get_rect(midbottom=(sw // 2, sh - 6)))


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="tuner.oscilloscope", description="Mind/scope sync oscilloscope.")
    parser.add_argument("--source", choices=("mic", "loopback"), default="mic")
    parser.add_argument("--view", choices=("xy", "time"), default="xy")
    args = parser.parse_args(argv)
    return Oscilloscope(source=args.source, view=args.view).run()


if __name__ == "__main__":
    raise SystemExit(main())
