"""Oscilloscope: captures the actual played audio and renders it as a
Lissajous X-Y figure or a time-domain sweep. Not a re-render of the
formula -- this reads from the soundcard ADC of whichever input device
you point it at.

The scope enumerates every input-capable device on the system at startup
(including microphones, audio-interface line-ins, and virtual-cable
'output capture' devices like VoiceMeeter / VB-CABLE / Stereo Mix when
they happen to be installed). Cycle through them at runtime with the
`D` / `B` hotkeys. The currently-selected device name is shown in the
HUD so you always know what is being captured.

Design intent: a 'mind-sync' exercise. The user watches the trace while
the binaural beat plays from the main tuner session. In X-Y mode, two
perfectly phase-locked tones draw a stable line; a binaural beat (Hz
offset on the right channel) draws a slowly-rotating ellipse. The
Lissajous can ONLY 'flatten' if L and R converge in frequency -- the
honest interpretation of 'flatten with the mind' is that the user
mentally tracks the rotation and, optionally, drops beat_hz toward zero
via the session hotkeys (their attention guides the hand). Microphone
or line-in modes pick up breath and room acoustics, so steady focused
breathing visibly stabilizes the trace -- real mind/body biofeedback.

Run standalone:  python -m tuner.oscilloscope
Or via tuner:    python -m tuner.tuner --scope
With CLI hint:   python -m tuner.tuner --scope --scope-device focusrite
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pygame
import sounddevice as sd


@dataclass
class InputDeviceInfo:
    index: int
    name: str
    samplerate: int
    channels: int
    hostapi: str


def list_input_devices() -> list[InputDeviceInfo]:
    """Enumerate every input-capable device on the system."""
    out: list[InputDeviceInfo] = []
    hostapis = sd.query_hostapis()
    for idx, dev in enumerate(sd.query_devices()):
        if int(dev.get("max_input_channels", 0)) < 1:
            continue
        sr = int(dev.get("default_samplerate") or 44100)
        ch = min(2, int(dev["max_input_channels"]))
        host_name = ""
        if 0 <= dev.get("hostapi", -1) < len(hostapis):
            host_name = hostapis[dev["hostapi"]]["name"]
        out.append(InputDeviceInfo(
            index=idx, name=dev["name"], samplerate=sr,
            channels=ch, hostapi=host_name,
        ))
    return out


def find_input_device(devices: list[InputDeviceInfo], substring: str) -> int:
    """Return the index in `devices` whose name contains `substring`
    (case-insensitive). -1 if no match.
    """
    s = substring.lower()
    for i, d in enumerate(devices):
        if s in d.name.lower():
            return i
    return -1


def default_input_index(devices: list[InputDeviceInfo]) -> int:
    """Best-effort default selection: the system default input device."""
    try:
        default = sd.default.device
        default_in = default[0] if isinstance(default, (list, tuple)) else default
        for i, d in enumerate(devices):
            if d.index == default_in:
                return i
    except Exception:
        pass
    return 0 if devices else -1


class Oscilloscope:
    def __init__(self, view: str = "xy", device_hint: str | None = None):
        self.view = view
        self.blocksize = 512
        self.buffer_seconds = 0.5
        self.samplerate = 44100
        self.buffer_n = int(self.samplerate * self.buffer_seconds)
        self.buffer = np.zeros((self.buffer_n, 2), dtype=np.float32)
        self.write_idx = 0

        self.gain = 2.0
        self.persistence_decay = 16
        self.window_samples = 2048

        self.devices: list[InputDeviceInfo] = list_input_devices()
        if not self.devices:
            self.cursor = -1
        elif device_hint:
            found = find_input_device(self.devices, device_hint)
            self.cursor = found if found >= 0 else default_input_index(self.devices)
        else:
            self.cursor = default_input_index(self.devices)

        self.stream: sd.InputStream | None = None
        self.screen: pygame.Surface | None = None
        self.persistence: pygame.Surface | None = None
        self.font: pygame.font.Font | None = None
        self.big_font: pygame.font.Font | None = None
        self.status: str = ""

    @property
    def current_device(self) -> InputDeviceInfo | None:
        if 0 <= self.cursor < len(self.devices):
            return self.devices[self.cursor]
        return None

    # ------------------------------------------------------------------
    # Audio capture
    # ------------------------------------------------------------------
    def _audio_callback(self, indata: np.ndarray, frames: int, _time_info, _status):
        del _time_info, _status  # required by sounddevice callback signature, unused
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

    def _close_stream(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def _open_stream(self) -> None:
        self._close_stream()
        dev = self.current_device
        if dev is None:
            self.status = "No input devices found on this system."
            return

        # Reset buffer for new sample rate
        self.samplerate = dev.samplerate
        self.buffer_n = int(self.samplerate * self.buffer_seconds)
        self.buffer = np.zeros((self.buffer_n, 2), dtype=np.float32)
        self.write_idx = 0

        try:
            self.stream = sd.InputStream(
                device=dev.index,
                channels=dev.channels,
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                dtype="float32",
                callback=self._audio_callback,
            )
            self.stream.start()
            self.status = (
                f"Device #{dev.index} ({dev.hostapi}): {dev.name}  "
                f"@ {self.samplerate} Hz, {dev.channels}ch"
            )
        except Exception as e:
            print(f"[scope] could not open device #{dev.index} '{dev.name}': {e}",
                  file=sys.stderr)
            self.status = f"Could not open '{dev.name}': {e}"
            self.stream = None

    def _cycle_device(self, step: int) -> None:
        if not self.devices:
            return
        self.cursor = (self.cursor + step) % len(self.devices)
        self._open_stream()
        if self.persistence is not None:
            self.persistence.fill((0, 0, 0, 0))

    # ------------------------------------------------------------------
    # Pygame loop
    # ------------------------------------------------------------------
    def run(self) -> int:
        pygame.init()
        self.screen = pygame.display.set_mode((960, 740), pygame.RESIZABLE)
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
                    running = self._handle_key(event.key, event.mod)
                elif event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                    self.persistence = pygame.Surface(event.size, pygame.SRCALPHA)
                    self.persistence.fill((0, 0, 0, 0))

            self._render_frame()
            pygame.display.flip()
            clock.tick(60)

        self._close_stream()
        pygame.quit()
        return 0

    def _handle_key(self, key: int, mod: int) -> bool:
        if key in (pygame.K_ESCAPE, pygame.K_q):
            return False
        elif key == pygame.K_x:
            self.view = "xy"
        elif key == pygame.K_t:
            self.view = "time"
        elif key == pygame.K_d:
            step = -1 if (mod & pygame.KMOD_SHIFT) else 1
            self._cycle_device(step)
        elif key == pygame.K_b:
            self._cycle_device(-1)
        elif key == pygame.K_r:
            # Re-enumerate devices (in case user plugged in something new)
            current_idx = self.devices[self.cursor].index if self.current_device else -1
            self.devices = list_input_devices()
            if not self.devices:
                self.cursor = -1
            else:
                self.cursor = next(
                    (i for i, d in enumerate(self.devices) if d.index == current_idx),
                    default_input_index(self.devices),
                )
            self._open_stream()
            self.status = f"Re-enumerated -- {len(self.devices)} input devices. {self.status}"
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

        decay = pygame.Surface((sw, sh), pygame.SRCALPHA)
        decay.fill((0, 0, 0, self.persistence_decay))
        self.persistence.blit(decay, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)

        if self.view == "xy":
            self._render_xy(samples, sw, sh)
        else:
            self._render_time(samples, sw, sh)

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

        dev_label = "(no device)"
        dev_pos = "[0/0]"
        if self.current_device is not None:
            dev_label = f"{self.current_device.name}  ({self.current_device.hostapi})"
            dev_pos = f"[{self.cursor + 1}/{len(self.devices)}]"

        lines = [
            f"View    : {self.view.upper()}    [X=Lissajous  T=time]",
            f"Input   : {dev_pos}  {dev_label}",
            f"          [D=next  Shift+D / B=prev  R=re-enumerate]",
            f"Gain    : {self.gain:5.2f}x   [+/-=adjust  C=clear trail]",
            f"L RMS   : {rms_l:.3f}    R RMS: {rms_r:.3f}",
            f"L<->R coherence: {corr:+.3f}",
        ]
        if self.status:
            lines.append(self.status)

        for i, line in enumerate(lines):
            surf = self.font.render(line, True, (210, 210, 230))
            self.screen.blit(surf, (12, 8 + i * 17))

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

        hint = self.font.render(
            "Esc/Q quit  |  D cycles input devices  |  Coherence near +1 = phase-locked line",
            True, (140, 140, 170),
        )
        self.screen.blit(hint, hint.get_rect(midbottom=(sw // 2, sh - 6)))


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="tuner.oscilloscope", description="Mind/scope sync oscilloscope.")
    parser.add_argument("--view", choices=("xy", "time"), default="xy")
    parser.add_argument("--device", metavar="SUBSTR", default=None,
                        help="Pick the first input device whose name contains SUBSTR (case-insensitive). "
                             "Defaults to the system default input. Use --list to see options.")
    parser.add_argument("--list", action="store_true",
                        help="List all input-capable devices and exit.")
    args = parser.parse_args(argv)

    if args.list:
        devs = list_input_devices()
        if not devs:
            print("No input devices found.")
            return 1
        for i, d in enumerate(devs):
            print(f"  [{i:2d}] dev#{d.index:3d}  {d.hostapi:24s}  {d.channels}ch @ {d.samplerate}Hz   {d.name}")
        return 0

    return Oscilloscope(view=args.view, device_hint=args.device).run()


if __name__ == "__main__":
    raise SystemExit(main())
