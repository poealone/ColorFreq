"""Fullscreen pygame photic-stimulation engine.

Drives a screen flicker locked to the audio beat frequency, displays a HUD,
processes hotkeys to retune the AudioEngine live, and shows a blocking PSE
warning splash before the session starts.
"""

from __future__ import annotations

import math
import time
from typing import Callable

import pygame

from .audio_engine import AudioEngine
from .color import beat_hz_to_color, carrier_hz_to_color
from .presets import AUDIO_MODES, BAND_ORDER, BANDS

TWO_PI = 2.0 * math.pi


class VisualEngine:
    def __init__(
        self,
        audio: AudioEngine,
        warning_text: str,
        fullscreen: bool = True,
        vsync: bool = True,
    ):
        self.audio = audio
        self.warning_text = warning_text
        self.fullscreen = fullscreen
        self.vsync = vsync
        self.screen: pygame.Surface | None = None
        self.refresh_hz: float = 60.0
        self._t0_ms: int = 0
        self._paused = False
        self._hud_visible = True
        self._quit = False
        self._font: pygame.font.Font | None = None
        self._small_font: pygame.font.Font | None = None

    def run(self) -> bool:
        """Returns True if a session ran to completion (or user quit normally);
        False if user declined the PSE splash."""
        pygame.init()
        pygame.mixer.quit()
        info = pygame.display.Info()
        flags = pygame.FULLSCREEN if self.fullscreen else 0
        try:
            self.screen = pygame.display.set_mode(
                (info.current_w, info.current_h), flags, vsync=1 if self.vsync else 0
            )
        except TypeError:
            self.screen = pygame.display.set_mode((info.current_w, info.current_h), flags)

        try:
            rate = pygame.display.get_current_refresh_rate()
            if rate and rate > 0:
                self.refresh_hz = float(rate)
        except Exception:
            pass

        pygame.display.set_caption("Brainwave Entrainment Tuner")
        pygame.mouse.set_visible(False)
        self._font = pygame.font.SysFont("consolas,couriernew,monospace", 22)
        self._small_font = pygame.font.SysFont("consolas,couriernew,monospace", 16)

        if not self._show_pse_splash():
            pygame.quit()
            return False

        self._t0_ms = pygame.time.get_ticks()
        self._main_loop()
        pygame.quit()
        return True

    def _show_pse_splash(self) -> bool:
        assert self.screen is not None and self._small_font is not None
        clock = pygame.time.Clock()
        lines = self.warning_text.splitlines()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_y:
                        return True
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        return False
            self.screen.fill((0, 0, 0))
            sw, sh = self.screen.get_size()
            line_h = self._small_font.get_linesize()
            total_h = line_h * len(lines)
            y = max(20, (sh - total_h) // 2)
            for line in lines:
                surf = self._small_font.render(line, True, (235, 235, 235))
                rect = surf.get_rect(midtop=(sw // 2, y))
                self.screen.blit(surf, rect)
                y += line_h
            footer = self._font.render("Press Y to begin   |   Q / Esc to exit", True, (255, 200, 80))
            self.screen.blit(footer, footer.get_rect(midbottom=(sw // 2, sh - 30)))
            pygame.display.flip()
            clock.tick(60)

    def _main_loop(self) -> None:
        assert self.screen is not None
        clock = pygame.time.Clock()
        while not self._quit:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._quit = True
                elif event.type == pygame.KEYDOWN:
                    self._handle_key(event.key, event.mod)

            if self._paused:
                self._render_paused()
                pygame.display.flip()
                clock.tick(self.refresh_hz)
                continue

            now_s = (pygame.time.get_ticks() - self._t0_ms) / 1000.0
            snap = self.audio.snapshot()

            visual_on = snap.visual_enabled
            beat = snap.beat_hz
            src_hz = beat if snap.visual_color_source == "beat" else snap.carrier_hz

            display_limited = beat * 2 > self.refresh_hz

            if not visual_on:
                self.screen.fill((0, 0, 0))
            elif display_limited:
                color = beat_hz_to_color(src_hz) if snap.visual_color_source == "beat" else carrier_hz_to_color(src_hz)
                dim = tuple(int(c * 0.18) for c in color)
                self.screen.fill(dim)
            else:
                brightness = 0.5 * (1.0 + math.sin(TWO_PI * beat * now_s))
                base = beat_hz_to_color(src_hz) if snap.visual_color_source == "beat" else carrier_hz_to_color(src_hz)
                px = tuple(max(0, min(255, int(c * brightness))) for c in base)
                self.screen.fill(px)

            if self._hud_visible:
                self._draw_hud(snap, now_s, display_limited)

            pygame.display.flip()
            clock.tick(max(self.refresh_hz, 60))

            if self.audio.is_finished():
                self._quit = True

    def _draw_hud(self, snap, now_s: float, display_limited: bool) -> None:
        assert self.screen is not None and self._small_font is not None and self._font is not None
        sw, sh = self.screen.get_size()
        from .presets import band_for_hz
        band = band_for_hz(snap.beat_hz)
        mins, secs = divmod(int(now_s), 60)
        lines = [
            f"Band   : {band}",
            f"Beat   : {snap.beat_hz:6.2f} Hz",
            f"Carrier: {snap.carrier_hz:6.1f} Hz",
            f"Mode   : {snap.audio_mode}",
            f"Volume : {snap.volume:.2f}    Envelope: {snap.isochronic_envelope}",
            f"Audio  : {'ON' if snap.audio_enabled else 'OFF'}    Visual: {'ON' if snap.visual_enabled else 'OFF'}",
            f"Color  : {snap.visual_color_source}",
            f"Elapsed: {mins:02d}:{secs:02d}    Refresh: {self.refresh_hz:.0f} Hz",
        ]
        if display_limited:
            lines.append("!! display-limited: flicker exceeds refresh/2; visual is static !!")

        panel_w = 460
        panel_h = self._small_font.get_linesize() * len(lines) + 16
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 140))
        self.screen.blit(panel, (16, 16))
        y = 24
        for line in lines:
            surf = self._small_font.render(line, True, (240, 240, 240))
            self.screen.blit(surf, (24, y))
            y += self._small_font.get_linesize()

        hint = self._small_font.render(
            "1-5 band  M mode  ^v beat  <> carrier  +/- vol  V A C E H  Space pause  Esc quit",
            True, (180, 180, 180),
        )
        self.screen.blit(hint, hint.get_rect(midbottom=(sw // 2, sh - 14)))

    def _render_paused(self) -> None:
        assert self.screen is not None and self._font is not None
        self.screen.fill((10, 10, 10))
        sw, sh = self.screen.get_size()
        msg = self._font.render("PAUSED  (Space to resume)", True, (255, 255, 255))
        self.screen.blit(msg, msg.get_rect(center=(sw // 2, sh // 2)))

    def _handle_key(self, key: int, mod: int) -> None:
        if key in (pygame.K_ESCAPE, pygame.K_q):
            self._quit = True
            return
        if key == pygame.K_SPACE:
            self._paused = not self._paused
            self.audio.update_params(audio_enabled=not self._paused)
            return

        snap = self.audio.snapshot()
        band_keys = {
            pygame.K_1: "delta", pygame.K_2: "theta", pygame.K_3: "alpha",
            pygame.K_4: "beta",  pygame.K_5: "gamma",
        }
        if key in band_keys:
            b = BANDS[band_keys[key]]
            self.audio.update_params(beat_hz=b.default_hz)
            return
        if key == pygame.K_m:
            idx = AUDIO_MODES.index(snap.audio_mode) if snap.audio_mode in AUDIO_MODES else 0
            self.audio.update_params(audio_mode=AUDIO_MODES[(idx + 1) % len(AUDIO_MODES)])
            return
        fine = bool(mod & pygame.KMOD_SHIFT)
        step = 0.1 if fine else 0.5
        if key == pygame.K_UP:
            self.audio.update_params(beat_hz=max(0.1, snap.beat_hz + step))
            return
        if key == pygame.K_DOWN:
            self.audio.update_params(beat_hz=max(0.1, snap.beat_hz - step))
            return
        if key == pygame.K_RIGHT:
            self.audio.update_params(carrier_hz=max(20.0, snap.carrier_hz + 10.0))
            return
        if key == pygame.K_LEFT:
            self.audio.update_params(carrier_hz=max(20.0, snap.carrier_hz - 10.0))
            return
        if key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.audio.update_params(volume=min(1.0, snap.volume + 0.05))
            return
        if key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.audio.update_params(volume=max(0.0, snap.volume - 0.05))
            return
        if key == pygame.K_v:
            self.audio.update_params(visual_enabled=not snap.visual_enabled)
            return
        if key == pygame.K_a:
            self.audio.update_params(audio_enabled=not snap.audio_enabled)
            return
        if key == pygame.K_c:
            self.audio.update_params(
                visual_color_source="carrier" if snap.visual_color_source == "beat" else "beat"
            )
            return
        if key == pygame.K_e:
            self.audio.update_params(
                isochronic_envelope="square" if snap.isochronic_envelope == "sine" else "sine"
            )
            return
        if key == pygame.K_h:
            self._hud_visible = not self._hud_visible
            return
