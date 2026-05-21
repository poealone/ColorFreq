"""Brainwave band presets and the SessionParams dataclass shared across modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Band:
    name: str
    low_hz: float
    high_hz: float
    default_hz: float


BANDS: dict[str, Band] = {
    "delta": Band("delta", 0.5, 4.0, 2.0),
    "theta": Band("theta", 4.0, 8.0, 6.0),
    "alpha": Band("alpha", 8.0, 13.0, 10.0),
    "beta":  Band("beta", 13.0, 30.0, 20.0),
    "gamma": Band("gamma", 30.0, 100.0, 40.0),
}

BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]
AUDIO_MODES = ["binaural", "isochronic", "monaural"]


def band_for_hz(hz: float) -> str:
    for name in BAND_ORDER:
        b = BANDS[name]
        if b.low_hz <= hz < b.high_hz:
            return name
    if hz < BANDS["delta"].low_hz:
        return "delta"
    return "gamma"


@dataclass
class SessionParams:
    audio_mode: str = "binaural"
    carrier_hz: float = 200.0
    beat_hz: float = 10.0
    volume: float = 0.3
    fade_in_s: float = 5.0
    fade_out_s: float = 5.0
    duration_s: float | None = None
    audio_enabled: bool = True
    visual_enabled: bool = True
    visual_color_source: str = "beat"     # "beat" | "carrier"
    isochronic_envelope: str = "sine"     # "sine" | "square"


def default_params() -> SessionParams:
    return SessionParams()
