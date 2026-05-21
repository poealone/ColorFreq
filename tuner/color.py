"""Wavelength <-> RGB mapping and Hz -> color helpers for the brainwave tuner.

The `wavelength_to_rgb` implementation is the Bruton/CIE piecewise mapping
extracted verbatim from the legacy colorfreq.py (float-tuple variant).
"""

from __future__ import annotations

import math


def wavelength_to_rgb(wavelength_nm: float) -> tuple[float, float, float]:
    gamma = 0.8
    factor = 0.0
    R = G = B = 0.0

    w = wavelength_nm
    if 380 <= w < 440:
        R = -(w - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif 440 <= w < 490:
        R = 0.0
        G = (w - 440) / (490 - 440)
        B = 1.0
    elif 490 <= w < 510:
        R = 0.0
        G = 1.0
        B = -(w - 510) / (510 - 490)
    elif 510 <= w < 580:
        R = (w - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif 580 <= w < 645:
        R = 1.0
        G = -(w - 645) / (645 - 580)
        B = 0.0
    elif 645 <= w <= 700:
        R = 1.0
        G = 0.0
        B = 0.0

    if 380 <= w < 420:
        factor = 0.3 + 0.7 * (w - 380) / (420 - 380)
    elif 420 <= w < 645:
        factor = 1.0
    elif 645 <= w <= 700:
        factor = 0.3 + 0.7 * (700 - w) / (700 - 645)

    R = max(R * factor, 0.0) ** gamma
    G = max(G * factor, 0.0) ** gamma
    B = max(B * factor, 0.0) ** gamma
    return (R, G, B)


def rgb_float_to_uint8(rgb: tuple[float, float, float]) -> tuple[int, int, int]:
    return (
        max(0, min(255, round(rgb[0] * 255))),
        max(0, min(255, round(rgb[1] * 255))),
        max(0, min(255, round(rgb[2] * 255))),
    )


def beat_hz_to_wavelength(beat_hz: float) -> float:
    """Log-scale map: [0.5, 100] Hz -> [700, 380] nm.

    Low Hz (delta) -> red; high Hz (gamma) -> violet. Each band lands on a
    recognizable hue: delta=deep red, theta=orange/yellow, alpha=green,
    beta=blue, gamma=violet.
    """
    f = max(0.5, min(100.0, beat_hz))
    log_min = math.log(0.5)
    log_max = math.log(100.0)
    t = (math.log(f) - log_min) / (log_max - log_min)
    return 700.0 - t * (700.0 - 380.0)


def carrier_hz_to_wavelength(carrier_hz: float) -> float:
    """Octave-based map (adapted from colorfreq.py:29-38)."""
    if carrier_hz < 20 or carrier_hz > 20000:
        return 550.0
    octave = int(math.log2(carrier_hz / 20))
    base = 20 * (2 ** octave)
    min_w = 380 + (700 - 380) / 10 * octave
    max_w = 380 + (700 - 380) / 10 * (octave + 1)
    norm = (carrier_hz - base) / base
    return min_w + norm * (max_w - min_w)


def beat_hz_to_color(beat_hz: float) -> tuple[int, int, int]:
    return rgb_float_to_uint8(wavelength_to_rgb(beat_hz_to_wavelength(beat_hz)))


def carrier_hz_to_color(carrier_hz: float) -> tuple[int, int, int]:
    return rgb_float_to_uint8(wavelength_to_rgb(carrier_hz_to_wavelength(carrier_hz)))


if __name__ == "__main__":
    for name, hz in [("delta", 2.0), ("theta", 6.0), ("alpha", 10.0), ("beta", 20.0), ("gamma", 40.0)]:
        print(f"{name:6s} {hz:5.1f} Hz -> wavelength {beat_hz_to_wavelength(hz):6.1f} nm -> RGB {beat_hz_to_color(hz)}")
