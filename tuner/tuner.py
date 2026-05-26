"""Brainwave Entrainment Tuner -- entry point.

Run with:   python -m tuner.tuner             (start a session)
            python -m tuner.tuner --list-devices   (enumerate audio outputs)

Safety gating: a photosensitive-epilepsy warning is shown twice -- once in
the console (must type I UNDERSTAND), once as a pygame splash (press Y).
"""

from __future__ import annotations

import argparse
import sys

from .audio_engine import AudioEngine
from .presets import AUDIO_MODES, BAND_ORDER, BANDS, SessionParams, default_params
from .protocols import EVIDENCE_LEGEND, PROTOCOLS, Protocol, categories, in_category
from .visual_engine import VisualEngine

PSE_WARNING_TEXT = """\
==============================================================================
                      PHOTOSENSITIVE EPILEPSY WARNING
==============================================================================
This program produces FLASHING LIGHTS and PULSED AUDIO at 0.5-100 Hz.
Flashing in the 3-30 Hz range -- especially 15-25 Hz -- can trigger seizures
in people with photosensitive epilepsy, even those with no prior history.

DO NOT USE THIS PROGRAM IF:
  - Personal or family history of epilepsy or seizures
  - Migraine aura, blackouts, or unexplained loss of consciousness
  - Under influence of alcohol, stimulants, hallucinogens, or sleep-deprived
  - Operating machinery, driving, or any setting where altered awareness is risky
  - Pregnant, heart condition, or any neurological disorder without physician OK

This software is for personal experimentation ONLY. NOT a medical device.
No clinical claims. Effects of brainwave entrainment are not consistently
validated by peer-reviewed research.

USE STEREO HEADPHONES for binaural mode. Start at LOW volume (0.2 or less).
STOP IMMEDIATELY if you feel dizzy, nauseous, disoriented, develop a headache,
or notice any visual or auditory disturbance.

Console: type   I UNDERSTAND   and press ENTER (anything else exits).
Pygame:  press  Y              to begin, or  Q / Esc  to exit.
=============================================================================="""


def show_console_safety_warning() -> bool:
    print(PSE_WARNING_TEXT)
    try:
        reply = input("\n> ").strip()
    except (EOFError, KeyboardInterrupt):
        return False
    return reply.upper() == "I UNDERSTAND"


def _prompt(label: str, default: str) -> str:
    try:
        raw = input(f"{label} [{default}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return default
    return raw if raw else default


def _prompt_float(label: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
    while True:
        raw = _prompt(label, f"{default}")
        try:
            v = float(raw)
        except ValueError:
            print("  enter a number")
            continue
        if lo is not None and v < lo:
            print(f"  must be >= {lo}")
            continue
        if hi is not None and v > hi:
            print(f"  must be <= {hi}")
            continue
        return v


def _prompt_bool(label: str, default: bool) -> bool:
    raw = _prompt(label, "Y" if default else "n").lower()
    if raw in ("", "y", "yes", "1", "true"):
        return True if default or raw else False
    if raw in ("n", "no", "0", "false"):
        return False
    return True if raw in ("y", "yes", "1", "true") else False


def _print_evidence_legend() -> None:
    print("\nEvidence tags:")
    for k, v in EVIDENCE_LEGEND.items():
        print(f"  [{k:5s}]  {v}")
    print()


def _select_protocol() -> Protocol | None:
    """Numbered catalog browse. Returns the chosen Protocol or None to bail."""
    _print_evidence_legend()
    print("=== Protocol Catalog ===")
    flat: list[Protocol] = []
    for cat in categories():
        print(f"\n[{cat}]")
        for p in in_category(cat):
            flat.append(p)
            idx = len(flat)
            print(f"  {idx:2d}) [{p.evidence:5s}]  {p.name}")
            print(f"           carrier={p.carrier_hz:>7.2f} Hz   beat={p.beat_hz:>5.2f} Hz   mode={p.audio_mode}")
    print()
    raw = _prompt("Pick a protocol number (blank to go back)", "")
    if not raw:
        return None
    try:
        i = int(raw)
        if 1 <= i <= len(flat):
            chosen = flat[i - 1]
            print(f"\n>> Selected: {chosen.name}  [{chosen.evidence}]")
            if chosen.claim:
                print(f"   Claim : {chosen.claim}")
            if chosen.source:
                print(f"   Source: {chosen.source}")
            if chosen.notes:
                print(f"   Notes : {chosen.notes}")
            return chosen
    except ValueError:
        pass
    print("  invalid selection")
    return None


def _prompt_band_or_custom() -> float:
    print("\nSelect target band:")
    for i, name in enumerate(BAND_ORDER, 1):
        b = BANDS[name]
        flag = "  [visual unreliable on 60 Hz display]" if name == "gamma" else ""
        print(f"  {i}) {name.capitalize():<6s} ({b.low_hz}-{b.high_hz} Hz, default {b.default_hz} Hz){flag}")
    print("  6) Custom (enter Hz)")
    band_choice = _prompt("Choice", "3")
    if band_choice in {"1", "2", "3", "4", "5"}:
        return BANDS[BAND_ORDER[int(band_choice) - 1]].default_hz
    if band_choice == "6":
        return _prompt_float("Custom beat Hz", 10.0, 0.1, 200.0)
    print("  unknown choice, defaulting to alpha 10 Hz")
    return BANDS["alpha"].default_hz


def _prompt_audio_mode(default: str = "binaural") -> str:
    print("\nSelect audio mode:")
    print("  1) Binaural   (requires headphones)")
    print("  2) Isochronic (speakers OK; strongest)")
    print("  3) Monaural   (speakers OK)")
    default_idx = {"binaural": "1", "isochronic": "2", "monaural": "3"}.get(default, "1")
    choice = _prompt("Choice", default_idx)
    return {"1": "binaural", "2": "isochronic", "3": "monaural"}.get(choice, default)


def menu_loop() -> SessionParams | None:
    print("\n=== Brainwave Entrainment Tuner ===\n")
    print("Setup mode:")
    print("  1) Custom (pick band + mode + carrier manually)")
    print("  2) Protocol catalog (curated presets with source citations)")
    mode_choice = _prompt("Choice", "1")

    audio_mode = "binaural"
    carrier_hz = 200.0
    beat_hz = BANDS["alpha"].default_hz
    envelope = "sine"

    if mode_choice == "2":
        protocol = _select_protocol()
        if protocol is None:
            print("  no protocol selected -- falling back to custom")
            beat_hz = _prompt_band_or_custom()
            audio_mode = _prompt_audio_mode()
            carrier_hz = _prompt_float("Carrier Hz", 200.0, 20.0, 4000.0)
        else:
            audio_mode = protocol.audio_mode
            carrier_hz = protocol.carrier_hz
            beat_hz = protocol.beat_hz
            envelope = protocol.envelope
            print("\n(You can override carrier or beat below; press ENTER to keep the protocol's values.)")
            carrier_hz = _prompt_float("Carrier Hz", carrier_hz, 20.0, 8000.0)
            beat_hz = _prompt_float("Beat Hz (0 = pure carrier, no entrainment)", beat_hz, 0.0, 200.0)
    else:
        beat_hz = _prompt_band_or_custom()
        audio_mode = _prompt_audio_mode()
        carrier_hz = _prompt_float("Carrier Hz", 200.0, 20.0, 4000.0)

    volume = _prompt_float("Volume 0.0-1.0", 0.1, 0.0, 1.0)
    duration_raw = _prompt("Duration in minutes (blank = until you quit)", "")
    duration_s: float | None = None
    if duration_raw:
        try:
            duration_s = max(1.0, float(duration_raw)) * 60.0
        except ValueError:
            duration_s = None
    fade_in = _prompt_float("Fade-in seconds", 5.0, 0.0, 60.0)
    fade_out = _prompt_float("Fade-out seconds", 5.0, 0.0, 60.0)
    visual_on = _prompt_bool("Visual on?", True)
    audio_on = _prompt_bool("Audio on?", True)

    return SessionParams(
        audio_mode=audio_mode,
        carrier_hz=carrier_hz,
        beat_hz=beat_hz,
        volume=volume,
        fade_in_s=fade_in,
        fade_out_s=fade_out,
        duration_s=duration_s,
        audio_enabled=audio_on,
        visual_enabled=visual_on,
        isochronic_envelope=envelope,
    )


def list_devices() -> int:
    try:
        import sounddevice as sd
    except Exception as e:
        print(f"sounddevice import failed: {e}")
        return 1
    print(sd.query_devices())
    print(f"\nDefault input/output: {sd.default.device}")
    return 0


def list_protocols() -> int:
    _print_evidence_legend()
    for cat in categories():
        print(f"=== {cat} ===")
        for p in in_category(cat):
            print(f"  {p.id:24s}  [{p.evidence:5s}]  {p.name}")
            print(f"  {'':24s}   carrier={p.carrier_hz:>7.2f} Hz   beat={p.beat_hz:>5.2f} Hz   mode={p.audio_mode}")
            if p.source:
                print(f"  {'':24s}   source: {p.source}")
        print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tuner", description="Brainwave entrainment tuner.")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    parser.add_argument("--list-protocols", action="store_true", help="Print the protocol catalog and exit.")
    parser.add_argument("--protocol", metavar="ID", help="Launch directly with a named protocol (see --list-protocols).")
    parser.add_argument("--gui", action="store_true", help="Launch the tkinter GUI control panel.")
    parser.add_argument("--scope", action="store_true", help="Launch the oscilloscope (captures audio from any input device).")
    parser.add_argument("--scope-device", metavar="SUBSTR", default=None,
                        help="Oscilloscope: pick first input device whose name contains SUBSTR (case-insensitive). "
                             "Defaults to system default input. Cycle with D/B at runtime.")
    parser.add_argument("--scope-view", choices=("xy", "time"), default="xy", help="Oscilloscope default view.")
    parser.add_argument("--scope-list-devices", action="store_true",
                        help="List input devices the scope can use and exit.")
    parser.add_argument("--windowed", action="store_true", help="Run visual in a window instead of fullscreen.")
    parser.add_argument("--no-vsync", action="store_true", help="Disable VSync (introduces tearing).")
    args = parser.parse_args(argv)

    if args.list_devices:
        return list_devices()
    if args.list_protocols:
        return list_protocols()
    if args.scope_list_devices:
        from .oscilloscope import list_input_devices
        devs = list_input_devices()
        if not devs:
            print("No input devices found.")
            return 1
        for i, d in enumerate(devs):
            print(f"  [{i:2d}] dev#{d.index:3d}  {d.hostapi:24s}  {d.channels}ch @ {d.samplerate}Hz   {d.name}")
        return 0
    if args.scope:
        from .oscilloscope import Oscilloscope
        return Oscilloscope(view=args.scope_view, device_hint=args.scope_device).run()
    if args.gui:
        from .gui import TunerGUI
        return TunerGUI().run()

    if not show_console_safety_warning():
        print("\nExiting -- safety confirmation declined.")
        return 1

    if args.protocol:
        from .protocols import by_id
        proto = by_id(args.protocol)
        if proto is None:
            print(f"Unknown protocol id: {args.protocol!r}. Use --list-protocols.")
            return 1
        print(f"\n>> Loaded protocol: {proto.name}  [{proto.evidence}]")
        if proto.source:
            print(f"   Source: {proto.source}")
        if proto.notes:
            print(f"   Notes : {proto.notes}")
        params = proto.to_params()
    else:
        params = menu_loop()
    if params is None:
        return 1

    print("\nLaunching pygame window... press Y on the splash to begin, Q/Esc to abort.\n")

    audio = AudioEngine(params)
    visual = VisualEngine(
        audio,
        warning_text=PSE_WARNING_TEXT,
        fullscreen=not args.windowed,
        vsync=not args.no_vsync,
    )

    audio.start()
    try:
        accepted = visual.run()
    finally:
        audio.stop(fade_out=True)

    if not accepted:
        print("Session aborted at safety splash.")
        return 1

    print("Session ended cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
