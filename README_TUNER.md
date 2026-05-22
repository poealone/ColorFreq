# Brainwave Entrainment Tuner

Open-loop audio + visual brainwave entrainment generator. Drives EEG-band rhythms (delta, theta, alpha, beta, gamma) via three audio modes and a synchronized fullscreen color flicker.

## Install

```
pip install -r requirements.txt
```

`sounddevice` and `pygame` are the new dependencies; the legacy scripts ([colorfreq.py](colorfreq.py), [colorfreqListener.py](colorfreqListener.py), [colorfreqHiFDetector.py](colorfreqHiFDetector.py)) still use pyaudio/matplotlib and are untouched.

## Run

```
python -m tuner.tuner
```

Optional flags:

- `--gui` — launch the tkinter GUI control panel (recommended for browsing the protocol catalog).
- `--scope` — launch the oscilloscope (mic or WASAPI loopback capture; not a digital re-render). See the **Oscilloscope** section below.
- `--list-devices` — print audio devices and exit (Windows multi-output troubleshooting).
- `--list-protocols` — print the full protocol catalog (id, Hz, source, evidence tag) and exit.
- `--protocol <id>` — launch directly into a named protocol from the catalog (e.g. `--protocol gamma40`, `--protocol schumann`, `--protocol focus_15`).
- `--windowed` — run the visual in a window instead of fullscreen.
- `--no-vsync` — disable VSync (introduces tearing, not recommended).

## Oscilloscope (mind/scope sync exercise)

`python -m tuner.tuner --scope` opens a real-signal oscilloscope. It does **not** re-render the formula — it captures the actual audio coming in from whichever input device you point it at, so what you see is the signal that's really there.

**Picking an input device.** On launch the scope enumerates every input-capable device on your system (microphones, audio-interface line-ins, virtual cables like VoiceMeeter / VB-CABLE / Stereo Mix when installed) and binds to the system default. Cycle through them at runtime with `D` (next) and `B` or `Shift+D` (previous); the current device name appears in the HUD.

Pick a device at launch with `--scope-device <substring>` (case-insensitive name match), or print the full list with `--scope-list-devices`:

```
py -3 -m tuner.tuner --scope --scope-device focusrite       # match Focusrite first
py -3 -m tuner.tuner --scope --scope-device voicemeeter     # capture VoiceMeeter
py -3 -m tuner.tuner --scope-list-devices                   # see what's available
```

**Capturing the binaural playing from the tuner itself.** A few options:

- **Microphone:** simplest. Speakers play → mic listens → scope sees the binaural plus any room sound and your breath. Real mind/body biofeedback path: steady focused breathing visibly stabilizes the trace.
- **Audio-interface line-in loopback (hardware):** if you have an audio interface, run a physical cable from a line-out back into a line-in and pick that line-in as the scope's source.
- **Virtual audio cable (software):** install [VB-CABLE](https://vb-audio.com/Cable/) or [VoiceMeeter](https://vb-audio.com/Voicemeeter/), route the tuner's playback *through* the virtual cable (e.g. set "VoiceMeeter Input" as Windows default playback), then point the scope at the matching "Output" capture device. Bit-perfect digital tap.
- **Windows Stereo Mix:** built-in but usually disabled. Enable in Sound Settings → Recording → right-click → Show Disabled Devices → Enable.

**Views:**
- **X-Y Lissajous** (default): X = left channel, Y = right channel. Two perfectly phase-locked tones draw a stable straight line; a binaural beat draws a slowly-rotating ellipse. The HUD shows L↔R **coherence** (Pearson correlation) — `+1.0` is a perfectly locked line, `0` is uncorrelated. The status indicator goes `BEATING → SYNCING → LOCKED` as coherence climbs.
- **Time-domain**: classic horizontal-sweep scope with L (green) and R (yellow) traces overlaid. Press `T` to switch, `X` to switch back.

**Hotkeys:**

| Key | Action |
|---|---|
| `X` / `T` | Lissajous / Time-domain view |
| `D` / `B` | Next / previous input device (Shift+D also goes back) |
| `R` | Re-enumerate input devices (after plugging something in) |
| `+` / `-` | Gain ±25% |
| `C` | Clear phosphor trail |
| `Esc` / `Q` | Quit |

**Honest framing — what "flattening with the mind" actually means.** A binaural beat with `f_L ≠ f_R` is deterministic — the Lissajous *will* keep rotating regardless of your mental state. The Lissajous can only collapse to a stable line if L and R are at the same frequency (i.e. `beat_hz = 0`). So there are two legitimate paths to "flatten" the scope while watching:

1. **Microphone source + steady breathing.** Mic picks up room acoustics, your breath, your posture. Stable focused breathing visibly reduces background noise and the trace becomes cleaner. This is real mind/body biofeedback.
2. **Intent-guided manipulation.** Open the main session, drop `beat_hz` toward zero via the hotkeys (`↓`) while watching the scope. The Lissajous slows its rotation and collapses to a line as you approach `beat_hz = 0`. The "mind" here is operating the keyboard — but if that's how you experience it, the feedback loop is real.

If you're looking for evidence of consciousness directly altering audio waveforms via attention alone, this tool won't provide it — but it's an interesting setup for honest interoceptive practice and a satisfying visualization of phase locking.

**Launching alongside a session:** open the GUI (`--gui`), click the **Open Oscilloscope** button — it spawns the scope as a separate process, so it runs alongside any active photic-flicker session.

## GUI

`python -m tuner.tuner --gui` opens a tkinter control panel:

- **Left**: protocol catalog as a tree, grouped by category, with the evidence tag and beat Hz visible on every row.
- **Right**: details pane (claim / source / notes for the selected protocol) plus session-parameter widgets (audio mode, envelope, carrier slider, beat slider, volume, fade-in/out, duration, audio/visual enable).
- **Start Session** button opens a modal PSE warning dialog the first time you click it (you must type `I UNDERSTAND`); then hides the GUI, launches the pygame fullscreen visual + audio session, and restores the GUI when you press `Esc`.
- **Evidence Legend** button (top-right) explains the `PR / EX / PH / FL / AUDIO` tags.

The GUI does not replace pygame — it's a launcher. The same in-window PSE confirmation splash + hotkeys + HUD still apply once the session starts. You can use `--gui --windowed` to launch the visual in a window instead of fullscreen.

## Bands

| Band  | Range (Hz) | Default Hz | Associated state           |
|-------|-----------:|-----------:|----------------------------|
| Delta | 0.5–4      | 2.0        | deep sleep                 |
| Theta | 4–8        | 6.0        | meditation, creativity     |
| Alpha | 8–13       | 10.0       | relaxed alertness          |
| Beta  | 13–30      | 20.0       | focused attention          |
| Gamma | 30–100     | 40.0       | high cognition (visual unreliable on 60 Hz displays — audio still entrains) |

## Audio modes

- **Binaural** — different tone in each ear (e.g. 200 Hz L, 210 Hz R → 10 Hz alpha beat). Requires **stereo headphones**.
- **Isochronic** — single tone gated on/off at the beat rate. Works on speakers, generally considered the strongest entrainer. Sine envelope (smooth) by default; square envelope (sharper) available via `E`.
- **Monaural** — two tones summed; beating is acoustic, not neural. Works on speakers.

## Hotkeys (in pygame window)

| Key                | Action |
|--------------------|--------|
| `Esc` / `Q`        | Quit with fade-out |
| `Space`            | Pause / resume |
| `1`–`5`            | Switch to delta / theta / alpha / beta / gamma |
| `M`                | Cycle audio mode |
| `↑` / `↓`          | Beat ±0.5 Hz (hold Shift for ±0.1) |
| `←` / `→`          | Carrier ±10 Hz |
| `+` / `-`          | Volume ±0.05 |
| `V` / `A`          | Toggle visual / audio |
| `C`                | Toggle color source (beat ↔ carrier) |
| `E`                | Toggle isochronic envelope (sine ↔ square) |
| `H`                | Toggle HUD |

## Safety

This program flashes light and pulses audio at 0.5–100 Hz. Photosensitive epilepsy can be triggered by flashing in the 3–30 Hz range — especially 15–25 Hz — even in people with no prior history. The console refuses to proceed unless you type `I UNDERSTAND`, and the pygame window blocks on a confirmation splash. Read both warnings in full before continuing.

This software is **not a medical device** and makes no clinical claims. Effects of brainwave entrainment are not consistently validated by peer-reviewed research. Use at low volume in a comfortably lit room; stop immediately if you feel dizzy, nauseous, disoriented, or develop a headache.

## Protocol catalog

The catalog ships a curated set of presets drawn from documented consciousness-exploration sources. Each preset is tagged with an evidence level so you know exactly what you're picking. **None of these are medical claims. Most "psychic frequency" attributions circulating online are folklore — we label them as such rather than pretending otherwise.**

### Evidence tags

| Tag      | Meaning |
|----------|---------|
| **PR**   | Peer-reviewed neuroscience finding |
| **EX**   | Experiential / historical practice — the *state* may be real but the Hz-to-effect link is not clinically established |
| **PH**   | Physics-real, but the entrainment claim is folklore (e.g. the Schumann cavity resonance is real physics; "optimal brain Hz" is unsupported) |
| **FL**   | Folklore / popular attribution. No primary source. Included for cultural completeness |
| **AUDIO**| Audio carrier-tone preference (not a brainwave entrainment beat). `beat_hz = 0` → pure tone, no entrainment |

### What we explicitly do NOT ship

- **"Monroe Focus 10 = X Hz" presets.** The Monroe Institute has never publicly published per-Focus-level binaural Hz tables. Online "Focus level → Hz" charts are fan reconstructions.
- **"i-Doser Astral Projection = Y Hz" presets.** i-Doser's `.drg` files are encrypted; the company publishes no Hz table. The only company-stated specific is that *Lucid Dream* uses "high to low theta" (~4–8 Hz beat) — that's represented by the `lucid_theta` preset.
- **"Russell Targ's secret remote viewing frequency."** Targ never published an entrainment Hz. He described the *state* RVers move through (beta → alpha → theta). The `targ_theta` and `targ_alpha` presets capture his state description, not a prescription.
- **A "Grinberg frequency."** Jacobo Grinberg's syntergic-theory and Faraday-cage telepathy work focused on inter-hemispheric coherence, not specific Hz. There is no honest Hz to attribute to him.

### Catalog (categorized)

**Standard EEG bands** — `delta`, `theta`, `alpha`, `beta`, `gamma`
General-purpose entries with peer-reviewed band associations (Berger; Hori et al. 1994; Xie et al. 2013; Niedermeyer; Llinás).

**Meditation & Gamma Binding** *(peer-reviewed)*
- `gamma40` — 40 Hz thalamocortical binding (Llinás 1991–2002)
- `compassion_gamma` — 35 Hz, compassion meditation in long-term meditators (Lutz, Greischar, Rawlings, Ricard, Davidson 2004, PNAS)
- `voss_lucid` — 40 Hz lucid-REM induction (Voss et al. 2014, Nat. Neurosci.) — original study used tACS, audio analogue is conjectural

**Lucid Dreaming & Hypnagogia**
- `lucid_theta` — low theta (5 Hz) for hypnagogic / dream onset (LaBerge experiential; i-Doser company description)
- `lucid_beta1` — 14 Hz parietal beta-1 marker of lucid REM (Holzinger, LaBerge, Levitan 2006) — describes a correlate, not a prescription

**Gateway / McDonnell 1983** *(declassified CIA memo)*
- `bentov` — 5.5 Hz isochronic, "Bentov acoustical resonance" (McDonnell §3)
- `earth_body` — 7.0 Hz, "earth-body resonance" claim (McDonnell §9) — *not* 7.83 Hz; the memo cites 6.8–7.5 Hz
- `monroe_beta_carrier` — 2877.3 Hz carrier (McDonnell §31, one specific Monroe OBE tape); beat 4 Hz is our pairing, not from the memo

**Remote Viewing State** *(Targ-described)*
- `targ_theta` — 6 Hz, theta-dominant RV state (Targ, *Limitless Mind* 2004)
- `targ_alpha` — 10.5 Hz, alpha en route to theta

**Earth Resonance** *(physics + folklore)*
- `schumann` — 7.83 Hz (Schumann 1952; König 1960s coupling claim). The cavity resonance is real EM physics; the natural field at the body is ~picotesla, far below any plausible entrainment threshold.

**Flow & Intuition** — for quieting the overthinking mind, letting intuition lead, coherent thought sequencing
- `coherent_flow` *(EX)* — 7.5 Hz isochronic, alpha-theta crossover. Combines the Aftanas/Golocheikine meditator EEG finding (frontal alpha + Fm-theta) with the Peniston-Kulkosky alpha-theta NFB tradition (Green & Green at Menninger). The EEG correlate of DMN-quiet / hypnagogic / intuitive states is well-documented; the *audio-entrainment-causes-state* link is observed-but-not-clinically-established (hence `EX`, not `PR`). A multi-stage version (alpha 10 → drift to 7.5 → hold 7.5) is on the roadmap; the single-Hz preset here parks at the destination. **Avoid driving / machinery during use.**
- `fm_theta` *(PR)* — 6 Hz isochronic, Frontal-midline theta signature of focused-attention meditation (Aftanas 2001; Kubota 2001; Lutz 2004). The Hz target is peer-reviewed; audio reproduction of the meditative state is not.
- `smr_calm_focus` *(PR)* — 13.5 Hz binaural, Sensorimotor rhythm. Sterman/Egner-Gruzelier NFB tradition — calm, motionless attention. Note: SMR is clinically supported via *neurofeedback* (closed-loop), not via passive audio entrainment. Useful as a "calm decisive" companion to `coherent_flow` for when intuition needs to translate into action.

**Solfeggio (audio carrier only, no entrainment)** — all `AUDIO` tagged, `beat_hz = 0`
- `hz_174` — 174 Hz "Foundation" / pain reduction
- `hz_285` — 285 Hz "Tissue regeneration"
- `hz_396` — 396 Hz "Liberating guilt"
- `hz_417` — 417 Hz "Facilitating change"
- `hz_432` — 432 Hz tuning (Verdi 1884 letter; modern conspiracy folklore)
- `hz_528` — 528 Hz "Love frequency" (Puleo c.1974 numerology; Horowitz 1999)
- `hz_639` — 639 Hz "Connecting / relationships"
- `hz_741` — 741 Hz "Awakening intuition"
- `hz_852` — 852 Hz "Returning to spiritual order"
- `hz_963` — 963 Hz "Divine consciousness / pineal"
- These are **audio carrier pitches**. The brainwave-entrainment beat is `0`, so the program emits a pure tone with no binaural difference. Conflating audio carrier Hz with EEG-band beat Hz is the most common entrainment-marketing error — we keep them clearly separate.

**Monroe Focus Levels (folklore)** — TMI has never publicly published per-Focus Hz; charts circulating online are fan reconstructions. We ship them so users who ask for them by name get them, with the `[FL]` tag prominent.
- `focus_10` — "Mind awake / body asleep" — folk 4 Hz
- `focus_12` — "Expanded awareness" — folk 10 Hz
- `focus_15` — "No time" — folk 8 Hz
- `focus_21` — "Bridge to non-physical" — folk 5 Hz
- `focus_27` — "Reception center / The Park" — folk 1.5 Hz

**i-Doser dose names (folklore)** — i-Doser does not publish Hz; `.drg` files are encrypted. These are fan-attributions from blog teardowns.
- `idoser_astral` — Astral Projection — folk 6.3 Hz
- `idoser_obe` — Out of Body — folk 4 Hz
- `idoser_french_roast` — Caffeine-like alertness — folk 18 Hz beta
- `idoser_quick_happy` — Quick mood lift — folk 10 Hz alpha

**Psi & Foresight (folklore)** — no primary source documents these Hz-to-effect claims.
- `foresight` — Foresight / Precognition — folk 7.5 Hz alpha-theta crossover
- `third_eye` — Third Eye / Pineal — folk 6.3 Hz beat on 936 Hz carrier
- `theta_gamma_crossing` — Theta-Gamma "psychic" Hz — folk 6.3 Hz
- `dmt_state` — DMT-like state — folk 3 Hz delta

### Sources (key references)

- Aftanas, L.I. & Golocheikine, S.A. (2001) — meditator EEG: frontal-midline theta + alpha
- Bentov, I. — referenced in McDonnell 1983 (4–7 Hz acoustical resonance claim)
- Berger, H. (1929) — first human EEG
- Cahn, B.R. & Polich, J. (2006) — meditation states & traits EEG meta-analysis
- Egner, T. & Gruzelier, J. (2003) — SMR neurofeedback & attentional performance
- Hori, T., Hayashi, M., Morikawa, T. (1994) — hypnagogic EEG taxonomy
- Holzinger, B., LaBerge, S., Levitan, L. (2006) — beta-1 parietal marker of lucid REM
- Horowitz, L. (1999) — *Healing Codes for the Biological Apocalypse* (528 Hz claim)
- König, H. (1960s) — Schumann-resonance brain-coupling claim
- Kubota, Y. et al. (2001) — Frontal-midline theta during meditation
- Llinás, R. & Paré, D. (1991); Llinás, R. (2002) — thalamocortical 40 Hz binding
- Lutz, A., Greischar, L., Rawlings, N., Ricard, M., Davidson, R. (2004) PNAS — gamma in compassion meditation
- Peniston, E.G. & Kulkosky, P.J. (1989, 1991) — alpha-theta NFB protocol
- Sterman, M.B. — sensorimotor rhythm neurofeedback tradition
- McDonnell, W. (1983) — *Analysis and Assessment of Gateway Process* (CIA, declassified 2003)
- Puleo, J. (c.1974) — Solfeggio numerology origin
- Schumann, W.O. (1952) — Earth-ionosphere cavity resonance prediction
- Targ, R. (2004) — *Limitless Mind* (RV state descriptions); Intuition Network interview
- Verdi, G. (1884) — letter on A=432 tuning
- Voss, U. et al. (2014) Nat. Neurosci. — 40 Hz tACS and lucid REM
- Xie, L. et al. (2013) Science — sleep and glymphatic clearance

## Verification ideas

- Record system loopback with Audacity, FFT each channel:
  - Binaural → L peak at carrier, R peak at carrier + beat.
  - Isochronic (sine) → carrier + sidebands at carrier ± beat.
  - Isochronic (square) → carrier + odd-harmonic sidebands.
  - Monaural → equal peaks at carrier and carrier + beat in mono.
- Phone slow-mo (240 fps) at the screen during alpha (10 Hz) → ~20 bright-frame transitions per 2 seconds.
