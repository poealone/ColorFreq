"""Curated catalog of brainwave-entrainment and consciousness-exploration presets.

Each Protocol carries an *evidence tag* so the user knows what they're picking:

    PR     Peer-reviewed neuroscience (Llinas, Lutz/Davidson, Voss, Xie, Hori, etc.)
    EX     Experiential / historical practice (subjective reports from documented
           sources -- Targ, Monroe, Lilly, LaBerge anecdotal). The state may be
           real; the *frequency-to-effect* link is not clinically established.
    PH     Physics-real but entrainment-link is folklore (e.g. the Schumann
           cavity resonance is real; the claim it is the "optimal brain Hz"
           is unsupported).
    FL     Pure folklore / popular attribution. No primary source documents the
           Hz claim. Included for cultural completeness so users who ask for
           these by name get them, with the tag visible.
    AUDIO  An audio carrier-tone preference (not a brainwave entrainment Hz).
           Beat = 0 -> the program emits a pure tone, no entrainment.

We deliberately do NOT label any preset as "Monroe Focus 10 = X Hz" or
"i-Doser Astral Projection = Y Hz". TMI never published Focus->Hz tables and
i-Doser keeps its .drg sequences proprietary -- any such claim circulating
online is folklore. The closest documented Monroe-program Hz numbers come
from the 1983 McDonnell "Gateway Process" memo (declassified by CIA in 2003),
which IS cited here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .presets import SessionParams


@dataclass(frozen=True)
class Protocol:
    id: str
    name: str
    category: str
    audio_mode: str          # "binaural" | "isochronic" | "monaural"
    carrier_hz: float
    beat_hz: float           # 0.0 = pure carrier tone, no entrainment
    envelope: str = "sine"   # for isochronic mode only
    evidence: str = "FL"     # PR | EX | PH | FL | AUDIO
    claim: str = ""
    source: str = ""
    notes: str = ""

    def to_params(self, base: SessionParams | None = None) -> SessionParams:
        p = SessionParams() if base is None else SessionParams(**base.__dict__)
        p.audio_mode = self.audio_mode
        p.carrier_hz = self.carrier_hz
        p.beat_hz = self.beat_hz
        p.isochronic_envelope = self.envelope
        return p


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

PROTOCOLS: list[Protocol] = [

    # --- Standard EEG bands -------------------------------------------------
    Protocol(
        id="delta",
        name="Delta -- deep sleep / SWS",
        category="Standard EEG bands",
        audio_mode="binaural", carrier_hz=150.0, beat_hz=2.0,
        evidence="PR",
        claim="Delta dominates NREM stage 3-4; glymphatic clearance, GH release",
        source="Xie et al. 2013 (Science) -- sleep & glymphatic clearance",
    ),
    Protocol(
        id="theta",
        name="Theta -- meditation / hypnagogia",
        category="Standard EEG bands",
        audio_mode="binaural", carrier_hz=200.0, beat_hz=6.0,
        evidence="PR",
        claim="Theta dominates Stage 1 sleep and hypnagogic state",
        source="Hori, Hayashi, Morikawa 1994 -- hypnagogic EEG taxonomy",
    ),
    Protocol(
        id="alpha",
        name="Alpha -- relaxed alertness",
        category="Standard EEG bands",
        audio_mode="binaural", carrier_hz=200.0, beat_hz=10.0,
        evidence="PR",
        claim="Alpha dominates eyes-closed wakeful relaxation",
        source="Berger 1929 (first EEG); standard textbook",
    ),
    Protocol(
        id="beta",
        name="Beta -- focused attention",
        category="Standard EEG bands",
        audio_mode="binaural", carrier_hz=220.0, beat_hz=18.0,
        evidence="PR",
        claim="Beta dominates active problem-solving / external attention",
        source="Standard EEG textbook (e.g. Niedermeyer)",
    ),
    Protocol(
        id="gamma",
        name="Gamma -- high cognition / binding",
        category="Standard EEG bands",
        audio_mode="isochronic", carrier_hz=240.0, beat_hz=40.0, envelope="sine",
        evidence="PR",
        claim="Gamma indexes feature-binding and conscious access",
        source="Llinas 1991-2002 thalamocortical binding",
        notes="On a 60 Hz display the visual flicker is display-limited.",
    ),

    # --- Meditation & gamma binding (strong PR) -----------------------------
    Protocol(
        id="gamma40",
        name="40 Hz Thalamocortical Binding",
        category="Meditation & Gamma Binding",
        audio_mode="isochronic", carrier_hz=200.0, beat_hz=40.0, envelope="sine",
        evidence="PR",
        claim="40 Hz indexes thalamocortical 'temporal binding' of conscious percepts",
        source="Llinas & Pare 1991; Llinas 2002 -- thalamocortical resonance",
        notes="Most-cited 'gamma binding' frequency. Display-limited on 60 Hz monitors.",
    ),
    Protocol(
        id="compassion_gamma",
        name="Compassion-Meditation Gamma (Tibetan monks)",
        category="Meditation & Gamma Binding",
        audio_mode="isochronic", carrier_hz=220.0, beat_hz=35.0, envelope="sine",
        evidence="PR",
        claim="Long-term meditators self-induce 25-42 Hz gamma during compassion practice",
        source="Lutz, Greischar, Rawlings, Ricard, Davidson 2004 (PNAS)",
        notes="The PNAS paper observed; it does not establish that exogenous 35 Hz entrainment reproduces the effect.",
    ),
    Protocol(
        id="voss_lucid",
        name="40 Hz Lucid-REM Induction",
        category="Meditation & Gamma Binding",
        audio_mode="isochronic", carrier_hz=200.0, beat_hz=40.0, envelope="sine",
        evidence="PR",
        claim="40 Hz tACS during REM increased self-reported lucidity in ~2/3 of subjects",
        source="Voss et al. 2014 (Nature Neuroscience)",
        notes="Original study used transcranial AC stimulation, not audio. Audio analogue is conjectural.",
    ),

    # --- Lucid dreaming & hypnagogia ----------------------------------------
    Protocol(
        id="lucid_theta",
        name="Lucid-Onset Low Theta",
        category="Lucid Dreaming & Hypnagogia",
        audio_mode="isochronic", carrier_hz=180.0, beat_hz=5.0, envelope="sine",
        evidence="EX",
        claim="Low theta (4-5 Hz) accompanies hypnagogic imagery and dream onset",
        source="LaBerge experiential reports; i-Doser 'Lucid Dream' company description ('high to low theta')",
    ),
    Protocol(
        id="lucid_beta1",
        name="Lucid-REM Beta-1 Parietal Marker",
        category="Lucid Dreaming & Hypnagogia",
        audio_mode="binaural", carrier_hz=220.0, beat_hz=14.0,
        evidence="PR",
        claim="Beta-1 (13-19 Hz) parietal activity distinguishes lucid from non-lucid REM",
        source="Holzinger, LaBerge & Levitan 2006",
        notes="The paper describes a correlate, not a prescription -- entraining beta-1 does not reliably induce lucidity.",
    ),

    # --- Gateway / McDonnell 1983 (declassified CIA memo) -------------------
    Protocol(
        id="bentov",
        name="Bentov Acoustical Resonance (4-7 Hz)",
        category="Gateway / McDonnell 1983",
        audio_mode="isochronic", carrier_hz=200.0, beat_hz=5.5, envelope="sine",
        evidence="EX",
        claim="Mechanical or acoustical vibrations 4-7 Hz for protracted periods may achieve consciousness shifts",
        source="McDonnell 1983 'Analysis and Assessment of Gateway Process' Sec. 3 (citing Bentov)",
    ),
    Protocol(
        id="earth_body",
        name="Earth-Body Resonance (Gateway)",
        category="Gateway / McDonnell 1983",
        audio_mode="binaural", carrier_hz=200.0, beat_hz=7.0,
        evidence="EX",
        claim="The body transfers energy into the ionospheric cavity at 6.8-7.5 Hz",
        source="McDonnell 1983 Sec. 9 -- earth/body resonance argument",
        notes="The cavity resonance is real physics (~7.83 Hz). The body-coupling claim is the McDonnell memo's interpretation, not validated.",
    ),
    Protocol(
        id="monroe_beta_carrier",
        name="Monroe Beta-Carrier OBE Tape",
        category="Gateway / McDonnell 1983",
        audio_mode="binaural", carrier_hz=2877.3, beat_hz=4.0,
        evidence="EX",
        claim="One specific Monroe OBE tape employed a Beta carrier at 'around 2877.3 CPS'",
        source="McDonnell 1983 Sec. 31",
        notes="McDonnell documents only the carrier. The 4 Hz beat here is a theta-band guess to give the carrier an entrainment partner; it is not from the memo.",
    ),

    # --- Remote viewing (Targ-described state) ------------------------------
    Protocol(
        id="targ_theta",
        name="RV Theta State (Targ-described)",
        category="Remote Viewing State",
        audio_mode="binaural", carrier_hz=200.0, beat_hz=6.0,
        evidence="EX",
        claim="Successful remote viewers shift toward theta (5-8 cps)",
        source="Targ, Intuition Network interview; 'Limitless Mind' 2004",
        notes="Targ described the state, not a prescription. He published no entrainment Hz.",
    ),
    Protocol(
        id="targ_alpha",
        name="RV Alpha State (Targ-described)",
        category="Remote Viewing State",
        audio_mode="binaural", carrier_hz=200.0, beat_hz=10.5,
        evidence="EX",
        claim="Viewers passed through alpha (9-14 cps) en route to theta-dominant RV state",
        source="Targ, same interview as above",
    ),

    # --- Earth resonance (physics + folklore) -------------------------------
    Protocol(
        id="schumann",
        name="Schumann Resonance (7.83 Hz)",
        category="Earth Resonance",
        audio_mode="binaural", carrier_hz=200.0, beat_hz=7.83,
        evidence="PH",
        claim="7.83 Hz is the Earth-ionosphere cavity fundamental; claimed to be the 'optimal brain frequency'",
        source="Schumann 1952 (cavity prediction); Konig 1960s (brain-coupling claim)",
        notes="The cavity resonance is real EM physics; the brain-entrainment claim is folklore. Natural field strength is ~picotesla.",
    ),

    # --- Solfeggio / 432 Hz / 528 Hz (AUDIO carrier only, no entrainment) ---
    Protocol(
        id="hz_432",
        name="432 Hz Pure Tone (audio tuning)",
        category="Solfeggio (audio carrier only)",
        audio_mode="monaural", carrier_hz=432.0, beat_hz=0.0,
        evidence="AUDIO",
        claim="Verdi (1884) preferred A ~ 432 for singers' comfort; modern claim of 'natural tuning'",
        source="Verdi letter to Bersezio 1884; ISO 16 set A=440 in 1955",
        notes="This is an audio carrier pitch, NOT a brainwave entrainment beat. Beat=0 -> pure tone, no entrainment.",
    ),
    Protocol(
        id="hz_528",
        name="528 Hz 'Love Frequency' Pure Tone",
        category="Solfeggio (audio carrier only)",
        audio_mode="monaural", carrier_hz=528.0, beat_hz=0.0,
        evidence="AUDIO",
        claim="'Love frequency', 'DNA repair', spiritual healing",
        source="Puleo c.1974 numerology; Horowitz 'Healing Codes for the Biological Apocalypse' 1999",
        notes="Folklore origin; no peer review; medieval Solfeggio chant did not use these Hz. Audio carrier only.",
    ),

    # --- Flow & Intuition (peer-reviewed correlates, entrainment causation unproven) ---
    Protocol(
        id="coherent_flow",
        name="Coherent Flow -- alpha-theta crossover",
        category="Flow & Intuition",
        audio_mode="isochronic", carrier_hz=200.0, beat_hz=7.5, envelope="sine",
        evidence="EX",
        claim="Targets the alpha-theta border (~7.5 Hz) -- a band associated in published meditation and neurofeedback research with reduced self-referential thought and hypnagogic, intuitive states. Audio entrainment is not a clinically validated method for reproducing those states.",
        source="Aftanas & Golocheikine 2001 (meditator EEG); Peniston & Kulkosky 1989 (alpha-theta NFB); Cahn & Polich 2006 (meta-analysis); Lutz et al. 2004 (PNAS, meditation EEG)",
        notes="EX -- multiple peer-reviewed correlates of DMN-quiet / intuitive states; the entrainment-causes-state link is observed but not clinically established. Avoid driving / machinery (alpha-theta can induce drowsiness). Isochronic pulsing -- contraindicated for seizure-prone users. Recommended 20-min max sessions. A stage-sequence (alpha 10 -> drift 10->7.5 -> hold 7.5) is a planned future feature; the single-Hz preset here holds the destination.",
    ),
    Protocol(
        id="fm_theta",
        name="Frontal-Midline Theta (mindfulness)",
        category="Flow & Intuition",
        audio_mode="isochronic", carrier_hz=200.0, beat_hz=6.0, envelope="sine",
        evidence="PR",
        claim="Frontal-midline theta (5-7 Hz) is documented during focused-attention meditation and sustained internal attention",
        source="Aftanas & Golocheikine 2001; Kubota et al. 2001; Lutz et al. 2004",
        notes="The Fm-theta signature is an observed correlate. Audio entrainment at this Hz is not proven to reproduce the meditative state, but the target Hz itself is peer-reviewed.",
    ),
    Protocol(
        id="smr_calm_focus",
        name="SMR -- Calm Focused Attention",
        category="Flow & Intuition",
        audio_mode="binaural", carrier_hz=220.0, beat_hz=13.5,
        evidence="PR",
        claim="Sensorimotor rhythm (12-15 Hz) is associated in NFB research with calm, motionless attention",
        source="Sterman SMR tradition; Egner & Gruzelier 2003 (attentional performance)",
        notes="SMR is clinically supported via neurofeedback (closed-loop training), NOT via passive audio entrainment. Useful as a 'calm decisive' counterpoint to alpha-theta when intuition needs to translate into action.",
    ),

    # --- Monroe Focus levels (FAN-ATTRIBUTED; TMI never published per-Focus Hz) ---
    Protocol(
        id="focus_10",
        name="Monroe Focus 10 -- 'Mind awake / body asleep'",
        category="Monroe Focus Levels (folklore)",
        audio_mode="binaural", carrier_hz=200.0, beat_hz=4.0,
        evidence="FL",
        claim="Body-asleep, mind-awake gateway state for guided exploration",
        source="Fan-circulating charts (e.g. theakan.com 'my classification, not TMI's')",
        notes="Popular attribution. The Monroe Institute has NEVER publicly published per-Focus-level binaural Hz tables. Any 'Focus 10 = X Hz' chart online is a fan reconstruction.",
    ),
    Protocol(
        id="focus_12",
        name="Monroe Focus 12 -- 'Expanded awareness'",
        category="Monroe Focus Levels (folklore)",
        audio_mode="binaural", carrier_hz=200.0, beat_hz=10.0,
        evidence="FL",
        claim="Wider perceptual frame, conscious exploration of imagery",
        source="Fan-circulating charts; TMI experiential descriptions only",
        notes="Popular attribution. No primary source documents this Hz claim.",
    ),
    Protocol(
        id="focus_15",
        name="Monroe Focus 15 -- 'No time'",
        category="Monroe Focus Levels (folklore)",
        audio_mode="binaural", carrier_hz=200.0, beat_hz=8.0,
        evidence="FL",
        claim="State of timelessness; intent-projection exercises",
        source="Fan-circulating charts; TMI experiential descriptions only",
        notes="Popular attribution. No primary source documents this Hz claim.",
    ),
    Protocol(
        id="focus_21",
        name="Monroe Focus 21 -- 'Bridge to non-physical'",
        category="Monroe Focus Levels (folklore)",
        audio_mode="binaural", carrier_hz=180.0, beat_hz=5.0,
        evidence="FL",
        claim="Edge of physical/non-physical; threshold for OBE-style exploration",
        source="Fan-circulating charts; TMI experiential descriptions only",
        notes="Popular attribution. No primary source documents this Hz claim.",
    ),
    Protocol(
        id="focus_27",
        name="Monroe Focus 27 -- 'Reception center / The Park'",
        category="Monroe Focus Levels (folklore)",
        audio_mode="binaural", carrier_hz=150.0, beat_hz=1.5,
        evidence="FL",
        claim="Constructed non-physical reception locus in Monroe cosmology",
        source="Fan-circulating charts; TMI experiential descriptions only",
        notes="Popular attribution. No primary source documents this Hz claim.",
    ),

    # --- i-Doser dose names (FAN-DECODED; i-Doser does not disclose Hz) ---
    Protocol(
        id="idoser_astral",
        name="i-Doser-style 'Astral Projection'",
        category="i-Doser dose names (folklore)",
        audio_mode="isochronic", carrier_hz=150.0, beat_hz=6.3, envelope="sine",
        evidence="FL",
        claim="Out-of-body / astral projection experience",
        source="Fan blog teardowns of .drg files; not company-disclosed",
        notes="Popular attribution. i-Doser publishes no Hz tables; .drg files are encrypted. Any 'i-Doser Astral = X Hz' claim is a guess unless backed by a drg2sbg dump.",
    ),
    Protocol(
        id="idoser_obe",
        name="i-Doser-style 'Out of Body'",
        category="i-Doser dose names (folklore)",
        audio_mode="isochronic", carrier_hz=180.0, beat_hz=4.0, envelope="sine",
        evidence="FL",
        claim="OBE / dissociation onset",
        source="Fan blog teardowns; not company-disclosed",
        notes="Popular attribution. No verified primary source.",
    ),
    Protocol(
        id="idoser_french_roast",
        name="i-Doser-style 'French Roast' (caffeine-like)",
        category="i-Doser dose names (folklore)",
        audio_mode="binaural", carrier_hz=220.0, beat_hz=18.0,
        evidence="FL",
        claim="Caffeine-like alertness boost",
        source="Fan blog teardowns; not company-disclosed",
        notes="Popular attribution. Beta-band beats; effect on alertness is plausible from general beta literature, but the specific Hz here is speculative.",
    ),
    Protocol(
        id="idoser_quick_happy",
        name="i-Doser-style 'Quick Happy'",
        category="i-Doser dose names (folklore)",
        audio_mode="binaural", carrier_hz=200.0, beat_hz=10.0,
        evidence="FL",
        claim="Quick mood lift / relaxed positivity",
        source="Fan blog teardowns; not company-disclosed",
        notes="Popular attribution. No verified primary source.",
    ),

    # --- Psi / Foresight / Third Eye (pure folklore) ------------------------
    Protocol(
        id="foresight",
        name="Foresight / Precognition (folklore)",
        category="Psi & Foresight (folklore)",
        audio_mode="binaural", carrier_hz=200.0, beat_hz=7.5,
        evidence="FL",
        claim="Enhanced precognition / future-sensing at alpha-theta crossover",
        source="Online entrainment marketing; no primary scientific source",
        notes="Pure folklore. No peer-reviewed study links any specific Hz to precognitive performance. Included because users ask for it.",
    ),
    Protocol(
        id="third_eye",
        name="Third Eye / Pineal Activation (folklore)",
        category="Psi & Foresight (folklore)",
        audio_mode="isochronic", carrier_hz=936.0, beat_hz=6.3, envelope="sine",
        evidence="FL",
        claim="Pineal-gland 'activation', third-eye opening, inner vision",
        source="New-age entrainment marketing",
        notes="Pure folklore. Pineal gland produces melatonin and is not known to respond to audio Hz. 936 Hz carrier is a Solfeggio-adjacent attribution.",
    ),
    Protocol(
        id="theta_gamma_crossing",
        name="Theta-Gamma 'Crossing' (folklore)",
        category="Psi & Foresight (folklore)",
        audio_mode="isochronic", carrier_hz=200.0, beat_hz=6.3, envelope="sine",
        evidence="FL",
        claim="A special 'psychic' frequency at the theta-gamma boundary",
        source="Online entrainment marketing (e.g. Itsu Sync, fan blogs)",
        notes="Folklore. Theta-gamma phase-amplitude coupling (Lisman & Jensen 2013) is real nested oscillation, NOT a single special tone at 6.3 Hz.",
    ),
    Protocol(
        id="dmt_state",
        name="DMT-like state (folklore)",
        category="Psi & Foresight (folklore)",
        audio_mode="isochronic", carrier_hz=120.0, beat_hz=3.0, envelope="sine",
        evidence="FL",
        claim="Reproducing DMT-induced visionary state via low-delta entrainment",
        source="Online entrainment marketing",
        notes="Pure folklore. No evidence audio entrainment reproduces serotonergic-psychedelic phenomenology.",
    ),

    # --- Solfeggio extended set (all AUDIO carrier only, beat=0) ------------
    Protocol(
        id="hz_174",
        name="174 Hz Pure Tone -- 'Foundation'",
        category="Solfeggio (audio carrier only)",
        audio_mode="monaural", carrier_hz=174.0, beat_hz=0.0,
        evidence="AUDIO",
        claim="'Foundation', pain reduction, sense of security",
        source="Solfeggio extended set; Horowitz lineage",
        notes="Audio carrier only, no entrainment. Folklore origin.",
    ),
    Protocol(
        id="hz_285",
        name="285 Hz Pure Tone -- 'Tissue regeneration'",
        category="Solfeggio (audio carrier only)",
        audio_mode="monaural", carrier_hz=285.0, beat_hz=0.0,
        evidence="AUDIO",
        claim="'Tissue and organ regeneration'",
        source="Solfeggio extended set; Horowitz lineage",
        notes="Audio carrier only. Folklore origin.",
    ),
    Protocol(
        id="hz_396",
        name="396 Hz Pure Tone -- 'Liberating guilt'",
        category="Solfeggio (audio carrier only)",
        audio_mode="monaural", carrier_hz=396.0, beat_hz=0.0,
        evidence="AUDIO",
        claim="'Liberating guilt and fear' (UT chant attribution)",
        source="Puleo/Horowitz Solfeggio set",
        notes="Audio carrier only. Folklore origin; no peer review.",
    ),
    Protocol(
        id="hz_417",
        name="417 Hz Pure Tone -- 'Facilitating change'",
        category="Solfeggio (audio carrier only)",
        audio_mode="monaural", carrier_hz=417.0, beat_hz=0.0,
        evidence="AUDIO",
        claim="'Undoing situations, facilitating change' (RE chant attribution)",
        source="Puleo/Horowitz Solfeggio set",
        notes="Audio carrier only. Folklore origin.",
    ),
    Protocol(
        id="hz_639",
        name="639 Hz Pure Tone -- 'Connecting / relationships'",
        category="Solfeggio (audio carrier only)",
        audio_mode="monaural", carrier_hz=639.0, beat_hz=0.0,
        evidence="AUDIO",
        claim="'Connecting/relationships' (FA chant attribution)",
        source="Puleo/Horowitz Solfeggio set",
        notes="Audio carrier only. Folklore origin.",
    ),
    Protocol(
        id="hz_741",
        name="741 Hz Pure Tone -- 'Awakening intuition'",
        category="Solfeggio (audio carrier only)",
        audio_mode="monaural", carrier_hz=741.0, beat_hz=0.0,
        evidence="AUDIO",
        claim="'Awakening intuition, expression/solutions' (SOL chant attribution)",
        source="Puleo/Horowitz Solfeggio set",
        notes="Audio carrier only. Folklore origin.",
    ),
    Protocol(
        id="hz_852",
        name="852 Hz Pure Tone -- 'Returning to spiritual order'",
        category="Solfeggio (audio carrier only)",
        audio_mode="monaural", carrier_hz=852.0, beat_hz=0.0,
        evidence="AUDIO",
        claim="'Returning to spiritual order' (LA chant attribution)",
        source="Puleo/Horowitz Solfeggio set",
        notes="Audio carrier only. Folklore origin.",
    ),
    Protocol(
        id="hz_963",
        name="963 Hz Pure Tone -- 'Divine consciousness / pineal'",
        category="Solfeggio (audio carrier only)",
        audio_mode="monaural", carrier_hz=963.0, beat_hz=0.0,
        evidence="AUDIO",
        claim="'Divine consciousness', pineal/crown activation",
        source="Puleo/Horowitz Solfeggio set",
        notes="Audio carrier only. Folklore origin.",
    ),
]


# Convenience lookups
def by_id(pid: str) -> Protocol | None:
    for p in PROTOCOLS:
        if p.id == pid:
            return p
    return None


def categories() -> list[str]:
    seen = []
    for p in PROTOCOLS:
        if p.category not in seen:
            seen.append(p.category)
    return seen


def in_category(cat: str) -> list[Protocol]:
    return [p for p in PROTOCOLS if p.category == cat]


EVIDENCE_LEGEND = {
    "PR": "Peer-reviewed neuroscience finding",
    "EX": "Experiential / historical practice -- state real, Hz-to-effect not clinically established",
    "PH": "Physics-real, but entrainment claim is folklore",
    "FL": "Folklore / popular attribution",
    "AUDIO": "Audio carrier-tone preference (no brainwave entrainment)",
}


if __name__ == "__main__":
    for cat in categories():
        print(f"\n=== {cat} ===")
        for p in in_category(cat):
            print(f"  [{p.evidence:5s}] {p.name}")
            print(f"           carrier={p.carrier_hz} Hz  beat={p.beat_hz} Hz  mode={p.audio_mode}")
            print(f"           source: {p.source}")
            if p.notes:
                print(f"           notes : {p.notes}")
