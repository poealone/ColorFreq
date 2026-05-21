"""Tkinter control-panel GUI for the brainwave entrainment tuner.

Launcher pattern: the GUI gathers SessionParams, then on Start it hides itself
and runs the existing pygame VisualEngine. When pygame closes, the GUI
restores and stays ready for another session.

Run via:  python -m tuner.tuner --gui
"""

from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext, ttk

from .audio_engine import AudioEngine
from .presets import AUDIO_MODES, SessionParams
from .protocols import (
    EVIDENCE_LEGEND,
    PROTOCOLS,
    Protocol,
    by_id,
    categories,
    in_category,
)
from .visual_engine import VisualEngine

# Imported lazily inside _start_session to avoid circular import at module load
_PSE_WARNING_TEXT: str | None = None


def _get_pse_text() -> str:
    global _PSE_WARNING_TEXT
    if _PSE_WARNING_TEXT is None:
        from .tuner import PSE_WARNING_TEXT
        _PSE_WARNING_TEXT = PSE_WARNING_TEXT
    return _PSE_WARNING_TEXT


class TunerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Brainwave Entrainment Tuner")
        self.root.geometry("1080x720")
        self.root.minsize(900, 600)

        self.params = SessionParams()
        self._safety_confirmed = False
        self._selected_protocol: Protocol | None = None

        self._build_styles()
        self._build_ui()
        self._populate_tree()
        self._refresh_params_into_widgets()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_styles(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("vista")  # Windows default; falls back if unavailable
        except tk.TclError:
            pass
        style.configure("Start.TButton", font=("Segoe UI", 14, "bold"), padding=10)
        style.configure("Status.TLabel", foreground="#555555")
        style.configure("Tag.TLabel", font=("Consolas", 10, "bold"))

    def _build_ui(self):
        # Shared StringVars used by callbacks during construction --------
        self.status_var = tk.StringVar(value="Ready. Select a protocol or use Custom settings.")

        # Top bar -------------------------------------------------------
        top = ttk.Frame(self.root, padding=(10, 8))
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(
            top, text="Brainwave Entrainment Tuner", font=("Segoe UI", 14, "bold")
        ).pack(side=tk.LEFT)
        ttk.Button(top, text="Evidence Legend", command=self._show_legend).pack(
            side=tk.RIGHT
        )

        # Status bar (built early so it exists when sub-builders fire callbacks)
        ttk.Label(
            self.root, textvariable=self.status_var, style="Status.TLabel",
            padding=(10, 4), relief=tk.SUNKEN, anchor=tk.W,
        ).pack(side=tk.BOTTOM, fill=tk.X)

        # Main paned ---------------------------------------------------
        paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 6))

        left = ttk.Frame(paned, padding=4)
        paned.add(left, weight=1)
        self._build_catalog_tree(left)

        right = ttk.Frame(paned, padding=4)
        paned.add(right, weight=2)
        self._build_details(right)
        self._build_controls(right)
        self._build_start_button(right)

    def _build_catalog_tree(self, parent: ttk.Frame):
        ttk.Label(parent, text="Protocol Catalog", font=("Segoe UI", 11, "bold")).pack(
            anchor=tk.W
        )
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        self.tree = ttk.Treeview(frame, columns=("tag", "beat"), show="tree headings", height=24)
        self.tree.heading("#0", text="Protocol")
        self.tree.heading("tag", text="Tag")
        self.tree.heading("beat", text="Beat Hz")
        self.tree.column("#0", width=300, anchor=tk.W)
        self.tree.column("tag", width=60, anchor=tk.CENTER)
        self.tree.column("beat", width=70, anchor=tk.E)

        vsb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

    def _build_details(self, parent: ttk.Frame):
        details = ttk.LabelFrame(parent, text="Details", padding=8)
        details.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))

        self.detail_name = tk.StringVar(value="(no protocol selected — using default Custom values)")
        self.detail_tag = tk.StringVar(value="")
        self.detail_claim = tk.StringVar(value="")
        self.detail_source = tk.StringVar(value="")
        self.detail_notes = tk.StringVar(value="")

        ttk.Label(details, text="Selected:", width=10, anchor=tk.W).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(details, textvariable=self.detail_name, font=("Segoe UI", 10, "bold"),
                  wraplength=560).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(details, text="Tag:", width=10, anchor=tk.W).grid(row=1, column=0, sticky=tk.W)
        ttk.Label(details, textvariable=self.detail_tag, style="Tag.TLabel").grid(row=1, column=1, sticky=tk.W)

        ttk.Label(details, text="Claim:", width=10, anchor=tk.W).grid(row=2, column=0, sticky=tk.NW)
        ttk.Label(details, textvariable=self.detail_claim, wraplength=560).grid(row=2, column=1, sticky=tk.W)

        ttk.Label(details, text="Source:", width=10, anchor=tk.W).grid(row=3, column=0, sticky=tk.NW)
        ttk.Label(details, textvariable=self.detail_source, wraplength=560,
                  foreground="#555").grid(row=3, column=1, sticky=tk.W)

        ttk.Label(details, text="Notes:", width=10, anchor=tk.W).grid(row=4, column=0, sticky=tk.NW)
        ttk.Label(details, textvariable=self.detail_notes, wraplength=560,
                  foreground="#a35a00").grid(row=4, column=1, sticky=tk.W)

        details.columnconfigure(1, weight=1)

    def _build_controls(self, parent: ttk.Frame):
        ctl = ttk.LabelFrame(parent, text="Session Parameters", padding=8)
        ctl.pack(side=tk.TOP, fill=tk.BOTH, expand=False, pady=(0, 6))

        # Audio mode
        ttk.Label(ctl, text="Audio mode").grid(row=0, column=0, sticky=tk.W)
        self.mode_var = tk.StringVar(value=self.params.audio_mode)
        modes_frame = ttk.Frame(ctl)
        modes_frame.grid(row=0, column=1, columnspan=3, sticky=tk.W)
        for i, m in enumerate(AUDIO_MODES):
            ttk.Radiobutton(
                modes_frame, text=m.capitalize(), variable=self.mode_var, value=m,
                command=self._on_mode_change,
            ).grid(row=0, column=i, padx=(0, 12))

        # Envelope
        ttk.Label(ctl, text="Envelope").grid(row=1, column=0, sticky=tk.W)
        self.envelope_var = tk.StringVar(value=self.params.isochronic_envelope)
        env_frame = ttk.Frame(ctl)
        env_frame.grid(row=1, column=1, columnspan=3, sticky=tk.W)
        self._env_radios = []
        for i, e in enumerate(("sine", "square")):
            r = ttk.Radiobutton(env_frame, text=e.capitalize(), variable=self.envelope_var, value=e)
            r.grid(row=0, column=i, padx=(0, 12))
            self._env_radios.append(r)

        # Sliders -- helper to make labeled sliders
        self._build_slider(ctl, 2, "Carrier Hz", "carrier_hz", 20.0, 4000.0, 0.5, fmt="{:.1f}")
        self._build_slider(ctl, 3, "Beat Hz",    "beat_hz",    0.0, 100.0,  0.1, fmt="{:.2f}")
        self._build_slider(ctl, 4, "Volume",     "volume",     0.0, 1.0,    0.01, fmt="{:.2f}")
        self._build_slider(ctl, 5, "Fade-in s",  "fade_in_s",  0.0, 30.0,   0.5, fmt="{:.1f}")
        self._build_slider(ctl, 6, "Fade-out s", "fade_out_s", 0.0, 30.0,   0.5, fmt="{:.1f}")

        # Duration
        ttk.Label(ctl, text="Duration").grid(row=7, column=0, sticky=tk.W, pady=(6, 0))
        self.duration_open_var = tk.BooleanVar(value=(self.params.duration_s is None))
        self.duration_minutes_var = tk.StringVar(value="20")
        dur_frame = ttk.Frame(ctl)
        dur_frame.grid(row=7, column=1, columnspan=3, sticky=tk.W, pady=(6, 0))
        ttk.Checkbutton(
            dur_frame, text="Open-ended", variable=self.duration_open_var,
            command=self._on_duration_open_change,
        ).grid(row=0, column=0, padx=(0, 10))
        self.duration_spin = ttk.Spinbox(
            dur_frame, from_=1, to=600, textvariable=self.duration_minutes_var, width=6
        )
        self.duration_spin.grid(row=0, column=1)
        ttk.Label(dur_frame, text="minutes").grid(row=0, column=2, padx=(4, 0))

        # Audio / visual enable
        self.audio_on_var = tk.BooleanVar(value=self.params.audio_enabled)
        self.visual_on_var = tk.BooleanVar(value=self.params.visual_enabled)
        toggles = ttk.Frame(ctl)
        toggles.grid(row=8, column=0, columnspan=4, sticky=tk.W, pady=(6, 0))
        ttk.Checkbutton(toggles, text="Audio on", variable=self.audio_on_var).pack(side=tk.LEFT, padx=(0, 14))
        ttk.Checkbutton(toggles, text="Visual on", variable=self.visual_on_var).pack(side=tk.LEFT)

        ctl.columnconfigure(1, weight=1)
        self._on_mode_change()
        self._on_duration_open_change()

    def _build_slider(self, parent, row, label, attr, lo, hi, step, fmt="{:.2f}"):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
        var = tk.DoubleVar(value=float(getattr(self.params, attr)))
        scale = ttk.Scale(parent, from_=lo, to=hi, variable=var, orient=tk.HORIZONTAL, length=420)
        scale.grid(row=row, column=1, sticky=tk.EW, padx=(6, 8))
        value_lbl = ttk.Label(parent, text=fmt.format(var.get()), width=8, anchor=tk.E)
        value_lbl.grid(row=row, column=2, sticky=tk.E)

        def _on_change(_evt=None, _attr=attr, _var=var, _lbl=value_lbl, _fmt=fmt):
            v = float(_var.get())
            _lbl.configure(text=_fmt.format(v))
            setattr(self.params, _attr, v)

        scale.configure(command=lambda _v, cb=_on_change: cb())
        # Hold references so we can refresh from outside
        if not hasattr(self, "_slider_refs"):
            self._slider_refs = {}
        self._slider_refs[attr] = (var, value_lbl, fmt)

    def _build_start_button(self, parent: ttk.Frame):
        bar = ttk.Frame(parent, padding=(0, 6))
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.start_btn = ttk.Button(
            bar, text="START SESSION", style="Start.TButton", command=self._on_start,
        )
        self.start_btn.pack(fill=tk.X)

        scope_bar = ttk.Frame(parent, padding=(0, 0))
        scope_bar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(
            scope_bar, text="Open Oscilloscope (mic / loopback)",
            command=self._on_open_scope,
        ).pack(fill=tk.X)

    # ------------------------------------------------------------------
    # Catalog tree
    # ------------------------------------------------------------------
    def _populate_tree(self):
        for cat in categories():
            parent = self.tree.insert("", "end", text=cat, open=True, values=("", ""))
            for p in in_category(cat):
                self.tree.insert(
                    parent, "end", iid=p.id, text=p.name,
                    values=(p.evidence, f"{p.beat_hz:.2f}"),
                )

    def _on_tree_select(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        iid = sel[0]
        proto = by_id(iid)
        if proto is None:
            return
        self._selected_protocol = proto
        self.detail_name.set(proto.name)
        self.detail_tag.set(f"[{proto.evidence}]  {EVIDENCE_LEGEND.get(proto.evidence, '')}")
        self.detail_claim.set(proto.claim or "—")
        self.detail_source.set(proto.source or "—")
        self.detail_notes.set(proto.notes or "")

        self.params = proto.to_params(self.params)
        self._refresh_params_into_widgets()
        self.status_var.set(f"Loaded protocol: {proto.name}  [{proto.evidence}]")

    # ------------------------------------------------------------------
    # Widget <-> params sync
    # ------------------------------------------------------------------
    def _refresh_params_into_widgets(self):
        self.mode_var.set(self.params.audio_mode)
        self.envelope_var.set(self.params.isochronic_envelope)
        for attr, (var, lbl, fmt) in getattr(self, "_slider_refs", {}).items():
            v = float(getattr(self.params, attr))
            var.set(v)
            lbl.configure(text=fmt.format(v))
        self.audio_on_var.set(self.params.audio_enabled)
        self.visual_on_var.set(self.params.visual_enabled)
        self._on_mode_change()

    def _collect_widgets_into_params(self) -> SessionParams:
        # Sliders already write through on_change; collect mode/envelope/duration/toggles here.
        p = SessionParams(**self.params.__dict__)
        p.audio_mode = self.mode_var.get()
        p.isochronic_envelope = self.envelope_var.get()
        p.audio_enabled = self.audio_on_var.get()
        p.visual_enabled = self.visual_on_var.get()
        if self.duration_open_var.get():
            p.duration_s = None
        else:
            try:
                p.duration_s = max(1.0, float(self.duration_minutes_var.get())) * 60.0
            except ValueError:
                p.duration_s = None
        return p

    def _on_mode_change(self):
        mode = self.mode_var.get()
        state = (tk.NORMAL if mode == "isochronic" else tk.DISABLED)
        for r in self._env_radios:
            r.configure(state=state)
        if mode == "binaural":
            self.status_var.set("Binaural mode — use STEREO HEADPHONES.")
        elif mode == "isochronic":
            self.status_var.set("Isochronic mode — works on speakers; envelope toggle active.")
        else:
            self.status_var.set("Monaural mode — works on speakers.")

    def _on_duration_open_change(self):
        if self.duration_open_var.get():
            self.duration_spin.configure(state=tk.DISABLED)
        else:
            self.duration_spin.configure(state=tk.NORMAL)

    # ------------------------------------------------------------------
    # Legend popup
    # ------------------------------------------------------------------
    def _show_legend(self):
        win = tk.Toplevel(self.root)
        win.title("Evidence Legend")
        win.transient(self.root)
        win.grab_set()
        win.geometry("520x280")
        frame = ttk.Frame(win, padding=14)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="Evidence tags", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W, pady=(0, 8))
        for tag, desc in EVIDENCE_LEGEND.items():
            line = ttk.Frame(frame)
            line.pack(fill=tk.X, pady=2)
            ttk.Label(line, text=f"[{tag}]", width=8, font=("Consolas", 10, "bold")).pack(side=tk.LEFT)
            ttk.Label(line, text=desc, wraplength=420, justify=tk.LEFT).pack(side=tk.LEFT)
        ttk.Button(frame, text="Close", command=win.destroy).pack(side=tk.BOTTOM, pady=(12, 0))

    # ------------------------------------------------------------------
    # Safety dialog
    # ------------------------------------------------------------------
    def _show_safety_dialog(self) -> bool:
        win = tk.Toplevel(self.root)
        win.title("Photosensitive Epilepsy Warning")
        win.transient(self.root)
        win.grab_set()
        win.geometry("760x560")

        frame = ttk.Frame(win, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        st = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=("Consolas", 9), height=20)
        st.insert(tk.END, _get_pse_text())
        st.configure(state=tk.DISABLED)
        st.pack(fill=tk.BOTH, expand=True)

        prompt = ttk.Frame(frame)
        prompt.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(prompt, text="Type  I UNDERSTAND  to enable the confirm button:").pack(anchor=tk.W)
        entry_var = tk.StringVar()
        entry = ttk.Entry(prompt, textvariable=entry_var, width=40)
        entry.pack(side=tk.LEFT, padx=(0, 10), pady=(4, 0))

        result = {"ok": False}
        confirm_btn = ttk.Button(prompt, text="I confirm", state=tk.DISABLED)
        cancel_btn = ttk.Button(prompt, text="Cancel")

        def _on_entry_change(*_):
            if entry_var.get().strip().upper() == "I UNDERSTAND":
                confirm_btn.configure(state=tk.NORMAL)
            else:
                confirm_btn.configure(state=tk.DISABLED)

        entry_var.trace_add("write", _on_entry_change)

        def _on_confirm():
            result["ok"] = True
            win.destroy()

        def _on_cancel():
            result["ok"] = False
            win.destroy()

        confirm_btn.configure(command=_on_confirm)
        cancel_btn.configure(command=_on_cancel)
        confirm_btn.pack(side=tk.LEFT, pady=(4, 0))
        cancel_btn.pack(side=tk.LEFT, padx=(8, 0), pady=(4, 0))

        entry.focus_set()
        self.root.wait_window(win)
        return result["ok"]

    # ------------------------------------------------------------------
    # Session launcher
    # ------------------------------------------------------------------
    def _on_start(self):
        if not self._safety_confirmed:
            if not self._show_safety_dialog():
                self.status_var.set("Session not started — safety confirmation declined.")
                return
            self._safety_confirmed = True

        params = self._collect_widgets_into_params()
        self.status_var.set("Launching pygame... press Y on the splash, Esc to exit.")
        self.root.update_idletasks()

        # Hide GUI while session runs
        self.root.withdraw()
        audio = AudioEngine(params)
        visual = VisualEngine(audio, warning_text=_get_pse_text(), fullscreen=True, vsync=True)
        audio.start()
        try:
            visual.run()
        finally:
            audio.stop(fade_out=True)

        self.root.deiconify()
        self.root.lift()
        self.status_var.set("Session ended. Ready for another.")

    def _on_open_scope(self):
        """Spawn the oscilloscope as a separate process so it can run alongside
        the GUI and any active session (pygame visual + audio)."""
        import subprocess
        import sys
        try:
            subprocess.Popen(
                [sys.executable, "-m", "tuner.tuner", "--scope"],
                cwd=None,
            )
            self.status_var.set("Oscilloscope launched in a new window.")
        except Exception as e:
            self.status_var.set(f"Could not launch oscilloscope: {e}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> int:
        self.root.mainloop()
        return 0


def main() -> int:
    return TunerGUI().run()


if __name__ == "__main__":
    raise SystemExit(main())
