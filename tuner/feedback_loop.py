"""Video feedback loop: live webcam viewer with a per-frame filter pipeline
(zoom, rotation, hue shift, brightness/contrast, color overlay, blur, digital
echo, kaleidoscope) designed for the camera-at-monitor optical-feedback
scrying setup.

The user points the webcam at the monitor showing this window. The closed
optical loop amplifies small artifacts into emergent recursive patterns --
a well-documented visual phenomenon (Bill Viola 'Information' 1973;
strange-face-in-the-mirror illusion: Caputo 2010, *Perception* 39).

The 'echo' filter blends the previous output frame back into the current
one in software, which both substitutes for the optical loop while dialing
in filters AND stacks with the optical loop for stronger amplification.

Run standalone:
    python -m tuner.feedback_loop
or via tuner:
    python -m tuner.tuner --feedback
    python -m tuner.tuner --feedback --feedback-camera 1 --feedback-fullscreen
    python -m tuner.tuner --feedback-list-cameras
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import cv2
import numpy as np
import pygame

CV_BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY

OVERLAY_TINTS: list[tuple[str, tuple[int, int, int] | None]] = [
    ("off",      None),
    ("red",      (40, 40, 200)),
    ("orange",   (40, 110, 220)),
    ("green",    (60, 200, 60)),
    ("cyan",     (200, 200, 40)),
    ("blue",     (200, 60, 40)),
    ("magenta",  (200, 40, 200)),
]


@dataclass
class CameraInfo:
    index: int
    width: int
    height: int


def list_cameras(max_probe: int = 10) -> list[CameraInfo]:
    """Probe indices 0..max_probe-1 and return the ones that yield a frame."""
    out: list[CameraInfo] = []
    for i in range(max_probe):
        cap = cv2.VideoCapture(i, CV_BACKEND)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                out.append(CameraInfo(index=i, width=w, height=h))
        cap.release()
    return out


@dataclass
class Filters:
    zoom: float = 1.0
    rotation_deg: float = 0.0
    hue_shift_deg: float = 0.0     # 0..180 (OpenCV HSV hue is 0..179)
    brightness: float = 0.0         # added: -1..+1
    contrast: float = 1.0           # multiplied: 0..3
    overlay_idx: int = 0            # index into OVERLAY_TINTS
    overlay_strength: float = 0.25
    blur_kernel: int = 1            # 1=no blur; odd numbers >= 3 = blur
    echo: float = 0.0               # 0..0.99 -- weight of previous output
    kaleidoscope: bool = False

    def reset(self) -> None:
        self.zoom = 1.0
        self.rotation_deg = 0.0
        self.hue_shift_deg = 0.0
        self.brightness = 0.0
        self.contrast = 1.0
        self.overlay_idx = 0
        self.overlay_strength = 0.25
        self.blur_kernel = 1
        self.echo = 0.0
        self.kaleidoscope = False


# ----------------------------------------------------------------------
# Per-filter helpers
# ----------------------------------------------------------------------

def apply_zoom(frame: np.ndarray, zoom: float) -> np.ndarray:
    if abs(zoom - 1.0) < 1e-3:
        return frame
    h, w = frame.shape[:2]
    if zoom > 1.0:
        # Center crop a smaller ROI, then upscale to original size
        new_w = max(1, int(w / zoom))
        new_h = max(1, int(h / zoom))
        x0 = (w - new_w) // 2
        y0 = (h - new_h) // 2
        roi = frame[y0:y0 + new_h, x0:x0 + new_w]
        return cv2.resize(roi, (w, h), interpolation=cv2.INTER_LINEAR)
    # zoom < 1.0: shrink and pad
    new_w = max(1, int(w * zoom))
    new_h = max(1, int(h * zoom))
    small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros_like(frame)
    x0 = (w - new_w) // 2
    y0 = (h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = small
    return canvas


def apply_rotation(frame: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 0.1:
        return frame
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def apply_hue_shift(frame: np.ndarray, hue_shift_deg: float) -> np.ndarray:
    if abs(hue_shift_deg) < 0.5:
        return frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # OpenCV hue is 0..179 (representing 0..360 deg in 2-deg steps)
    shift = int(round(hue_shift_deg / 2.0)) % 180
    hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int32) + shift) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_brightness_contrast(frame: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    if abs(brightness) < 1e-3 and abs(contrast - 1.0) < 1e-3:
        return frame
    beta = brightness * 127.0
    return cv2.convertScaleAbs(frame, alpha=contrast, beta=beta)


def apply_overlay(frame: np.ndarray, tint_bgr: tuple[int, int, int] | None, strength: float) -> np.ndarray:
    if tint_bgr is None or strength <= 1e-3:
        return frame
    overlay = np.full_like(frame, tint_bgr, dtype=np.uint8)
    return cv2.addWeighted(frame, 1.0 - strength, overlay, strength, 0.0)


def apply_blur(frame: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1:
        return frame
    k = kernel if kernel % 2 == 1 else kernel + 1
    return cv2.GaussianBlur(frame, (k, k), 0)


def apply_echo(frame: np.ndarray, prev: np.ndarray | None, echo: float) -> np.ndarray:
    if echo <= 1e-3 or prev is None or prev.shape != frame.shape:
        return frame
    return cv2.addWeighted(frame, 1.0 - echo, prev, echo, 0.0)


def apply_kaleidoscope(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    half_w = w // 2
    left = frame[:, :half_w]
    mirrored = cv2.flip(left, 1)
    out = frame.copy()
    out[:, half_w:half_w + mirrored.shape[1]] = mirrored
    return out


def run_pipeline(frame: np.ndarray, prev_out: np.ndarray | None, f: Filters) -> np.ndarray:
    out = apply_zoom(frame, f.zoom)
    out = apply_rotation(out, f.rotation_deg)
    out = apply_hue_shift(out, f.hue_shift_deg)
    out = apply_brightness_contrast(out, f.brightness, f.contrast)
    overlay_name, overlay_bgr = OVERLAY_TINTS[f.overlay_idx % len(OVERLAY_TINTS)]
    out = apply_overlay(out, overlay_bgr, f.overlay_strength)
    out = apply_blur(out, f.blur_kernel)
    out = apply_echo(out, prev_out, f.echo)
    if f.kaleidoscope:
        out = apply_kaleidoscope(out)
    return out


# ----------------------------------------------------------------------
# Main class
# ----------------------------------------------------------------------

class FeedbackLoop:
    def __init__(self, camera_index: int | None = None, fullscreen: bool = False):
        self.cameras: list[CameraInfo] = list_cameras()
        if not self.cameras:
            self.cursor = -1
        else:
            self.cursor = 0
            if camera_index is not None:
                for i, c in enumerate(self.cameras):
                    if c.index == camera_index:
                        self.cursor = i
                        break

        self.cap: cv2.VideoCapture | None = None
        self.filters = Filters()
        self.paused = False
        self.hud_visible = True
        self.fullscreen = fullscreen
        self.status: str = ""
        self.prev_out: np.ndarray | None = None

        self.screen: pygame.Surface | None = None
        self.font: pygame.font.Font | None = None
        self.windowed_size = (1280, 720)

    @property
    def current_camera(self) -> CameraInfo | None:
        if 0 <= self.cursor < len(self.cameras):
            return self.cameras[self.cursor]
        return None

    def _open_camera(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cam = self.current_camera
        if cam is None:
            self.status = "No cameras detected. Connect a webcam and press R to re-enumerate."
            return
        try:
            self.cap = cv2.VideoCapture(cam.index, CV_BACKEND)
            if not self.cap.isOpened():
                raise RuntimeError("VideoCapture.isOpened() returned False")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.height)
            self.status = f"Camera #{cam.index}  {cam.width}x{cam.height}"
            self.prev_out = None
        except Exception as e:
            print(f"[feedback] could not open camera index {cam.index}: {e}", file=sys.stderr)
            self.status = f"Could not open camera #{cam.index}: {e}"
            self.cap = None

    def _cycle_camera(self, step: int) -> None:
        if not self.cameras:
            return
        self.cursor = (self.cursor + step) % len(self.cameras)
        self._open_camera()

    def _toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        flags = pygame.FULLSCREEN if self.fullscreen else pygame.RESIZABLE
        info = pygame.display.Info()
        size = (info.current_w, info.current_h) if self.fullscreen else self.windowed_size
        self.screen = pygame.display.set_mode(size, flags)

    # ------------------------------------------------------------------
    # Pygame loop
    # ------------------------------------------------------------------
    def run(self) -> int:
        pygame.init()
        flags = pygame.FULLSCREEN if self.fullscreen else pygame.RESIZABLE
        if self.fullscreen:
            info = pygame.display.Info()
            self.screen = pygame.display.set_mode((info.current_w, info.current_h), flags)
        else:
            self.screen = pygame.display.set_mode(self.windowed_size, flags)
        pygame.display.set_caption("Video Feedback Loop -- scrying setup")
        self.font = pygame.font.SysFont("consolas,couriernew,monospace", 14)
        self._open_camera()

        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key, event.mod)
                elif event.type == pygame.VIDEORESIZE and not self.fullscreen:
                    self.windowed_size = event.size
                    self.screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)

            self._render_frame()
            pygame.display.flip()
            clock.tick(60)

        if self.cap is not None:
            self.cap.release()
        pygame.quit()
        return 0

    def _handle_key(self, key: int, mod: int) -> bool:
        shift = bool(mod & pygame.KMOD_SHIFT)
        f = self.filters
        if key in (pygame.K_ESCAPE, pygame.K_q):
            return False
        elif key == pygame.K_F11:
            self._toggle_fullscreen()
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_0:
            f.reset()
            self.prev_out = None
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            f.zoom = min(8.0, f.zoom * 1.10)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            f.zoom = max(0.25, f.zoom / 1.10)
        elif key == pygame.K_LEFTBRACKET:
            f.rotation_deg = (f.rotation_deg - 5.0) % 360.0
        elif key == pygame.K_RIGHTBRACKET:
            f.rotation_deg = (f.rotation_deg + 5.0) % 360.0
        elif key == pygame.K_h:
            f.hue_shift_deg = (f.hue_shift_deg + (-5.0 if shift else 5.0)) % 360.0
        elif key == pygame.K_b:
            f.brightness = max(-1.0, min(1.0, f.brightness + (-0.05 if shift else 0.05)))
        elif key == pygame.K_k:
            f.contrast = max(0.05, min(3.0, f.contrast + (-0.05 if shift else 0.05)))
        elif key == pygame.K_o:
            f.overlay_idx = (f.overlay_idx + 1) % len(OVERLAY_TINTS)
        elif key == pygame.K_t:
            f.overlay_strength = max(0.0, min(1.0, f.overlay_strength + (-0.05 if shift else 0.05)))
        elif key == pygame.K_f:
            if shift:
                f.blur_kernel = max(1, f.blur_kernel - 2)
            else:
                f.blur_kernel = min(31, f.blur_kernel + 2)
        elif key == pygame.K_e:
            f.echo = max(0.0, min(0.99, f.echo + (-0.05 if shift else 0.05)))
        elif key == pygame.K_m:
            f.kaleidoscope = not f.kaleidoscope
        elif key == pygame.K_c:
            step = -1 if shift else 1
            self._cycle_camera(step)
        elif key == pygame.K_r:
            self.cameras = list_cameras()
            if self.cameras:
                self.cursor = self.cursor % len(self.cameras) if len(self.cameras) else 0
            self._open_camera()
        elif key == pygame.K_i:
            self.hud_visible = not self.hud_visible
        return True

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render_frame(self) -> None:
        assert self.screen is not None
        sw, sh = self.screen.get_size()

        if self.cap is None or self.paused:
            self.screen.fill((10, 10, 16))
            if self.hud_visible:
                self._draw_hud(sw, sh, no_signal=True)
            return

        ok, frame_bgr = self.cap.read()
        if not ok or frame_bgr is None:
            self.screen.fill((20, 6, 6))
            if self.hud_visible:
                self._draw_hud(sw, sh, no_signal=True)
            return

        processed = run_pipeline(frame_bgr, self.prev_out, self.filters)
        self.prev_out = processed.copy()

        # Convert BGR (OpenCV) -> RGB (pygame), transpose for surfarray
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        # Resize to fit window while preserving aspect ratio
        fh, fw = rgb.shape[:2]
        scale = min(sw / fw, sh / fh)
        new_w = max(1, int(fw * scale))
        new_h = max(1, int(fh * scale))
        if (new_w, new_h) != (fw, fh):
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        surf = pygame.image.frombuffer(rgb.tobytes(), (new_w, new_h), "RGB")
        self.screen.fill((0, 0, 0))
        self.screen.blit(surf, ((sw - new_w) // 2, (sh - new_h) // 2))

        if self.hud_visible:
            self._draw_hud(sw, sh, no_signal=False)

    def _draw_hud(self, sw: int, sh: int, no_signal: bool) -> None:
        assert self.font is not None
        f = self.filters
        cam = self.current_camera
        cam_label = "(no camera)" if cam is None else f"#{cam.index}  {cam.width}x{cam.height}"
        cam_pos = "[0/0]" if not self.cameras else f"[{self.cursor + 1}/{len(self.cameras)}]"
        overlay_name = OVERLAY_TINTS[f.overlay_idx % len(OVERLAY_TINTS)][0]

        lines = [
            f"Camera : {cam_pos}  {cam_label}    [C=next  Shift+C=prev  R=re-enum]",
            f"Zoom   : {f.zoom:5.2f}x    Rotation: {f.rotation_deg:6.1f} deg    Kaleidoscope: {'ON' if f.kaleidoscope else 'off'}",
            f"Hue    : {f.hue_shift_deg:5.1f} deg    Bright: {f.brightness:+.2f}    Contrast: {f.contrast:.2f}",
            f"Overlay: {overlay_name:8s} strength {f.overlay_strength:.2f}    Blur: {f.blur_kernel:2d}    Echo: {f.echo:.2f}",
            f"State  : {'PAUSED' if self.paused else 'LIVE'}    Fullscreen: {'ON' if self.fullscreen else 'off'}    [F11 toggle, Space pause, 0 reset]",
        ]
        if self.status:
            lines.append(self.status)
        if no_signal:
            lines.append("!! NO CAMERA SIGNAL -- check connection, press R to re-enumerate !!")

        panel_w = min(sw - 24, 880)
        panel_h = self.font.get_linesize() * len(lines) + 16
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 165))
        self.screen.blit(panel, (12, 12))
        y = 20
        for line in lines:
            surf = self.font.render(line, True, (230, 230, 240))
            self.screen.blit(surf, (24, y))
            y += self.font.get_linesize()

        hint = self.font.render(
            "+/- zoom  | [/] rotate  | H hue  | B bright  | K contrast  | O overlay  | T strength  | F blur  | E echo  | M kaleidoscope  | I HUD  | Esc quit",
            True, (160, 160, 190),
        )
        self.screen.blit(hint, hint.get_rect(midbottom=(sw // 2, sh - 8)))


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="tuner.feedback_loop", description="Video feedback loop / scrying setup.")
    parser.add_argument("--camera", type=int, default=None, help="Camera index (see --list).")
    parser.add_argument("--list", action="store_true", help="List cameras and exit.")
    parser.add_argument("--fullscreen", action="store_true", help="Start fullscreen.")
    args = parser.parse_args(argv)

    if args.list:
        cams = list_cameras()
        if not cams:
            print("No cameras detected.")
            return 1
        for c in cams:
            print(f"  index {c.index}: {c.width}x{c.height}")
        return 0

    return FeedbackLoop(camera_index=args.camera, fullscreen=args.fullscreen).run()


if __name__ == "__main__":
    raise SystemExit(main())
