"""
trainer.py
──────────
Interactive gaze trainer — collect labelled iris samples, compute
optimal thresholds, and save them to eye_tracker_config.json.

Run:
    python trainer.py

HOW TO USE
───────────
A live camera preview opens.  Press keys to label your current gaze:

  ↑  (Up arrow)    — you are looking UP     → label sample "up"
  ↓  (Down arrow)  — you are looking DOWN   → label sample "down"
  ←→ (Left/Right)  — you are looking CENTER → label sample "center"
  S  — save & compute thresholds from collected samples
  R  — reset / clear all samples
  Q  — quit without saving

Each key-press captures 30 consecutive iris readings and averages them
to make one robust sample.  Aim for ≥ 15 samples per direction.

After pressing S the script prints the computed thresholds and writes
them to eye_tracker_config.json which main.py loads automatically.
"""

import cv2
import numpy as np
import time
import sys

from gaze_detector import GazeDetector
import config as cfg_module


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

SAMPLES_PER_LABEL  = 30   # frames averaged into one sample
CAMERA_INDEX       = 0
WIN_NAME           = "Gaze Trainer  –  press ↑↓←→ to label  |  S=save  R=reset  Q=quit"


# ─────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────

C = {
    "bg"     : (18, 18, 28),
    "up"     : (80, 220,  80),
    "down"   : (80, 100, 220),
    "center" : (200, 200, 200),
    "accent" : (  0, 200, 255),
    "warn"   : ( 30, 165, 255),
    "white"  : (240, 240, 240),
    "red"    : ( 60,  60, 220),
}

KEY_UP     = 82    # OpenCV key code on Linux  (↑)
KEY_DOWN   = 84    # (↓)
KEY_LEFT   = 81    # (←)
KEY_RIGHT  = 83    # (→)


# ─────────────────────────────────────────────────────────────
# Threshold computation
# ─────────────────────────────────────────────────────────────

def compute_thresholds(samples: dict) -> tuple[float, float] | None:
    """
    Given collected samples dict  {"up": [...], "down": [...], "center": [...]}
    (each value is a list of (rx, ry) floats),
    compute (up_thresh, down_thresh) that separates the clusters best.

    Strategy: midpoint between the mean of each adjacent pair.
    """
    up_vals     = [ry for _, ry in samples.get("up",     [])]
    down_vals   = [ry for _, ry in samples.get("down",   [])]
    center_vals = [ry for _, ry in samples.get("center", [])]

    if not up_vals or not down_vals:
        return None

    up_mean     = float(np.mean(up_vals))
    down_mean   = float(np.mean(down_vals))
    center_mean = float(np.mean(center_vals)) if center_vals else \
                  (up_mean + down_mean) / 2

    # Guard: up should be < center < down
    up_mean     = min(up_mean,     center_mean - 0.01)
    down_mean   = max(down_mean,   center_mean + 0.01)

    up_thresh   = (up_mean   + center_mean) / 2
    down_thresh = (center_mean + down_mean) / 2

    # Clamp to a sane range
    up_thresh   = max(0.20, min(0.49, up_thresh))
    down_thresh = max(0.51, min(0.80, down_thresh))

    return up_thresh, down_thresh


# ─────────────────────────────────────────────────────────────
# Overlay drawing helpers
# ─────────────────────────────────────────────────────────────

def draw_ui(canvas, samples, collecting_label, progress, last_status):
    h, w = canvas.shape[:2]

    # Sidebar background
    sidebar_x = w - 260
    overlay = canvas.copy()
    cv2.rectangle(overlay, (sidebar_x, 0), (w, h), C["bg"], -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    x0 = sidebar_x + 12

    # Title
    cv2.putText(canvas, "GAZE TRAINER", (x0, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, C["accent"], 2)
    cv2.line(canvas, (sidebar_x, 42), (w, 42), (60, 60, 80), 1)

    # Sample counts
    labels = [("up",     "↑ Up",     C["up"]),
              ("center", "● Center", C["center"]),
              ("down",   "↓ Down",   C["down"])]

    y = 70
    for key, display, col in labels:
        n = len(samples.get(key, []))
        bar_filled = min(n, 15)
        for i in range(15):
            bx = x0 + i * 15
            by = y + 12
            color = col if i < bar_filled else (50, 50, 60)
            cv2.rectangle(canvas, (bx, by), (bx + 12, by + 8), color, -1)
        cv2.putText(canvas, f"{display}: {n}", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
        y += 44

    cv2.line(canvas, (sidebar_x, y), (w, y), (60, 60, 80), 1)
    y += 14

    # Collecting indicator
    if collecting_label:
        pct = int(progress * 100)
        cv2.putText(canvas, f"Recording {collecting_label}…", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C["warn"], 1)
        y += 20
        # Progress bar
        bar_w = 220
        cv2.rectangle(canvas, (x0, y), (x0 + bar_w, y + 10), (50, 50, 60), -1)
        cv2.rectangle(canvas, (x0, y), (x0 + int(bar_w * progress), y + 10),
                      C["warn"], -1)
        cv2.putText(canvas, f"{pct}%", (x0 + bar_w + 5, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, C["white"], 1)
        y += 22
    else:
        cv2.putText(canvas, "Ready", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C["up"], 1)
        y += 20

    # Controls
    y += 10
    cv2.line(canvas, (sidebar_x, y), (w, y), (60, 60, 80), 1)
    y += 16
    controls = [
        ("↑", "look up, press ↑"),
        ("↓", "look down, press ↓"),
        ("←/→", "look center, press ←/→"),
        ("S", "Save thresholds"),
        ("R", "Reset samples"),
        ("Q", "Quit"),
    ]
    for key_label, desc in controls:
        cv2.putText(canvas, f"[{key_label}] {desc}", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["white"], 1)
        y += 18

    # Last status message
    if last_status:
        cv2.putText(canvas, last_status, (x0, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C["accent"], 1)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print(__doc__)

    cam = cv2.VideoCapture(CAMERA_INDEX)
    if not cam.isOpened():
        print("[ERROR] Cannot open camera.", file=sys.stderr)
        sys.exit(1)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_FPS, 30)

    detector = GazeDetector()

    # Load existing config so we start from last known thresholds
    current_cfg = cfg_module.load()
    detector.UP_THRESH   = current_cfg["up_thresh"]
    detector.DOWN_THRESH = current_cfg["down_thresh"]

    samples: dict[str, list[tuple[float, float]]] = {
        "up": [], "center": [], "down": []
    }

    collecting_label: str | None = None
    collect_frames: list[tuple[float, float]] = []
    last_status = "Press an arrow key to start labelling."

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 900, 480)

    print("[trainer] Window open. Follow on-screen instructions.")

    while True:
        ret, frame = cam.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        direction, debug_frame, gaze = detector.process(frame)

        # ── Collect samples ───────────────────────────────────
        if collecting_label is not None:
            if gaze is not None:
                collect_frames.append(gaze)
            progress = len(collect_frames) / SAMPLES_PER_LABEL

            if len(collect_frames) >= SAMPLES_PER_LABEL:
                avg = (
                    float(np.mean([g[0] for g in collect_frames])),
                    float(np.mean([g[1] for g in collect_frames])),
                )
                samples[collecting_label].append(avg)
                n = len(samples[collecting_label])
                last_status = (
                    f"Saved sample #{n} for '{collecting_label}'  "
                    f"ry={avg[1]:.3f}  (need ≥15 per direction)"
                )
                print(f"[trainer] {last_status}")
                collecting_label = None
                collect_frames   = []
                progress         = 0.0
        else:
            progress = 0.0

        # ── Draw UI ───────────────────────────────────────────
        # Resize debug to 640 wide, paste sidebar
        canvas = cv2.resize(debug_frame, (640, 480))
        draw_ui(canvas, samples, collecting_label, progress, last_status)
        cv2.imshow(WIN_NAME, canvas)

        # ── Key handling ─────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q') or key == 27:
            break

        elif key == KEY_UP:
            if collecting_label is None:
                collecting_label = "up"
                collect_frames   = []
                last_status = "Look UP and hold still…"

        elif key == KEY_DOWN:
            if collecting_label is None:
                collecting_label = "down"
                collect_frames   = []
                last_status = "Look DOWN and hold still…"

        elif key in (KEY_LEFT, KEY_RIGHT):
            if collecting_label is None:
                collecting_label = "center"
                collect_frames   = []
                last_status = "Look STRAIGHT AHEAD (center) and hold still…"

        elif key in (ord('r'), ord('R')):
            samples = {"up": [], "center": [], "down": []}
            collecting_label = None
            collect_frames   = []
            last_status = "Samples reset."
            print("[trainer] Samples reset.")

        elif key in (ord('s'), ord('S')):
            result = compute_thresholds(samples)
            if result is None:
                last_status = "Need ≥1 sample each for UP and DOWN first."
                print(f"[trainer] {last_status}")
            else:
                up_t, down_t = result
                current_cfg["up_thresh"]   = up_t
                current_cfg["down_thresh"] = down_t
                cfg_module.save(current_cfg)
                cfg_module.show(current_cfg)
                last_status = f"Saved!  UP<{up_t:.3f}  DOWN>{down_t:.3f}"
                print(f"[trainer] {last_status}")
                # Update live detector so you can test immediately
                detector.UP_THRESH   = up_t
                detector.DOWN_THRESH = down_t

    cam.release()
    cv2.destroyAllWindows()
    print("[trainer] Done.")


if __name__ == "__main__":
    main()
