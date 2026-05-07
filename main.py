"""
main.py
───────
Entry point for the eye-tracker.

Run with:
    python main.py [--camera N] [--no-preview] [--calibrate]

Controls (in any OpenCV window)
────────────────────────────────
  Q       → quit
  Space   → pause / resume
  C       → run calibration
  ESC     → emergency stop (same as Q)
  +/-     → increase / decrease scroll sensitivity
"""

import cv2
import argparse
import time
import sys

from gaze_detector   import GazeDetector, GazeDirection
from screen_controller import ScreenController
from calibration     import Calibrator
from overlay         import StatusOverlay


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Gaze-based scroll & click controller – built from scratch."
    )
    p.add_argument("--camera",      type=int, default=0,
                   help="Camera index (default 0)")
    p.add_argument("--no-preview",  action="store_true",
                   help="Hide the annotated camera preview window")
    p.add_argument("--calibrate",   action="store_true",
                   help="Run calibration on startup")
    p.add_argument("--rekognition", action="store_true",
                   help="Use Amazon Rekognition for high-accuracy face detection")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────

PREVIEW_WIN = "Eye Tracker – Camera Preview"


def main():
    args = parse_args()

    # ── Camera ────────────────────────────────────────────────
    print(f"[main] Opening camera {args.camera} …")
    cam = cv2.VideoCapture(args.camera)
    if not cam.isOpened():
        print("[ERROR] Could not open camera. Check --camera index.", file=sys.stderr)
        sys.exit(1)

    # Prefer a decent resolution
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_FPS, 30)

    print("[main] Camera ready.")

    # ── Components ────────────────────────────────────────────
    detector    = GazeDetector(use_rekognition=args.rekognition)
    controller  = ScreenController()
    overlay     = StatusOverlay()

    # ── Optional camera preview window ────────────────────────
    if not args.no_preview:
        cv2.namedWindow(PREVIEW_WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(PREVIEW_WIN, 480, 360)

    # ── Calibration ───────────────────────────────────────────
    if args.calibrate:
        print("[main] Running calibration …")
        cal = Calibrator(detector)
        cal.run(cam)
        print("[main] Calibration complete.")

    # ── State ─────────────────────────────────────────────────
    paused         = False
    running        = True
    frame_count    = 0

    print("[main] Tracking started. Press Q or ESC in any window to quit.")
    print("       Space = pause/resume   C = calibrate   +/- = scroll speed")

    while running:
        ret, frame = cam.read()
        if not ret:
            print("[WARN] Failed to read frame – retrying …")
            time.sleep(0.05)
            continue

        # Mirror horizontally (more intuitive for the user)
        frame = cv2.flip(frame, 1)
        frame_count += 1

        # ── Process ───────────────────────────────────────────
        direction = GazeDirection.UNKNOWN
        gaze_ratios = None
        dwell_progress = 0.0

        if not paused:
            direction, debug_frame, gaze_ratios = detector.process(frame)
            dwell_progress = controller.update(direction, gaze_ratios)
        else:
            debug_frame = frame.copy()
            cv2.putText(debug_frame, "PAUSED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)

        # ── Render windows ────────────────────────────────────
        if not args.no_preview:
            cv2.imshow(PREVIEW_WIN, debug_frame)

        overlay.draw(direction, dwell_progress,
                     controller.status_message, paused)

        # ── Key handling ─────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q'), 27):        # Q / ESC
            running = False

        elif key == ord(' '):                       # Space
            paused = not paused
            print(f"[main] {'Paused' if paused else 'Resumed'}")

        elif key in (ord('c'), ord('C')):           # C – calibrate
            print("[main] Starting calibration …")
            cal = Calibrator(detector)
            cal.run(cam)
            print("[main] Calibration done.")

        elif key == ord('+'):
            from screen_controller import SCROLL_AMOUNT
            import screen_controller as sc
            sc.SCROLL_AMOUNT = min(sc.SCROLL_AMOUNT + 1, 20)
            print(f"[main] Scroll amount → {sc.SCROLL_AMOUNT}")

        elif key == ord('-'):
            import screen_controller as sc
            sc.SCROLL_AMOUNT = max(sc.SCROLL_AMOUNT - 1, 1)
            print(f"[main] Scroll amount → {sc.SCROLL_AMOUNT}")

    # ── Cleanup ───────────────────────────────────────────────
    cam.release()
    overlay.close()
    cv2.destroyAllWindows()
    print("[main] Exited cleanly.")


if __name__ == "__main__":
    main()
