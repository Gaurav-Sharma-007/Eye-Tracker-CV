"""
calibration.py
──────────────
Optional one-time calibration that lets the user look at
reference points (TL, TR, BL, BR, Centre) to learn their
personal gaze-ratio range and improve accuracy.

Usage
──────
cal = Calibrator(detector)
cal.run(cam)          # blocking; opens a fullscreen window
offsets = cal.offsets # (mean_x_offset, mean_y_offset) in ratio units
"""

import cv2
import numpy as np
import time


CALIBRATION_POINTS = [
    (0.1,  0.1,  "Top-Left"),
    (0.9,  0.1,  "Top-Right"),
    (0.5,  0.5,  "Centre"),
    (0.1,  0.9,  "Bottom-Left"),
    (0.9,  0.9,  "Bottom-Right"),
]

COLLECT_DURATION = 2.5   # seconds of data collected per point
COUNTDOWN        = 1.5   # seconds of "look here" before collecting


class Calibrator:
    def __init__(self, detector):
        self.detector  = detector
        self.offsets   = (0.0, 0.0)   # (rx_offset, ry_offset)
        self._done     = False

    @property
    def is_done(self):
        return self._done

    def run(self, cam: cv2.VideoCapture):
        """
        Display calibration dots fullscreen and collect gaze samples.
        Modifies detector thresholds based on collected data.
        """
        sw = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        sh = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # We'll draw on a separate calibration canvas in a named window
        win = "Calibration – Eye Tracker"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        screen_w = 1920
        screen_h = 1080

        collected = {}

        for target_rx, target_ry, label in CALIBRATION_POINTS:
            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            dot_x  = int(target_rx * screen_w)
            dot_y  = int(target_ry * screen_h)

            # Countdown phase
            start = time.time()
            while time.time() - start < COUNTDOWN:
                ret, frame = cam.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                _, _, _ = self.detector.process(frame)

                canvas[:] = (15, 15, 25)
                remaining = COUNTDOWN - (time.time() - start)
                self._draw_dot(canvas, dot_x, dot_y, (0, 200, 255), 24)
                cv2.putText(canvas, f"Look at the dot: {label}",
                            (screen_w // 2 - 200, screen_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 200, 200), 2)
                cv2.putText(canvas, f"Starting in {remaining:.1f}s",
                            (screen_w // 2 - 150, screen_h // 2 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 1)
                cv2.imshow(win, canvas)
                if cv2.waitKey(1) & 0xFF == 27:   # ESC to abort
                    cv2.destroyWindow(win)
                    return

            # Collection phase
            ratios_x, ratios_y = [], []
            start = time.time()
            while time.time() - start < COLLECT_DURATION:
                ret, frame = cam.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                _, _, gaze = self.detector.process(frame)

                canvas[:] = (15, 15, 25)
                progress = (time.time() - start) / COLLECT_DURATION
                self._draw_dot(canvas, dot_x, dot_y, (0, 255, 100), 24)
                self._draw_progress_ring(canvas, dot_x, dot_y, progress)

                cv2.putText(canvas, "Keep looking…",
                            (screen_w // 2 - 120, screen_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)
                cv2.imshow(win, canvas)
                cv2.waitKey(1)

                if gaze is not None:
                    ratios_x.append(gaze[0])
                    ratios_y.append(gaze[1])

            if ratios_x:
                collected[(target_rx, target_ry)] = (
                    float(np.median(ratios_x)),
                    float(np.median(ratios_y)),
                )

        cv2.destroyWindow(win)

        if not collected:
            return

        # ── Derive per-user thresholds ──────────────────────────
        # Map screen target ratios → measured gaze ratios
        # Use top/bottom measured ratios to set UP/DOWN thresholds
        top_ys    = [v[1] for (_, ty), v in collected.items() if ty < 0.3]
        bottom_ys = [v[1] for (_, ty), v in collected.items() if ty > 0.7]
        mid_ys    = [v[1] for (_, ty), v in collected.items() if 0.3 <= ty <= 0.7]

        if top_ys and bottom_ys:
            top_measured    = float(np.mean(top_ys))
            bottom_measured = float(np.mean(bottom_ys))
            mid_measured    = float(np.mean(mid_ys)) if mid_ys else \
                              (top_measured + bottom_measured) / 2

            # Set thresholds at midpoints
            self.detector.UP_THRESH   = (top_measured + mid_measured) / 2
            self.detector.DOWN_THRESH = (mid_measured + bottom_measured) / 2

        self._done = True
        print("[Calibration] Done. "
              f"UP<{self.detector.UP_THRESH:.2f}, "
              f"DOWN>{self.detector.DOWN_THRESH:.2f}")

    # ── Drawing helpers ───────────────────────────────────────

    @staticmethod
    def _draw_dot(canvas, x, y, color, radius):
        cv2.circle(canvas, (x, y), radius + 6, (255, 255, 255), 2)
        cv2.circle(canvas, (x, y), radius, color, -1)
        cv2.circle(canvas, (x, y), 6, (0, 0, 0), -1)

    @staticmethod
    def _draw_progress_ring(canvas, x, y, progress):
        axes   = (36, 36)
        angle  = int(360 * progress)
        cv2.ellipse(canvas, (x, y), axes, -90, 0, angle, (0, 220, 120), 3)
