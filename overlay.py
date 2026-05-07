"""
overlay.py
──────────
Renders a translucent, always-on-top status overlay using OpenCV
in a small named window.  Shows:
  • Current gaze direction (with arrow icon)
  • Dwell-click progress ring
  • Scrolling status
  • FPS counter
  • Keyboard shortcuts

The window is kept small (~320×180) and positioned in the corner.
"""

import cv2
import numpy as np
import time


WINDOW_NAME = "Eye Tracker – Status"
WIN_W, WIN_H = 320, 200


PALETTE = {
    "bg"      : (18, 18, 30),
    "border"  : (60, 60, 90),
    "up"      : (80, 220, 80),
    "down"    : (80, 100, 220),
    "center"  : (200, 200, 200),
    "unknown" : (80, 80, 80),
    "accent"  : (0, 200, 255),
    "white"   : (240, 240, 240),
    "green"   : (50, 220, 100),
    "warn"    : (30, 165, 255),
}

ARROW = {
    "up"     : "↑",
    "down"   : "↓",
    "left"   : "←",
    "right"  : "→",
    "center" : "●",
    "unknown": "?",
}


class StatusOverlay:
    def __init__(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WIN_W, WIN_H)
        # Position in top-right corner (rough estimate)
        cv2.moveWindow(WINDOW_NAME, 1580, 20)

        self._fps_times  = []
        self._last_draw  = 0.0

    def draw(self, direction: str, dwell_progress: float,
             status_msg: str, paused: bool):
        """Render one frame of the overlay."""
        now = time.time()

        # FPS counter
        self._fps_times.append(now)
        self._fps_times = [t for t in self._fps_times if now - t < 1.0]
        fps = len(self._fps_times)

        canvas = np.full((WIN_H, WIN_W, 3), PALETTE["bg"], dtype=np.uint8)

        # Border
        cv2.rectangle(canvas, (1, 1), (WIN_W - 2, WIN_H - 2),
                      PALETTE["border"], 1)

        # Title bar
        cv2.rectangle(canvas, (1, 1), (WIN_W - 2, 28), (30, 30, 50), -1)
        cv2.putText(canvas, "Eye Tracker", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, PALETTE["accent"], 1)
        cv2.putText(canvas, f"{fps} fps", (WIN_W - 65, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, PALETTE["white"], 1)

        if paused:
            cv2.putText(canvas, "PAUSED  (Space to resume)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        PALETTE["warn"], 1)
        else:
            # Gaze direction
            col = PALETTE.get(direction, PALETTE["unknown"])
            label = f"Gaze: {direction.upper()}"
            cv2.putText(canvas, label, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

            # Dwell progress ring
            if dwell_progress > 0.0:
                cx, cy, r = WIN_W - 48, 80, 28
                # Background ring
                cv2.circle(canvas, (cx, cy), r, (50, 50, 70), 3)
                # Progress arc
                angle = int(360 * dwell_progress)
                cv2.ellipse(canvas, (cx, cy), (r, r), -90, 0, angle,
                            PALETTE["green"], 3)
                pct = int(dwell_progress * 100)
                cv2.putText(canvas, f"{pct}%", (cx - 16, cy + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            PALETTE["white"], 1)
                cv2.putText(canvas, "Dwell", (cx - 22, cy + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            PALETTE["white"], 1)

            # Status message
            if status_msg:
                cv2.putText(canvas, status_msg, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            PALETTE["white"], 1)

        # Shortcuts legend
        cv2.line(canvas, (6, WIN_H - 45), (WIN_W - 6, WIN_H - 45),
                 PALETTE["border"], 1)
        cv2.putText(canvas, "C=calibrate  Space=pause  Q=quit",
                    (6, WIN_H - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (120, 120, 130), 1)
        cv2.putText(canvas, "ESC=emergency stop",
                    (6, WIN_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (120, 120, 130), 1)

        cv2.imshow(WINDOW_NAME, canvas)

    def close(self):
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except Exception:
            pass
