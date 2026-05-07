"""
gaze_detector.py
────────────────
Pure-OpenCV gaze estimation built from scratch.

Pipeline
─────────
1. Haar-cascade face detection          → face ROI
2. Haar-cascade eye detection inside face → eye ROIs
3. Adaptive threshold + contour analysis → iris / pupil centre
4. Normalised iris position inside eye box → gaze ratio
5. Kalman-style exponential smoothing    → stable direction
"""

import cv2
import numpy as np
import os
import config as _cfg


# ─────────────────────────────────────────────────────────────
# Paths to the built-in OpenCV Haar cascade XML files
# ─────────────────────────────────────────────────────────────
def _cv2_data_path(filename: str) -> str:
    """Locate OpenCV's bundled Haar cascade XML files."""
    cv2_data = cv2.data.haarcascades          # works for pip-installed opencv
    return os.path.join(cv2_data, filename)


FACE_CASCADE_PATH = _cv2_data_path("haarcascade_frontalface_default.xml")
EYE_CASCADE_PATH  = _cv2_data_path("haarcascade_eye_tree_eyeglasses.xml")

# Fallback cascade (no eyeglasses variant)
EYE_CASCADE_FALLBACK = _cv2_data_path("haarcascade_eye.xml")


# ─────────────────────────────────────────────────────────────
# GazeDirection enum-like constants
# ─────────────────────────────────────────────────────────────
class GazeDirection:
    CENTER   = "center"
    UP       = "up"
    DOWN     = "down"
    LEFT     = "left"
    RIGHT    = "right"
    UNKNOWN  = "unknown"


# ─────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────

def _find_iris_center(eye_gray: np.ndarray):
    """
    Locate the iris/pupil centre inside a grey-scale eye crop.

    Strategy (no external library):
    ─────────────────────────────────
    • Brighten contrast with CLAHE.
    • Blur to suppress noise.
    • Use HoughCircles to find iris-like circles.
    • Fall back to the centroid of the darkest contour region.

    Returns (cx_normalised, cy_normalised) in [0, 1] × [0, 1],
    or None if detection fails.
    """
    h, w = eye_gray.shape

    if h < 10 or w < 10:
        return None

    # ── Enhance contrast ──────────────────────────────────────
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(eye_gray)

    # ── Blur ──────────────────────────────────────────────────
    blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)

    # ── HoughCircles (iris detection) ─────────────────────────
    min_r = int(min(h, w) * 0.15)
    max_r = int(min(h, w) * 0.45)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=w // 2,
        param1=50,
        param2=20,
        minRadius=max(min_r, 5),
        maxRadius=max(max_r, 10),
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Pick the circle whose centre is closest to the middle of the eye
        best = None
        best_dist = float("inf")
        mid_x, mid_y = w / 2, h / 2
        for c in circles[0]:
            cx, cy, _ = c
            dist = ((cx - mid_x) ** 2 + (cy - mid_y) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best = (cx, cy)
        if best:
            return best[0] / w, best[1] / h

    # ── Fallback: darkest contour centroid ────────────────────
    _, thresh = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV)

    # Remove eyelid artefacts: focus on the centre vertical strip
    mask = np.zeros_like(thresh)
    margin_x = int(w * 0.15)
    margin_y = int(h * 0.15)
    mask[margin_y : h - margin_y, margin_x : w - margin_x] = 255
    thresh = cv2.bitwise_and(thresh, mask)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Largest contour → pupil
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 30:
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return cx / w, cy / h


def _smooth(prev, curr, alpha=0.35):
    """Exponential moving average smoothing."""
    if prev is None:
        return curr
    return alpha * curr + (1 - alpha) * prev


# ─────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────

class GazeDetector:
    """
    Detect gaze direction from a BGR webcam frame.

    Usage
    ──────
    detector = GazeDetector()
    direction, debug_frame, gaze_point = detector.process(frame)
    """

    # Minimum confidence: how many eyes must agree
    MIN_EYES = 1

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        self.eye_cascade  = cv2.CascadeClassifier(EYE_CASCADE_PATH)

        # Fallback if the eyeglasses variant is missing
        if self.eye_cascade.empty():
            self.eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_FALLBACK)

        if self.face_cascade.empty():
            raise RuntimeError(
                "Could not load face Haar cascade. "
                "Check your opencv-python installation."
            )
        if self.eye_cascade.empty():
            raise RuntimeError(
                "Could not load eye Haar cascade. "
                "Check your opencv-python installation."
            )

        # Load thresholds from saved config (falls back to defaults)
        self.reload_config()

        # Smoothed gaze ratios (exponential moving average)
        self._smooth_x: float | None = None
        self._smooth_y: float | None = None

        # Smoothed screen gaze point (pixels)
        self._smooth_gx: float | None = None
        self._smooth_gy: float | None = None

        # For drawing / diagnostics
        self._last_face  = None
        self._last_eyes  = []

    def reload_config(self):
        """Re-read thresholds from eye_tracker_config.json."""
        cfg = _cfg.load()
        self.UP_THRESH   = cfg["up_thresh"]
        self.DOWN_THRESH = cfg["down_thresh"]
        self.LEFT_THRESH  = cfg.get("left_thresh",  0.38)
        self.RIGHT_THRESH = cfg.get("right_thresh", 0.62)
        print(f"[detector] Thresholds loaded: UP<{self.UP_THRESH:.3f}  "
              f"DOWN>{self.DOWN_THRESH:.3f}")

    # ── Public API ────────────────────────────────────────────

    def process(self, frame: np.ndarray):
        """
        Analyse one BGR frame.

        Returns
        ───────
        direction    : GazeDirection constant
        debug_frame  : annotated BGR frame for preview window
        gaze_ratios  : (rx, ry) normalised in [0,1] or None
        """
        debug = frame.copy()
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)

        fh, fw = frame.shape[:2]

        # ── Detect face ───────────────────────────────────────
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        if len(faces) == 0:
            cv2.putText(debug, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return GazeDirection.UNKNOWN, debug, None

        # Take the largest face
        face = max(faces, key=lambda r: r[2] * r[3])
        fx, fy, fw_f, fh_f = face
        self._last_face = face

        cv2.rectangle(debug, (fx, fy), (fx + fw_f, fy + fh_f), (0, 200, 0), 2)

        face_gray = gray[fy : fy + fh_f, fx : fx + fw_f]

        # ── Detect eyes within the TOP 60 % of face ──────────
        # Restrict to upper portion to avoid nose/mouth detections
        eye_roi_gray = face_gray[: int(fh_f * 0.60), :]

        eyes = self.eye_cascade.detectMultiScale(
            eye_roi_gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(25, 15),
        )

        if len(eyes) == 0:
            cv2.putText(debug, "No eyes detected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
            return GazeDirection.UNKNOWN, debug, None

        # Sort eyes left-to-right; keep at most 2
        eyes = sorted(eyes, key=lambda e: e[0])[:2]
        self._last_eyes = eyes

        iris_ratios_x = []
        iris_ratios_y = []

        for ex, ey, ew, eh in eyes:
            # Absolute coordinates in full frame
            abs_ex = fx + ex
            abs_ey = fy + ey

            cv2.rectangle(debug,
                          (abs_ex, abs_ey),
                          (abs_ex + ew, abs_ey + eh),
                          (255, 180, 0), 2)

            eye_gray_crop = face_gray[ey : ey + eh, ex : ex + ew]
            result = _find_iris_center(eye_gray_crop)

            if result is None:
                continue

            rx, ry = result
            iris_ratios_x.append(rx)
            iris_ratios_y.append(ry)

            # Draw iris centre on debug frame
            iris_px = int(abs_ex + rx * ew)
            iris_py = int(abs_ey + ry * eh)
            cv2.circle(debug, (iris_px, iris_py), 4, (0, 255, 255), -1)

        if len(iris_ratios_x) < self.MIN_EYES:
            cv2.putText(debug, "Iris not found", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
            return GazeDirection.UNKNOWN, debug, None

        avg_x = float(np.mean(iris_ratios_x))
        avg_y = float(np.mean(iris_ratios_y))

        # Smooth
        self._smooth_x = _smooth(self._smooth_x, avg_x)
        self._smooth_y = _smooth(self._smooth_y, avg_y)

        sx = self._smooth_x
        sy = self._smooth_y

        # ── Classify direction ────────────────────────────────
        if sy < self.UP_THRESH:
            v_dir = GazeDirection.UP
        elif sy > self.DOWN_THRESH:
            v_dir = GazeDirection.DOWN
        else:
            v_dir = GazeDirection.CENTER

        if sx < self.LEFT_THRESH:
            h_dir = GazeDirection.LEFT
        elif sx > self.RIGHT_THRESH:
            h_dir = GazeDirection.RIGHT
        else:
            h_dir = GazeDirection.CENTER

        # Primary direction: vertical overrides horizontal for scrolling
        if v_dir != GazeDirection.CENTER:
            direction = v_dir
        elif h_dir != GazeDirection.CENTER:
            direction = h_dir
        else:
            direction = GazeDirection.CENTER

        # ── Annotate debug frame ──────────────────────────────
        color_map = {
            GazeDirection.UP:     (0, 255, 0),
            GazeDirection.DOWN:   (0, 0, 255),
            GazeDirection.LEFT:   (255, 165, 0),
            GazeDirection.RIGHT:  (255, 165, 0),
            GazeDirection.CENTER: (200, 200, 200),
            GazeDirection.UNKNOWN:(0, 0, 255),
        }
        col = color_map.get(direction, (255, 255, 255))
        cv2.putText(debug, f"Gaze: {direction.upper()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        cv2.putText(debug, f"rx={sx:.2f}  ry={sy:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

        # Draw gaze ratio bar (vertical)
        bar_x, bar_y, bar_h = frame.shape[1] - 25, 80, 120
        cv2.rectangle(debug, (bar_x, bar_y), (bar_x + 12, bar_y + bar_h),
                      (80, 80, 80), -1)
        dot_y = int(bar_y + sy * bar_h)
        cv2.circle(debug, (bar_x + 6, dot_y), 6, col, -1)

        return direction, debug, (sx, sy)
