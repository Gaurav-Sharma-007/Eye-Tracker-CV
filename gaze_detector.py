"""
gaze_detector.py
────────────────
Gaze direction detector.

Two modes (auto-selected at startup):

Mode A – CNN  (when models/gaze_model.keras exists)
    • Face crop via Amazon Rekognition OR OpenCV Haar cascade
    • 224×224 resize → MobileNetV2 → softmax over 5 classes
    • Prediction is smoothed with a temporal buffer

Mode B – Classic CV  (fallback, no trained model)
    • Haar cascades → iris localisation (HoughCircles + contour fallback)
    • Normalised iris position → threshold-based direction

Classes: up | down | straight | left | right
The "straight" class maps to GazeDirection.CENTER.
"""

import cv2
import numpy as np
import os
import json
import collections
import time
from concurrent.futures import ThreadPoolExecutor

# ─────────────────────────────────────────────────────────────
# Optional imports (graceful degradation)
# ─────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

import config as _cfg


# ─────────────────────────────────────────────────────────────
# Haar cascade paths
# ─────────────────────────────────────────────────────────────
def _cv2_data_path(filename: str) -> str:
    return os.path.join(cv2.data.haarcascades, filename)

FACE_CASCADE_PATH    = _cv2_data_path("haarcascade_frontalface_default.xml")
EYE_CASCADE_PATH     = _cv2_data_path("haarcascade_eye_tree_eyeglasses.xml")
EYE_CASCADE_FALLBACK = _cv2_data_path("haarcascade_eye.xml")

MODEL_DIR   = os.path.join(os.path.dirname(__file__), "models")
IDX_PATH    = os.path.join(MODEL_DIR, "class_indices.json")

# Prefer best checkpoint; fall back to final saved model
_CANDIDATE_MODELS = [
    os.path.join(MODEL_DIR, "best_gaze_model.keras"),
    os.path.join(MODEL_DIR, "gaze_model.keras"),
]
MODEL_PATH = next((p for p in _CANDIDATE_MODELS if os.path.exists(p)), _CANDIDATE_MODELS[-1])

IMG_H, IMG_W = 224, 224


# ─────────────────────────────────────────────────────────────
# GazeDirection constants
# ─────────────────────────────────────────────────────────────
class GazeDirection:
    CENTER   = "center"
    UP       = "up"
    DOWN     = "down"
    LEFT     = "left"
    RIGHT    = "right"
    UNKNOWN  = "unknown"


# ─────────────────────────────────────────────────────────────
# Classic CV helpers (Mode B)
# ─────────────────────────────────────────────────────────────

def _find_iris_center(eye_gray: np.ndarray):
    h, w = eye_gray.shape
    if h < 10 or w < 10:
        return None

    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(eye_gray)
    blurred  = cv2.GaussianBlur(enhanced, (7, 7), 0)

    min_r = max(int(min(h, w) * 0.15), 5)
    max_r = max(int(min(h, w) * 0.45), 10)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=w // 2, param1=50, param2=20,
        minRadius=min_r, maxRadius=max_r,
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        mid_x, mid_y = w / 2, h / 2
        best, best_dist = None, float("inf")
        for c in circles[0]:
            cx, cy, _ = c
            d = ((cx - mid_x) ** 2 + (cy - mid_y) ** 2) ** 0.5
            if d < best_dist:
                best_dist, best = d, (cx, cy)
        if best:
            return best[0] / w, best[1] / h

    _, thresh = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV)
    mask = np.zeros_like(thresh)
    mx, my = int(w * 0.15), int(h * 0.15)
    mask[my:h-my, mx:w-mx] = 255
    thresh = cv2.bitwise_and(thresh, mask)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 30:
        return None
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    return M["m10"] / M["m00"] / w, M["m01"] / M["m00"] / h


def _smooth(prev, curr, alpha=0.35):
    if prev is None:
        return curr
    return alpha * curr + (1 - alpha) * prev


# ─────────────────────────────────────────────────────────────
# GazeDetector
# ─────────────────────────────────────────────────────────────

class GazeDetector:
    """
    Detect gaze direction from a live BGR frame.

    Automatically uses the CNN model (Mode A) if available,
    falling back to the classic CV pipeline (Mode B).
    """

    # Mode B thresholds (overridden by config on init)
    UP_THRESH    = 0.38
    DOWN_THRESH  = 0.62
    LEFT_THRESH  = 0.38
    RIGHT_THRESH = 0.62
    MIN_EYES     = 1

    # CNN confidence smoothing buffer length
    CNN_SMOOTH   = 5
    DEFAULT_REKO_INTERVAL_S = 0.75
    REKO_CACHE_TTL_S = 2.5

    def __init__(self, use_rekognition: bool = False,
                 rekognition_region: str | None = None,
                 rekognition_interval: float | None = None):
        cfg = _cfg.load()
        self.UP_THRESH    = cfg["up_thresh"]
        self.DOWN_THRESH  = cfg["down_thresh"]
        self.LEFT_THRESH  = cfg.get("left_thresh",  0.38)
        self.RIGHT_THRESH = cfg.get("right_thresh", 0.62)
        self._reko_interval_s = max(
            0.1,
            rekognition_interval
            if rekognition_interval is not None
            else self.DEFAULT_REKO_INTERVAL_S,
        )

        # ── Haar cascades (used in Mode B AND as fallback face detector) ──
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        self.eye_cascade  = cv2.CascadeClassifier(EYE_CASCADE_PATH)
        if self.eye_cascade.empty():
            self.eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_FALLBACK)

        # ── Try to load CNN model (Mode A) ────────────────────────────────
        self._cnn_model    = None
        self._idx_to_class = None
        self._mode         = "B"

        if _TF_AVAILABLE and os.path.exists(MODEL_PATH):
            try:
                self._cnn_model = tf.keras.models.load_model(MODEL_PATH)
                with open(IDX_PATH) as f:
                    data = json.load(f)
                self._idx_to_class = data["idx_to_class"]
                self._mode = "A"
                model_name = os.path.basename(MODEL_PATH)
                print(f"[detector] Mode A – CNN model loaded: {model_name}")
                print(f"[detector] Classes: {list(self._idx_to_class.values())}")
                print(f"[detector] Input shape: {self._cnn_model.input_shape}")
            except Exception as e:
                print(f"[detector] Could not load CNN model ({e}) – falling back to Mode B")

        if self._mode == "B":
            print("[detector] Mode B – classic CV (Haar + iris detection)")

        # ── Optional Rekognition for face crop ────────────────────────────
        self._reko = None
        self._reko_executor = None
        self._reko_future = None
        self._reko_last_request = 0.0
        self._reko_last_result = None
        if use_rekognition and self._mode == "A":
            try:
                from rekognition_face import RekognitionFaceDetector
                self._reko = RekognitionFaceDetector(region=rekognition_region)
                if not self._reko.available:
                    self._reko = None
                else:
                    self._reko_executor = ThreadPoolExecutor(max_workers=1)
                    print(
                        "[detector] Rekognition async mode "
                        f"({self._reko_interval_s:.2f}s throttle)"
                    )
            except ImportError:
                pass

        # ── Smoothing state ───────────────────────────────────────────────
        self._smooth_x: float | None = None
        self._smooth_y: float | None = None

        # CNN: circular buffer of predicted class indices for majority vote
        self._pred_buffer: collections.deque = collections.deque(
            maxlen=self.CNN_SMOOTH
        )

    # ── Public API ────────────────────────────────────────────────────────

    def reload_config(self):
        cfg = _cfg.load()
        self.UP_THRESH    = cfg["up_thresh"]
        self.DOWN_THRESH  = cfg["down_thresh"]
        self.LEFT_THRESH  = cfg.get("left_thresh",  0.38)
        self.RIGHT_THRESH = cfg.get("right_thresh", 0.62)
        print(f"[detector] Thresholds reloaded: UP<{self.UP_THRESH:.3f} DOWN>{self.DOWN_THRESH:.3f}")

    def process(self, frame: np.ndarray):
        """
        Analyse one BGR frame.

        Returns
        ────────
        direction    : GazeDirection constant string
        debug_frame  : annotated BGR frame
        gaze_ratios  : (rx, ry) in [0,1] or None
        """
        if self._mode == "A":
            return self._process_cnn(frame)
        else:
            return self._process_classic(frame)

    def close(self):
        if self._reko_executor is not None:
            self._reko_executor.shutdown(wait=False, cancel_futures=True)
            self._reko_executor = None

    def _detect_face_haar(self, frame: np.ndarray, debug: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )
        if len(faces) == 0:
            return None

        face_box = max(faces, key=lambda r: r[2] * r[3])
        fx, fy, fw, fh = face_box
        cv2.rectangle(debug, (fx, fy), (fx + fw, fy + fh), (0, 200, 0), 2)
        return face_box

    def _poll_rekognition(self, frame: np.ndarray):
        now = time.time()

        if self._reko_future is not None and self._reko_future.done():
            try:
                face_box, landmarks = self._reko_future.result()
                if face_box is not None:
                    self._reko_last_result = (face_box, landmarks, now)
            except Exception as e:
                print(f"[rekognition] Async detect failed: {e}")
            finally:
                self._reko_future = None

        if (
            self._reko_executor is not None
            and self._reko_future is None
            and now - self._reko_last_request >= self._reko_interval_s
        ):
            self._reko_last_request = now
            self._reko_future = self._reko_executor.submit(
                self._reko.detect, frame.copy()
            )

        if self._reko_last_result is None:
            return None, {}

        face_box, landmarks, detected_at = self._reko_last_result
        if now - detected_at > self.REKO_CACHE_TTL_S:
            return None, {}
        return face_box, landmarks

    # ── Mode A – CNN ──────────────────────────────────────────────────────

    def _process_cnn(self, frame: np.ndarray):
        debug = frame.copy()
        h, w  = frame.shape[:2]

        # ── 1. Detect Face ───────────────────────────────────────────
        face_box = None
        face_crop = None
        landmarks = {}

        if self._reko is not None:
            face_box, landmarks = self._poll_rekognition(frame)
            if face_box:
                fx, fy, fw, fh = face_box
                cv2.rectangle(debug, (fx, fy), (fx+fw, fy+fh), (0, 200, 255), 2)
                cv2.putText(debug, "Rekognition async", (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
            else:
                face_box = self._detect_face_haar(frame, debug)
        else:
            face_box = self._detect_face_haar(frame, debug)

        if face_box is None:
            cv2.putText(debug, "No face", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return GazeDirection.UNKNOWN, debug, None

        # ── 2. Extract Eye Crops ─────────────────────────────────────
        # The model was trained on tight eye crops (approx 224x224 after resize).
        # We try to extract both eyes, flip the right eye to match the left-eye template.
        eye_crops = []
        fx, fy, fw, fh = face_box
        
        if landmarks.get("left_eye") and landmarks.get("right_eye"):
            # Use high-accuracy landmarks from Rekognition
            for side in ["right_eye", "left_eye"]:
                ex, ey = landmarks[side]
                # Define a crop around the eye (approx 0.5x face width)
                # Training data includes the eyebrow, so we shift the crop slightly up.
                ew = int(fw * 0.45)
                eh = ew
                ex1, ey1 = max(0, ex - ew//2), max(0, ey - int(eh * 0.6))
                ex2, ey2 = min(w, ex1 + ew), min(h, ey1 + eh)
                crop = frame[ey1:ey2, ex1:ex2]
                if crop.size > 0:
                    # Flip the person's right eye (left side of image) to look like a left eye
                    if side == "right_eye":
                        crop = cv2.flip(crop, 1)
                    eye_crops.append(crop)
                    cv2.rectangle(debug, (ex1, ey1), (ex2, ey2), (255, 255, 0), 1)
        else:
            # Fall back to Haar eye detection
            face_gray = cv2.cvtColor(frame[fy:fy+fh, fx:fx+fw], cv2.COLOR_BGR2GRAY)
            eye_roi_h = int(fh * 0.5)
            eyes = self.eye_cascade.detectMultiScale(
                face_gray[:eye_roi_h, :], scaleFactor=1.1, minNeighbors=5
            )
            
            # Filter and sort eyes by X position
            eyes = sorted([e for e in eyes if e[2] > fw*0.15], key=lambda e: e[0])
            for i, (ex, ey, ew_e, eh_e) in enumerate(eyes[:2]):
                # Expand the eye box to match training context (include eyebrow)
                span = int(ew_e * 1.8)
                cx, cy = fx + ex + ew_e//2, fy + ey + eh_e//2
                x1, y1 = max(0, cx - span//2), max(0, cy - int(span * 0.55))
                x2, y2 = min(w, x1 + span), min(h, y1 + span)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    # Decide if we should flip (to match left-eye training template)
                    should_flip = False
                    if len(eyes) > 1:
                        if i == 0: should_flip = True
                    else:
                        # If only one eye, check if it's on the left side of the face
                        if ex < fw * 0.4: should_flip = True
                        
                    if should_flip:
                        crop = cv2.flip(crop, 1)
                    eye_crops.append(crop)
                    cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 255, 0), 1)

        # Fallback if no eyes found: just take the top half of face as a single crop
        if not eye_crops:
            span = int(fw * 0.8)
            cx, cy = fx + fw//2, fy + fh//3
            x1, y1 = max(0, cx - span//2), max(0, cy - int(span * 0.5))
            x2, y2 = min(w, x1 + span), min(h, y1 + span)
            eye_crops = [frame[y1:y2, x1:x2]]
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 100, 255), 1)

        # ── 3. CNN Inference (Average over eye crops) ──────────────
        batch = []
        for ec in eye_crops:
            inp = cv2.resize(ec, (IMG_W, IMG_H))
            inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32)
            inp = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
            batch.append(inp)

        probs = self._cnn_model.predict(np.asarray(batch), verbose=0)
        avg_probs = np.mean(probs, axis=0)
        pred_idx  = int(np.argmax(avg_probs))
        confidence = float(avg_probs[pred_idx])

        self._pred_buffer.append(pred_idx)

        # Majority vote over buffer
        counts = collections.Counter(self._pred_buffer)
        voted_idx = counts.most_common(1)[0][0]
        raw_class = self._idx_to_class.get(str(voted_idx), "unknown").lower()

        # Map "straight" or others to GazeDirection constants
        if raw_class == "straight":
            direction = GazeDirection.CENTER
        elif hasattr(GazeDirection, raw_class.upper()):
            direction = getattr(GazeDirection, raw_class.upper())
        else:
            direction = GazeDirection.UNKNOWN

        # ── 4. Debug Overlay ──────────────────────────────────────────
        col_map = {
            GazeDirection.UP:      (80, 220,  80),
            GazeDirection.DOWN:    (80, 100, 220),
            GazeDirection.LEFT:    (255, 165,  0),
            GazeDirection.RIGHT:   (255, 165,  0),
            GazeDirection.CENTER:  (200, 200, 200),
            GazeDirection.UNKNOWN: ( 80,  80,  80),
        }
        col = col_map.get(direction, (255, 255, 255))
        cv2.putText(debug, f"CNN: {direction.upper()} ({confidence*100:.0f}%)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2)
        cv2.putText(debug, f"Mode A | eyes={len(eye_crops)} | smooth={self.CNN_SMOOTH}f",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        return direction, debug, None   # gaze_ratios not needed in Mode A

    # ── Mode B – Classic CV ───────────────────────────────────────────────

    def _process_classic(self, frame: np.ndarray):
        debug = frame.copy()
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        if len(faces) == 0:
            cv2.putText(debug, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return GazeDirection.UNKNOWN, debug, None

        face = max(faces, key=lambda r: r[2] * r[3])
        fx, fy, fw_f, fh_f = face
        cv2.rectangle(debug, (fx, fy), (fx+fw_f, fy+fh_f), (0, 200, 0), 2)

        face_gray    = gray[fy:fy+fh_f, fx:fx+fw_f]
        eye_roi_gray = face_gray[:int(fh_f * 0.60), :]

        eyes = self.eye_cascade.detectMultiScale(
            eye_roi_gray, scaleFactor=1.05, minNeighbors=4, minSize=(25, 15)
        )
        if len(eyes) == 0:
            cv2.putText(debug, "No eyes detected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
            return GazeDirection.UNKNOWN, debug, None

        eyes = sorted(eyes, key=lambda e: e[0])[:2]
        iris_ratios_x, iris_ratios_y = [], []

        for ex, ey, ew, eh in eyes:
            abs_ex, abs_ey = fx + ex, fy + ey
            cv2.rectangle(debug, (abs_ex, abs_ey), (abs_ex+ew, abs_ey+eh), (255, 180, 0), 2)
            result = _find_iris_center(face_gray[ey:ey+eh, ex:ex+ew])
            if result is None:
                continue
            rx, ry = result
            iris_ratios_x.append(rx)
            iris_ratios_y.append(ry)
            cv2.circle(debug, (int(abs_ex + rx*ew), int(abs_ey + ry*eh)), 4, (0, 255, 255), -1)

        if len(iris_ratios_x) < self.MIN_EYES:
            cv2.putText(debug, "Iris not found", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
            return GazeDirection.UNKNOWN, debug, None

        avg_x = float(np.mean(iris_ratios_x))
        avg_y = float(np.mean(iris_ratios_y))
        self._smooth_x = _smooth(self._smooth_x, avg_x)
        self._smooth_y = _smooth(self._smooth_y, avg_y)
        sx, sy = self._smooth_x, self._smooth_y

        v_dir = GazeDirection.UP   if sy < self.UP_THRESH   else \
                GazeDirection.DOWN if sy > self.DOWN_THRESH  else GazeDirection.CENTER
        h_dir = GazeDirection.LEFT  if sx < self.LEFT_THRESH  else \
                GazeDirection.RIGHT if sx > self.RIGHT_THRESH else GazeDirection.CENTER

        direction = v_dir if v_dir != GazeDirection.CENTER else h_dir

        col_map = {GazeDirection.UP:(0,255,0), GazeDirection.DOWN:(0,0,255),
                   GazeDirection.LEFT:(255,165,0), GazeDirection.RIGHT:(255,165,0),
                   GazeDirection.CENTER:(200,200,200), GazeDirection.UNKNOWN:(80,80,80)}
        col = col_map.get(direction, (255, 255, 255))
        cv2.putText(debug, f"Gaze: {direction.upper()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        cv2.putText(debug, f"rx={sx:.2f}  ry={sy:.2f}  Mode B",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        # Vertical gaze bar
        bar_x, bar_y, bar_h = frame.shape[1]-25, 80, 120
        cv2.rectangle(debug, (bar_x, bar_y), (bar_x+12, bar_y+bar_h), (80, 80, 80), -1)
        cv2.circle(debug, (bar_x+6, int(bar_y + sy*bar_h)), 6, col, -1)

        return direction, debug, (sx, sy)
