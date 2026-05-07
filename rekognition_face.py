"""
rekognition_face.py
────────────────────
Face and eye-region detection using Amazon Rekognition.

Why?
────
Rekognition's DetectFaces API gives you sub-pixel landmark positions
(left/right eye centres, nose, mouth corners) and a precise bounding box
for each face.  This is far more accurate than OpenCV Haar cascades,
especially in poor lighting or at off-angles.

For gaze direction, we:
  1. Send the current camera frame to Rekognition.
  2. Get the face bounding box + eye landmarks.
  3. Crop a wider "forehead+eyes" region around both eye landmarks.
  4. Pass that crop to the Keras CNN classifier.

Setup
──────
1.  pip install boto3
2.  Configure AWS credentials:
        aws configure
    or set env vars AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION

3.  Make sure your IAM user has the rekognition:DetectFaces permission.

Usage (called from gaze_detector.py)
──────────────────────────────────────
    from rekognition_face import RekognitionFaceDetector
    rfd = RekognitionFaceDetector()
    box, landmarks = rfd.detect(bgr_frame)
    # box = (x, y, w, h) in pixels or None
    # landmarks = {"left_eye": (x,y), "right_eye": (x,y), ...} or {}
"""

import io
import cv2
import numpy as np

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class RekognitionFaceDetector:
    """
    Wraps Amazon Rekognition DetectFaces to return face bounding box
    and eye landmark positions in pixel coordinates.

    Falls back gracefully to None if Rekognition is unavailable or
    credentials are missing.
    """

    def __init__(self, region: str = "us-east-1"):
        self._client = None
        self._available = False

        if not BOTO3_AVAILABLE:
            print("[rekognition] boto3 not installed – face detection disabled.")
            return

        try:
            self._client = boto3.client("rekognition", region_name=region)
            # Quick connectivity test
            self._client.list_collections(MaxResults=1)
            self._available = True
            print(f"[rekognition] Connected (region={region})")
        except Exception as e:
            print(f"[rekognition] Not available ({type(e).__name__}): {e}")
            print("[rekognition] Falling back to OpenCV Haar cascades.")

    @property
    def available(self) -> bool:
        return self._available

    def detect(self, bgr_frame: np.ndarray):
        """
        Detect the largest face in `bgr_frame` (BGR uint8).

        Returns
        ────────
        box       : (x, y, w, h) in pixels  or  None
        landmarks : dict with keys "left_eye", "right_eye"  or  {}
        """
        if not self._available:
            return None, {}

        h, w = bgr_frame.shape[:2]

        # Encode frame as JPEG for the API call
        _, buf = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_bytes = io.BytesIO(buf.tobytes()).getvalue()

        try:
            resp = self._client.detect_faces(
                Image={"Bytes": image_bytes},
                Attributes=["DEFAULT"],      # includes landmarks
            )
        except Exception as e:
            print(f"[rekognition] API error: {e}")
            return None, {}

        if not resp.get("FaceDetails"):
            return None, {}

        # Pick the largest face by bounding box area
        best = max(
            resp["FaceDetails"],
            key=lambda f: f["BoundingBox"]["Width"] * f["BoundingBox"]["Height"],
        )

        bb = best["BoundingBox"]
        x = int(bb["Left"]   * w)
        y = int(bb["Top"]    * h)
        bw = int(bb["Width"]  * w)
        bh = int(bb["Height"] * h)
        # Clamp to frame bounds
        x  = max(0, x);  y  = max(0, y)
        bw = min(bw, w - x);  bh = min(bh, h - y)
        box = (x, y, bw, bh)

        # Extract eye landmarks
        landmarks = {}
        for lm in best.get("Landmarks", []):
            lm_type = lm["Type"]
            px = int(lm["X"] * w)
            py = int(lm["Y"] * h)
            if lm_type == "eyeLeft":
                landmarks["left_eye"] = (px, py)
            elif lm_type == "eyeRight":
                landmarks["right_eye"] = (px, py)

        return box, landmarks

    @staticmethod
    def eye_region_from_landmarks(
        bgr_frame: np.ndarray,
        landmarks: dict,
        pad_factor: float = 0.8,
    ) -> np.ndarray | None:
        """
        Crop the brow+eye band using Rekognition eye landmark positions.

        Returns a BGR crop suitable for the CNN classifier, or None.
        """
        left  = landmarks.get("left_eye")
        right = landmarks.get("right_eye")
        if left is None or right is None:
            return None

        h, w = bgr_frame.shape[:2]
        eye_dist = abs(right[0] - left[0])
        pad = int(eye_dist * pad_factor)

        mid_x = (left[0] + right[0]) // 2
        mid_y = (left[1] + right[1]) // 2

        x1 = max(0, mid_x - int(eye_dist * 1.0))
        x2 = min(w, mid_x + int(eye_dist * 1.0))
        y1 = max(0, mid_y - pad)
        y2 = min(h, mid_y + pad)

        crop = bgr_frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None
