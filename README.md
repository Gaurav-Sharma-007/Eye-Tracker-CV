# Eye Tracker - Gaze-Controlled Scroll & Click

A webcam-based gaze controller that classifies where you are looking and turns
that into desktop actions.

The current app uses a hybrid detection stack:

- **Mode A - CNN classifier:** loads `models/best_gaze_model.keras` or
  `models/gaze_model.keras` when available, detects/crops the eye region, and
  predicts one of `up`, `down`, `left`, `right`, or `straight`.
- **Mode B - classic CV fallback:** uses OpenCV Haar cascades, iris
  localization, smoothing, and ratio thresholds when TensorFlow or a trained
  model is unavailable.
- **Optional Amazon Rekognition:** with `--rekognition`, Rekognition landmarks
  can be used for a more accurate face/eye crop before CNN inference.

---

## How It Works

```text
Camera frame
    |
    v
Face detection
    |-- default: OpenCV Haar cascade
    `-- optional: Amazon Rekognition landmarks
    |
    v
Eye crop extraction
    |
    v
Mode A, if trained model exists:
    MobileNetV2-style Keras model -> 5-class softmax
    Temporal majority vote smoothing

Mode B, fallback:
    Haar eye detection
    CLAHE + Gaussian blur
    HoughCircles iris candidate
    Fallback darkest-contour centroid
    Normalized iris position (rx, ry)
    Threshold-based direction
    |
    v
Direction: up | down | left | right | center | unknown
    |
    |-- up/down   -> continuous scroll
    `-- center    -> dwell timer -> left click
```

`straight` from the trained model is mapped to the app's `center` direction.
Left/right are detected and displayed, but the current controller only performs
actions for up, down, and center.

---

## Project Structure

```text
eye-tracker/
├── main.py                  # App entry point and main loop
├── gaze_detector.py         # CNN + classic CV gaze detection pipeline
├── screen_controller.py     # xdotool scroll and dwell-click actions
├── calibration.py           # 5-point classic-CV threshold calibration
├── trainer.py               # Interactive threshold trainer for classic CV
├── config.py                # Loads/saves eye_tracker_config.json
├── overlay.py               # OpenCV status overlay
├── rekognition_face.py      # Optional Amazon Rekognition face landmarks
├── requirements.txt
├── models/
│   ├── best_gaze_model.keras
│   ├── gaze_model.keras
│   └── class_indices.json
└── data/
    ├── down/
    ├── left/
    ├── right/
    ├── straight/
    └── up/
```

---

## Installation

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt
```

On Linux/X11, install `xdotool` for scroll and click automation:

```bash
sudo apt-get install xdotool
```

Optional Rekognition setup:

```bash
pip install boto3
aws configure
```

Your AWS credentials need permission for `rekognition:DetectFaces`.

---

## Usage

```bash
# Basic run, camera 0, preview window shown
python main.py

# Hide camera preview
python main.py --no-preview

# Use a different camera index
python main.py --camera 1

# Run 5-point calibration on startup
python main.py --calibrate

# Use Amazon Rekognition for face/eye landmarks before CNN inference
python main.py --rekognition

# Override the AWS region for Rekognition
python main.py --rekognition --rekognition-region us-east-1

# Reduce Rekognition refresh frequency if your network is slow
python main.py --rekognition --rekognition-interval 1.5
```

### Keyboard Shortcuts

| Key       | Action                 |
|-----------|------------------------|
| `Q`/`ESC` | Quit                   |
| `Space`   | Pause / resume         |
| `C`       | Run calibration        |
| `+`       | Increase scroll amount |
| `-`       | Decrease scroll amount |

---

## Gaze Actions

| Gaze direction | App behavior                         |
|----------------|--------------------------------------|
| Up             | Scroll up continuously while held    |
| Down           | Scroll down continuously while held  |
| Center         | Start dwell timer, then left-click   |
| Left / Right   | Detected and shown; no action mapped |
| Unknown        | No action                            |

The dwell-click progress ring is shown in the status overlay.

---

## Calibration and Training

### 5-Point Calibration

```bash
python main.py --calibrate
```

Or press `C` while the app is running.

Calibration shows dots at the corners and center of the screen, collects gaze
samples, and adjusts the classic-CV up/down thresholds for the current user.
This mainly affects **Mode B**, because the CNN path does not return iris ratio
coordinates.

### Interactive Threshold Trainer

```bash
python trainer.py
```

Use the live trainer to collect labeled classic-CV samples:

| Key             | Label captured        |
|-----------------|-----------------------|
| Up arrow        | Looking up            |
| Down arrow      | Looking down          |
| Left/right arrow| Looking center        |
| `S`             | Save computed thresholds |
| `R`             | Reset samples         |
| `Q`/`ESC`       | Quit                  |

Saved values are written to `eye_tracker_config.json` and loaded automatically
by `main.py`.

---

## Configuration

Runtime settings are loaded from `eye_tracker_config.json` when it exists.
Missing values fall back to defaults from `config.py`.

| Setting           | Default | Description                            |
|-------------------|---------|----------------------------------------|
| `up_thresh`       | `0.38`  | Classic-CV iris Y threshold for up     |
| `down_thresh`     | `0.62`  | Classic-CV iris Y threshold for down   |
| `left_thresh`     | `0.38`  | Classic-CV iris X threshold for left   |
| `right_thresh`    | `0.62`  | Classic-CV iris X threshold for right  |
| `scroll_amount`   | `5`     | Scroll notches per scroll event        |
| `scroll_interval` | `0.12`  | Seconds between scroll events          |
| `dwell_time_s`    | `2.0`   | Seconds of stable center gaze to click |
| `dwell_zone_px`   | `80`    | Gaze movement radius before reset      |
| `stable_frames`   | `4`     | Matching frames required before action |
| `click_cooldown`  | `1.5`   | Seconds before another dwell click     |

---

## Models and Data

The trained gaze model is loaded automatically from `models/` in this order:

1. `models/best_gaze_model.keras`
2. `models/gaze_model.keras`

Class mappings live in `models/class_indices.json`:

```text
down, left, right, straight, up
```

The `data/` directory contains labeled gaze images used for training and
experimentation, grouped by direction.

---

## Dependencies

| Package          | Purpose                                      |
|------------------|----------------------------------------------|
| `opencv-python`  | Camera capture, Haar cascades, UI windows    |
| `numpy`          | Numeric processing and contour math          |
| `tensorflow`     | Keras gaze classifier inference              |
| `scikit-learn`   | Model training utilities                     |
| `matplotlib`     | Training plots                               |
| `boto3`          | Optional Amazon Rekognition integration      |
| `xdotool`        | Linux desktop scroll and click automation    |

---

## Notes

- The desktop-control layer is Linux/X11-oriented because it uses `xdotool`.
- If TensorFlow or the model files are missing, the app falls back to classic
  OpenCV gaze detection automatically.
- If Rekognition is requested but unavailable, the detector falls back to OpenCV
  face detection.
- Rekognition uses your AWS configured/default region unless
  `--rekognition-region` is provided.
- Rekognition runs asynchronously and is throttled so cloud latency does not
  block the live camera loop. Increase `--rekognition-interval` on slow
  networks, or omit `--rekognition` for fully local real-time tracking.
