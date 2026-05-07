# Eye Tracker – Gaze-Controlled Scroll & Click

> **Built entirely from scratch** using OpenCV's built-in Haar cascades + pure
> NumPy math.  No pre-existing eye-tracking library used.

---

## How it works

```
Camera frame
    │
    ▼
Face detection  (Haar cascade – haarcascade_frontalface_default.xml)
    │
    ▼
Eye detection   (Haar cascade – haarcascade_eye_tree_eyeglasses.xml)
    │
    ▼
Iris localisation:
  1. CLAHE contrast enhancement
  2. Gaussian blur
  3. HoughCircles → iris circle candidate
  4. Fallback: adaptive threshold → darkest contour centroid
    │
    ▼
Normalised iris position (rx, ry) in [0,1]×[0,1]
    │
    ▼
Exponential moving average smoothing
    │
    ├── ry < 0.38 → LOOK UP   → scroll up
    ├── ry > 0.62 → LOOK DOWN → scroll down
    └── centered  → dwell timer → click after 2 s
```

---

## Project structure

```
eye-tracker/
├── main.py              # Entry point / main loop
├── gaze_detector.py     # Core CV pipeline (face → eye → iris → direction)
├── screen_controller.py # Scroll & dwell-click logic
├── calibration.py       # Optional per-user calibration
├── overlay.py           # Status overlay window
└── requirements.txt
```

---

## Installation

```bash
# 1. Create and activate a virtual environment (already done)
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
# Basic (camera 0, preview window shown)
python main.py

# Hide camera preview
python main.py --no-preview

# Different camera index
python main.py --camera 1

# Run calibration on startup (recommended for best accuracy)
python main.py --calibrate
```

### Keyboard shortcuts (in any open window)

| Key     | Action                             |
|---------|------------------------------------|
| `Q`/`ESC` | Quit                             |
| `Space` | Pause / resume                     |
| `C`     | Run calibration                    |
| `+`     | Increase scroll speed              |
| `-`     | Decrease scroll speed              |

---

## Gaze actions

| You do          | App does                                |
|-----------------|-----------------------------------------|
| Look **up**     | Scroll up (continuous while held)       |
| Look **down**   | Scroll down (continuous while held)     |
| Look **center** | Start 2-second dwell timer → left-click |

The dwell-click progress ring appears in the status overlay.

---

## Calibration (recommended)

Press **C** at any time to start the 5-point calibration.  
You'll see dots at the corners and centre of your screen.  
Look at each dot while it collects ~2.5 s of data.  
The detector then auto-adjusts its **UP** and **DOWN** thresholds to your specific eyes and face geometry.

---

## Tuning

Edit constants at the top of `screen_controller.py`:

| Constant          | Default | Description                         |
|-------------------|---------|-------------------------------------|
| `SCROLL_AMOUNT`   | `3`     | Scroll units per trigger             |
| `SCROLL_INTERVAL_S` | `0.12` | Seconds between scroll events       |
| `DWELL_TIME_S`    | `2.0`   | Seconds of gaze required to click   |
| `DWELL_ZONE_PX`   | `80`    | Pixel radius of dwell zone          |
| `STABLE_FRAMES`   | `3`     | Frames required before acting       |

Edit constants in `gaze_detector.py`:

| Constant     | Default | Description                    |
|--------------|---------|--------------------------------|
| `UP_THRESH`  | `0.38`  | Iris-Y ratio to classify "up"  |
| `DOWN_THRESH`| `0.62`  | Iris-Y ratio to classify "down"|

---

## Dependencies

| Package        | Purpose                              |
|----------------|--------------------------------------|
| `opencv-python`| Camera capture, Haar cascades, CV math |
| `numpy`        | Array math, contour analysis         |
| `pyautogui`    | Scroll & click simulation            |
| `Pillow`       | pyautogui screenshot support on Linux|
