"""
config.py
─────────
Loads and saves per-user thresholds to eye_tracker_config.json.
This persists calibration and training results across sessions.
"""

import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "eye_tracker_config.json")

DEFAULTS = {
    "up_thresh":        0.38,
    "down_thresh":      0.62,
    "left_thresh":      0.38,
    "right_thresh":     0.62,
    "scroll_amount":    5,
    "dwell_time_s":     2.0,
    "dwell_zone_px":    80,
    "stable_frames":    4,
    "scroll_interval":  0.12,
    "click_cooldown":   1.5,
}


def load() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                data = json.load(f)
            # Fill in any missing keys with defaults
            merged = {**DEFAULTS, **data}
            return merged
        except Exception as e:
            print(f"[config] Could not load config ({e}), using defaults.")
    return dict(DEFAULTS)


def save(cfg: dict):
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[config] Saved to {CONFIG_PATH}")
    except Exception as e:
        print(f"[config] Could not save config: {e}")


def show(cfg: dict):
    print("\n── Current thresholds ───────────────────────────────")
    print(f"  UP_THRESH   : {cfg['up_thresh']:.3f}  (iris Y < this → look up)")
    print(f"  DOWN_THRESH : {cfg['down_thresh']:.3f}  (iris Y > this → look down)")
    print(f"  DWELL_TIME  : {cfg['dwell_time_s']:.1f}s")
    print(f"  DWELL_ZONE  : {cfg['dwell_zone_px']}px")
    print(f"  STABLE_FRAMES: {cfg['stable_frames']}")
    print("─────────────────────────────────────────────────────\n")
