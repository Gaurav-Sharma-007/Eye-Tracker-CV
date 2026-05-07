"""
screen_controller.py
────────────────────
Maps gaze direction → system scroll / click actions using xdotool
via subprocess.  No X display connection happens at import time.

Scroll  : xdotool key --clearmodifiers button4/5
Click   : xdotool click 1  (triggered after DWELL_TIME seconds of fixed gaze)

Dwell-click logic
──────────────────
• We track a "dwell zone" (a small screen rectangle around the current gaze).
• If the gaze stays inside the zone for ≥ DWELL_TIME seconds → left-click once.
• After a click we pause for CLICK_COOLDOWN seconds before the next click.
"""

import time
import subprocess
import shutil
import sys
import config as _cfg


# ─────────────────────────────────────────────────────────────
# Verify xdotool is available
# ─────────────────────────────────────────────────────────────

def _check_xdotool():
    if shutil.which("xdotool") is None:
        print(
            "[ERROR] xdotool is not installed.\n"
            "  Install it with:  sudo apt-get install xdotool\n"
            "  Then re-run main.py.",
            file=sys.stderr,
        )
        sys.exit(1)

_check_xdotool()


# ─────────────────────────────────────────────────────────────
# Tuning constants  (overridden from config at ScreenController init)
# ─────────────────────────────────────────────────────────────

_C = _cfg.load()

SCROLL_AMOUNT       = _C.get("scroll_amount",    5)
SCROLL_INTERVAL_S   = _C.get("scroll_interval",  0.12)

DWELL_TIME_S        = _C.get("dwell_time_s",     2.0)
DWELL_ZONE_PX       = _C.get("dwell_zone_px",    80)
CLICK_COOLDOWN_S    = _C.get("click_cooldown",   1.5)

# Higher = less jitter but more lag.  6 ≈ 200 ms at 30 fps.
STABLE_FRAMES       = _C.get("stable_frames",    6)


# ─────────────────────────────────────────────────────────────
# Low-level X actions via xdotool subprocess
# ─────────────────────────────────────────────────────────────

def _xdo(*args):
    """Run xdotool with the given arguments. Fire-and-forget."""
    subprocess.Popen(
        ["xdotool"] + list(args),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _get_screen_size():
    """Return (width, height) of the primary display via xdotool."""
    try:
        out = subprocess.check_output(
            ["xdotool", "getdisplaygeometry"], text=True
        ).strip()
        w, h = map(int, out.split())
        return w, h
    except Exception:
        return 1920, 1080      # safe fallback


def _scroll(direction: str, amount: int):
    """
    Scroll up (button4) or down (button5) at the current pointer position.
    Each click() call is one scroll notch, so we call it `amount` times.
    """
    btn = "4" if direction == "up" else "5"
    for _ in range(amount):
        _xdo("click", "--clearmodifiers", btn)


def _click_at(x: int, y: int):
    """Move pointer to (x, y) and left-click."""
    _xdo("mousemove", str(x), str(y))
    _xdo("click", "1")


# ─────────────────────────────────────────────────────────────
# ScreenController
# ─────────────────────────────────────────────────────────────

class ScreenController:
    """
    Receives gaze direction events and translates them into
    scroll and click actions on the host system.
    """

    def __init__(self):
        self._last_scroll_time   = 0.0
        self._last_click_time    = 0.0

        # Direction stability filter
        self._direction_buffer   = []
        self._stable_direction   = None

        # Dwell-click state
        self._dwell_start        = None
        self._dwell_anchor       = None   # (gx, gy) screen position when dwell began
        self._dwell_progress     = 0.0   # 0.0 → 1.0

        # Screen size (for mapping gaze ratios → screen coords)
        self._screen_w, self._screen_h = _get_screen_size()

        self.status_message = ""

    # ── Public API ────────────────────────────────────────────

    def update(self, direction: str, gaze_ratios):
        """
        Call once per frame.

        Parameters
        ──────────
        direction    : GazeDirection constant string
        gaze_ratios  : (rx, ry) in [0,1]×[0,1]  or  None
        """
        now = time.time()

        # ── Stability filter ──────────────────────────────────
        self._direction_buffer.append(direction)
        if len(self._direction_buffer) > STABLE_FRAMES:
            self._direction_buffer.pop(0)

        if (len(self._direction_buffer) == STABLE_FRAMES and
                len(set(self._direction_buffer)) == 1):
            stable = self._direction_buffer[0]
        else:
            stable = None

        self._stable_direction = stable

        # ── Scroll ────────────────────────────────────────────
        if stable in ("up", "down"):
            if now - self._last_scroll_time >= SCROLL_INTERVAL_S:
                _scroll(stable, SCROLL_AMOUNT)
                self._last_scroll_time = now
                self.status_message = f"Scrolling {'↑' if stable == 'up' else '↓'}"

        # ── Dwell-to-click ────────────────────────────────────
        if gaze_ratios is not None and stable == "center":
            rx, ry = gaze_ratios
            gx = int(rx * self._screen_w)
            gy = int(ry * self._screen_h)

            if self._dwell_anchor is None:
                self._dwell_anchor   = (gx, gy)
                self._dwell_start    = now
                self._dwell_progress = 0.0
            else:
                ax, ay = self._dwell_anchor
                dist = ((gx - ax) ** 2 + (gy - ay) ** 2) ** 0.5

                if dist > DWELL_ZONE_PX:
                    # Gaze moved — reset dwell
                    self._dwell_anchor   = (gx, gy)
                    self._dwell_start    = now
                    self._dwell_progress = 0.0
                    self.status_message  = "Dwell reset"
                else:
                    elapsed = now - self._dwell_start
                    self._dwell_progress = min(elapsed / DWELL_TIME_S, 1.0)

                    if elapsed >= DWELL_TIME_S and \
                            (now - self._last_click_time) >= CLICK_COOLDOWN_S:
                        self._fire_click(ax, ay, now)
        else:
            # Not centered — reset dwell
            if stable in ("up", "down", "left", "right"):
                self._dwell_anchor   = None
                self._dwell_start    = None
                self._dwell_progress = 0.0

        return self._dwell_progress

    @property
    def dwell_progress(self):
        return self._dwell_progress

    # ── Private ───────────────────────────────────────────────

    def _fire_click(self, x: int, y: int, now: float):
        _click_at(x, y)
        self._last_click_time = now
        self._dwell_anchor    = None
        self._dwell_start     = None
        self._dwell_progress  = 0.0
        self.status_message   = f"Clicked at ({x}, {y})"
        print(f"[CLICK] ({x}, {y})")
