"""Exposure control for either camera source.

One ExposureControl per camera. EXPOSURE_MODE decides what it does:
    'auto'    : nothing; the camera keeps its own auto-exposure.
    'locked'  : lock_after_warmup() fixes exposure and gain once.
    'managed' : lock as above, then manage_step() nudges exposure toward the
                target brightness every EXPOSURE_CHECK_FRAMES, except while
                hold is set (exposure_hold, set outside LANE_FOLLOW).

Wiring (in main.build_system): start the CameraStream first so frames flow,
call lock_after_warmup() from the main thread, and have the camera thread call
manage_step(result.mean_v, exposure_hold.is_set()) after each processed frame.
manage_step also records each frame's brightness, which becomes the target
when EXPOSURE_TARGET_V is None.

Gain is treated as a multiplier (exposure x gain ~ brightness), as libcamera's
AnalogueGain is. Webcam gain may use a different scale; see config.py.
"""
import time


class ExposureControl:
    def __init__(self, source, name, cfg):
        self.source = source
        self.name = name                     # 'front' or 'side', for messages
        self.cfg = cfg
        self.exposure_us = None              # current locked values
        self.gain = None
        self.target_v = cfg.EXPOSURE_TARGET_V
        self.locked = False
        self._last_mean_v = None
        self._frames = 0

    def lock_after_warmup(self):
        # Lets auto-exposure settle for CAM_WARMUP_S, then fixes those values,
        # capped at EXPOSURE_MAX_US with gain raised to keep the brightness.
        if self.cfg.EXPOSURE_MODE == "auto":
            return
        time.sleep(self.cfg.CAM_WARMUP_S)
        exposure_us, gain = self.source.get_exposure()
        if not readback_ok(exposure_us, gain):
            exposure_us, gain = self._fallback_values(exposure_us, gain)
            if exposure_us is None:
                return
        exposure_us, gain = cap_exposure(exposure_us, gain, self.cfg)
        self.source.set_exposure(exposure_us, gain)
        self.exposure_us, self.gain = exposure_us, gain
        if self.target_v is None:
            self.target_v = self._last_mean_v
        self.locked = True

    def manage_step(self, mean_v, hold):
        # Called by the camera thread once per frame with that frame's brightness.
        self._last_mean_v = mean_v
        if not self._managing() or hold:
            return
        self._frames += 1
        if self._frames < self.cfg.EXPOSURE_CHECK_FRAMES:
            return
        self._frames = 0
        exposure_us, gain = nudge(self.exposure_us, self.gain, mean_v, self.target_v, self.cfg)
        if (exposure_us, gain) != (self.exposure_us, self.gain):
            self.source.set_exposure(exposure_us, gain)
            self.exposure_us, self.gain = exposure_us, gain

    def _managing(self):
        return (self.cfg.EXPOSURE_MODE == "managed" and self.locked
                and self.target_v is not None)

    def _fallback_values(self, exposure_us, gain):
        # Readback unusable: use the fixed bench values, or stay on auto if unset.
        fixed_us, fixed_gain = self.cfg.USB_LOCK_EXPOSURE_US, self.cfg.USB_LOCK_GAIN
        if fixed_us is None or fixed_gain is None:
            print(f"Exposure '{self.name}': readback unusable ({exposure_us}, {gain}) and "
                  "USB_LOCK_EXPOSURE_US / USB_LOCK_GAIN not set; staying on auto-exposure")
            return None, None
        print(f"Exposure '{self.name}': readback unusable ({exposure_us}, {gain}); "
              f"using fixed {fixed_us} us, gain {fixed_gain}")
        return fixed_us, fixed_gain


def readback_ok(exposure_us, gain):
    # A usable reading has a positive exposure and some gain value.
    return exposure_us is not None and gain is not None and exposure_us > 0


def cap_exposure(exposure_us, gain, cfg):
    # Caps exposure at EXPOSURE_MAX_US and raises gain by the same factor
    # (up to EXPOSURE_GAIN_MAX) so the image keeps its brightness.
    if exposure_us > cfg.EXPOSURE_MAX_US:
        gain = min(gain * exposure_us / cfg.EXPOSURE_MAX_US, cfg.EXPOSURE_GAIN_MAX)
        exposure_us = cfg.EXPOSURE_MAX_US
    return exposure_us, gain


def nudge(exposure_us, gain, mean_v, target_v, cfg):
    # One EXPOSURE_STEP toward target_v. Brightening: exposure first, then gain
    # once exposure is at the cap. Darkening: gain first, then exposure.
    if abs(mean_v - target_v) <= cfg.EXPOSURE_TOL_V:
        return exposure_us, gain
    factor = 1.0 + cfg.EXPOSURE_STEP
    if mean_v < target_v:
        if exposure_us < cfg.EXPOSURE_MAX_US:
            return min(exposure_us * factor, cfg.EXPOSURE_MAX_US), gain
        return exposure_us, min(gain * factor, cfg.EXPOSURE_GAIN_MAX)
    if gain > cfg.EXPOSURE_GAIN_MIN:
        return exposure_us, max(gain / factor, cfg.EXPOSURE_GAIN_MIN)
    return max(exposure_us / factor, cfg.EXPOSURE_MIN_US), gain