"""Health checks: one small test per warning, each in its own helper.

Inputs that are None (a sensor not built or not yet reporting) raise no
warning of their own; staleness is the safety check's job.
"""
from robot_types import HealthWarning as W


def check_health(front, side, imu, lane, road_yaw, fps, overruns, motor_error, now, cfg):
    # fps: {'front': x, 'side': y} for the cameras that exist.
    # overruns: control-loop overruns in the recent window (not a lifetime total).
    # motor_error: SabertoothDriver.serial_error.
    checks = (
        (W.IMU_INVALID, _imu_invalid(imu, now, cfg)),
        (W.IMU_NOT_CALIBRATED, _imu_not_calibrated(imu)),
        (W.IMU_DRIFT_HIGH, _imu_drift_high(imu, cfg)),
        (W.ROAD_YAW_STALE, _road_yaw_stale(road_yaw, now, cfg)),
        (W.CAM_FPS_LOW, _cam_fps_low(fps, cfg)),
        (W.CAM_TOO_DARK, _cam_too_dark(front, side, cfg)),
        (W.CAM_TOO_BRIGHT, _cam_too_bright(front, side, cfg)),
        (W.CAMS_DISAGREE, _cams_disagree(lane, cfg)),
        (W.LANE_CONF_LOW, _lane_conf_low(lane, cfg)),
        (W.LOOP_OVERRUN, overruns > 0),
        (W.MOTOR_SERIAL_ERROR, bool(motor_error)),
    )
    return [warning for warning, failed in checks if failed]


def _imu_invalid(imu, now, cfg):
    # Failing reads, or a sample older than MAX_OBS_AGE_S (IMU thread stalled).
    return imu is not None and (not imu.valid or now - imu.t > cfg.MAX_OBS_AGE_S)


def _imu_not_calibrated(imu):
    return imu is not None and imu.calib_gyro < 3


def _imu_drift_high(imu, cfg):
    return imu is not None and abs(imu.drift_dps) > cfg.IMU_DRIFT_WARN_DPS


def _road_yaw_stale(road_yaw, now, cfg):
    if road_yaw is None:
        return False
    return not road_yaw.valid or now - road_yaw.t_anchor > cfg.ROAD_YAW_STALE_S


def _cam_fps_low(fps, cfg):
    return any(rate < cfg.CAM_FPS_MIN for rate in fps.values())


def _brightness(front, side):
    return [r.mean_v for r in (front, side) if r is not None]


def _cam_too_dark(front, side, cfg):
    return any(v < cfg.DARK_WARN_V for v in _brightness(front, side))


def _cam_too_bright(front, side, cfg):
    return any(v > cfg.BRIGHT_WARN_V for v in _brightness(front, side))


def _cams_disagree(lane, cfg):
    return lane is not None and lane.cam_gap_m is not None and abs(lane.cam_gap_m) > cfg.CAM_GAP_WARN_M


def _lane_conf_low(lane, cfg):
    return lane is not None and lane.conf < cfg.LANE_CONF_WARN
