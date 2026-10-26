"""Front camera calibration from photos of a printed chessboard.

Usually run through the guided screen, tools/calib_ui.py, which takes the
photos, enters the measured spots and calls these functions. The same steps
work from the command line on photos you already have (run from autonomous/):

    python -m tools.calibrate fit   calibration/front_20261025_140000 [--apply]
    python -m tools.calibrate check photo.png [--near-m 0.87 --lateral-m 0.05]
    python -m tools.calibrate lens  calibration/lens_20261025_150000 [--apply]

--apply writes the results into config.py (config.py.bak keeps the old file).

The method (design doc, "Front camera calibration"):
  * One letter-size board, photographed at several spots in front of the
    robot. Each spot is read off one tape measure run forward from the
    rear-axle mark along the robot's centreline: near_m is the tape reading
    at the board's nearest row of inner corners, lateral_m how far the
    board's centre sits right (+) or left (-) of the tape.
  * All corners together fit ONE homography: lane-image pixels -> metres
    from the rear axle. FRONT_BANDS are recomputed from it.
  * Each "fit" board is also checked against a fit made WITHOUT it, so a
    mis-measured or crooked board stands out (with a hint of which way).
  * The "check" photo is never used in the fit; it verifies the result, and
    its spot is taped on the floor for the 30-second pre-session check.
  * Lens distortion is optional: if boards near the image edges fit much
    worse than central ones, take the lens photos (lens subcommand) and refit.

Session folder (written by calib_ui.py; can be made by hand):
    session.json    {"square_m", "pattern", "road_width_m", "frame_size", "created"}
    positions.csv   image,near_m,lateral_m,role     (role: fit | check)
    board_01.png    full FRONT_SIZE frames
    result.json, preview.png    written by fit

Assumes the boards lie flat on the mat and the camera does not move between
photos. Bands keep the rows of the current FRONT_BANDS unless --new-rows.
"""
import argparse
import csv
import datetime
import json
import os
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

# Let `python tools/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402

SESSION_JSON = "session.json"
POSITIONS_CSV = "positions.csv"
RESULT_JSON = "result.json"
PREVIEW_PNG = "preview.png"
POSITION_FIELDS = ("image", "near_m", "lateral_m", "role")
ROLES = ("fit", "check")


# ---------------------------------------------------------------- corners

def find_board_corners(image, pattern, fast=False):
    # Inner corners (N, 2) in image pixels, refined to sub-pixel, or None.
    # fast=True gives up quickly when there is no board (live preview).
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    if fast:
        flags += cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, tuple(pattern), flags=flags)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    return corners.reshape(-1, 2)


def order_grid(corners, pattern):
    # (rows, cols, 2) grid with rows running across the image, row 0 nearest
    # the robot (lowest in the image) and column 0 on the left.
    cols, rows = pattern
    grid = corners.reshape(rows, cols, 2)
    along_row = grid[0, -1] - grid[0, 0]
    if abs(along_row[0]) < abs(along_row[1]):          # rows run up the image
        grid = grid.transpose(1, 0, 2)
    if grid[0, :, 1].mean() < grid[-1, :, 1].mean():   # row 0 is the far one
        grid = grid[::-1]
    if grid[:, 0, 0].mean() > grid[:, -1, 0].mean():   # column 0 is on the right
        grid = grid[:, ::-1]
    return np.ascontiguousarray(grid)


def ground_points(rows, cols, near_m, square_m, lateral_m):
    # (rows, cols, 2) of (forward_m, lateral_m) for each inner corner;
    # lateral + = right, board centre at lateral_m.
    r, c = np.mgrid[0:rows, 0:cols]
    forward = near_m + r * square_m
    lateral = lateral_m + (c - (cols - 1) / 2.0) * square_m
    return np.dstack([forward, lateral]).astype(np.float64)


# ---------------------------------------------------------------- lens and pixel spaces

@dataclass
class Lens:
    camera_matrix: np.ndarray            # 3 x 3, frame pixels
    dist_coeffs: np.ndarray

    def undistort(self, points_frame_px):
        # Frame pixels -> where an ideal (distortion-free) lens would put them.
        pts = np.asarray(points_frame_px, np.float64).reshape(-1, 1, 2)
        out = cv2.undistortPoints(pts, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)
        return out.reshape(-1, 2)


def lens_from_config(cfg):
    # Lens from FRONT_CAM_MATRIX / FRONT_DIST_COEFFS, or None if not measured.
    if cfg.FRONT_CAM_MATRIX is None or cfg.FRONT_DIST_COEFFS is None:
        return None
    return Lens(np.array(cfg.FRONT_CAM_MATRIX, np.float64),
                np.array(cfg.FRONT_DIST_COEFFS, np.float64))


def to_lane_px(points, image_shape, lane_size):
    # Full-frame pixels -> FRONT_LANE_SIZE pixels.
    scale_x = lane_size[0] / image_shape[1]
    scale_y = lane_size[1] / image_shape[0]
    return np.asarray(points, np.float64) * np.array([scale_x, scale_y])


def frame_to_lane(points, image_shape, lane_size, lens=None):
    # Frame pixels -> lane pixels, lens-corrected when a lens is given.
    pts = lens.undistort(points) if lens is not None else np.asarray(points, np.float64)
    return to_lane_px(pts, image_shape, lane_size)


@dataclass
class GroundMap:
    """Raw lane-image pixels (what perception sees) -> (forward_m, lateral_m)."""
    H: np.ndarray                        # corrected lane px -> metres
    lane_size: tuple
    frame_size: tuple                    # (width, height) of the full frame
    lens: object = None

    def to_ground(self, points_lane_px):
        pts = np.asarray(points_lane_px, np.float64).reshape(-1, 2)
        if self.lens is not None:
            shape = (self.frame_size[1], self.frame_size[0])
            frame = pts * np.array([self.frame_size[0] / self.lane_size[0],
                                    self.frame_size[1] / self.lane_size[1]])
            pts = frame_to_lane(frame, shape, self.lane_size, self.lens)
        return image_to_ground(self.H, pts)


def image_to_ground(H, points_px):
    # (N, 2) image points -> (N, 2) (forward_m, lateral_m).
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def fit_ground(grid_px, ground):
    # Homography from image pixels to (forward_m, lateral_m), and the RMS
    # error (m) of the corners mapped back through it. Any shapes that
    # reshape to (N, 2) work, so several boards can be stacked.
    src = np.asarray(grid_px, np.float64).reshape(-1, 2)
    dst = np.asarray(ground, np.float64).reshape(-1, 2)
    H, _ = cv2.findHomography(src, dst, 0)
    mapped = image_to_ground(H, src)
    rms = float(np.sqrt(np.mean(np.sum((mapped - dst) ** 2, axis=1))))
    return H, rms


# ---------------------------------------------------------------- boards and errors

@dataclass
class Board:
    name: str                            # image file name
    near_m: float
    lateral_m: float
    role: str                            # 'fit' or 'check'
    grid: np.ndarray = None              # (rows, cols, 2) corrected lane px; None = not found
    ground: np.ndarray = None            # (rows, cols, 2) metres
    stats: dict = field(default_factory=dict)


def detect_board(image, pattern, lane_size, lens=None, fast=False):
    # Ordered (rows, cols, 2) grid in corrected lane px, or None.
    corners = find_board_corners(image, pattern, fast)
    if corners is None:
        return None
    return order_grid(frame_to_lane(corners, image.shape, lane_size, lens), pattern)


def board_errors(H, grid, ground):
    # How far the board's corners land from where they should, through H:
    # rms and worst distance, and the mean miss (forward, lateral) in m.
    # A mean miss means the whole board sits off its entered spot.
    mapped = image_to_ground(H, grid)
    diff = mapped - np.asarray(ground, np.float64).reshape(-1, 2)
    dist = np.sqrt(np.sum(diff ** 2, axis=1))
    return {"rms_m": float(np.sqrt(np.mean(dist ** 2))), "max_m": float(dist.max()),
            "offset_fwd_m": float(diff[:, 0].mean()), "offset_lat_m": float(diff[:, 1].mean())}


def fit_boards(boards):
    # One homography from every found 'fit' board; each fit board is scored
    # against a fit made without it (so its own error is not hidden), each
    # 'check' board against the full fit. Fills board.stats; returns (H, rms).
    fit = [b for b in boards if b.role == "fit" and b.grid is not None]
    if not fit:
        raise ValueError("no fit boards were found in the photos")
    H, rms = fit_ground([b.grid for b in fit], [b.ground for b in fit])
    for b in fit:
        others = [o for o in fit if o is not b]
        H_wo = fit_ground([o.grid for o in others], [o.ground for o in others])[0] if others else H
        b.stats = board_errors(H_wo, b.grid, b.ground)
    for b in boards:
        if b.role == "check" and b.grid is not None:
            b.stats = board_errors(H, b.grid, b.ground)
    return H, rms


def coverage_rows(boards):
    # (top, bottom) lane rows reached by the found fit boards' corners.
    ys = np.concatenate([b.grid[..., 1].ravel() for b in boards
                         if b.role == "fit" and b.grid is not None])
    return int(np.floor(ys.min())), int(np.ceil(ys.max()))


def lens_hint(boards, lane_size, cfg):
    # Mean error of boards near the image centre vs near the edges. A lens
    # that bends lines shows up as edge boards fitting much worse.
    centre = np.array(lane_size, np.float64) / 2.0
    scored = []
    for b in boards:
        if b.role == "fit" and b.grid is not None and b.stats:
            offset = np.abs(b.grid.reshape(-1, 2).mean(axis=0) - centre) / centre
            scored.append((float(offset.max()), b.stats["rms_m"]))
    if len(scored) < 4:
        return None
    scored.sort()
    half = len(scored) // 2
    inner = float(np.mean([e for _, e in scored[:half]]))
    outer = float(np.mean([e for _, e in scored[half:]]))
    suggest = outer > cfg.CALIB_LENS_EDGE_RATIO * inner and outer > cfg.CALIB_BOARD_RMS_WARN_M
    return {"inner_rms_m": inner, "outer_rms_m": outer, "suggest": bool(suggest)}


# ---------------------------------------------------------------- bands

def band_rows(top_row, bottom_row, count, fill):
    # count (y_top, y_bottom) rows between top_row and bottom_row, farthest first.
    slot = (bottom_row - top_row) / count
    rows = []
    for k in range(count):
        y_top = int(round(top_row + k * slot + slot * (1 - fill) / 2))
        rows.append((y_top, y_top + max(1, int(round(slot * fill)))))
    return rows


def bands_for_rows(to_ground, rows, width, road_width_m):
    # FRONT_BANDS entries (y_top, y_bottom, forward_m, half_road_px, m_per_px)
    # for the given rows; to_ground maps raw lane px -> metres.
    img_cx = (width - 1) / 2.0
    half_span = width / 4.0
    bands = []
    for y_top, y_bottom in rows:
        y_mid = (y_top + y_bottom) / 2.0
        centre, left, right = to_ground(
            [(img_cx, y_mid), (img_cx - half_span, y_mid), (img_cx + half_span, y_mid)])
        m_per_px = (right[1] - left[1]) / (2 * half_span)
        half_road_px = int(round(road_width_m / 2 / m_per_px))
        bands.append((int(y_top), int(y_bottom), round(float(centre[0]), 4), half_road_px,
                      round(float(m_per_px), 6)))
    return bands


def make_bands(to_ground, top_row, bottom_row, count, fill, width, road_width_m):
    # count bands between top_row and bottom_row (farthest first).
    return bands_for_rows(to_ground, band_rows(top_row, bottom_row, count, fill), width,
                          road_width_m)


def camera_check(to_ground, bands, width):
    # Where straight ahead of the image centre lands on the ground: the
    # lateral offset at the nearest band, and the yaw implied by near vs far.
    img_cx = (width - 1) / 2.0
    far_band, near_band = bands[0], bands[-1]
    far, near = to_ground([(img_cx, (far_band[0] + far_band[1]) / 2),
                           (img_cx, (near_band[0] + near_band[1]) / 2)])
    yaw_deg = np.degrees(np.arctan2(far[1] - near[1], far[0] - near[0]))
    return float(near[1]), float(yaw_deg)


def choose_rows(cfg, coverage, new_rows=False, top_row=None, bottom_row=None):
    # Band rows: explicit rows if given; else the current FRONT_BANDS rows
    # (so recalibrating never moves the bands silently); else spread over
    # the rows the boards covered.
    if top_row is not None or bottom_row is not None:
        top = top_row if top_row is not None else coverage[0]
        bottom = bottom_row if bottom_row is not None else coverage[1]
        return band_rows(top, bottom, cfg.CALIB_BAND_COUNT, cfg.CALIB_BAND_FILL)
    if cfg.FRONT_BANDS is not None and not new_rows:
        return [(b[0], b[1]) for b in cfg.FRONT_BANDS]
    return band_rows(coverage[0], coverage[1], cfg.CALIB_BAND_COUNT, cfg.CALIB_BAND_FILL)


# ---------------------------------------------------------------- session files

def new_session_dir(cfg, kind="front", root=None):
    # A new, empty session folder under CALIB_DIR, named by the time.
    root = root or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                cfg.CALIB_DIR)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(root, f"{kind}_{stamp}")
    os.makedirs(path)
    return path


def write_session_meta(folder, cfg, square_m=None, road_width_m=None):
    meta = {"square_m": square_m if square_m is not None else cfg.CALIB_SQUARE_M,
            "pattern": list(cfg.CALIB_PATTERN),
            "road_width_m": road_width_m if road_width_m is not None else cfg.ROAD_WIDTH_M,
            "frame_size": list(cfg.FRONT_SIZE),
            "created": datetime.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(folder, SESSION_JSON), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def read_session(folder):
    # (meta, rows): rows are dicts with image, near_m, lateral_m (floats), role.
    with open(os.path.join(folder, SESSION_JSON)) as f:
        meta = json.load(f)
    path = os.path.join(folder, POSITIONS_CSV)
    rows = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["role"] not in ROLES:
                    raise ValueError(f"{POSITIONS_CSV}: role must be fit or check, "
                                     f"got {row['role']!r}")
                rows.append({"image": row["image"], "near_m": float(row["near_m"]),
                             "lateral_m": float(row["lateral_m"]), "role": row["role"]})
    return meta, rows


def write_positions(folder, rows):
    # Rewritten after every kept photo, so a crash loses nothing.
    tmp = os.path.join(folder, POSITIONS_CSV + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=POSITION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in POSITION_FIELDS})
    os.replace(tmp, os.path.join(folder, POSITIONS_CSV))


def load_boards(folder, meta, rows, lane_size, lens=None):
    # Board per positions.csv row; grid is None where the board wasn't found.
    pattern = tuple(meta["pattern"])
    boards = []
    for row in rows:
        image = cv2.imread(os.path.join(folder, row["image"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(os.path.join(folder, row["image"]))
        board = Board(row["image"], row["near_m"], row["lateral_m"], row["role"])
        board.grid = detect_board(image, pattern, lane_size, lens)
        if board.grid is not None:
            board.ground = ground_points(board.grid.shape[0], board.grid.shape[1], row["near_m"],
                                         meta["square_m"], row["lateral_m"])
        boards.append(board)
    return boards


# ---------------------------------------------------------------- the fit

def fit_session(folder, cfg, new_rows=False, top_row=None, bottom_row=None):
    # Everything except printing: a result dict (see report_lines), also
    # written to result.json with a preview.png.
    meta, rows = read_session(folder)
    lane_size = tuple(cfg.FRONT_LANE_SIZE)
    lens = lens_from_config(cfg)
    boards = load_boards(folder, meta, rows, lane_size, lens)
    found_fit = [b for b in boards if b.role == "fit" and b.grid is not None]
    if len(found_fit) < cfg.CALIB_MIN_FIT_BOARDS:
        raise ValueError(f"{len(found_fit)} fit boards found; at least "
                         f"{cfg.CALIB_MIN_FIT_BOARDS} are needed")
    H, rms = fit_boards(boards)
    gmap = GroundMap(H, lane_size, tuple(meta["frame_size"]), lens)
    coverage = coverage_rows(boards)
    rows_used = choose_rows(cfg, coverage, new_rows, top_row, bottom_row)
    bands = bands_for_rows(gmap.to_ground, rows_used, lane_size[0], meta["road_width_m"])
    offset_m, yaw_deg = camera_check(gmap.to_ground, bands, lane_size[0])
    result = {
        "folder": folder, "H": H.tolist(), "rms_m": rms, "bands": bands,
        "old_bands": [list(b) for b in cfg.FRONT_BANDS] if cfg.FRONT_BANDS else None,
        "coverage_rows": list(coverage),
        "extrapolated": [i for i, (t, b) in enumerate(rows_used)
                         if t < coverage[0] or b > coverage[1]],
        "boards": [_board_summary(b) for b in boards],
        "lens_used": lens is not None, "lens_hint": lens_hint(boards, lane_size, cfg),
        "camera_offset_m": offset_m, "camera_yaw_deg": yaw_deg,
        "square_m": meta["square_m"], "road_width_m": meta["road_width_m"],
    }
    result["check"] = _check_summary(result["boards"], cfg)
    _save_result(folder, result, boards, bands, lane_size, lens)
    return result


def _board_summary(board):
    return {"name": board.name, "role": board.role, "near_m": board.near_m,
            "lateral_m": board.lateral_m, "found": board.grid is not None, **board.stats}


def _check_summary(boards, cfg):
    # The held-back board(s): error and PASS, plus the spot to tape.
    checks = [b for b in boards if b["role"] == "check" and b["found"]]
    if not checks:
        return None
    worst = max(checks, key=lambda b: b["rms_m"])
    return {"name": worst["name"], "near_m": worst["near_m"], "lateral_m": worst["lateral_m"],
            "rms_m": worst["rms_m"], "max_m": worst["max_m"],
            "passed": worst["rms_m"] <= cfg.CALIB_HOLDOUT_TOL_M}


def _save_result(folder, result, boards, bands, lane_size, lens):
    with open(os.path.join(folder, RESULT_JSON), "w") as f:
        json.dump(result, f, indent=2)
    first = next(b for b in boards if b.grid is not None)
    image = cv2.imread(os.path.join(folder, first.name), cv2.IMREAD_COLOR)
    lane_img = cv2.resize(image, lane_size, interpolation=cv2.INTER_AREA)
    grids = [b.grid for b in boards if b.grid is not None]
    if lens is not None:                 # show raw-pixel positions on the raw image
        grids = [corrected_to_raw(g, image.shape, lane_size, lens) for g in grids]
    cv2.imwrite(os.path.join(folder, PREVIEW_PNG), draw_preview(lane_img, grids, bands))


def corrected_to_raw(grid, image_shape, lane_size, lens):
    # Lens-corrected lane px back to raw lane px, for drawing on the raw image.
    h, w = image_shape[:2]
    frame = grid.reshape(-1, 2) * np.array([w / lane_size[0], h / lane_size[1]])
    norm = cv2.undistortPoints(frame.reshape(-1, 1, 2), lens.camera_matrix, None)
    raw, _ = cv2.projectPoints(cv2.convertPointsToHomogeneous(norm), np.zeros(3), np.zeros(3),
                               lens.camera_matrix, lens.dist_coeffs)
    return to_lane_px(raw.reshape(-1, 2), image_shape, lane_size).reshape(grid.shape)


def draw_preview(lane_img, grids, bands):
    # Corners of every board, band rows and band labels on the lane-size image.
    img = lane_img.copy()
    for grid in grids:
        for x, y in np.asarray(grid).reshape(-1, 2):
            cv2.circle(img, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)
    width = img.shape[1]
    for y_top, y_bottom, forward_m, _, _ in bands:
        cv2.rectangle(img, (0, y_top), (width - 1, y_bottom), (0, 255, 0), 1)
        cv2.putText(img, f"{forward_m * 100:.0f} cm", (4, y_bottom - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
    x_mid = int(round((width - 1) / 2))
    cv2.line(img, (x_mid, 0), (x_mid, img.shape[0] - 1), (255, 255, 0), 1)
    return img


# ---------------------------------------------------------------- what the numbers mean

def board_note(board, cfg):
    # (level, short note) for one board: level 'ok', 'warn' or 'bad'.
    if not board["found"]:
        return "bad", "board not found: retake with the whole board in view"
    fwd, lat = board["offset_fwd_m"], board["offset_lat_m"]
    if max(abs(fwd), abs(lat)) >= cfg.CALIB_BOARD_OFFSET_WARN_M:
        return "warn", _offset_hint(fwd, lat)
    if board["rms_m"] > cfg.CALIB_BOARD_RMS_WARN_M:
        return "warn", "fits poorly: is the board flat and square to the tape?"
    return "ok", "good"


def _offset_hint(fwd, lat):
    # The direction the board seems to sit from where it was entered.
    return f"looks {direction(fwd, lat)} than entered: re-measure or retake"


def direction(fwd, lat):
    # A mean miss as words: "6 mm farther and 2 mm right" (the bigger part
    # first; a part under half the other is left out).
    parts = []
    if abs(fwd) >= abs(lat) / 2:
        parts.append(f"{abs(fwd) * 1000:.0f} mm {'farther' if fwd > 0 else 'nearer'}")
    if abs(lat) >= abs(fwd) / 2:
        parts.append(f"{abs(lat) * 1000:.0f} mm {'right' if lat > 0 else 'left'}")
    return " and ".join(parts)


def report_lines(result, cfg):
    # The fit result as (level, text) lines, for the console and the screen.
    lines = [("info", f"{sum(b['found'] for b in result['boards'] if b['role'] == 'fit')} fit "
                      f"boards, corner error {result['rms_m'] * 1000:.1f} mm RMS")]
    for b in result["boards"]:
        level, note = board_note(b, cfg)
        err = f"{b['rms_m'] * 1000:.1f} mm" if b["found"] else "--"
        lines.append((level, f"{b['name']} ({b['role']}, {b['near_m'] * 100:.1f} cm, "
                             f"{b['lateral_m'] * 100:+.1f} cm): {err}, {note}"))
    lines += _check_lines(result.get("check"), cfg)
    if result["extrapolated"]:
        nums = ", ".join(str(i + 1) for i in result["extrapolated"])
        lines.append(("warn", f"band(s) {nums} lie outside the rows the boards covered "
                              f"({result['coverage_rows'][0]}-{result['coverage_rows'][1]}): "
                              "add a board nearer or farther"))
    hint = result["lens_hint"]
    if hint and hint["suggest"] and not result["lens_used"]:
        lines.append(("warn", f"edge boards fit {hint['outer_rms_m'] * 1000:.1f} mm vs centre "
                              f"{hint['inner_rms_m'] * 1000:.1f} mm: lens distortion likely; "
                              "run the lens step, then fit again"))
    offset_mm = round(result["camera_offset_m"] * 1000) + 0     # + 0 turns -0 into 0
    lines.append(("info", f"camera centre column hits the ground {offset_mm:+d} mm from the "
                          f"centreline, camera yaw {result['camera_yaw_deg']:+.1f} deg"))
    return lines


def _check_lines(check, cfg):
    if check is None:
        return [("warn", "no check photo: the fit is not verified")]
    verdict = "PASS" if check["passed"] else "FAIL"
    level = "ok" if check["passed"] else "bad"
    return [(level, f"check photo: {check['rms_m'] * 1000:.1f} mm average, "
                    f"{check['max_m'] * 1000:.1f} mm worst (limit "
                    f"{cfg.CALIB_HOLDOUT_TOL_M * 1000:.0f} mm): {verdict}")]


def band_lines(result):
    # One line per band: forward distance, old -> new when there were old bands.
    old = result["old_bands"]
    lines = []
    for i, band in enumerate(result["bands"]):
        text = f"band {i + 1}: rows {band[0]}-{band[1]}, {band[2] * 100:.1f} cm"
        if old and i < len(old) and tuple(old[i][:2]) == tuple(band[:2]):
            text += f" (was {old[i][2] * 100:.1f} cm, {(band[2] - old[i][2]) * 1000:+.0f} mm)"
        lines.append(text)
    return lines


def fit_config_values(result):
    # The config.py values APPLY writes for a board session.
    values = {"FRONT_BANDS": [tuple(b) for b in result["bands"]],
              "FRONT_GROUND_H": [[round(v, 9) for v in row] for row in result["H"]],
              "CALIB_SQUARE_M": result["square_m"], "ROAD_WIDTH_M": result["road_width_m"]}
    if result.get("check"):
        values["CALIB_CHECK_NEAR_M"] = result["check"]["near_m"]
        values["CALIB_CHECK_LATERAL_M"] = result["check"]["lateral_m"]
    return values


# ---------------------------------------------------------------- pre-session check

def expected_corners(cfg, near_m, lateral_m, frame_shape):
    # Where a board at this spot should appear in the raw frame (frame px),
    # from the saved FRONT_GROUND_H; None before the first calibration.
    if cfg.FRONT_GROUND_H is None:
        return None
    rows, cols = cfg.CALIB_PATTERN[1], cfg.CALIB_PATTERN[0]
    ground = ground_points(rows, cols, near_m, cfg.CALIB_SQUARE_M, lateral_m)
    H_inv = np.linalg.inv(np.array(cfg.FRONT_GROUND_H, np.float64))
    lane_size = tuple(cfg.FRONT_LANE_SIZE)
    lane = image_to_ground(H_inv, ground).reshape(rows, cols, 2)
    lens = lens_from_config(cfg)
    if lens is not None:
        lane = corrected_to_raw(lane, frame_shape, lane_size, lens)
    h, w = frame_shape[:2]
    return lane.reshape(-1, 2) * np.array([w / lane_size[0], h / lane_size[1]])


def check_photo(image, cfg, near_m=None, lateral_m=None):
    # One board at the taped check spot, against the saved FRONT_GROUND_H:
    # error dict with 'passed', or raises ValueError with what is missing.
    if cfg.FRONT_GROUND_H is None:
        raise ValueError("FRONT_GROUND_H is not set: run the full calibration first")
    near_m = cfg.CALIB_CHECK_NEAR_M if near_m is None else near_m
    lateral_m = cfg.CALIB_CHECK_LATERAL_M if lateral_m is None else lateral_m
    if near_m is None or lateral_m is None:
        raise ValueError("the check spot is not set (CALIB_CHECK_NEAR_M / CALIB_CHECK_LATERAL_M)")
    lane_size = tuple(cfg.FRONT_LANE_SIZE)
    grid = detect_board(image, cfg.CALIB_PATTERN, lane_size, lens_from_config(cfg))
    if grid is None:
        raise ValueError("board not found: the whole board must be in view")
    ground = ground_points(grid.shape[0], grid.shape[1], near_m, cfg.CALIB_SQUARE_M, lateral_m)
    errors = board_errors(np.array(cfg.FRONT_GROUND_H, np.float64), grid, ground)
    errors["passed"] = errors["rms_m"] <= cfg.CALIB_CHECK_TOL_M
    errors["grid"] = grid
    return errors


def check_lines(errors, cfg):
    verdict = "PASS" if errors["passed"] else "FAIL"
    lines = [("ok" if errors["passed"] else "bad",
              f"{verdict}: {errors['rms_m'] * 1000:.1f} mm average, "
              f"{errors['max_m'] * 1000:.1f} mm "
              f"worst (limit {cfg.CALIB_CHECK_TOL_M * 1000:.0f} mm)")]
    if not errors["passed"]:
        lines.append(("info", f"the board reads "
                              f"{direction(errors['offset_fwd_m'], errors['offset_lat_m'])} "
                              "than the marks"))
        lines.append(("info", "check the board sits exactly on the marks; if it does, the camera "
                              "has moved: run the full calibration"))
    return lines


# ---------------------------------------------------------------- lens (optional)

def lens_cells(corners, frame_size, n=3):
    # The n x n image cells (row, col) that a board's corners reach.
    w, h = frame_size
    cols = np.clip((corners[:, 0] / w * n).astype(int), 0, n - 1)
    rows = np.clip((corners[:, 1] / h * n).astype(int), 0, n - 1)
    return set(zip(rows.tolist(), cols.tolist()))


def is_new_view(corners, previous, min_px):
    # True if the board moved at least min_px (centre) or changed size by
    # 20% since every earlier photo, so photos aren't near-duplicates.
    centre, size = corners.mean(axis=0), np.ptp(corners, axis=0).max()
    for prev in previous:
        moved = np.linalg.norm(prev.mean(axis=0) - centre)
        resized = abs(np.ptp(prev, axis=0).max() - size) / max(size, 1.0)
        if moved < min_px and resized < 0.2:
            return False
    return True


def calibrate_lens(corner_sets, pattern, square_m, frame_size):
    # Camera matrix and distortion from boards seen at many angles.
    # Returns (Lens, rms_px).
    cols, rows = pattern
    obj = np.zeros((rows * cols, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    image_pts = [np.asarray(c, np.float32).reshape(-1, 1, 2) for c in corner_sets]
    rms, K, dist, _, _ = cv2.calibrateCamera([obj] * len(image_pts), image_pts,
                                             tuple(frame_size), None, None)
    return Lens(K, dist.reshape(-1)), float(rms)


def lens_impact_mm(lens, cfg):
    # How much the lens correction changes the band measurements, in mm on
    # the ground. Most of a smooth lens bend is absorbed by the homography
    # anyway, so this is the part it can't absorb: fit one homography from
    # raw to corrected points across the band rows; the largest leftover,
    # times m_per_px, is what using the lens values would change. Below
    # CALIB_LENS_NEGLIGIBLE_MM it isn't worth applying.
    if cfg.FRONT_BANDS is None:
        return None
    lane_w, lane_h = cfg.FRONT_LANE_SIZE
    fw, fh = cfg.FRONT_SIZE
    raw, scales = [], []
    for y_top, y_bottom, _, _, m_per_px in cfg.FRONT_BANDS:
        for y in (y_top, y_bottom):
            xs = np.linspace(0, fw - 1, 9)
            raw.append(np.stack([xs, np.full_like(xs, y * fh / lane_h)], axis=1))
            scales.append(np.full(len(xs), m_per_px * lane_w / fw * 1000))   # mm per frame px
    raw, scales = np.concatenate(raw), np.concatenate(scales)
    corrected = lens.undistort(raw)
    H, _ = cv2.findHomography(raw, corrected, 0)
    leftover = np.linalg.norm(image_to_ground(H, raw) - corrected, axis=1)
    return float((leftover * scales).max())


def lens_config_values(lens):
    return {"FRONT_CAM_MATRIX": [[round(float(v), 6) for v in row] for row in lens.camera_matrix],
            "FRONT_DIST_COEFFS": [round(float(v), 8) for v in lens.dist_coeffs]}


def lens_folder_corners(folder, pattern):
    # Corners from every PNG in a lens folder (photos without a board are skipped).
    sets, size = [], None
    for name in sorted(os.listdir(folder)):
        if name.endswith(".png") and name != PREVIEW_PNG:
            image = cv2.imread(os.path.join(folder, name), cv2.IMREAD_COLOR)
            corners = find_board_corners(image, pattern)
            if corners is not None:
                sets.append(corners)
                size = (image.shape[1], image.shape[0])
    return sets, size


# ---------------------------------------------------------------- command line

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Front camera calibration (see the module docstring).")
    sub = p.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="fit FRONT_BANDS from a board session folder")
    fit.add_argument("folder")
    fit.add_argument("--new-rows", action="store_true",
                     help="spread the bands over the rows the boards covered, not the current rows")
    fit.add_argument("--top-row", type=int, default=None, help="highest band row (lane px)")
    fit.add_argument("--bottom-row", type=int, default=None, help="lowest band row (lane px)")
    fit.add_argument("--apply", action="store_true", help="write the results into config.py")
    check = sub.add_parser("check", help="pre-session check: one photo at the taped spot")
    check.add_argument("image")
    check.add_argument("--near-m", type=float, default=None, help="default CALIB_CHECK_NEAR_M")
    check.add_argument("--lateral-m", type=float, default=None,
                       help="default CALIB_CHECK_LATERAL_M")
    lens = sub.add_parser("lens", help="lens distortion from a folder of angled board photos")
    lens.add_argument("folder")
    lens.add_argument("--apply", action="store_true", help="write the results into config.py")
    return p.parse_args(argv)


def print_lines(lines):
    marks = {"ok": "  ok ", "warn": "  !! ", "bad": "  XX ", "info": "     "}
    for level, text in lines:
        print(marks[level] + text)


def run_fit(args, cfg):
    result = fit_session(args.folder, cfg, args.new_rows, args.top_row, args.bottom_row)
    print_lines(report_lines(result, cfg))
    print("\n" + "\n".join("     " + line for line in band_lines(result)))
    print(f"\nPreview: {os.path.join(args.folder, PREVIEW_PNG)}")
    return fit_config_values(result)


def run_lens(args, cfg):
    sets, size = lens_folder_corners(args.folder, cfg.CALIB_PATTERN)
    if len(sets) < cfg.CALIB_LENS_MIN_PHOTOS:
        raise SystemExit(f"{len(sets)} photos with a board found; "
                         f"{cfg.CALIB_LENS_MIN_PHOTOS} needed")
    lens, rms = calibrate_lens(sets, cfg.CALIB_PATTERN, cfg.CALIB_SQUARE_M, size)
    impact = lens_impact_mm(lens, cfg)
    print(f"Lens fit: {len(sets)} photos, {rms:.2f} px RMS"
          + ("  (high: retake blurry or bent photos)" if rms > cfg.CALIB_LENS_RMS_WARN_PX else ""))
    if impact is not None:
        negligible = impact < cfg.CALIB_LENS_NEGLIGIBLE_MM
        print(f"Correction changes band measurements by up to {impact:.1f} mm"
              + ("  (negligible: no need to apply)" if negligible else ""))
    return lens_config_values(lens)


def main(argv=None):
    from tools.config_writer import config_path, write_values

    args = parse_args(argv)
    try:
        if args.command == "check":
            image = cv2.imread(args.image, cv2.IMREAD_COLOR)
            if image is None:
                raise SystemExit(f"Cannot read {args.image}")
            errors = check_photo(image, config, args.near_m, args.lateral_m)
            print_lines(check_lines(errors, config))
            from tools.calib_status import record_check
            record_check(config, errors["passed"], errors["rms_m"], "check")
            return
        values = run_fit(args, config) if args.command == "fit" else run_lens(args, config)
    except ValueError as e:
        raise SystemExit(f"Calibration stopped: {e}")
    if args.apply:
        backup = write_values(config_path(), values)
        print(f"\nWritten to config.py (old file kept as {os.path.basename(backup)})")
        if args.command == "lens":
            print("Now fit the board session again so the bands use the lens values.")
    else:
        print("\nAdd --apply to write these into config.py, or paste:")
        for name, value in values.items():
            print(f"{name} = {value!r}")


if __name__ == "__main__":
    main()
