"""Front camera calibration from a printed chessboard: computes FRONT_BANDS.

Setup (robot on a flat surface, camera mounted as it will drive):
  1. Print a chessboard; the default is 10 x 7 squares (9 x 6 inner corners).
     Measure one square on the print: printers rescale.
  2. Lay it flat in front of the robot, long side across the robot, centred
     on the robot's centreline, its edges square to the robot.
  3. Measure from the rear axle (midpoint) forward to the NEAREST row of inner
     corners (where four squares meet). That is --near-m.
  4. Record a few frames (tools/record.py) or save one PNG from the camera.
  5. Run from the robot/ folder:
       python -m tools.calibrate recordings/calib_1/frame_000040.png \
           --near-m 0.30 --square-m 0.025 --road-width-m 0.30

It finds the inner corners, fits the ground plane (a homography from
lane-image pixels to metres from the rear axle) and splits the chosen rows
into bands. By default the bands span the rows the board covers; widen them
with --top-row/--bottom-row (FRONT_LANE_SIZE pixels) after checking the
preview. Prints FRONT_BANDS to paste into config.py, saves a preview image
next to the input and shows it.

Assumes the lens has little distortion; check the reprojection error printed.
"""
import argparse
import os
import sys

import cv2
import numpy as np

# Let `python tools/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402


def find_board_corners(image, pattern):
    # Inner corners (N, 2) in image pixels, refined to sub-pixel, or None.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)
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


def fit_ground(grid_px, ground):
    # Homography from image pixels to (forward_m, lateral_m), and the RMS
    # error (m) of the corners mapped back through it.
    src = grid_px.reshape(-1, 2).astype(np.float64)
    dst = ground.reshape(-1, 2)
    H, _ = cv2.findHomography(src, dst, 0)
    mapped = image_to_ground(H, src)
    rms = float(np.sqrt(np.mean(np.sum((mapped - dst) ** 2, axis=1))))
    return H, rms


def image_to_ground(H, points_px):
    # (N, 2) image points -> (N, 2) (forward_m, lateral_m).
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def make_bands(H, top_row, bottom_row, count, fill, width, road_width_m):
    # count bands between top_row and bottom_row (farthest first), each
    # (y_top, y_bottom, forward_m, half_road_px, m_per_px).
    img_cx = (width - 1) / 2.0
    half_span = width / 4.0
    slot = (bottom_row - top_row) / count
    bands = []
    for k in range(count):
        y_top = int(round(top_row + k * slot + slot * (1 - fill) / 2))
        y_bottom = y_top + max(1, int(round(slot * fill)))
        y_mid = (y_top + y_bottom) / 2.0
        centre, left, right = image_to_ground(
            H, [(img_cx, y_mid), (img_cx - half_span, y_mid), (img_cx + half_span, y_mid)])
        m_per_px = (right[1] - left[1]) / (2 * half_span)
        half_road_px = int(round(road_width_m / 2 / m_per_px))
        bands.append((y_top, y_bottom, round(float(centre[0]), 4), half_road_px,
                      round(float(m_per_px), 6)))
    return bands


def camera_check(H, bands, width):
    # Where straight ahead of the image centre lands on the ground: the
    # lateral offset at the nearest band, and the yaw implied by near vs far.
    img_cx = (width - 1) / 2.0
    far_band, near_band = bands[0], bands[-1]
    far, near = image_to_ground(H, [(img_cx, (far_band[0] + far_band[1]) / 2),
                                    (img_cx, (near_band[0] + near_band[1]) / 2)])
    yaw_deg = np.degrees(np.arctan2(far[1] - near[1], far[0] - near[0]))
    return float(near[1]), float(yaw_deg)


def to_lane_px(points, image_shape, lane_size):
    # Full-frame pixels -> FRONT_LANE_SIZE pixels.
    scale_x = lane_size[0] / image_shape[1]
    scale_y = lane_size[1] / image_shape[0]
    return points * np.array([scale_x, scale_y])


def draw_preview(lane_img, grid_px, bands):
    # Corners, band rows and band labels on the lane-size image.
    img = lane_img.copy()
    for x, y in grid_px.reshape(-1, 2):
        cv2.circle(img, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)
    width = img.shape[1]
    for y_top, y_bottom, forward_m, _, _ in bands:
        cv2.rectangle(img, (0, y_top), (width - 1, y_bottom), (0, 255, 0), 1)
        cv2.putText(img, f"{forward_m * 100:.0f} cm", (4, y_bottom - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
    x_mid = int(round((width - 1) / 2))
    cv2.line(img, (x_mid, 0), (x_mid, img.shape[0] - 1), (255, 255, 0), 1)
    return img


def calibrate(image, args, lane_size):
    # Everything except printing and showing: returns (bands, report, preview).
    corners = find_board_corners(image, tuple(args.pattern))
    if corners is None:
        raise SystemExit("Chessboard not found: check --pattern, lighting and that "
                         "the whole board is in view")
    grid = order_grid(to_lane_px(corners, image.shape, lane_size), tuple(args.pattern))
    ground = ground_points(grid.shape[0], grid.shape[1], args.near_m, args.square_m,
                           args.lateral_m)
    H, rms = fit_ground(grid, ground)
    top = args.top_row if args.top_row is not None else int(np.floor(grid[:, :, 1].min()))
    bottom = args.bottom_row if args.bottom_row is not None else int(np.ceil(grid[:, :, 1].max()))
    bands = make_bands(H, top, bottom, args.bands, args.fill, lane_size[0], args.road_width_m)
    offset_m, yaw_deg = camera_check(H, bands, lane_size[0])
    lane_img = cv2.resize(image, lane_size, interpolation=cv2.INTER_AREA)
    report = {"rms_m": rms, "offset_m": offset_m, "yaw_deg": yaw_deg}
    return bands, report, draw_preview(lane_img, grid, bands)


def print_result(bands, report):
    print(f"\nCorner fit error: {report['rms_m'] * 1000:.1f} mm RMS "
          "(above ~3 mm: board not flat, wrong --square-m, or lens distortion)")
    print(f"Image centre column hits the ground {report['offset_m'] * 1000:+.1f} mm "
          f"from the centreline (+ = right), camera yaw {report['yaw_deg']:+.1f} deg")
    print("  Large values mean the camera is off-centre or turned, or the board was.")
    print("\n# Paste into config.py (cameras and geometry)")
    print("FRONT_BANDS = [")
    for band in bands:
        print(f"    {band},")
    print("]")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compute FRONT_BANDS from a chessboard image.")
    p.add_argument("image", help="a frame from tools/record.py (full FRONT_SIZE)")
    p.add_argument("--near-m", type=float, required=True,
                   help="rear axle to the nearest row of inner corners (m)")
    p.add_argument("--road-width-m", type=float, required=True,
                   help="full road width, edge to edge (m)")
    p.add_argument("--square-m", type=float, default=0.025, help="measured square size (m)")
    p.add_argument("--pattern", type=int, nargs=2, default=(9, 6),
                   metavar=("COLS", "ROWS"), help="inner corners across and along")
    p.add_argument("--lateral-m", type=float, default=0.0,
                   help="board centre right of the robot centreline (m)")
    p.add_argument("--bands", type=int, default=5, help="number of bands")
    p.add_argument("--fill", type=float, default=0.75, help="fraction of each slot a band covers")
    p.add_argument("--top-row", type=int, default=None, help="highest band row (lane px)")
    p.add_argument("--bottom-row", type=int, default=None, help="lowest band row (lane px)")
    p.add_argument("--no-window", action="store_true", help="don't open the preview window")
    return p.parse_args(argv)


def main():
    args = parse_args()
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Cannot read {args.image}")
    bands, report, preview = calibrate(image, args, tuple(config.FRONT_LANE_SIZE))
    print_result(bands, report)
    out = os.path.splitext(args.image)[0] + "_calibration.png"
    cv2.imwrite(out, preview)
    print(f"\nPreview saved to {out}")
    if not args.no_window:
        cv2.imshow("calibration (any key to close)", cv2.resize(preview, None, fx=2, fy=2))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
