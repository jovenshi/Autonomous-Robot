"""Print-ready calibration chessboard (vector PDF, no dependencies).

    python -m tools.make_board                  # 38 mm squares on 8.5 x 11 in (letter)
    python -m tools.make_board --square-mm 55 --paper-mm 431.8 279.4   # tabloid

The board has CALIB_PATTERN inner corners (5 x 4 -> 6 x 5 squares), centred
on the page. Print at 100% / "actual size" (NOT "fit to page"), then measure
one printed square with a ruler and enter that on the calibration WELCOME
screen: printers rescale a little.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402

PT_PER_MM = 72 / 25.4


def board_pdf(pattern, square_mm, paper_mm):
    cols, rows = pattern[0] + 1, pattern[1] + 1          # squares across, along
    page_w, page_h = paper_mm[0] * PT_PER_MM, paper_mm[1] * PT_PER_MM
    sq = square_mm * PT_PER_MM
    x0, y0 = (page_w - cols * sq) / 2, (page_h - rows * sq) / 2
    if x0 < 0 or y0 < 0:
        raise SystemExit("board does not fit on the paper: use a smaller --square-mm")
    ops = ["0 g"]
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                ops.append(f"{x0 + c * sq:.3f} {y0 + r * sq:.3f} {sq:.3f} {sq:.3f} re f")
    stream = "\n".join(ops).encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w:.3f} {page_h:.3f}] "
            f"/Contents 4 0 R >>".encode(),
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"]
    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    margins = (x0 / PT_PER_MM, y0 / PT_PER_MM)
    return out, margins


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--square-mm", type=float, default=38.0)
    p.add_argument("--paper-mm", type=float, nargs=2, default=(279.4, 215.9),
                   metavar=("W", "H"), help="landscape paper size (default 11 x 8.5 in)")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "docs",
                                                 "calibration_board.pdf"))
    args = p.parse_args()
    data, (mx, my) = board_pdf(config.CALIB_PATTERN, args.square_mm, args.paper_mm)
    with open(args.out, "wb") as f:
        f.write(data)
    print(f"Wrote {os.path.abspath(args.out)}: {args.square_mm:g} mm squares, white margin "
          f"{mx:.0f} mm left/right, {my:.0f} mm top/bottom. Print at 100%.")


if __name__ == "__main__":
    main()
