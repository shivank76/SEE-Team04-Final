"""Printable checkerboard for the A4 camera calibration (OpenCV, Zhang 2002).

A4 landscape, 10 x 7 squares of 25 mm = 9 x 6 inner corners (odd x even, so the
board orientation is unambiguous). Vector PDF; print at 100 % / actual size.
Writes target/checkerboard_9x6_25mm.pdf.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PAGE_W, PAGE_H = 297.0, 210.0  # mm, A4 landscape
COLS, ROWS = 10, 7             # squares
SQUARE = 25.0                  # mm
SCALE_BAR = 100.0              # mm, for checking the print scale

OUT = Path(__file__).resolve().parent.parent / "target" / "checkerboard_9x6_25mm.pdf"


def main():
    fig = plt.figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(0, PAGE_H)
    ax.set_aspect("equal")
    ax.axis("off")

    x0 = (PAGE_W - COLS * SQUARE) / 2
    y0 = (PAGE_H - ROWS * SQUARE) / 2
    for i in range(COLS):
        for j in range(ROWS):
            if (i + j) % 2 == 0:
                ax.add_patch(Rectangle((x0 + i * SQUARE, y0 + j * SQUARE),
                                       SQUARE, SQUARE, color="black", lw=0))

    # Scale bar and print note in the bottom margin (>= 17.5 mm white border).
    yb = y0 / 2
    ax.plot([x0, x0 + SCALE_BAR], [yb, yb], color="black", lw=0.8)
    for x in (x0, x0 + SCALE_BAR):
        ax.plot([x, x], [yb - 2, yb + 2], color="black", lw=0.8)
    ax.text(x0 + SCALE_BAR + 4, yb,
            f"{SCALE_BAR:.0f} mm  |  {COLS}x{ROWS} squares, {SQUARE:.0f} mm, "
            f"{COLS - 1}x{ROWS - 1} inner corners  |  print at 100 % (actual size)",
            va="center", fontsize=7)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
