"""
Rebuilds mascot/assets/frame_NN.png from the original raw sprite tiles.

Only needed if you want to re-tweak the transparency/threshold, or
swap in a different sprite sheet later — mascot.py itself never runs
this; it just reads the already-built frame_NN.png files in assets/.

Why this exists: Windows' "-transparentcolor" trick tkinter uses (see
mascot.py) is a HARD on/off color key — a pixel is either fully see-
through or fully opaque, nothing in between. The original sprite tiles
have plenty of ordinary, correct semi-transparent edge pixels (normal
anti-aliasing), but naively alpha-blending those onto a magenta
background bakes in a blended, magenta-tinted halo color at every edge
— and since transparentcolor can't key out anything except the exact
magenta value, that halo stays fully visible as a magenta fringe around
the sprite.

Fix: threshold the alpha channel to pure 0/255 before compositing, so
every pixel is either fully the sprite's own color or fully the key
color — no in-between blend is ever produced, so there's no fringe left
to show through.
"""

from pathlib import Path
from PIL import Image

RAW_TILES_DIR = Path("raw_tiles")          # put your original tile000.png etc. here
OUT_DIR       = Path(__file__).resolve().parent / "assets"

KEY_COLOR       = (255, 0, 255)   # must match TRANSPARENT_KEY in mascot.py
CANVAS_SIZE     = (112, 123)      # must match your sprite's frame size
ALPHA_THRESHOLD = 120             # 0-255: raise this to shrink the visible
                                   # sprite silhouette slightly (trims more
                                   # of the soft edge); lower it to keep more
                                   # of the soft edge (at the risk of a
                                   # faint fringe reappearing)

# Frame indices to skip entirely (blank/degenerate frames from the
# original GIF split, if any -- inspect your own raw tiles and adjust).
SKIP_INDICES = {6, 13, 20}


def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tiles = sorted(RAW_TILES_DIR.glob("tile*.png"))
    kept = 0
    for tile_path in tiles:
        # e.g. "tile006.png" -> 6
        idx = int(tile_path.stem.replace("tile", ""))
        if idx in SKIP_INDICES:
            continue

        im = Image.open(tile_path).convert("RGBA")
        _, _, _, a = im.split()
        hard_mask = a.point(lambda v: 255 if v >= ALPHA_THRESHOLD else 0)

        bg = Image.new("RGB", CANVAS_SIZE, KEY_COLOR)
        bg.paste(im.convert("RGB"), (0, 0), hard_mask)
        bg.save(OUT_DIR / f"frame_{kept:02d}.png")
        kept += 1

    print(f"Wrote {kept} frames to {OUT_DIR}")


if __name__ == "__main__":
    build()
