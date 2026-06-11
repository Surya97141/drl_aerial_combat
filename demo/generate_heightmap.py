"""
Generate a 512x512 greyscale heightmap PNG for the Babylon.js terrain.
Produces a realistic-looking landscape with mountains, valleys, and a
flat combat arena in the centre.

Run once before starting the demo:
    python demo/generate_heightmap.py
"""

import numpy as np
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit("pip install Pillow")


def fbm(shape, octaves=6, H=0.7, seed=42):
    """Fractional Brownian Motion heightmap via summed Perlin-like noise."""
    rng = np.random.default_rng(seed)
    h   = np.zeros(shape)
    freq, amp = 1.0, 1.0
    for _ in range(octaves):
        # Random phase tilt noise at increasing frequency
        x = np.linspace(0, freq * 2 * np.pi, shape[1])
        y = np.linspace(0, freq * 2 * np.pi, shape[0])
        xx, yy = np.meshgrid(x, y)
        phase  = rng.random((2,)) * 2 * np.pi
        h += amp * (np.sin(xx + phase[0]) * np.cos(yy + phase[1]))
        freq *= 2.0
        amp  *= H
    return h


def generate(out_path: Path, size=512, seed=42):
    h = fbm((size, size), octaves=7, H=0.65, seed=seed)
    # Normalise 0-1
    h = (h - h.min()) / (h.max() - h.min())

    # Flatten the combat arena in the centre (radius = 30% of size)
    cx, cy = size // 2, size // 2
    yy, xx  = np.ogrid[:size, :size]
    dist    = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    r_flat  = size * 0.28
    r_blend = size * 0.38
    blend   = np.clip((dist - r_flat) / (r_blend - r_flat), 0, 1)
    # Keep centre low (altitude ~0.1) so aircraft have room to fly
    flat_h  = 0.08 * np.ones_like(h)
    h       = flat_h * (1 - blend) + h * blend

    # Boost peripheral mountains
    h = np.where(dist > r_blend, h * 1.3, h)
    h = np.clip(h, 0, 1)

    img = Image.fromarray((h * 255).astype(np.uint8), mode='L')
    img.save(str(out_path))
    print(f"Heightmap saved -> {out_path}  ({size}x{size}px)")


if __name__ == "__main__":
    out = Path(__file__).parent / "static" / "assets" / "heightmap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    generate(out)
