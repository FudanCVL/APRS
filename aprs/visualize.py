from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "aprs.visualize requires opencv-python (`pip install opencv-python`)."
    ) from exc

from .geometry import sphere_to_erp_px


def draw_box_on_erp(erp: np.ndarray, box_norm: Tuple[float, float, float, float],
                    color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 3
                    ) -> np.ndarray:
    img = erp.copy()
    h, w = img.shape[:2]
    x, y, bw, bh = box_norm
    p0 = (int(x * w), int(y * h))
    p1 = (int((x + bw) * w), int((y + bh) * h))
    cv2.rectangle(img, p0, p1, color, thickness)
    return img


def mark_orientations(erp: np.ndarray,
                      points: Sequence[Tuple[float, float, Tuple[int, int, int], str]],
                      radius: int = 10) -> np.ndarray:
    img = erp.copy()
    h, w = img.shape[:2]
    for theta, phi, color, label in points:
        u, v = sphere_to_erp_px(theta, phi, w, h)
        u, v = int(u), int(v)
        cv2.circle(img, (u, v), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (u, v), radius + 3, (255, 255, 255), 2, cv2.LINE_AA)
        if label:
            cv2.putText(img, label, (u + radius + 4, v),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return img


def make_grid(images: Sequence[np.ndarray], titles: Optional[Sequence[str]] = None,
              cols: int = 2, cell_w: int = 640, pad: int = 8,
              bg: int = 30) -> np.ndarray:
    def _fit(im: np.ndarray) -> np.ndarray:
        h, w = im.shape[:2]
        scale = cell_w / w
        return cv2.resize(im, (cell_w, max(1, int(h * scale))))

    cells = [_fit(im) for im in images]
    cell_h = max(c.shape[0] for c in cells)
    rows = (len(cells) + cols - 1) // cols
    header = 26 if titles else 0
    canvas = np.full((rows * (cell_h + header) + pad, cols * (cell_w + pad) + pad, 3),
                     bg, dtype=np.uint8)
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        y0 = r * (cell_h + header) + pad + header
        x0 = c * (cell_w + pad) + pad
        canvas[y0:y0 + cell.shape[0], x0:x0 + cell.shape[1]] = cell
        if titles and i < len(titles):
            cv2.putText(canvas, titles[i], (x0, y0 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    return canvas
