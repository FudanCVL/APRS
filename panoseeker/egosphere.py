from __future__ import annotations

import sys
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "panoseeker.egosphere requires opencv-python (`pip install opencv-python`)."
    ) from exc

from PIL import Image, ImageDraw, ImageFont

from aprs import config
from aprs.geometry import normalize_theta
from aprs.projection import backproject_view



_BG_COLOR = (28, 30, 35)          # unexplored frontier (RGB)
_GRID_RGBA = (255, 255, 255, 35)  # translucent lat/lon grid
_NODE_FILL = "#FFFFFF"
_INK = "#111111"                  # outlines, ticks, axis labels
_CROSS_RGBA = (225, 29, 72, 255)  # rose crosshair (#E11D48)


def _load_bold_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "Times New Roman Bold.ttf",
        "timesbd.ttf",
        "LiberationSerif-Bold.ttf",
        "DejaVuSans-Bold.ttf",
    ]
    if sys.platform == "win32":
        candidates.insert(0, "timesbd.ttf")
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:  # older Pillow without size support
        return ImageFont.load_default()


class EgoSphere:

    def __init__(self, cfg: config.CanvasConfig = config.CanvasConfig(),
                 obs: config.ObsConfig = config.ObsConfig()):
        self.cfg = cfg
        self.obs = obs
        self.reset()


    def reset(self, init_theta: float = 0.0) -> None:
        h, w = self.cfg.height, self.cfg.width
        self.canvas = np.zeros((h, w, 3), dtype=np.uint8)   # BGR (OpenCV)
        self.explored = np.zeros((h, w), dtype=bool)
        self._theta_anchor = init_theta
        self.trajectory: List[Tuple[float, float]] = []

    def _to_relative(self, theta_abs: float) -> float:
        return normalize_theta(theta_abs - self._theta_anchor)


    def update(self, view: np.ndarray, theta_abs: float, phi: float) -> None:
        theta_rel = self._to_relative(theta_abs)
        rgb, mask = backproject_view(
            view, theta_rel, phi,
            canvas_w=self.cfg.width, canvas_h=self.cfg.height,
            fov_h=self.obs.fov_h, fov_v=self.obs.fov_v)
        self.canvas[mask] = rgb[mask]
        self.explored |= mask
        self.trajectory.append((theta_rel, phi))


    def render_prompt(self, scale: int = 2, draw_grid: bool = True,
                      draw_trajectory: bool = True, draw_crosshair: bool = True,
                      draw_axes: bool = True, target_width: int = 1024
                      ) -> np.ndarray:
        s = max(1, int(scale))
        map_w, map_h = self.cfg.width * s, self.cfg.height * s
        m_l, m_r, m_t, m_b = 65 * s, 30 * s, 30 * s, 45 * s
        out_w, out_h = map_w + m_l + m_r, map_h + m_t + m_b


        rgb = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (map_w, map_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(self.explored.astype(np.uint8), (map_w, map_h),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        map_canvas = np.full((map_h, map_w, 3), _BG_COLOR, dtype=np.uint8)
        map_canvas[mask] = rgb[mask]

        map_img = Image.fromarray(map_canvas).convert("RGBA")
        if draw_grid:
            self._draw_grid(ImageDraw.Draw(map_img), map_w, map_h, s)

        out = Image.new("RGBA", (out_w, out_h), (255, 255, 255, 0))
        out.paste(map_img, (m_l, m_t))
        draw = ImageDraw.Draw(out)
        draw.rectangle([(m_l, m_t), (m_l + map_w, m_t + map_h)],
                       outline=_INK, width=2 * s)

        def to_px(theta_rel: float, phi: float) -> Tuple[int, int]:
            px = int(((theta_rel + 180.0) / 360.0) * map_w) % map_w
            py = int(((90.0 - phi) / 180.0) * map_h)
            return px + m_l, py + m_t

        if draw_trajectory:
            self._draw_trajectory(draw, to_px, s)
        if draw_crosshair and self.trajectory:
            self._draw_crosshair(draw, to_px(*self.trajectory[-1]), s)
        if draw_axes:
            self._draw_axes(draw, map_w, map_h, m_l, m_t, s)


        bbox = out.getbbox()
        if bbox:
            pad = 10 * s
            bbox = (max(0, bbox[0] - pad), max(0, bbox[1] - pad),
                    min(out_w, bbox[2] + pad), min(out_h, bbox[3] + pad))
            out = out.crop(bbox)
        flat = Image.new("RGBA", out.size, (255, 255, 255, 255))
        flat.paste(out, (0, 0), out)
        flat = flat.convert("RGB")
        tw = target_width
        th = int(flat.height * tw / flat.width)
        flat = flat.resize((tw, th), Image.LANCZOS)
        return cv2.cvtColor(np.array(flat), cv2.COLOR_RGB2BGR)


    def _draw_grid(self, d: ImageDraw.ImageDraw, w: int, h: int, s: int) -> None:
        for lat in range(-60, 90, 30):                       # interior parallels
            y = int(((90.0 - lat) / 180.0) * h)
            d.line([(0, y), (w, y)], fill=_GRID_RGBA, width=1 * s)
        for lon in range(-150, 180, 30):                     # interior meridians
            x = int(((lon + 180.0) / 360.0) * w)
            d.line([(x, 0), (x, h)], fill=_GRID_RGBA, width=1 * s)

    def _draw_trajectory(self, d: ImageDraw.ImageDraw, to_px, s: int) -> None:
        r = 20 * s
        font = _load_bold_font(18 * s)
        for i, (theta_rel, phi) in enumerate(self.trajectory, 1):
            x, y = to_px(theta_rel, phi)
            d.ellipse([x - r, y - r, x + r, y + r],
                      fill=_NODE_FILL, outline=_INK, width=2 * s)
            d.text((x, y - s), str(i), fill=_INK, font=font, anchor="mm")

    def _draw_crosshair(self, d: ImageDraw.ImageDraw, center: Tuple[int, int],
                        s: int) -> None:
        cx, cy = center
        arm, w = 18 * s, 2 * s
        d.line([(cx - arm, cy), (cx + arm, cy)], fill=_CROSS_RGBA, width=w)
        d.line([(cx, cy - arm), (cx, cy + arm)], fill=_CROSS_RGBA, width=w)
        d.ellipse([cx - arm - 6 * s, cy - arm - 6 * s,
                   cx + arm + 6 * s, cy + arm + 6 * s], outline=_CROSS_RGBA, width=w)

    def _draw_axes(self, d: ImageDraw.ImageDraw, w: int, h: int,
                   m_l: int, m_t: int, s: int) -> None:
        font = _load_bold_font(18 * s)
        for lat in range(-90, 91, 30):                       # latitude labels (left)
            y = m_t + int(((90.0 - lat) / 180.0) * h)
            d.text((m_l - 10 * s, y), f"{lat}°", fill=_INK, font=font, anchor="rm")
            d.line([(m_l - 6 * s, y), (m_l, y)], fill=_INK, width=2 * s)
        for lon in range(-180, 181, 60):                     # longitude labels (bottom)
            x = m_l + int(((lon + 180.0) / 360.0) * w)
            d.text((x, m_t + h + 12 * s), f"{lon}°", fill=_INK, font=font, anchor="ma")
            d.line([(x, m_t + h), (x, m_t + h + 6 * s)], fill=_INK, width=2 * s)


    @property
    def coverage(self) -> float:
        return float(self.explored.mean())
