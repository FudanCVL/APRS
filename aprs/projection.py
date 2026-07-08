from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency hint
    raise ImportError(
        "aprs.projection requires opencv-python (`pip install opencv-python`)."
    ) from exc

from . import config



# Low-level, pixel-free primitives (all angles in radians).                    #

def focal_lengths(fov_h_deg: float, fov_v_deg: float, width: int, height: int
                  ) -> Tuple[float, float]:
    fh = width / (2.0 * np.tan(np.radians(fov_h_deg) / 2.0))
    fv = height / (2.0 * np.tan(np.radians(fov_v_deg) / 2.0))
    return fh, fv


def perspective_to_sphere(dx, dy, theta0_rad, phi0_rad):
    dx = np.asarray(dx, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    rho = np.sqrt(dx * dx + dy * dy)
    c = np.arctan(rho)
    sin_c, cos_c = np.sin(c), np.cos(c)

    # Guard rho == 0 (the tangent point itself) to avoid 0/0.
    safe = np.where(rho == 0.0, 1.0, rho)
    sin_phi = cos_c * np.sin(phi0_rad) + np.where(
        rho == 0.0, 0.0, dy * sin_c * np.cos(phi0_rad) / safe)
    phi = np.arcsin(np.clip(sin_phi, -1.0, 1.0))

    num = dx * sin_c
    den = rho * np.cos(phi0_rad) * cos_c - dy * np.sin(phi0_rad) * sin_c
    theta = theta0_rad + np.arctan2(num, den)
    return theta, phi


def sphere_to_perspective(theta_rad, phi_rad, theta0_rad, phi0_rad):
    theta_rad = np.asarray(theta_rad, dtype=np.float64)
    phi_rad = np.asarray(phi_rad, dtype=np.float64)
    dtheta = theta_rad - theta0_rad
    cos_c = (np.sin(phi0_rad) * np.sin(phi_rad)
             + np.cos(phi0_rad) * np.cos(phi_rad) * np.cos(dtheta))
    safe = np.where(cos_c == 0.0, 1e-12, cos_c)
    dx = np.cos(phi_rad) * np.sin(dtheta) / safe
    dy = (np.cos(phi0_rad) * np.sin(phi_rad)
          - np.sin(phi0_rad) * np.cos(phi_rad) * np.cos(dtheta)) / safe
    return dx, dy, cos_c



# Image-level forward / inverse projection.                                    #

def render_perspective(erp: np.ndarray, theta0: float, phi0: float,
                       fov_h: float = config.FOV_H, fov_v: float = config.FOV_V,
                       out_w: int = config.OBS_W, out_h: int = config.OBS_H,
                       interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    H, W = erp.shape[:2]
    th0, ph0 = np.radians(theta0), np.radians(phi0)
    fh, fv = focal_lengths(fov_h, fov_v, out_w, out_h)

    xs = np.arange(out_w, dtype=np.float64)
    ys = np.arange(out_h, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    cx, cy = (out_w - 1) / 2.0, (out_h - 1) / 2.0
    dx = (xx - cx) / fh
    dy = (cy - yy) / fv          # image y grows downward -> flip so dy points up

    theta, phi = perspective_to_sphere(dx, dy, th0, ph0)

    # Map spherical coordinates to source ERP pixels.
    u = (theta + np.pi) / (2.0 * np.pi) * W
    v = (np.pi / 2.0 - phi) / np.pi * H
    map_x = u.astype(np.float32)
    map_y = np.clip(v, 0, H - 1).astype(np.float32)

    return cv2.remap(erp, map_x, map_y, interpolation,
                     borderMode=cv2.BORDER_WRAP)


def backproject_view(view: np.ndarray, theta0: float, phi0: float,
                     canvas_w: int = config.CANVAS_W, canvas_h: int = config.CANVAS_H,
                     fov_h: float = config.FOV_H, fov_v: float = config.FOV_V,
                     interpolation: int = cv2.INTER_LINEAR
                     ) -> Tuple[np.ndarray, np.ndarray]:
    vh, vw = view.shape[:2]
    th0, ph0 = np.radians(theta0), np.radians(phi0)
    fh, fv = focal_lengths(fov_h, fov_v, vw, vh)
    tan_h = np.tan(np.radians(fov_h) / 2.0)
    tan_v = np.tan(np.radians(fov_v) / 2.0)

    js = np.arange(canvas_w, dtype=np.float64)
    is_ = np.arange(canvas_h, dtype=np.float64)
    xx, yy = np.meshgrid(js, is_)
    theta = (xx + 0.5) / canvas_w * 2.0 * np.pi - np.pi
    phi = np.pi / 2.0 - (yy + 0.5) / canvas_h * np.pi

    dx, dy, cos_c = sphere_to_perspective(theta, phi, th0, ph0)
    valid = (cos_c > 0) & (np.abs(dx) <= tan_h) & (np.abs(dy) <= tan_v)

    cx, cy = (vw - 1) / 2.0, (vh - 1) / 2.0
    px = dx * fh + cx
    py = cy - dy * fv            # flip dy (up) back to image y (down)
    # Only sample inside the view; clamp elsewhere and drop via the mask.
    map_x = np.where(valid, px, 0).astype(np.float32)
    map_y = np.where(valid, py, 0).astype(np.float32)

    sampled = cv2.remap(view, map_x, map_y, interpolation,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = valid
    rgb = np.where(mask[..., None], sampled, 0).astype(view.dtype)
    return rgb, mask
