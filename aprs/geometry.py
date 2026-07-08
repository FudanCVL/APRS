from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def normalize_theta(theta: float) -> float:
    return (theta + 180.0) % 360.0 - 180.0


def clamp_phi(phi: float, limit: float = 89.9) -> float:
    return max(-limit, min(limit, phi))


def pixel_to_sphere(xn: float, yn: float) -> Tuple[float, float]:
    return 360.0 * xn - 180.0, 90.0 - 180.0 * yn


def sphere_to_pixel(theta: float, phi: float) -> Tuple[float, float]:
    xn = (normalize_theta(theta) + 180.0) / 360.0
    yn = (90.0 - phi) / 180.0
    return xn, yn


def sphere_to_erp_px(theta, phi, width: int, height: int):
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    xn = ((theta + 180.0) % 360.0) / 360.0
    yn = np.clip((90.0 - phi) / 180.0, 0.0, 1.0)
    u = np.clip(xn * width, 0, width - 1).astype(np.int32)
    v = np.clip(yn * height, 0, height - 1).astype(np.int32)
    return u, v


def geodesic_distance(theta1, phi1, theta2, phi2) -> float:
    p1, p2 = math.radians(phi1), math.radians(phi2)
    dl = math.radians(theta1 - theta2)
    cos_d = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)
    return math.acos(max(-1.0, min(1.0, cos_d)))
