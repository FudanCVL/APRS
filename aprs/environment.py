from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "aprs.environment requires opencv-python (`pip install opencv-python`)."
    ) from exc

from . import config
from .geometry import clamp_phi, geodesic_distance, normalize_theta
from .projection import render_perspective


@dataclass
class EnvState:

    theta: float
    phi: float
    step_count: int = 0
    done: bool = False
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    path_length: float = 0.0     # cumulative geodesic distance travelled (radians)


class PanoramicEnvironment:

    def __init__(self, image: "np.ndarray | str",
                 obs: config.ObsConfig = config.ObsConfig(),
                 target: Optional[Tuple[float, float]] = None):
        if isinstance(image, str):
            erp = cv2.imread(image, cv2.IMREAD_COLOR)
            if erp is None:
                raise FileNotFoundError(f"Could not read panorama: {image}")
            self.erp = erp
        else:
            self.erp = image
        self.obs = obs
        self.target = target
        self.state: Optional[EnvState] = None


    def reset(self, init_theta: float = 0.0, init_phi: float = 0.0) -> np.ndarray:
        theta = normalize_theta(init_theta)
        phi = clamp_phi(init_phi, config.PITCH_LIMIT)
        self.state = EnvState(theta=theta, phi=phi, trajectory=[(theta, phi)])
        return self.observe()

    def observe(self) -> np.ndarray:
        self._require_state()
        return render_perspective(
            self.erp, self.state.theta, self.state.phi,
            fov_h=self.obs.fov_h, fov_v=self.obs.fov_v,
            out_w=self.obs.width, out_h=self.obs.height)

    def step(self, dtheta: float, dphi: float) -> np.ndarray:
        self._require_state()
        prev = (self.state.theta, self.state.phi)
        self.state.theta = normalize_theta(self.state.theta + dtheta)
        self.state.phi = clamp_phi(self.state.phi + dphi, config.PITCH_LIMIT)
        self.state.step_count += 1
        self.state.trajectory.append((self.state.theta, self.state.phi))
        self.state.path_length += geodesic_distance(
            prev[0], prev[1], self.state.theta, self.state.phi)
        return self.observe()

    def stop(self) -> None:
        self._require_state()
        self.state.done = True


    @property
    def orientation(self) -> Tuple[float, float]:
        self._require_state()
        return self.state.theta, self.state.phi

    def distance_to_target(self) -> Optional[float]:
        if self.target is None:
            return None
        self._require_state()
        return geodesic_distance(self.state.theta, self.state.phi, *self.target)

    def is_target_centered(self, tau: float = config.SUCCESS_TAU_RAD) -> Optional[bool]:
        d = self.distance_to_target()
        return None if d is None else d < tau

    def _require_state(self) -> None:
        if self.state is None:
            raise RuntimeError("Call reset() before interacting with the environment.")
