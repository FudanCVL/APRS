from dataclasses import dataclass

FOV_H = 120.0          # horizontal field of view (degrees)
FOV_V = 90.0           # vertical field of view (degrees)
OBS_W = 1024           # observation width  (pixels)
OBS_H = 768            # observation height (pixels)

CANVAS_W = 1024
CANVAS_H = 512


# Horizontal steps (degrees): Small / Medium / Large / U-turn.
H_ACTIONS = {"small": 30.0, "medium": 60.0, "large": 120.0, "uturn": 180.0}
# Vertical steps (degrees): Small / Medium / Large.
V_ACTIONS = {"small": 30.0, "medium": 60.0, "large": 90.0}

PITCH_LIMIT = 89.9

SUCCESS_TAU_RAD = 0.3


@dataclass(frozen=True)
class ObsConfig:

    fov_h: float = FOV_H
    fov_v: float = FOV_V
    width: int = OBS_W
    height: int = OBS_H


@dataclass(frozen=True)
class CanvasConfig:

    width: int = CANVAS_W
    height: int = CANVAS_H
    grid_step: float = 30.0
