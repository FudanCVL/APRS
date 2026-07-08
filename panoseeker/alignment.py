from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from aprs import PanoramicEnvironment, config
from aprs.geometry import clamp_phi, normalize_theta
from aprs.projection import focal_lengths, perspective_to_sphere

logger = logging.getLogger(__name__)


@dataclass
class SegResult:
    mask: Optional[np.ndarray] = None
    refined_theta: float = 0.0
    refined_phi: float = 0.0
    bbox_centroid: Tuple[float, float] = (0.0, 0.0)
    aligned_view: Optional[np.ndarray] = None
    alignment_steps: List[Tuple[float, float, np.ndarray, np.ndarray]] = field(default_factory=list)


class Segmenter:
    def align_and_segment(self, env: PanoramicEnvironment, view: np.ndarray,
                          bbox: Tuple[float, float, float, float],
                          instruction: str) -> SegResult:
        raise NotImplementedError


class StubSegmenter(Segmenter):
    def align_and_segment(self, env: PanoramicEnvironment, view: np.ndarray,
                          bbox: Tuple[float, float, float, float],
                          instruction: str) -> SegResult:
        xmin, ymin, xmax, ymax = bbox
        vh, vw = view.shape[:2]
        cx_norm = ((xmin + xmax) / 2.0) / vw
        cy_norm = ((ymin + ymax) / 2.0) / vh

        theta_t, phi_t = env.orientation
        fh, fv = focal_lengths(config.FOV_H, config.FOV_V, vw, vh)
        cx, cy = cx_norm * vw, cy_norm * vh
        dx = (cx - (vw - 1) / 2.0) / fh
        dy = ((vh - 1) / 2.0 - cy) / fv
        theta_star, phi_star = perspective_to_sphere(
            dx, dy, math.radians(theta_t), math.radians(phi_t)
        )
        theta_star = normalize_theta(np.degrees(theta_star))
        phi_star = clamp_phi(np.degrees(phi_star), config.PITCH_LIMIT)

        dtheta_star = normalize_theta(theta_star - theta_t)
        dphi_star = phi_star - phi_t

        aligned_view = env.step(dtheta_star, dphi_star)
        refined_theta, refined_phi = env.orientation

        return SegResult(
            mask=None,
            refined_theta=refined_theta,
            refined_phi=refined_phi,
            bbox_centroid=(theta_star, phi_star),
            aligned_view=aligned_view,
        )


class SAM3Segmenter(Segmenter):
    def __init__(self, sam3_checkpoint: str = None, device: str = "cuda:0",
                 max_centering_steps: int = 5, centering_threshold: float = 2.0):
        try:
            from sam3.model_builder import build_sam3_camera_predictor
        except (ImportError, ModuleNotFoundError) as e:
            raise ImportError(
                f"SAM3 not found. Please ensure SAM3 is cloned.\n"
                f"Error: {e}\n"
                f"Clone: git clone https://github.com/facebookresearch/sam3"
            )

        self.device = device
        self.max_centering_steps = max_centering_steps
        self.centering_threshold = centering_threshold

        if sam3_checkpoint is None:
            sam3_checkpoint = "weights/sam3/sam3.pt"

        logger.info(f"Loading SAM3 model from {sam3_checkpoint}")
        self.model = build_sam3_camera_predictor(sam3_checkpoint, mode="eval", device=device)
        self.inference_state = None
        logger.info("SAM3 model loaded")

    def align_and_segment(self, env: PanoramicEnvironment, view: np.ndarray,
                          bbox: Tuple[float, float, float, float],
                          instruction: str) -> SegResult:
        vh, vw = view.shape[:2]
        theta_t, phi_t = env.orientation

        xmin_norm, ymin_norm, xmax_norm, ymax_norm = bbox
        logger.debug(f"VLM bbox (0-1000 range): ({xmin_norm}, {ymin_norm}, {xmax_norm}, {ymax_norm})")

        mx = max(xmin_norm, ymin_norm, xmax_norm, ymax_norm)
        if mx <= 1.0:
            xmin = xmin_norm * vw
            ymin = ymin_norm * vh
            xmax = xmax_norm * vw
            ymax = ymax_norm * vh
        elif mx <= 1000:
            xmin = xmin_norm / 1000.0 * vw
            ymin = ymin_norm / 1000.0 * vh
            xmax = xmax_norm / 1000.0 * vw
            ymax = ymax_norm / 1000.0 * vh
        else:
            xmin, ymin, xmax, ymax = xmin_norm, ymin_norm, xmax_norm, ymax_norm

        logger.debug(f"VLM bbox (pixels): ({xmin:.1f}, {ymin:.1f}, {xmax:.1f}, {ymax:.1f})")

        view_rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        h, w = view_rgb.shape[:2]

        logger.debug(f"Loading frame into SAM3: {view_rgb.shape}")
        self.inference_state = self.model.load_first_frame(view_rgb)

        bbox_norm = np.array([[xmin / w, ymin / h, xmax / w, ymax / h]], dtype=np.float32)

        logger.debug(f"Adding box prompt to SAM3: {bbox_norm}")
        _, out_obj_ids, _, out_mask_logits = self.model.add_new_points_or_box(
            self.inference_state, frame_idx=0, obj_id=0, box=bbox_norm
        )

        mask = self._extract_mask(out_mask_logits, h, w)
        logger.debug(f"Initial mask extracted: {mask.shape}, area={mask.sum()}")

        M = cv2.moments(mask)
        if M["m00"] < 1:
            logger.warning("Empty mask, cannot compute centroid")
            return SegResult(
                mask=mask,
                refined_theta=theta_t,
                refined_phi=phi_t,
                bbox_centroid=(theta_t, phi_t),
                aligned_view=view,
                alignment_steps=[(theta_t, phi_t, view.copy())],
            )

        cx_pixel = M["m10"] / M["m00"]
        cy_pixel = M["m01"] / M["m00"]

        fh, fv = focal_lengths(config.FOV_H, config.FOV_V, w, h)
        dx = cx_pixel - (w - 1) / 2.0
        dy = (h - 1) / 2.0 - cy_pixel

        theta_star, phi_star = perspective_to_sphere(
            dx / fh, dy / fv, math.radians(theta_t), math.radians(phi_t)
        )
        theta_star = normalize_theta(np.degrees(theta_star))
        phi_star = clamp_phi(np.degrees(phi_star), config.PITCH_LIMIT)

        dtheta_star = normalize_theta(theta_star - theta_t)
        dphi_star = phi_star - phi_t

        logger.debug(f"Mask centroid: ({cx_pixel:.1f}, {cy_pixel:.1f})")
        logger.debug(f"Target θ*={theta_star:.1f}°, φ*={phi_star:.1f}°")
        logger.debug(f"Angular adjustment: Δθ*={dtheta_star:.1f}°, Δφ*={dphi_star:.1f}°")

        aligned_view = env.step(dtheta_star, dphi_star)
        refined_theta, refined_phi = env.orientation

        aligned_view_rgb = cv2.cvtColor(aligned_view, cv2.COLOR_BGR2RGB)
        out_obj_ids, out_mask_logits = self.model.track(aligned_view_rgb, self.inference_state)
        aligned_mask = self._extract_mask(out_mask_logits, h, w)

        alignment_steps = [(theta_t, phi_t, view.copy(), mask.copy()), (refined_theta, refined_phi, aligned_view.copy(), aligned_mask.copy())]

        final_mask = aligned_mask
        if self.max_centering_steps > 0:
            refined_theta, refined_phi, alignment_steps, final_mask = self._iterative_centering(
                env, aligned_mask, refined_theta, refined_phi, instruction, alignment_steps
            )

        return SegResult(
            mask=final_mask,
            refined_theta=refined_theta,
            refined_phi=refined_phi,
            bbox_centroid=(theta_star, phi_star),
            aligned_view=aligned_view,
            alignment_steps=alignment_steps,
        )

    def _extract_mask(self, mask_logits, h: int, w: int) -> np.ndarray:
        if isinstance(mask_logits, (list, tuple)):
            mask_logits = mask_logits[0]
        if hasattr(mask_logits, 'detach'):
            mask_np = (mask_logits > 0).detach().cpu().numpy()
        else:
            mask_np = np.array(mask_logits) > 0
        mask_np = np.squeeze(mask_np)
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        return (mask_np.astype(np.uint8) * 255).astype(np.uint8)

    def _iterative_centering(self, env: PanoramicEnvironment, mask: np.ndarray,
                             curr_theta: float, curr_phi: float,
                             instruction: str, alignment_steps: List) -> Tuple[float, float, List, np.ndarray]:
        vh, vw = mask.shape
        fh, fv = focal_lengths(config.FOV_H, config.FOV_V, vw, vh)
        for step in range(self.max_centering_steps):
            M = cv2.moments(mask)
            if M["m00"] < 1:
                logger.debug(f"Centering step {step}: Lost object (empty mask)")
                break
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            dx_pixel = cx - vw / 2.0
            dy_pixel = cy - vh / 2.0
            d_theta = (dx_pixel / vw) * config.FOV_H
            d_phi = -(dy_pixel / vh) * config.FOV_V
            logger.debug(f"Centering step {step}: Offset=({dx_pixel:.1f}, {dy_pixel:.1f}) -> dAng=({d_theta:.2f}, {d_phi:.2f})")
            if abs(d_theta) < self.centering_threshold and abs(d_phi) < self.centering_threshold:
                logger.debug(f"Centering step {step}: Converged!")
                break
            view = env.step(d_theta, d_phi)
            curr_theta, curr_phi = env.orientation
            view_rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
            out_obj_ids, out_mask_logits = self.model.track(view_rgb, self.inference_state)
            mask = self._extract_mask(out_mask_logits, vh, vw)
            alignment_steps.append((curr_theta, curr_phi, view.copy(), mask.copy()))
        return curr_theta, curr_phi, alignment_steps, mask
