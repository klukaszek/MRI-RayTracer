from __future__ import annotations

import math
from typing import Optional
import numpy as np


class OrbitalCamera:
    def __init__(
        self,
        initial_target: Optional[np.ndarray] = None,
        initial_radius: float = 2.0,
        initial_phi: float = math.pi * 0.5,
        initial_theta: float = 0.0,
        min_radius: float = 0.1,
        max_radius: float = 100.0,
        min_phi: float = 0.01,
        max_phi: float = math.pi - 0.01,
        aspect: float = 16.0 / 9.0,
        fovY_radians: float = math.radians(55.0),
        near: float = 0.1,
        far: float = 1000.0,
        world_up: Optional[np.ndarray] = None,
    ):
        self._initial_target = (initial_target.astype(np.float32) if initial_target is not None
                                else np.array([0.0, 0.0, 0.0], dtype=np.float32))
        self._initial_radius = float(initial_radius)
        self._initial_phi = float(initial_phi)
        self._initial_theta = float(initial_theta)
        self._initial_min_radius = float(min_radius)
        self._initial_max_radius = float(max_radius)
        self._initial_min_phi = float(min_phi)
        self._initial_max_phi = float(max_phi)

        # Apply initial state
        self.target = self._initial_target.copy()
        self.radius = self._initial_radius
        self.phi = self._initial_phi
        self.theta = self._initial_theta
        self.min_radius = self._initial_min_radius
        self.max_radius = self._initial_max_radius
        self.min_phi = self._initial_min_phi
        self.max_phi = self._initial_max_phi

        self.fovY_radians = float(fovY_radians)
        self.aspect = float(aspect)
        self.near = float(near)
        self.far = float(far)
        self.world_up = (world_up.astype(np.float32) if world_up is not None
                         else np.array([0.0, 1.0, 0.0], dtype=np.float32))

    def reset(self):
        self.target = self._initial_target.copy()
        self.radius = self._initial_radius
        self.phi = self._initial_phi
        self.theta = self._initial_theta
        self.min_radius = self._initial_min_radius
        self.max_radius = self._initial_max_radius
        self.min_phi = self._initial_min_phi
        self.max_phi = self._initial_max_phi

    def _base_frame(self):
        wu = self.world_up
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(wu, ref))) > 0.999:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        r = np.cross(ref, wu)
        rn = float(np.linalg.norm(r))
        if rn < 1e-6:
            r = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            rn = 1.0
        r = (r / rn).astype(np.float32)
        f = np.cross(wu, r).astype(np.float32)
        fn = float(np.linalg.norm(f))
        if fn > 0:
            f = (f / fn).astype(np.float32)
        return r, f, wu

    def get_eye_position(self) -> np.ndarray:
        r, f, u = self._base_frame()
        s = math.sin(self.phi)
        c = math.cos(self.phi)
        dir_vec = (s * math.cos(self.theta)) * r + (s * math.sin(self.theta)) * f + c * u
        eye = self.target + self.radius * dir_vec.astype(np.float32)
        return eye.astype(np.float32)

    def get_basis(self):
        eye = self.get_eye_position()
        forward = (self.target - eye)
        fn = float(np.linalg.norm(forward))
        if fn < 1e-6:
            forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            forward = (forward / fn).astype(np.float32)
        wu = self.world_up
        right = np.cross(forward, wu)
        rn = float(np.linalg.norm(right))
        if rn < 1e-6:
            right, _, _ = self._base_frame()
            rn = float(np.linalg.norm(right))
        if rn > 0:
            right = (right / rn).astype(np.float32)
        up = np.cross(right, forward).astype(np.float32)
        if float(np.dot(up, wu)) < 0.0:
            up = -up
            right = -right
        return eye.astype(np.float32), right, up, forward

    def orbit(self, d_theta: float, d_phi: float):
        self.theta += float(d_theta)
        self.phi = max(self.min_phi, min(self.max_phi, self.phi + float(d_phi)))

    def pan(self, dx: float, dy: float, viewport_height: float | None = None):
        eye, right, up, _ = self.get_basis()
        pixels = float(viewport_height) if (viewport_height is not None and viewport_height > 0) else 720.0
        view_height_world = 2.0 * self.radius * math.tan(max(1e-3, self.fovY_radians * 0.5))
        px_to_world = view_height_world / max(1.0, pixels)
        self.target = (self.target
                       - right * (float(dx) * px_to_world)
                       + up * (float(dy) * px_to_world)).astype(np.float32)

    def zoom(self, factor: float):
        self.radius = max(self.min_radius, min(self.max_radius, self.radius * float(factor)))

    def set_fov_degrees(self, fov_deg: float):
        self.fovY_radians = math.radians(float(fov_deg))

    def set_aspect(self, aspect: float):
        self.aspect = float(aspect)

