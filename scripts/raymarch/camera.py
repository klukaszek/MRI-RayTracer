from __future__ import annotations

import math
from typing import Optional
import numpy as np


# Orbital camera matching the TypeScript shape (simplified for this project)
class OrbitalCamera:
    def __init__(
        self,
        initial_target: Optional[np.ndarray] = None,
        initial_radius: float = 2.0,
        initial_phi: float = math.pi * 0.5,   # from +Y
        initial_theta: float = 0.0,           # around Y in XZ plane
        min_radius: float = 0.1,
        max_radius: float = 100.0,
        min_phi: float = 0.01,
        max_phi: float = math.pi - 0.01,
        aspect: float = 16.0 / 9.0,
        fovY_radians: float = math.radians(55.0),
        near: float = 0.1,
        far: float = 1000.0,
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

        # Projection params (kept for completeness)
        self.fovY_radians = float(fovY_radians)
        self.aspect = float(aspect)
        self.near = float(near)
        self.far = float(far)

    def reset(self):
        self.target = self._initial_target.copy()
        self.radius = self._initial_radius
        self.phi = self._initial_phi
        self.theta = self._initial_theta
        self.min_radius = self._initial_min_radius
        self.max_radius = self._initial_max_radius
        self.min_phi = self._initial_min_phi
        self.max_phi = self._initial_max_phi

    def get_eye_position(self) -> np.ndarray:
        # Spherical coordinates relative to target
        s = math.sin(self.phi)
        c = math.cos(self.phi)
        x = self.target[0] + self.radius * s * math.cos(self.theta)
        y = self.target[1] + self.radius * c
        z = self.target[2] + self.radius * s * math.sin(self.theta)
        return np.array([x, y, z], dtype=np.float32)

    def get_basis(self):
        eye = self.get_eye_position()
        forward = (self.target - eye)
        fn = float(np.linalg.norm(forward))
        if fn < 1e-6:
            forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            forward = (forward / fn).astype(np.float32)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        rn = float(np.linalg.norm(right))
        if rn < 1e-6:
            alt = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            right = np.cross(forward, alt)
            rn = float(np.linalg.norm(right))
        if rn > 0:
            right = (right / rn).astype(np.float32)
        up = np.cross(right, forward).astype(np.float32)
        return eye.astype(np.float32), right, up, forward

    # Controls
    def orbit(self, d_theta: float, d_phi: float):
        self.theta += float(d_theta)
        self.phi = max(self.min_phi, min(self.max_phi, self.phi + float(d_phi)))

    def pan(self, dx: float, dy: float):
        # Pan in screen-space using current basis; scale by radius and FOV
        eye, right, up, forward = self.get_basis()
        # Convert pixel delta to world units: assume dy positive is down in screen coords
        # Use a conservative scale: fraction of view height at current radius.
        pixels = max(1.0, 720.0)  # nominal height to normalize if real viewport not provided
        view_height_world = 2.0 * self.radius * math.tan(max(1e-3, self.fovY_radians * 0.5))
        px_to_world = view_height_world / pixels
        self.target = (self.target
                       - right * (float(dx) * px_to_world)
                       + up * (float(dy) * px_to_world)).astype(np.float32)

    def zoom(self, factor: float):
        self.radius = max(self.min_radius, min(self.max_radius, self.radius * float(factor)))

    def set_fov_degrees(self, fov_deg: float):
        self.fovY_radians = math.radians(float(fov_deg))

    def set_aspect(self, aspect: float):
        self.aspect = float(aspect)
