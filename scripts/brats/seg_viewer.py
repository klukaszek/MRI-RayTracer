from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import nibabel as nib
import slangpy as spy

# Reuse the VDB viewer's orbital camera for consistent controls
import sys as _sys
_HERE = Path(__file__).parent
_sys.path.append((_HERE / "vdb_viewer").as_posix())
from camera import OrbitalCamera  # type: ignore


def load_seg_nii(seg_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = nib.load(seg_path.as_posix())
    data = img.get_fdata(dtype=np.float32)
    # Map to label integers 0..N (BraTS labels are typically 0,1,2,4)
    labels = np.rint(data).astype(np.uint32)

    # NIfTI uses LPS/RAS conventions; we’ll just use voxel spacing for aspect and center in world.
    hdr = img.header
    zooms = np.array(hdr.get_zooms()[:3], dtype=np.float32)
    # Reorder to X-fastest layout for shader: buffer index = x + y*dimX + z*dimX*dimY
    # Nibabel returns array typically in (X,Y,Z) order but with Fortran strides; make C-contiguous x-fastest.
    # Ensure shape is (X,Y,Z), then flatten C-order.
    if labels.ndim != 3:
        raise RuntimeError(f"Expected 3D labels, got shape {labels.shape}")
    dimx, dimy, dimz = labels.shape
    linear = np.ascontiguousarray(labels.transpose(2, 1, 0).reshape(-1))
    return linear, np.array([dimx, dimy, dimz], dtype=np.uint32), zooms


class SegViewer:
    def __init__(self, seg_path: Path, up: str = "Y"):
        self.window = spy.Window(title="BraTS Seg Viewer (Slang)", resizable=True)
        self.device = spy.Device(enable_debug_layers=True, compiler_options={"include_paths": [Path(__file__).parent]})
        self.surface = self.device.create_surface(self.window)
        self.surface.configure(width=self.window.width, height=self.window.height)
        self.ui = spy.ui.Context(self.device)

        program = self.device.load_program(str(Path(__file__).parent / "seg_rt.slang"), ["seg_main"])
        self.kernel = self.device.create_compute_kernel(program)

        self.output_texture = None
        self.label_buffer = None
        self.vol_dims = None
        self.voxel_size = None
        self.vol_min = None

        # Camera state (match VDB viewer behavior)
        self.fov_deg = 70.0
        up_map = {
            "X": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "Y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "Z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "-X": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            "-Y": np.array([0.0, -1.0, 0.0], dtype=np.float32),
            "-Z": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        }
        world_up = up_map.get(up.upper(), up_map["Y"]) if isinstance(up, str) else up_map["Y"]
        self.camera = OrbitalCamera(
            initial_radius=3.0,
            initial_phi=math.radians(50),
            initial_theta=math.radians(25),
            world_up=world_up,
        )
        try:
            self.camera.set_fov_degrees(self.fov_deg)
        except Exception:
            pass

        # UI params
        self.bg_color = np.array([0.05, 0.06, 0.08], dtype=np.float32)
        self.step_size = 0.5
        self.use_grayscale = False

        # BraTS label LUT (0..7), with alpha in w
        # Standard label semantics:
        #   0: Background
        #   1: NCR/NET (non‑enhancing/necrotic core)
        #   2: Edema (ED)
        #   4: Enhancing tumor (ET)
        # Common visualization: 1=blue, 2=green, 4=red
        self.lut = np.zeros((8, 4), dtype=np.float32)
        self.lut[0] = [0.0, 0.0, 0.0, 0.0]   # background transparent
        self.lut[1] = [0.0, 0.4, 1.0, 0.9]   # blue: NCR/NET
        self.lut[2] = [0.0, 0.8, 0.0, 0.7]   # green: edema
        self.lut[3] = [0.0, 0.0, 0.0, 0.0]   # unused
        self.lut[4] = [1.0, 0.1, 0.1, 0.9]   # red: enhancing
        # Ensure contiguous for param upload
        self.lut = np.ascontiguousarray(self.lut, dtype=np.float32)

        # Mouse/UI interaction state (mirror VDB pattern)
        self._lmb = False
        self._last = None
        self._press_pos = None
        self._orbit_dragging = False
        self._pan_dragging = False
        self.drag_deadzone_px = 6.0
        self.pan_speed = 0.2

        # Build UI
        self._init_ui()

        # Load segmentation
        self.load_seg(seg_path)

        # Bind events
        self.window.on_mouse_event = self.on_mouse
        self.window.on_keyboard_event = self.on_keyboard
        self.window.on_resize = self.on_resize

    def _init_ui(self):
        screen = self.ui.screen
        window = spy.ui.Window(screen, "Seg Settings", size=spy.float2(400, 260))
        # Volume group
        self.volume_group = spy.ui.Group(window, "Volume")
        self.info = spy.ui.Text(self.volume_group, "(no volume)")
        spy.ui.Button(self.volume_group, "Load Seg...", callback=self.on_click_load)
        spy.ui.Button(self.volume_group, "Frame Volume", callback=self.on_click_frame)

        # Controls
        self.controls_group = spy.ui.Group(window, "Controls")
        self.fov_slider = spy.ui.SliderFloat(self.controls_group, "FOV (deg)", value=self.fov_deg, min=20.0, max=100.0)
        self.step_slider = spy.ui.SliderFloat(self.controls_group, "Step", value=self.step_size, min=1e-4, max=2.0)
        self.pan_slider = spy.ui.SliderFloat(self.controls_group, "Pan Speed", value=self.pan_speed, min=0.05, max=2.0)
        self.gray_checkbox = spy.ui.CheckBox(self.controls_group, "Greyscale labels", value=self.use_grayscale)
        spy.ui.Text(self.controls_group, "Orbit: LMB | Pan: Shift+LMB | Zoom: Wheel")

    def _create_uint_buffer(self, linear: np.ndarray):
        for kwargs in (
            dict(size=linear.nbytes, usage=spy.BufferUsage.shader_resource),
            dict(element_count=linear.size, element_size=4, usage=spy.BufferUsage.shader_resource),
            dict(element_count=linear.size, struct_size=4, usage=spy.BufferUsage.shader_resource),
            dict(element_count=linear.size, usage=spy.BufferUsage.shader_resource),
        ):
            try:
                return self.device.create_buffer(**kwargs)
            except TypeError:
                continue
        return self.device.create_buffer(linear.size)

    def load_seg(self, seg_path: Path):
        linear, dims, voxel = load_seg_nii(seg_path)
        self.vol_dims = dims.astype(np.uint32)
        # Normalize to unit-ish cube
        max_dim = float(max(int(dims[0]), int(dims[1]), int(dims[2])))
        scale = np.float32(1.8 / max_dim)
        self.voxel_size = (voxel.astype(np.float32) * scale).astype(np.float32)
        extent = self.voxel_size * self.vol_dims.astype(np.float32)
        self.vol_min = -0.5 * extent
        # Upload
        self.label_buffer = self._create_uint_buffer(linear)
        self.label_buffer.copy_from_numpy(linear)
        # Step based on voxel size
        vmin = float(np.min(self.voxel_size))
        self.step_size = 0.75 * vmin
        try:
            self.step_slider.min = max(1e-6, 0.05 * vmin)
            self.step_slider.max = max(self.step_slider.min * 2.0, 5.0 * vmin)
            self.step_slider.value = self.step_size
        except Exception:
            pass
        self.info.text = (
            f"File: {seg_path.name}\n"
            f"Dims: {int(dims[0])} x {int(dims[1])} x {int(dims[2])}\n"
            f"Voxel: {self.voxel_size[0]:.4g}, {self.voxel_size[1]:.4g}, {self.voxel_size[2]:.4g}"
        )

    def on_click_load(self):
        try:
            path = spy.platform.open_file_dialog()
        except Exception as e:
            self.info.text = f"Open dialog error: {e}"
            return
        if path:
            p = Path(path)
            if p.suffixes[-2:] == [".nii", ".gz"] or p.suffix.lower() == ".nii":
                self.load_seg(p)
                self.frame_volume()
            else:
                self.info.text = f"Not a NIfTI file: {p.name}"

    def on_click_frame(self):
        self.frame_volume()

    def frame_volume(self):
        if self.vol_dims is None or self.voxel_size is None:
            return
        extent = self.voxel_size * self.vol_dims.astype(np.float32)
        self.vol_min = -0.5 * extent
        center = (self.vol_min + 0.5 * extent).astype(np.float32)
        self.camera.target = center
        diag = float(np.linalg.norm(extent))
        if diag > 0:
            self.camera.radius = max(self.camera.min_radius, 0.6 * diag)

    def on_keyboard(self, event: spy.KeyboardEvent):
        if self.ui.handle_keyboard_event(event):
            return
        if event.type == spy.KeyboardEventType.key_press and event.key == spy.KeyCode.escape:
            self.window.close()

    def on_mouse(self, event: spy.MouseEvent):
        # Keep camera FOV in sync with UI for consistent pan speed
        try:
            self.camera.set_fov_degrees(float(self.fov_slider.value))
        except Exception:
            pass
        if self.ui.handle_mouse_event(event):
            return
        if event.type == spy.MouseEventType.button_down and event.button == spy.MouseButton.left:
            self._lmb = True
            self._last = event.pos
            self._press_pos = event.pos
            self._orbit_dragging = False
        elif event.type == spy.MouseEventType.button_up and event.button == spy.MouseButton.left:
            self._lmb = False
            self._orbit_dragging = False
            self._pan_dragging = False
        elif event.type == spy.MouseEventType.move:
            # Shift + drag = pan, LMB drag = orbit
            is_shift = False
            try:
                is_shift = event.has_modifier(spy.KeyModifier.shift) if hasattr(event, 'has_modifier') else False
            except Exception:
                is_shift = False
            if self._pan_dragging and not is_shift:
                self._pan_dragging = False
                self._orbit_dragging = False
                self._press_pos = event.pos
                self._last = event.pos
                return
            if is_shift and getattr(self, "_lmb", False):
                if not self._pan_dragging and self._press_pos is not None:
                    pdx = float(event.pos.x - self._press_pos.x)
                    pdy = float(event.pos.y - self._press_pos.y)
                    if (pdx * pdx + pdy * pdy) >= (self.drag_deadzone_px * self.drag_deadzone_px):
                        self._pan_dragging = True
                        self._last = event.pos
                if self._pan_dragging and self._last is not None:
                    dx = float(event.pos.x - self._last.x)
                    dy = float(event.pos.y - self._last.y)
                    speed = float(self.pan_slider.value) if hasattr(self, 'pan_slider') else self.pan_speed
                    self.camera.pan(dx * speed, dy * speed, viewport_height=float(self.window.height))
                    self._last = event.pos
                self._orbit_dragging = False
            elif getattr(self, "_lmb", False):
                if not self._orbit_dragging and self._press_pos is not None:
                    pdx = float(event.pos.x - self._press_pos.x)
                    pdy = float(event.pos.y - self._press_pos.y)
                    if (pdx * pdx + pdy * pdy) >= (self.drag_deadzone_px * self.drag_deadzone_px):
                        self._orbit_dragging = True
                        self._last = event.pos
                if self._orbit_dragging and self._last is not None:
                    dx = float(event.pos.x - self._last.x)
                    dy = float(event.pos.y - self._last.y)
                    self.camera.orbit(dx * 0.01, dy * 0.01)
                    self._last = event.pos
        elif hasattr(event, "is_scroll") and event.is_scroll():
            sy = float(event.scroll.y)
            self.camera.zoom(pow(1.15, -sy))

    def run(self):
        timer = spy.Timer()
        while not self.window.should_close():
            self.window.process_events()
            if not self.surface.config:
                continue
            surface_tex = self.surface.acquire_next_image()
            if not surface_tex:
                continue
            if (
                self.output_texture is None
                or self.output_texture.width != surface_tex.width
                or self.output_texture.height != surface_tex.height
            ):
                self.output_texture = self.device.create_texture(
                    format=spy.Format.rgba16_float,
                    width=surface_tex.width,
                    height=surface_tex.height,
                    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                    label="seg_output",
                )

            self.fov_deg = float(self.fov_slider.value)
            self.step_size = float(self.step_slider.value)
            # SlangPy UI CheckBox exposes .value in newer builds; fall back to .checked if present
            try:
                self.use_grayscale = bool(self.gray_checkbox.value)
            except Exception:
                self.use_grayscale = bool(getattr(self.gray_checkbox, 'checked', False))

            if self.label_buffer is not None:
                # Camera basis from OrbitalCamera
                eye, gU, gV, gW = self.camera.get_basis()
                # Build LUT for this frame (optionally greyscale for labels > 0)
                if self.use_grayscale:
                    lut_use = np.zeros_like(self.lut)
                    lut_use[0] = [0.0, 0.0, 0.0, 0.0]
                    lut_use[1:] = [1.0, 1.0, 1.0, 0.9]
                else:
                    lut_use = self.lut

                params = {
                    "imageSize": (np.uint32(self.output_texture.width), np.uint32(self.output_texture.height)),
                    "fovY": np.float32(math.radians(self.fov_deg)),
                    "eye": eye.astype(np.float32),
                    "U": gU.astype(np.float32),
                    "V": gV.astype(np.float32),
                    "W": gW.astype(np.float32),
                    "volMin": (self.vol_min.astype(np.float32) if isinstance(self.vol_min, np.ndarray) else np.array([0,0,0], dtype=np.float32)),
                    "voxelSize": self.voxel_size.astype(np.float32),
                    "dims": self.vol_dims.astype(np.uint32),
                    "stepSize": np.float32(self.step_size),
                    "bgColor": self.bg_color.astype(np.float32),
                    # Provide as a Python list-of-4-floats per element (length 8)
                    "lutColorAlpha": [tuple(map(float, row)) for row in lut_use.tolist()],
                }
                ce = self.device.create_command_encoder()
                self.kernel.dispatch(
                    thread_count=[self.output_texture.width, self.output_texture.height, 1],
                    vars={
                        "gOutput": self.output_texture,
                        "gLabels": self.label_buffer,
                        "gParams": params,
                    },
                    command_encoder=ce,
                )
                ce.blit(surface_tex, self.output_texture)
                self.ui.begin_frame(surface_tex.width, surface_tex.height)
                self.ui.end_frame(surface_tex, ce)
                self.device.submit_command_buffer(ce.finish())
            self.surface.present()
            del surface_tex

    def on_resize(self, width: int, height: int):
        # Keep surface in sync with window pixel size (avoids DPI mismatch for UI)
        self.device.wait()
        if width > 0 and height > 0:
            self.surface.configure(width=width, height=height)
        else:
            self.surface.unconfigure()


def find_default_seg(root: Path) -> Path | None:
    # Try a common BraTS case path for convenience
    for case in root.glob("BraTS-GLI-*/"):
        cand = list(case.glob("*-seg.nii.gz"))
        if cand:
            return cand[0]
    return None


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Render BraTS segmentation NIfTI using Slang")
    p.add_argument("--seg", type=str, default=None, help="Path to *-seg.nii.gz")
    p.add_argument("--data_root", type=str, default=str(Path(__file__).parent / "data" / "BraTS-2023"))
    p.add_argument("--up", type=str, default="Y", help="World up axis: X|Y|Z|-X|-Y|-Z")
    args = p.parse_args()
    seg_path = Path(args.seg) if args.seg else find_default_seg(Path(args.data_root))
    if not seg_path or not seg_path.exists():
        raise SystemExit("No segmentation NIfTI found. Pass --seg path to *-seg.nii.gz")
    app = SegViewer(seg_path, up=args.up)
    app.run()
