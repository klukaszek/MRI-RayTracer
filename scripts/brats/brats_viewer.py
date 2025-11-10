from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import nibabel as nib
import slangpy as spy

# Reuse the VDB viewer orbital camera for consistent controls
import sys as _sys
_HERE = Path(__file__).parent
_sys.path.append((_HERE / "vdb_viewer").as_posix())
from camera import OrbitalCamera  # type: ignore


MOD_SUFFIXES = {
    "t1n": "T1n",
    "t1c": "T1c",
    "t2w": "T2w",
    "t2f": "FLAIR",
}


def load_nifti_float(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = nib.load(path.as_posix())
    data = img.get_fdata(dtype=np.float32)
    # Normalize per volume to [0,1] robustly
    vmin = float(np.percentile(data, 1.0))
    vmax = float(np.percentile(data, 99.5))
    if vmax <= vmin:
        vmax = float(np.max(data))
        vmin = float(np.min(data))
    rng = max(1e-6, vmax - vmin)
    norm = np.clip((data - vmin) / rng, 0.0, 1.0).astype(np.float32)
    hdr = img.header
    zooms = np.array(hdr.get_zooms()[:3], dtype=np.float32)
    if norm.ndim != 3:
        raise RuntimeError(f"Expected 3D volume, got shape {norm.shape}")
    dimx, dimy, dimz = norm.shape
    linear = np.ascontiguousarray(norm.transpose(2, 1, 0).reshape(-1))
    return linear, np.array([dimx, dimy, dimz], dtype=np.uint32), zooms


def load_seg_uint(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = nib.load(path.as_posix())
    data = img.get_fdata(dtype=np.float32)
    labels = np.rint(data).astype(np.uint32)
    hdr = img.header
    zooms = np.array(hdr.get_zooms()[:3], dtype=np.float32)
    if labels.ndim != 3:
        raise RuntimeError(f"Expected 3D labels, got shape {labels.shape}")
    dimx, dimy, dimz = labels.shape
    linear = np.ascontiguousarray(labels.transpose(2, 1, 0).reshape(-1))
    return linear, np.array([dimx, dimy, dimz], dtype=np.uint32), zooms


class BraTSViewer:
    def __init__(self, case_dir: Path, up: str = "Y"):
        self.window = spy.Window(title="BraTS Viewer (Slang)", resizable=True)
        self.device = spy.Device(enable_debug_layers=True, compiler_options={"include_paths": [Path(__file__).parent]})
        self.surface = self.device.create_surface(self.window)
        self.surface.configure(width=self.window.width, height=self.window.height)
        self.ui = spy.ui.Context(self.device)

        # Load main entry point only (overlay path removed)
        program = self.device.load_program(str(Path(__file__).parent / "brats_rt.slang"), ["brats_main"])
        self.kernel = self.device.create_compute_kernel(program)
        # Load precompute kernel if present
        try:
            program_pre = self.device.load_program(str(Path(__file__).parent / "brats_rt.slang"), ["inr_precompute"])
            self.kernel_inr_pre = self.device.create_compute_kernel(program_pre)
        except Exception:
            self.kernel_inr_pre = None
        self.kernel_inr = None

        self.output_texture: Optional[spy.Texture] = None
        self.inr_overlay_tex: Optional[spy.Texture] = None  # unused in simplified path
        self.inr_volume_tex: Optional[spy.Texture] = None   # 3D precomputed INR probabilities
        self.buffers: Dict[str, spy.Buffer] = {}
        self.seg_buffer: Optional[spy.Buffer] = None
        self.vol_dims: Optional[np.ndarray] = None
        self.voxel_size: Optional[np.ndarray] = None
        self.vol_min: Optional[np.ndarray] = None
        # INR (implicit network) weights and toggle
        self.inr_weights: Dict[str, spy.Buffer] = {}
        self.show_inr: bool = False
        # INR dynamic dims (deduced from NPZ)
        self.inr_in_dim: int = 0
        self.inr_h0: int = 0
        self.inr_h1: int = 0
        self.inr_h2: int = 0
        self.inr_h3: int = 0
        self.inr_out_dim: int = 0
        self.inr_kff: int = 0  # fourier features per-axis
        self.inr_alpha: float = 0.4
        self.inr_stride: int = 4
        # Min intensity threshold for INR evaluation
        self.inr_min_intensity: float = 0.05

        # Camera
        up_map = {
            "X": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "Y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "Z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "-X": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            "-Y": np.array([0.0, -1.0, 0.0], dtype=np.float32),
            "-Z": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        }
        world_up = up_map.get(up.upper(), up_map["Y"]) if isinstance(up, str) else up_map["Y"]
        self.fov_deg = 70.0
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

        # Intensity controls
        self.enabled = {k: True for k in MOD_SUFFIXES.values()}
        self.weights = {k: 1.0 for k in MOD_SUFFIXES.values()}
        self.ww = 1.0
        self.wl = 0.5
        self.intensity_alpha = 0.4
        # Tone/edge shaping
        self.gamma = 1.0
        self.grad_boost = 1.5
        self.grad_scale = 1.0

        # Ray near/far clipping along ray parameter t (world units)
        # far_t = 0 disables far clipping
        self.near_t = 0.0
        self.far_t = 0.0

        # Segmentation controls
        self.show_seg = True
        self.use_grayscale_seg = False
        self.lut = np.zeros((8, 4), dtype=np.float32)
        self.lut[0] = [0.0, 0.0, 0.0, 0.0]
        self.lut[1] = [0.0, 0.4, 1.0, 0.9]  # NCR/NET blue
        self.lut[2] = [0.0, 0.8, 0.0, 0.7]  # Edema green
        self.lut[3] = [0.0, 0.0, 0.0, 0.0]
        self.lut[4] = [1.0, 0.1, 0.1, 0.9]  # Enhancing red
        self.lut = np.ascontiguousarray(self.lut, dtype=np.float32)

        # Raymarching
        self.step_size = 0.05
        self.bg_color = np.array([0.0, 0.0, 0.0])
        # self.bg_color = np.array([0.05, 0.06, 0.08], dtype=np.float32)
        self.pan_speed = 0.2
        self.drag_deadzone_px = 6.0
        self._lmb = False
        self._orbit_dragging = False
        self._pan_dragging = False
        self._last = None
        self._press_pos = None

        self._init_ui()
        self.load_dir(case_dir)

        # Bind events
        self.window.on_mouse_event = self.on_mouse
        self.window.on_keyboard_event = self.on_keyboard
        self.window.on_resize = self.on_resize

    def _init_ui(self):
        screen = self.ui.screen
        window = spy.ui.Window(screen, "BraTS Settings", size=spy.float2(480, 500))

        # Volume group
        self.volume_group = spy.ui.Group(window, "Volume")
        self.info = spy.ui.Text(self.volume_group, "(no case)")
        spy.ui.Button(self.volume_group, "Load NIfTI...", callback=self.on_click_load_file)
        spy.ui.Button(self.volume_group, "Frame Volume", callback=self.on_click_frame)

        # Camera group
        self.cam_group = spy.ui.Group(window, "Camera")
        self.fov_slider = spy.ui.SliderFloat(self.cam_group, "FOV (deg)", value=self.fov_deg, min=20.0, max=100.0)
        # Near/Far sliders; updated after volume load based on volume size
        self.near_slider = spy.ui.SliderFloat(self.cam_group, "Near T", value=self.near_t, min=0.0, max=10.0)
        self.far_slider = spy.ui.SliderFloat(self.cam_group, "Far T (0=off)", value=self.far_t, min=0.0, max=10.0)
        # Pan speed for Shift+LMB panning
        try:
            self.pan_slider = spy.ui.SliderFloat(self.cam_group, "Pan Speed", value=self.pan_speed, min=0.01, max=2.0)
        except Exception:
            # For older UI builds that may not support default value prop ordering
            self.pan_slider = spy.ui.SliderFloat(self.cam_group, "Pan Speed", min=0.01, max=2.0, value=self.pan_speed)
        spy.ui.Button(self.cam_group, "Reset Camera", callback=self.on_click_reset_camera)

        # Intensity group
        self.int_group = spy.ui.Group(window, "Intensity")
        self.step_slider = spy.ui.SliderFloat(self.int_group, "Step", value=self.step_size, min=0.001, max=2.0)
        self.alpha_slider = spy.ui.SliderFloat(self.int_group, "Intensity Alpha", value=self.intensity_alpha, min=0.0, max=1.0)
        self.ww_slider = spy.ui.SliderFloat(self.int_group, "Window Width", value=self.ww, min=0.01, max=2.0)
        self.wl_slider = spy.ui.SliderFloat(self.int_group, "Window Level", value=self.wl, min=0.0, max=1.0)
        self.gamma_slider = spy.ui.SliderFloat(self.int_group, "Gamma", value=self.gamma, min=0.2, max=3.0)
        self.grad_boost_slider = spy.ui.SliderFloat(self.int_group, "Edge Boost", value=self.grad_boost, min=0.0, max=4.0)
        self.grad_scale_slider = spy.ui.SliderFloat(self.int_group, "Edge Scale", value=self.grad_scale, min=0.1, max=4.0)
        # Modality toggles and weights
        self.mod_check: Dict[str, spy.ui.CheckBox] = {}
        self.mod_weight: Dict[str, spy.ui.SliderFloat] = {}
        for key in ("T1n", "T1c", "T2w", "FLAIR"):
            self.mod_check[key] = spy.ui.CheckBox(self.int_group, f"Enable {key}", value=self.enabled.get(key, False))
            self.mod_weight[key] = spy.ui.SliderFloat(self.int_group, f"Weight {key}", value=self.weights.get(key, 1.0), min=0.0, max=2.0)

        # Segmentation group
        self.seg_group = spy.ui.Group(window, "Segmentation")
        # Ground truth segmentation overlay controls
        self.seg_check = spy.ui.CheckBox(self.seg_group, "Show Ground Truth", value=self.show_seg)
        self.seg_gray = spy.ui.CheckBox(self.seg_group, "Greyscale Ground Truth", value=self.use_grayscale_seg)
        # INR overlay controls
        self.inr_check = spy.ui.CheckBox(self.seg_group, "Show INR Prediction", value=self.show_inr)
        spy.ui.Button(self.seg_group, "Load INR Weights", callback=self.on_click_load_inr)
        # INR probability mask controls
        try:
            self.inr_alpha_slider = spy.ui.SliderFloat(self.seg_group, "INR Alpha", value=self.inr_alpha, min=0.0, max=1.0)
        except Exception:
            self.inr_alpha_slider = spy.ui.SliderFloat(self.seg_group, "INR Alpha", min=0.0, max=1.0, value=self.inr_alpha)
        try:
            # Expanded stride range to allow coarser base sampling.
            self.inr_stride_slider = spy.ui.SliderFloat(self.seg_group, "INR Stride (steps)", value=float(self.inr_stride), min=1.0, max=64.0)
        except Exception:
            self.inr_stride_slider = spy.ui.SliderFloat(self.seg_group, "INR Stride (steps)", min=1.0, max=64.0, value=float(self.inr_stride))
        spy.ui.Text(window, "Orbit: LMB | Pan: Shift+LMB | Zoom: Wheel")

    def _create_float_buffer(self, linear: np.ndarray):
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

    def load_dir(self, case_dir: Path):
        # Find modality files
        files = list(Path(case_dir).glob("*.nii.gz"))
        mod_files: Dict[str, Path] = {}
        seg_file: Optional[Path] = None
        # Accept both BraTS-style ("-t1c") and MU-Glioma-style ("_t1c", "_brain_t1c") suffixes.
        # Also accept segmentation names like "-seg", "_seg", "_tumorMask".
        for f in files:
            name = f.name.lower()
            # Known segmentation filename endings
            if name.endswith(("-seg.nii.gz", "_seg.nii.gz", "-tumormask.nii.gz", "_tumormask.nii.gz")):
                seg_file = f
                continue
            # Known modality suffixes with either '-' or '_' separator
            for suf, key in MOD_SUFFIXES.items():
                if name.endswith(f"-{suf}.nii.gz") or name.endswith(f"_{suf}.nii.gz"):
                    mod_files[key] = f
        if not mod_files:
            raise RuntimeError(
                "No modality volumes found in "
                f"{case_dir}. Expected files ending with one of: "
                "-t1n/_t1n, -t1c/_t1c, -t2w/_t2w, -t2f/_t2f"
            )

        # Load first modality to get dims/voxels
        first_key = next(iter(mod_files))
        lin0, dims, vox = load_nifti_float(mod_files[first_key])
        self.vol_dims = dims.astype(np.uint32)
        max_dim = float(max(int(dims[0]), int(dims[1]), int(dims[2])))
        scale = np.float32(1.8 / max_dim)
        self.voxel_size = (vox.astype(np.float32) * scale).astype(np.float32)
        extent = self.voxel_size * self.vol_dims.astype(np.float32)
        self.vol_min = -0.5 * extent

        # Upload placeholder for all buffers, then replace ones present
        one = np.zeros((1,), dtype=np.float32)
        empty_f = self._create_float_buffer(one)
        self.buffers = {k: empty_f for k in ("T1n", "T1c", "T2w", "FLAIR")}

        # Upload first and others
        buf0 = self._create_float_buffer(lin0)
        buf0.copy_from_numpy(lin0)
        self.buffers[first_key] = buf0
        for key, path in mod_files.items():
            if key == first_key:
                continue
            lin, d2, v2 = load_nifti_float(path)
            if not np.all(d2 == dims):
                raise RuntimeError(f"Dim mismatch for {key}: {tuple(d2)} vs {tuple(dims)}")
            self.buffers[key] = self._create_float_buffer(lin)
            self.buffers[key].copy_from_numpy(lin)

        # Segmentation (optional)
        self.seg_buffer = None
        if seg_file is not None:
            slin, sdims, svox = load_seg_uint(seg_file)
            if not np.all(sdims == dims):
                self.info.text = f"Seg dims mismatch, ignoring seg: {tuple(sdims)} vs {tuple(dims)}"
            else:
                self.seg_buffer = self._create_uint_buffer(slin)
                self.seg_buffer.copy_from_numpy(slin)
        # Enable/disable seg UI based on availability
        try:
            has_seg = self.seg_buffer is not None
            # Toggle checkbox enabled state if supported
            if hasattr(self.seg_check, 'enabled'):
                self.seg_check.enabled = has_seg
            if hasattr(self.seg_gray, 'enabled'):
                self.seg_gray.enabled = has_seg
            # If no seg, uncheck
            if not has_seg:
                if hasattr(self.seg_check, 'value'):
                    self.seg_check.value = False
                if hasattr(self.seg_gray, 'value'):
                    self.seg_gray.value = False
                self.show_seg = False
        except Exception:
            pass

        # Update UI text
        lines = [f"Dir: {case_dir.name}", f"Dims: {int(dims[0])} x {int(dims[1])} x {int(dims[2])}",
                 f"Voxel: {self.voxel_size[0]:.4g}, {self.voxel_size[1]:.4g}, {self.voxel_size[2]:.4g}"]
        for key in ("T1n", "T1c", "T2w", "FLAIR"):
            if key in mod_files:
                lines.append(f"{key}: {mod_files[key].name}")
        if seg_file is not None:
            lines.append(f"Seg: {seg_file.name}")
        self.info.text = "\n".join(lines)

        # Set step and near/far ranges based on voxel size and volume extent
        vmin = float(np.min(self.voxel_size))
        self.step_size = 0.75 * vmin
        # Default gradient scale ~ inverse voxel to keep [0,1]
        try:
            self.grad_scale = float(1.0)
            self.grad_scale_slider.value = self.grad_scale
        except Exception:
            pass
        try:
            # Update step slider bounds
            self.step_slider.min = max(1e-6, 0.05 * vmin)
            self.step_slider.max = max(self.step_slider.min * 2.0, 5.0 * vmin)
            self.step_slider.value = self.step_size
            # Update near/far slider ranges based on volume diagonal length
            extent = self.voxel_size * self.vol_dims.astype(np.float32)
            diag = float(np.linalg.norm(extent))
            self.near_slider.min = 0.0
            self.near_slider.max = max(1.0, 0.75 * diag)
            self.near_slider.value = 0.0
            self.far_slider.min = 0.0
            self.far_slider.max = max(1.0, 1.25 * diag)
            # Keep far disabled by default
            self.far_slider.value = 0.0
        except Exception:
            pass

        # Frame
        self.frame_volume()
        # Create persistent tiny fallback buffers (avoid per-frame allocations)
        try:
            self._empty_float = self._create_float_buffer(np.zeros((1,), dtype=np.float32))
            self._empty_uint = self._create_uint_buffer(np.zeros((1,), dtype=np.uint32))
        except Exception:
            # Fallback if device not ready; will lazy-create on first use
            self._empty_float = None
            self._empty_uint = None

    def _create_weight_buffer(self, arr: np.ndarray) -> spy.Buffer:
        arr = np.ascontiguousarray(arr.astype(np.float32).reshape(-1))
        buf = self._create_float_buffer(arr)
        buf.copy_from_numpy(arr)
        return buf

    def on_click_load_inr(self):
        # Load INR .npz (expects W_0..W_4 and b_0..b_4)
        try:
            path = spy.platform.open_file_dialog()
        except Exception:
            path = str((Path(__file__).parent.parent.parent / "artifacts" / "inr_brats23.npz").resolve())
        if not path:
            return
        p = Path(path)
        if not p.exists() or p.suffix.lower() != ".npz":
            self.info.text = f"Not an .npz file: {p.name}"
            return
        d = np.load(p)
        keys = [f"W_{i}" for i in range(5)] + [f"b_{i}" for i in range(5)]
        if not all(k in d for k in keys):
            self.info.text = f"Missing expected keys in {p.name}"
            return
        # Upload
        self.inr_weights.clear()
        for k in keys:
            arr = np.ascontiguousarray(d[k].astype(np.float32))
            self.inr_weights[k] = self._create_weight_buffer(arr)
        # Infer dims from weight shapes; expect row-major (in,out)
        try:
            w0 = d["W_0"]; b0 = d["b_0"]
            w1 = d["W_1"]; b1 = d["b_1"]
            w2 = d["W_2"]; b2 = d["b_2"]
            w3 = d["W_3"]; b3 = d["b_3"]
            w4 = d["W_4"]; b4 = d["b_4"]
            in0, h0 = int(w0.shape[0]), int(w0.shape[1])
            h0b = int(b0.shape[0])
            h1in, h1 = int(w1.shape[0]), int(w1.shape[1])
            h2in, h2 = int(w2.shape[0]), int(w2.shape[1])
            h3in, h3 = int(w3.shape[0]), int(w3.shape[1])
            h4in, outd = int(w4.shape[0]), int(w4.shape[1])
            # Basic consistency checks
            if not (h0 == h0b == h1in and h1 == int(b1.shape[0]) == h2in and h2 == int(b2.shape[0]) == h3in and h3 == int(b3.shape[0]) == h4in and outd == int(b4.shape[0])):
                self.info.text = f"INR dims inconsistent in {p.name}"
            # Deduce K_FF from input dim: in0 = 3 + 3*2*Kff + 4
            kff_num = in0 - (3 + 4)
            kff = 0
            if kff_num % 6 == 0 and kff_num >= 0:
                kff = kff_num // 6
            self.inr_in_dim = in0
            self.inr_h0, self.inr_h1, self.inr_h2, self.inr_h3 = h0, h1, h2, h3
            self.inr_out_dim = outd
            self.inr_kff = int(kff)
        except Exception as e:
            # If anything goes wrong, keep dims zeroed to disable INR path
            self.inr_in_dim = self.inr_h0 = self.inr_h1 = self.inr_h2 = self.inr_h3 = self.inr_out_dim = self.inr_kff = 0
            self.info.text = f"Loaded INR but failed to parse dims: {e}"
        # Do not auto-enable INR overlay on load to avoid sudden GPU cost
        self.show_inr = False
        try:
            self.inr_check.value = False
        except Exception:
            pass
        self.info.text = self.info.text + f"\nINR loaded: {p.name} (in={self.inr_in_dim}, h=({self.inr_h0},{self.inr_h1},{self.inr_h2},{self.inr_h3}), out={self.inr_out_dim}, kff={self.inr_kff})"
        # Precompute INR volume if kernel available and dims reasonable
        if self.kernel_inr_pre is not None and self.vol_dims is not None and self.inr_out_dim > 0:
            try:
                dx, dy, dz = map(int, self.vol_dims.tolist())
                # Allocate or reallocate 3D texture
                self.inr_volume_tex = self.device.create_texture(
                    format=spy.Format.rgba16_float,
                    width=dx,
                    height=dy,
                    depth=dz,
                    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                    label="inr_volume"
                )
                self.kernel_inr_pre.dispatch(
                    thread_count=[dx, dy, dz],
                    vars={
                        "gIntensity0": self.inr_weights and self.buffers.get("T1n", self._empty_float) or self._empty_float,
                        "gIntensity1": self.buffers.get("T1c", self._empty_float),
                        "gIntensity2": self.buffers.get("T2w", self._empty_float),
                        "gIntensity3": self.buffers.get("FLAIR", self._empty_float),
                        "gW0": self.inr_weights.get("W_0", self._empty_float),
                        "gB0": self.inr_weights.get("b_0", self._empty_float),
                        "gW1": self.inr_weights.get("W_1", self._empty_float),
                        "gB1": self.inr_weights.get("b_1", self._empty_float),
                        "gW2": self.inr_weights.get("W_2", self._empty_float),
                        "gB2": self.inr_weights.get("b_2", self._empty_float),
                        "gW3": self.inr_weights.get("W_3", self._empty_float),
                        "gB3": self.inr_weights.get("b_3", self._empty_float),
                        "gW4": self.inr_weights.get("W_4", self._empty_float),
                        "gB4": self.inr_weights.get("b_4", self._empty_float),
                        "gINRVolumeOut": self.inr_volume_tex,
                        "gParams": {
                            "imageSize": (np.uint32(1), np.uint32(1)),
                            "fovY": np.float32(1.0),
                            "eye": np.array([0,0,0], dtype=np.float32),
                            "U": np.array([1,0,0], dtype=np.float32),
                            "V": np.array([0,1,0], dtype=np.float32),
                            "W": np.array([0,0,1], dtype=np.float32),
                            "volMin": (self.vol_min.astype(np.float32) if isinstance(self.vol_min, np.ndarray) else np.array([0,0,0], dtype=np.float32)),
                            "voxelSize": (self.voxel_size.astype(np.float32) if isinstance(self.voxel_size, np.ndarray) else np.array([1,1,1], dtype=np.float32)),
                            "dims": self.vol_dims.astype(np.uint32),
                            "stepSize": np.float32(1.0),
                            "nearT": np.float32(0.0),
                            "farT": np.float32(0.0),
                            "bgColor": np.array([0,0,0], dtype=np.float32),
                            "volEnabled": (np.uint32(1), np.uint32(1), np.uint32(1), np.uint32(1)),
                            "volWeight": (np.float32(1.0), np.float32(1.0), np.float32(1.0), np.float32(1.0)),
                            "ww": np.float32(self.ww),
                            "wl": np.float32(self.wl),
                            "intensityAlpha": np.float32(self.intensity_alpha),
                            "gamma": np.float32(self.gamma),
                            "gradBoost": np.float32(0.0),
                            "gradScale": np.float32(0.0),
                            "showSeg": np.uint32(0),
                            "showINR": np.uint32(0),
                            "inrKFF": np.uint32(self.inr_kff),
                            "inrInDim": np.uint32(self.inr_in_dim),
                            "inrH0": np.uint32(self.inr_h0),
                            "inrH1": np.uint32(self.inr_h1),
                            "inrH2": np.uint32(self.inr_h2),
                            "inrH3": np.uint32(self.inr_h3),
                            "inrOutDim": np.uint32(self.inr_out_dim),
                            "inrAlpha": np.float32(self.inr_alpha),
                            "inrStepStride": np.uint32(1),
                            "inrMinIntensity": np.float32(0.0),
                            "inrOverlayWidth": np.uint32(0),
                            "inrOverlayHeight": np.uint32(0),
                            "precomputedINR": np.uint32(1),
                        },
                    }
                )
                self.info.text += "\nPrecomputed INR volume generated"
            except Exception as e:
                self.info.text += f"\nINR precompute failed: {e}"

    def on_click_load_file(self):
        # Let user pick a NIfTI file; then load all .nii/.nii.gz in its directory
        try:
            path = spy.platform.open_file_dialog()
        except Exception as e:
            self.info.text = f"Open file dialog error: {e}"
            return
        if path:
            p = Path(path)
            # Accept only .nii.gz per BraTS convention
            is_niigz = (p.suffixes[-2:] == [".nii", ".gz"]) if p.suffixes else False
            if not p.exists() or not p.is_file() or not is_niigz:
                self.info.text = f"Not a NIfTI file: {p.name}"
                return
            self.load_dir(p.parent)

    def on_click_frame(self):
        self.frame_volume()

    def on_click_reset_camera(self):
        # Reset orbital camera orientation/zoom and restore FOV, near/far, and pan speed defaults
        try:
            self.camera.reset()
        except Exception:
            pass
        # Defaults
        default_fov = 70.0
        default_pan = 0.2
        # Update UI widgets if present
        try:
            self.fov_slider.value = default_fov
            self.camera.set_fov_degrees(default_fov)
        except Exception:
            pass
        try:
            self.near_t = 0.0
            self.far_t = 0.0
            self.near_slider.value = 0.0
            self.far_slider.value = 0.0
        except Exception:
            pass
        try:
            self.pan_speed = default_pan
            if hasattr(self, 'pan_slider') and self.pan_slider is not None:
                self.pan_slider.value = default_pan
        except Exception:
            pass
        # Reframe to current volume for a sensible radius/center
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

    def on_resize(self, width: int, height: int):
        self.device.wait()
        if width > 0 and height > 0:
            self.surface.configure(width=width, height=height)
        else:
            self.surface.unconfigure()

    def run(self):
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
                    label="brats_output",
                )
                # Invalidate INR overlay on resize
                self.inr_overlay_tex = None

            # Update settings from UI
            self.fov_deg = float(self.fov_slider.value)
            self.step_size = float(self.step_slider.value)
            self.intensity_alpha = float(self.alpha_slider.value)
            self.ww = float(self.ww_slider.value)
            self.wl = float(self.wl_slider.value)
            self.gamma = float(self.gamma_slider.value)
            self.grad_boost = float(self.grad_boost_slider.value)
            self.grad_scale = float(self.grad_scale_slider.value)
            # Near/Far and pan speed from UI
            try:
                self.near_t = max(0.0, float(self.near_slider.value))
                self.far_t = max(0.0, float(self.far_slider.value))
                self.pan_speed = float(self.pan_slider.value)
            except Exception:
                pass
            for key in ("T1n", "T1c", "T2w", "FLAIR"):
                try:
                    self.enabled[key] = bool(self.mod_check[key].value)
                except Exception:
                    self.enabled[key] = bool(getattr(self.mod_check[key], 'checked', False))
                self.weights[key] = float(self.mod_weight[key].value)
            try:
                self.show_seg = bool(self.seg_check.value)
                self.use_grayscale_seg = bool(self.seg_gray.value)
                self.show_inr = bool(self.inr_check.value)
                self.inr_alpha = float(self.inr_alpha_slider.value)
                # Treat stride slider as integer in [1, 64]
                self.inr_stride = max(1, min(64, int(round(float(self.inr_stride_slider.value)))))
                # Screen-space stride
                if hasattr(self, 'inr_screen_stride_slider'):
                    self.inr_screen_stride = max(1, min(128, int(round(float(self.inr_screen_stride_slider.value)))))
            except Exception:
                self.show_seg = bool(getattr(self.seg_check, 'checked', True))
                self.use_grayscale_seg = bool(getattr(self.seg_gray, 'checked', False))
                self.show_inr = bool(getattr(self, 'show_inr', False))

            # Camera basis
            eye, gU, gV, gW = self.camera.get_basis()

            # Build LUT (optional greyscale)
            if self.use_grayscale_seg:
                lut_use = np.zeros_like(self.lut)
                lut_use[0] = [0.0, 0.0, 0.0, 0.0]
                lut_use[1:] = [1.0, 1.0, 1.0, 0.9]
            else:
                lut_use = self.lut

            # volEnabled and volWeight packs
            en = [1 if self.enabled.get(k, False) else 0 for k in ("T1n", "T1c", "T2w", "FLAIR")]
            wt = [float(self.weights.get(k, 0.0)) for k in ("T1n", "T1c", "T2w", "FLAIR")]

            params = {
                "imageSize": (np.uint32(self.output_texture.width), np.uint32(self.output_texture.height)),
                "fovY": np.float32(math.radians(self.fov_deg)),
                "eye": eye.astype(np.float32),
                "U": gU.astype(np.float32),
                "V": gV.astype(np.float32),
                "W": gW.astype(np.float32),
                "volMin": (self.vol_min.astype(np.float32) if isinstance(self.vol_min, np.ndarray) else np.array([0,0,0], dtype=np.float32)),
                "voxelSize": (self.voxel_size.astype(np.float32) if isinstance(self.voxel_size, np.ndarray) else np.array([1,1,1], dtype=np.float32)),
                "dims": (self.vol_dims.astype(np.uint32) if isinstance(self.vol_dims, np.ndarray) else np.array([1,1,1], dtype=np.uint32)),
                "stepSize": np.float32(self.step_size),
                "nearT": np.float32(self.near_t),
                "farT": np.float32(self.far_t),
                "bgColor": self.bg_color.astype(np.float32),
                "volEnabled": tuple(map(np.uint32, en)),
                "volWeight": tuple(map(np.float32, wt)),
                "ww": np.float32(self.ww),
                "wl": np.float32(self.wl),
                "intensityAlpha": np.float32(self.intensity_alpha),
                "gamma": np.float32(self.gamma),
                "gradBoost": np.float32(self.grad_boost),
                "gradScale": np.float32(self.grad_scale),
                "showSeg": np.uint32(1 if (self.show_seg and self.seg_buffer is not None) else 0),
                "showINR": np.uint32(1 if (self.show_inr and len(self.inr_weights) == 10 and self.inr_out_dim > 0) else 0),
                # INR dims and controls
                "inrKFF": np.uint32(max(0, self.inr_kff)),
                "inrInDim": np.uint32(max(0, self.inr_in_dim)),
                "inrH0": np.uint32(max(0, self.inr_h0)),
                "inrH1": np.uint32(max(0, self.inr_h1)),
                "inrH2": np.uint32(max(0, self.inr_h2)),
                "inrH3": np.uint32(max(0, self.inr_h3)),
                "inrOutDim": np.uint32(max(0, self.inr_out_dim)),
                "inrAlpha": np.float32(self.inr_alpha),
                "inrStepStride": np.uint32(max(1, self.inr_stride)),
                "inrMinIntensity": np.float32(0.05),
                # Prefer differentiable INR overlay pipeline in shader
                "useDiffINR": np.uint32(1),
                "lutColorAlpha": [tuple(map(float, row)) for row in lut_use.tolist()],
                "precomputedINR": np.uint32(1 if (self.inr_volume_tex is not None) else 0),
            }

            ce = self.device.create_command_encoder()

            # Ensure INR overlay texture exists when INR is enabled
            inr_enabled = bool(params["showINR"]) and (self.kernel_inr is not None)
            if inr_enabled:
                ow = max(1, int(math.ceil(self.output_texture.width / max(1, self.inr_screen_stride))))
                oh = max(1, int(math.ceil(self.output_texture.height / max(1, self.inr_screen_stride))))
                if (self.inr_overlay_tex is None or self.inr_overlay_tex.width != ow or self.inr_overlay_tex.height != oh):
                    self.inr_overlay_tex = self.device.create_texture(
                        format=spy.Format.rgba16_float,
                        width=ow,
                        height=oh,
                        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                        label="inr_overlay",
                    )
                # Dispatch overlay generator (one thread per overlay pixel)
                if self.kernel_inr is not None:
                    try:
                        self.kernel_inr.dispatch(
                            thread_count=[self.inr_overlay_tex.width, self.inr_overlay_tex.height, 1],
                            vars={
                                "gIntensity0": _buf_or_empty("T1n"),
                                "gIntensity1": _buf_or_empty("T1c"),
                                "gIntensity2": _buf_or_empty("T2w"),
                                "gIntensity3": _buf_or_empty("FLAIR"),
                                "gLabels": (self.seg_buffer if self.seg_buffer is not None else (self._empty_uint if getattr(self, "_empty_uint", None) is not None else self._create_uint_buffer(np.zeros((1,), dtype=np.uint32)))),
                                "gW0": (self.inr_weights.get("W_0") if inr_enabled else self._empty_float),
                                "gB0": (self.inr_weights.get("b_0") if inr_enabled else self._empty_float),
                                "gW1": (self.inr_weights.get("W_1") if inr_enabled else self._empty_float),
                                "gB1": (self.inr_weights.get("b_1") if inr_enabled else self._empty_float),
                                "gW2": (self.inr_weights.get("W_2") if inr_enabled else self._empty_float),
                                "gB2": (self.inr_weights.get("b_2") if inr_enabled else self._empty_float),
                                "gW3": (self.inr_weights.get("W_3") if inr_enabled else self._empty_float),
                                "gB3": (self.inr_weights.get("b_3") if inr_enabled else self._empty_float),
                                "gW4": (self.inr_weights.get("W_4") if inr_enabled else self._empty_float),
                                "gB4": (self.inr_weights.get("b_4") if inr_enabled else self._empty_float),
                                "gINROverlay": self.inr_overlay_tex,
                                "gINROverlayRW": self.inr_overlay_tex,
                                "gParams": {
                                    **params,
                                    "inrOverlayWidth": np.uint32(self.inr_overlay_tex.width),
                                    "inrOverlayHeight": np.uint32(self.inr_overlay_tex.height),
                                },
                            },
                            command_encoder=ce,
                        )
                    except Exception:
                        pass
            # Bind buffers; fall back to small buffer if missing
            def _buf_or_empty(key: str) -> spy.Buffer:
                if key in self.buffers and self.buffers[key] is not None:
                    return self.buffers[key]
                # create small empty once
                if getattr(self, "_empty_float", None) is None:
                    self._empty_float = self._create_float_buffer(np.zeros((1,), dtype=np.float32))
                return self._empty_float

            self.kernel.dispatch(
                thread_count=[self.output_texture.width, self.output_texture.height, 1],
                vars={
                    "gOutput": self.output_texture,
                    "gINROverlay": (self.inr_overlay_tex if (inr_enabled and self.inr_overlay_tex is not None) else self.output_texture),
                    "gIntensity0": _buf_or_empty("T1n"),
                    "gIntensity1": _buf_or_empty("T1c"),
                    "gIntensity2": _buf_or_empty("T2w"),
                    "gIntensity3": _buf_or_empty("FLAIR"),
                    "gLabels": (self.seg_buffer if self.seg_buffer is not None else (self._empty_uint if getattr(self, "_empty_uint", None) is not None else self._create_uint_buffer(np.zeros((1,), dtype=np.uint32)))),
                    # Bind INR buffers only when overlay is active
                    "gW0": (self.inr_weights.get("W_0") if params["showINR"] else self._empty_float),
                    "gB0": (self.inr_weights.get("b_0") if params["showINR"] else self._empty_float),
                    "gW1": (self.inr_weights.get("W_1") if params["showINR"] else self._empty_float),
                    "gB1": (self.inr_weights.get("b_1") if params["showINR"] else self._empty_float),
                    "gW2": (self.inr_weights.get("W_2") if params["showINR"] else self._empty_float),
                    "gB2": (self.inr_weights.get("b_2") if params["showINR"] else self._empty_float),
                    "gW3": (self.inr_weights.get("W_3") if params["showINR"] else self._empty_float),
                    "gB3": (self.inr_weights.get("b_3") if params["showINR"] else self._empty_float),
                    "gW4": (self.inr_weights.get("W_4") if params["showINR"] else self._empty_float),
                    "gB4": (self.inr_weights.get("b_4") if params["showINR"] else self._empty_float),
                    "gINRVolume": (self.inr_volume_tex if self.inr_volume_tex is not None else self.output_texture),
                    "gParams": {
                        **params,
                        "inrOverlayWidth": np.uint32(self.inr_overlay_tex.width if (inr_enabled and self.inr_overlay_tex is not None) else 0),
                        "inrOverlayHeight": np.uint32(self.inr_overlay_tex.height if (inr_enabled and self.inr_overlay_tex is not None) else 0),
                    },
                },
                command_encoder=ce,
            )
            ce.blit(surface_tex, self.output_texture)
            self.ui.begin_frame(surface_tex.width, surface_tex.height)
            self.ui.end_frame(surface_tex, ce)
            self.device.submit_command_buffer(ce.finish())

            self.surface.present()
            del surface_tex


def find_case_dir(root: Path) -> Optional[Path]:
    # pick first BraTS-GLI-* directory containing .nii.gz files
    for case in sorted(root.glob("BraTS-GLI-*/")):
        if any(case.glob("*.nii.gz")):
            return case
    return None


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Render BraTS case (multi-volume) using Slang")
    p.add_argument("--dir", type=str, default=None, help="Path to BraTS case directory")
    p.add_argument("--data_root", type=str, default=str(Path(__file__).parent / "data" / "BraTS-2023"))
    p.add_argument("--up", type=str, default="Y", help="World up axis: X|Y|Z|-X|-Y|-Z")
    args = p.parse_args()
    case_dir = Path(args.dir) if args.dir else find_case_dir(Path(args.data_root))
    if not case_dir or not case_dir.exists():
        raise SystemExit("No BraTS case directory found. Pass --dir to a case folder.")
    app = BraTSViewer(case_dir, up=args.up)
    app.run()
