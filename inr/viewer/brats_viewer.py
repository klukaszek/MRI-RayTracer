from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import nibabel as nib
import slangpy as spy

from camera import OrbitalCamera 

# --- Path Setup for Imports ---
# Assuming structure:
#   root/
#    inr/
#     model.py
#     ...
#    viewer/
#      brats_viewer.py
#      ...

_SCRIPT_DIR = Path(__file__).resolve().parent
# Try resolving project root (up 2 levels from scripts/brats)
_PROJECT_ROOT = _SCRIPT_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

try:
    import inr.model as inr_model
    HAS_JAX = True
except ImportError as e:
    print(f"Warning: Could not import 'inr.model'. Inference disabled. Error: {e}")
    HAS_JAX = False

MOD_SUFFIXES = {
    "t1n": "T1n",
    "t1c": "T1c",
    "t2w": "T2w",
    "t2f": "FLAIR",
}

def load_nifti_float(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    img = nib.load(path.as_posix())
    data = img.get_fdata(dtype=np.float32)
    
    # --- Visualization Normalization [0, 1] ---
    vmin = float(np.percentile(data, 1.0))
    vmax = float(np.percentile(data, 99.5))
    if vmax <= vmin:
        vmax = float(np.max(data))
        vmin = float(np.min(data))
    rng = max(1e-6, vmax - vmin)
    norm = np.clip((data - vmin) / rng, 0.0, 1.0).astype(np.float32)
    
    hdr = img.header
    zooms = np.array(hdr.get_zooms()[:3], dtype=np.float32)
    dimx, dimy, dimz = norm.shape
    
    # Flatten to linear buffer (Z-major scanline to match shader)
    linear = np.ascontiguousarray(norm.transpose(2, 1, 0).reshape(-1))
    return linear, norm, np.array([dimx, dimy, dimz], dtype=np.uint32), zooms


def load_seg_uint(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = nib.load(path.as_posix())
    data = img.get_fdata(dtype=np.float32)
    labels = np.rint(data).astype(np.uint32)
    dimx, dimy, dimz = labels.shape
    linear = np.ascontiguousarray(labels.transpose(2, 1, 0).reshape(-1))
    return linear, np.array([dimx, dimy, dimz], dtype=np.uint32), np.array(img.header.get_zooms()[:3])


class BraTSViewer:
    def __init__(self, case_dir: Path, up: str = "Y"):
        self.window = spy.Window(title="BraTS Viewer (Slang) - Offline Inference", resizable=True)
        self.device = spy.Device(enable_debug_layers=True, compiler_options={"include_paths": [Path(__file__).parent]})
        self.surface = self.device.create_surface(self.window)
        self.surface.configure(width=self.window.width, height=self.window.height)
        self.ui = spy.ui.Context(self.device)

        program = self.device.load_program(str(Path(__file__).parent / "brats_rt.slang"), ["brats_main"])
        self.kernel = self.device.create_compute_kernel(program)

        self.output_texture: Optional[spy.Texture] = None
        self.buffers: Dict[str, spy.Buffer] = {}
        self.raw_volumes: Dict[str, np.ndarray] = {} # Store RAW float32 for inference
        
        self.seg_buffer: Optional[spy.Buffer] = None
        self.pred_buffer: Optional[spy.Buffer] = None
        
        self.vol_dims: Optional[np.ndarray] = None
        self.voxel_size: Optional[np.ndarray] = None
        self.vol_min: Optional[np.ndarray] = None
        
        self.show_seg = True
        self.show_pred = False
        
        # Camera Setup
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
        self.camera = OrbitalCamera(initial_radius=3.0, world_up=world_up)
        self.camera.set_fov_degrees(self.fov_deg)

        # Interaction State
        self.pan_speed = 0.2
        self._lmb = False
        self._is_dragging = False
        self._press_pos = None
        self._last = None
        self._drag_threshold = 5.0 # Pixels deadzone

        self.enabled = {k: True for k in MOD_SUFFIXES.values()}
        self.weights = {k: 1.0 for k in MOD_SUFFIXES.values()}
        self.ww = 1.0
        self.wl = 0.5
        self.intensity_alpha = 0.4
        self.gamma = 1.0
        self.grad_boost = 1.5
        self.grad_scale = 1.0
        self.near_t = 0.0
        self.far_t = 0.0
        self.step_size = 0.05
        self.bg_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Fixed LUT for classes 0, 1, 2, 3
        self.lut = np.zeros((8, 4), dtype=np.float32)
        self.lut[0] = [0.0, 0.0, 0.0, 0.0]
        self.lut[1] = [0.0, 0.4, 1.0, 0.9] # NCR/NET
        self.lut[2] = [0.0, 0.8, 0.0, 0.7] # Edema
        self.lut[3] = [1.0, 0.1, 0.1, 0.9] # Enhancing
        self.lut[4] = [1.0, 0.1, 0.1, 0.9] # Backup
        self.lut = np.ascontiguousarray(self.lut, dtype=np.float32)

        self._init_ui()
        self.load_dir(case_dir)

        self.window.on_mouse_event = self.on_mouse
        self.window.on_keyboard_event = self.on_keyboard
        self.window.on_resize = self.on_resize

    def _init_ui(self):
        screen = self.ui.screen
        window = spy.ui.Window(screen, "BraTS Settings", size=spy.float2(480, 500))

        self.volume_group = spy.ui.Group(window, "Volume")
        self.info = spy.ui.Text(self.volume_group, "(no case)")
        spy.ui.Button(self.volume_group, "Load NIfTI...", callback=self.on_click_load_file)
        spy.ui.Button(self.volume_group, "Frame Volume", callback=self.on_click_frame)

        self.cam_group = spy.ui.Group(window, "Camera")
        self.fov_slider = spy.ui.SliderFloat(self.cam_group, "FOV (deg)", value=self.fov_deg, min=20.0, max=100.0)
        self.near_slider = spy.ui.SliderFloat(self.cam_group, "Near T", value=self.near_t, min=0.0, max=10.0)
        self.far_slider = spy.ui.SliderFloat(self.cam_group, "Far T (0=off)", value=self.far_t, min=0.0, max=10.0)
        
        self.int_group = spy.ui.Group(window, "Intensity")
        self.step_slider = spy.ui.SliderFloat(self.int_group, "Step", value=self.step_size, min=0.001, max=2.0)
        self.alpha_slider = spy.ui.SliderFloat(self.int_group, "Intensity Alpha", value=self.intensity_alpha, min=0.0, max=1.0)
        self.ww_slider = spy.ui.SliderFloat(self.int_group, "Window Width", value=self.ww, min=0.01, max=2.0)
        self.wl_slider = spy.ui.SliderFloat(self.int_group, "Window Level", value=self.wl, min=0.0, max=1.0)
        
        self.mod_check: Dict[str, spy.ui.CheckBox] = {}
        for key in ("T1n", "T1c", "T2w", "FLAIR"):
            self.mod_check[key] = spy.ui.CheckBox(self.int_group, f"Enable {key}", value=self.enabled.get(key, False))

        self.seg_group = spy.ui.Group(window, "Segmentation")
        self.seg_check = spy.ui.CheckBox(self.seg_group, "Show Ground Truth", value=self.show_seg)
        self.pred_check = spy.ui.CheckBox(self.seg_group, "Show INR Prediction", value=self.show_pred)
        spy.ui.Button(self.seg_group, "Load INR & Predict...", callback=self.on_click_load_inr)

    def _create_float_buffer(self, linear: np.ndarray):
        return self.device.create_buffer(element_count=linear.size, struct_size=4, usage=spy.BufferUsage.shader_resource)

    def _create_uint_buffer(self, linear: np.ndarray):
        return self.device.create_buffer(element_count=linear.size, struct_size=4, usage=spy.BufferUsage.shader_resource)

    def load_dir(self, case_dir: Path):
        files = list(Path(case_dir).glob("*.nii.gz"))
        mod_files: Dict[str, Path] = {}
        seg_file: Optional[Path] = None
        
        for f in files:
            name = f.name.lower()
            if name.endswith(("-seg.nii.gz", "_seg.nii.gz", "tumormask.nii.gz")):
                seg_file = f
                continue
            for suf, key in MOD_SUFFIXES.items():
                if name.endswith(f"-{suf}.nii.gz") or name.endswith(f"_{suf}.nii.gz"):
                    mod_files[key] = f
        if not mod_files:
            raise RuntimeError("No modality volumes found.")

        first_key = next(iter(mod_files))
        lin0, raw0, dims, vox = load_nifti_float(mod_files[first_key])
        self.vol_dims = dims.astype(np.uint32)
        max_dim = float(max(dims))
        scale = np.float32(1.8 / max_dim)
        self.voxel_size = (vox * scale).astype(np.float32)
        self.vol_min = -0.5 * (self.voxel_size * self.vol_dims.astype(np.float32))

        self.raw_volumes = {}
        
        empty = np.zeros(1, dtype=np.float32)
        self.buffers = {k: self._create_float_buffer(empty) for k in MOD_SUFFIXES.values()}
        
        # Process first
        self.raw_volumes[first_key] = raw0
        buf0 = self._create_float_buffer(lin0)
        buf0.copy_from_numpy(lin0)
        self.buffers[first_key] = buf0
        
        for key, path in mod_files.items():
            if key == first_key: continue
            lin, raw, d2, _ = load_nifti_float(path)
            if not np.all(d2 == dims): raise RuntimeError("Dim mismatch")
            self.raw_volumes[key] = raw
            b = self._create_float_buffer(lin)
            b.copy_from_numpy(lin)
            self.buffers[key] = b

        self.seg_buffer = None
        if seg_file:
            slin, sdims, _ = load_seg_uint(seg_file)
            if np.all(sdims == dims):
                self.seg_buffer = self._create_uint_buffer(slin)
                self.seg_buffer.copy_from_numpy(slin)

        self.info.text = f"Case: {case_dir.name}\n{int(dims[0])}x{int(dims[1])}x{int(dims[2])}"
        self.frame_volume()
        
        self.pred_buffer = None
        self.show_pred = False
        try: self.pred_check.value = False
        except: pass

        self._empty_float = self._create_float_buffer(np.zeros(1, dtype=np.float32))
        self._empty_uint = self._create_uint_buffer(np.zeros(1, dtype=np.uint32))

    def on_click_load_inr(self):
        if not HAS_JAX:
            self.info.text = "JAX not installed."
            return
            
        try: path = spy.platform.open_file_dialog()
        except: return
        if not path: return
        p = Path(path)
        
        try:
            params, config_raw = inr_model.model_load(p)
            train_config = config_raw.get('config', config_raw)
            
            fourier_freqs = 10
            if 'FOURIER_FREQS' in train_config:
                fourier_freqs = int(train_config['FOURIER_FREQS'])
            elif 'fourier_freqs' in train_config:
                fourier_freqs = int(train_config['fourier_freqs'])
            
            self.info.text = f"Inference (K={fourier_freqs})..."
            self.window.process_events() 
            
            req_keys = ["T1n", "T1c", "T2w", "FLAIR"]
            if not all(k in self.raw_volumes for k in req_keys):
                self.info.text = "Missing required modalities."
                return
            
            # --- PREPROCESSING (Correct Z-Score Normalization) ---
            processed_mods = []
            for k in req_keys:
                arr = self.raw_volumes[k] # float32 raw data
                mask = arr != 0
                if mask.any():
                    mu = arr[mask].mean()
                    sigma = arr[mask].std() + 1e-6
                    arr = (arr - mu) / sigma
                processed_mods.append(arr)
            
            mods_np = np.stack(processed_mods, axis=0) # (4, H, W, D)
            
            # Run Inference
            case_data = {"mods": mods_np, "seg": None}
            pred_vol, _ = inr_model.predict_volume(params, case_data, fourier_freqs=fourier_freqs)
            
            # Upload to GPU
            # Flatten Z-major to match shader layout
            pred_linear = np.ascontiguousarray(pred_vol.transpose(2, 1, 0).reshape(-1))
            self.pred_buffer = self._create_uint_buffer(pred_linear.astype(np.uint32))
            self.pred_buffer.copy_from_numpy(pred_linear.astype(np.uint32))
            
            self.show_pred = True
            try: self.pred_check.value = True
            except: pass
            
            self.info.text = "Inference Done."
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.info.text = f"Inference Failed: {e}"

    def on_click_load_file(self):
        try: path = spy.platform.open_file_dialog(); 
        except: return
        if path: self.load_dir(Path(path).parent)

    def on_click_frame(self): self.frame_volume()
    def on_click_reset_camera(self): self.camera.reset(); self.frame_volume()
    
    def frame_volume(self):
        if self.vol_dims is None: return
        center = (self.vol_min + 0.5 * self.voxel_size * self.vol_dims.astype(np.float32))
        self.camera.target = center
        self.camera.radius = np.linalg.norm(self.voxel_size * self.vol_dims.astype(np.float32)) * 0.8

    def on_mouse(self, event):
        if self.ui.handle_mouse_event(event): return
        
        if event.type == spy.MouseEventType.button_down:
            if event.button == spy.MouseButton.left:
                self._lmb = True
                self._press_pos = event.pos
                self._last = event.pos
                self._is_dragging = False
                
        elif event.type == spy.MouseEventType.button_up:
            if event.button == spy.MouseButton.left:
                self._lmb = False
                self._is_dragging = False

        elif event.type == spy.MouseEventType.move and self._lmb:
            # Deadzone logic to prevent jumps on click
            if not self._is_dragging:
                dx = event.pos.x - self._press_pos.x
                dy = event.pos.y - self._press_pos.y
                if (dx*dx + dy*dy) > (self._drag_threshold * self._drag_threshold):
                    self._is_dragging = True
                    # Reset last to avoid jump
                    self._last = event.pos 
            
            if self._is_dragging:
                dx = event.pos.x - self._last.x
                dy = event.pos.y - self._last.y
                
                if event.has_modifier(spy.KeyModifier.shift): 
                    self.camera.pan(dx * self.pan_speed, dy * self.pan_speed, self.window.height)
                else: 
                    self.camera.orbit(dx * 0.01, dy * 0.01)
                
                self._last = event.pos

        elif event.is_scroll(): 
            self.camera.zoom(pow(1.1, -event.scroll.y))

    def on_keyboard(self, e): 
        if not self.ui.handle_keyboard_event(e) and e.key == spy.KeyCode.escape: self.window.close()
    def on_resize(self, w, h): self.surface.configure(width=w, height=h) if w > 0 else None

    def run(self):
        while not self.window.should_close():
            self.window.process_events()
            if not self.surface.config: continue
            tex = self.surface.acquire_next_image()
            if not tex: continue
            
            if not self.output_texture or self.output_texture.width != tex.width:
                self.output_texture = self.device.create_texture(
                    format=spy.Format.rgba16_float, 
                    width=tex.width, 
                    height=tex.height,
                    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access
                )

            self.step_size = float(self.step_slider.value)
            try: 
                self.show_seg = bool(self.seg_check.value)
                self.show_pred = bool(self.pred_check.value)
            except: pass
            
            eye, gU, gV, gW = self.camera.get_basis()
            
            en = [1 if self.enabled.get(k) else 0 for k in ("T1n", "T1c", "T2w", "FLAIR")]
            wt = [float(self.weights.get(k)) for k in ("T1n", "T1c", "T2w", "FLAIR")]
            
            params = {
                "imageSize": (tex.width, tex.height),
                "fovY": math.radians(self.fov_deg),
                "eye": eye.astype(np.float32), 
                "U": gU.astype(np.float32), 
                "V": gV.astype(np.float32), 
                "W": gW.astype(np.float32),
                "volMin": self.vol_min if self.vol_min is not None else np.zeros(3, dtype=np.float32),
                "voxelSize": self.voxel_size if self.voxel_size is not None else np.ones(3, dtype=np.float32),
                "dims": self.vol_dims if self.vol_dims is not None else np.ones(3, dtype=np.uint32),
                "stepSize": self.step_size,
                "nearT": self.near_t, "farT": self.far_t,
                "bgColor": self.bg_color,
                "volEnabled": tuple(map(np.uint32, en)),
                "volWeight": tuple(map(float, wt)),
                "ww": float(self.ww_slider.value), "wl": float(self.wl_slider.value),
                "intensityAlpha": float(self.alpha_slider.value),
                "gamma": 1.0, "gradBoost": 1.5, "gradScale": 1.0,
                "showSeg": 1 if (self.show_seg and self.seg_buffer) else 0,
                "showPred": 1 if (self.show_pred and self.pred_buffer) else 0,
                "lutColorAlpha": [tuple(map(float, row)) for row in self.lut.tolist()]
            }

            ce = self.device.create_command_encoder()
            def _b(k): return self.buffers.get(k, self._empty_float)
            
            self.kernel.dispatch(
                thread_count=[tex.width, tex.height, 1],
                vars={
                    "gOutput": self.output_texture,
                    "gIntensity0": _b("T1n"), "gIntensity1": _b("T1c"), 
                    "gIntensity2": _b("T2w"), "gIntensity3": _b("FLAIR"),
                    "gLabels": self.seg_buffer or self._empty_uint,
                    "gPreds": self.pred_buffer or self._empty_uint,
                    "gParams": params
                },
                command_encoder=ce
            )
            
            # Correct Blit order: dst, src
            ce.blit(tex, self.output_texture)
            
            self.ui.begin_frame(tex.width, tex.height)
            self.ui.end_frame(tex, ce)
            self.device.submit_command_buffer(ce.finish())
            self.surface.present()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str)
    p.add_argument("--data_root", type=str, default=str(Path(__file__).parent / "data" / "BraTS-2023"))
    p.add_argument("--up", type=str, default="Y", help="World up axis: X|Y|Z|-X|-Y|-Z")
    args = p.parse_args()
    case = Path(args.dir) if args.dir else None
    if not case or not case.exists():
        root = Path(args.data_root)
        if root.exists():
            for d in root.glob("BraTS-GLI-*"):
                if any(d.glob("*.nii.gz")): case = d; break
    
    if not case: raise SystemExit("No case found.")
    BraTSViewer(case, up=args.up).run()