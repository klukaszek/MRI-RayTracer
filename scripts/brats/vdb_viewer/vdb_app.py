from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import slangpy as spy
import openvdb as vdb

from .camera import OrbitalCamera

HERE = Path(__file__).parent


def load_vdb_dense(path: Path):
    md = vdb.readAllGridMetadata(path.as_posix())
    if not md:
        raise RuntimeError("No grids in VDB file")
    name = md[0].name or "density"
    grid = vdb.read(path.as_posix(), name)
    mn, mx = grid.evalActiveVoxelBoundingBox()
    dims = (mx[0] - mn[0] + 1, mx[1] - mn[1] + 1, mx[2] - mn[2] + 1)
    # Allocate X,Y,Z (x-fastest) layout and copy from VDB
    arr = np.zeros((dims[0], dims[1], dims[2]), dtype=np.float32)
    grid.copyToArray(arr, mn)
    # Normalize/improve range a bit
    vmin = arr.min(); vmax = arr.max()
    scale = 1.0 if vmax <= 0 else 1.0 / float(vmax)
    arr *= scale
    # Build world transform from VDB transform; assume voxelSize is uniform-ish
    vs = grid.transform.voxelSize()
    voxel_size = np.array([float(vs[0]), float(vs[1]), float(vs[2])], dtype=np.float32)
    return arr, np.array(mn, dtype=np.int32), voxel_size


class VDBApp:
    def __init__(self, vdb_path: Path | None = None, up: str = "Y"):
        self.window = spy.Window(title="VDB Volume (Slang/Metal)", resizable=True)
        self.device = spy.Device(enable_debug_layers=True, compiler_options={"include_paths": [HERE]})
        self.surface = self.device.create_surface(self.window)
        self.surface.configure(width=self.window.width, height=self.window.height)
        self.ui = spy.ui.Context(self.device)

        program = self.device.load_program(str(HERE / "vdb_rt.slang"), ["volume_main"])
        self.kernel = self.device.create_compute_kernel(program)

        self.output_texture = None
        self.volume_buffer = None
        self.vol_dims = None
        self.vol_min = None
        self.voxel_size = None

        # Input & interaction state (set early for UI init safety)
        self._lmb = False
        self._last = None
        self._press_pos = None
        self._orbit_dragging = False
        self._pan_dragging = False
        self.drag_deadzone_px = 6.0
        self.pan_speed = 0.2
        self.fps_avg = 0.0
        # Dropdown open states
        self.stats_open = True
        self.volume_open = True
        self.controls_open = True

        up_map = {
            "X": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "Y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "Z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "-X": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            "-Y": np.array([0.0, -1.0, 0.0], dtype=np.float32),
            "-Z": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        }
        world_up = up_map.get(up.upper(), up_map["Y"]) if isinstance(up, str) else up_map["Y"]
        self.camera = OrbitalCamera(initial_radius=3.2, initial_phi=math.radians(50), initial_theta=math.radians(25), world_up=world_up)
        self.fov_deg = 70.0
        # Keep camera's FOV in sync with UI/default
        try:
            self.camera.set_fov_degrees(self.fov_deg)
        except Exception:
            pass

        self.sigma_scale = 8.0
        self.step_size = 0.75  # in world units; updated after load
        self.albedo = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.bg_color = np.array([0.1, 0.12, 0.16], dtype=np.float32)

        self.setup_ui()

        if vdb_path is not None:
            self.load_vdb(vdb_path)

        self.window.on_keyboard_event = self.on_keyboard_event
        self.window.on_mouse_event = self.on_mouse_event
        self.window.on_resize = self.on_resize

    def setup_ui(self):
        screen = self.ui.screen
        window = spy.ui.Window(screen, "VDB Volume Settings", size=spy.float2(420, 300))

        # Stats dropdown header (button toggles visibility) + contents
        self.stats_group = spy.ui.Group(window, "Stats")
        self.fps_text = spy.ui.Text(self.stats_group, "FPS: 0.0 (0.00 ms)")

        # Volume dropdown header (button toggles visibility) + contents
        self.volume_group = spy.ui.Group(window, "Volume")
        self.volume_info = spy.ui.Text(self.volume_group, "Volume: (none)")
        spy.ui.Button(self.volume_group, "Load VDB...", callback=self.on_click_load)
        spy.ui.Button(self.volume_group, "Frame Volume", callback=self.on_click_frame)

        # Render controls
        self.controls_group = spy.ui.Group(window, "Controls")
        self.fov_slider = spy.ui.SliderFloat(self.controls_group, "FOV Y (deg)", value=self.fov_deg, min=20.0, max=90.0)
        self.sigma_slider = spy.ui.SliderFloat(self.controls_group, "Sigma Scale", value=self.sigma_scale, min=0.1, max=64.0)
        # Default to a fine-grained range; refined after VDB load
        self.step_slider = spy.ui.SliderFloat(self.controls_group, "Step (world)", value=self.step_size, min=1e-4, max=0.1)
        self.pan_slider = spy.ui.SliderFloat(self.controls_group, "Pan Speed", value=self.pan_speed, min=0.05, max=2.0)
        spy.ui.Text(self.controls_group, "Orbit: LMB | Pan: Shift+LMB | Zoom: Wheel")

        # Initial visibility from state
        self.stats_group.visible = bool(self.stats_open)
        self.volume_group.visible = bool(self.volume_open)
        self.controls_group.visible = bool(self.controls_open)

    def on_click_load(self):
        try:
            path = spy.platform.open_file_dialog()
        except Exception as e:
            self.volume_info.text = f"Open dialog error: {e}"
            return
        if path:
            self.load_vdb(Path(path))

    def on_click_frame(self):
        self.frame_volume()

    def on_toggle_stats(self):
        self.stats_open = not self.stats_open
        self.stats_group.visible = bool(self.stats_open)
        try:
            self.stats_header.label = "▼ Stats" if self.stats_open else "▶ Stats"
        except Exception:
            pass

    def on_toggle_volume(self):
        self.volume_open = not self.volume_open
        self.volume_group.visible = bool(self.volume_open)
        try:
            self.volume_header.label = "▼ Volume" if self.volume_open else "▶ Volume"
        except Exception:
            pass

    def _create_float_buffer(self, linear: np.ndarray):
        # Try multiple create_buffer signatures across slangpy versions
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
        # Fallback
        return self.device.create_buffer(linear.size)

    def load_vdb(self, path: Path):
        try:
            arr, mn, voxel_size = load_vdb_dense(path)
            # Upload as linear buffer (x-fastest layout)
            dimx, dimy, dimz = arr.shape
            self.vol_dims = np.array([dimx, dimy, dimz], dtype=np.uint32)
            # Normalize volume to a ~unit cube for viewing
            max_dim = float(max(dimx, dimy, dimz))
            scale = np.float32(1.8 / max_dim)
            self.voxel_size = (voxel_size.astype(np.float32) * scale).astype(np.float32)
            # Center and frame the volume in world space
            extent = self.voxel_size * self.vol_dims.astype(np.float32)
            self.vol_min = -0.5 * extent
            self.frame_volume()
            # Linearize with X-fastest layout expected by shader:
            # buffer index = x + y*dimX + z*dimX*dimY
            # Use C-order flatten of (Z,Y,X) view to achieve X-fastest layout
            linear = np.ascontiguousarray(arr.transpose(2, 1, 0).reshape(-1))
            self.volume_buffer = self._create_float_buffer(linear)
            self.volume_buffer.copy_from_numpy(linear)
            # Set a reasonable step: fraction of smallest voxel, and tune slider range around it
            vmin = float(np.min(self.voxel_size))
            self.step_size = 0.75 * vmin
            try:
                # Dynamically adapt slider range to voxel size
                self.step_slider.min = max(1e-6, 0.05 * vmin)
                self.step_slider.max = max(self.step_slider.min * 2.0, 5.0 * vmin)
                self.step_slider.value = self.step_size
            except Exception:
                pass
            # Pretty volume info string
            vs = self.voxel_size
            self.volume_info.text = (
                f"Volume: {path.name}\n"
                f"  Dims: {dimx} x {dimy} x {dimz}\n"
                f"  Voxel: ({vs[0]:.4g}, {vs[1]:.4g}, {vs[2]:.4g}) world units\n"
                f"  Bounds min: ({self.vol_min[0]:.3f}, {self.vol_min[1]:.3f}, {self.vol_min[2]:.3f})"
            )
        except Exception as e:
            self.volume_info.text = f"Volume: Error — {e}"

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

    def on_keyboard_event(self, event: spy.KeyboardEvent):
        if self.ui.handle_keyboard_event(event):
            return
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape:
                self.window.close()

    def on_mouse_event(self, event: spy.MouseEvent):
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
                # Prefer explicit bit check if available
                is_shift = event.has_modifier(spy.KeyModifier.shift) if hasattr(event, 'has_modifier') else False
            except Exception:
                is_shift = False
            # If we were panning and shift was released, reset references to avoid jump
            if self._pan_dragging and not is_shift:
                self._pan_dragging = False
                self._orbit_dragging = False
                self._press_pos = event.pos
                self._last = event.pos
                return
            if is_shift and getattr(self, "_lmb", False):
                # Enforce a small deadzone before starting pan
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
                # When panning, do not accumulate orbit dragging
                self._orbit_dragging = False
            elif getattr(self, "_lmb", False):
                # Enforce a small deadzone before starting orbit to avoid accidental micro-drags
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
        timer = spy.Timer()
        while not self.window.should_close():
            self.window.process_events()
            # self.ui.process_events()
            elapsed = max(1e-6, timer.elapsed_s()); timer.reset()
            fps_raw = 1.0 / elapsed
            self.fps_avg = 0.95 * self.fps_avg + 0.05 * fps_raw
            ms = elapsed * 1000.0
            self.fps_text.text = f"FPS: {fps_raw:.2f} ({ms:.2f} ms)"

            if not self.surface.config:
                continue
            surface_texture = self.surface.acquire_next_image()
            if not surface_texture:
                continue

            if (
                self.output_texture is None
                or self.output_texture.width != surface_texture.width
                or self.output_texture.height != surface_texture.height
            ):
                self.output_texture = self.device.create_texture(
                    format=spy.Format.rgba16_float,
                    width=surface_texture.width,
                    height=surface_texture.height,
                    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                    label="vdb_rt_output",
                )

            eye, gU, gV, gW = self.camera.get_basis()
            self.sigma_scale = float(self.sigma_slider.value)
            self.step_size = float(self.step_slider.value)

            command_encoder = self.device.create_command_encoder()
            if self.volume_buffer is not None:
                params = {
                    "imageSize": (np.uint32(self.output_texture.width), np.uint32(self.output_texture.height)),
                    "fovY": np.float32(math.radians(float(self.fov_slider.value))),
                    "eye": eye.astype(np.float32),
                    "U": gU.astype(np.float32),
                    "V": gV.astype(np.float32),
                    "W": gW.astype(np.float32),
                    "volMin": (self.vol_min.astype(np.float32) if isinstance(self.vol_min, np.ndarray) else np.array(self.vol_min if self.vol_min is not None else [0.0,0.0,0.0], dtype=np.float32)),
                    "voxelSize": self.voxel_size.astype(np.float32),
                    "dims": self.vol_dims.astype(np.uint32),
                    "stepSize": np.float32(self.step_size),
                    "sigmaScale": np.float32(self.sigma_scale),
                    "albedo": self.albedo.astype(np.float32),
                    "bgColor": self.bg_color.astype(np.float32),
                }
                self.kernel.dispatch(
                    thread_count=[self.output_texture.width, self.output_texture.height, 1],
                    vars={
                        "gOutput": self.output_texture,
                        "gVolume": self.volume_buffer,
                        "gParams": params,
                    },
                    command_encoder=command_encoder,
                )
                command_encoder.blit(surface_texture, self.output_texture)

            self.ui.begin_frame(surface_texture.width, surface_texture.height)
            self.ui.end_frame(surface_texture, command_encoder)

            self.device.submit_command_buffer(command_encoder.finish())
            del surface_texture
            self.surface.present()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simple VDB volume renderer (Slang)")
    parser.add_argument("--vdb", type=str, default=None, help="Path to .vdb file (defaults to CloudPack sample)")
    parser.add_argument("--up", type=str, default="Y", help="World up axis: X|Y|Z|-X|-Y|-Z")
    args = parser.parse_args()
    default_vdb = HERE.parent / "CloudPack" / "CloudPackVDB" / "cloud_01_variant_0000.vdb"
    vdb_path = Path(args.vdb) if args.vdb else (default_vdb if default_vdb.exists() else None)
    app = VDBApp(vdb_path, up=args.up)
    app.run()
