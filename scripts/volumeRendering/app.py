# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import gzip
from pathlib import Path
import numpy as np
import slangpy as spy

# Reuse orbital camera from raymarch example
import sys
HERE = Path(__file__).parent
sys.path.append(str(HERE.parent / "raymarch"))
from camera import OrbitalCamera

class App:
    def __init__(self):
        self.window = spy.Window(title="Volume Rendering (SlangPy)", resizable=True)
        self.device = spy.Device(enable_debug_layers=True, compiler_options={"include_paths": [HERE]})
        self.surface = self.device.create_surface(self.window)
        self.surface.configure(width=self.window.width, height=self.window.height)

        self.ui = spy.ui.Context(self.device)

        # Output
        self.output_texture = None

        # Load shader program
        program = self.device.load_program("volume_render.slang", ["volume_cs"])
        self.kernel = self.device.create_compute_kernel(program)

        # Camera
        self.camera = OrbitalCamera(initial_radius=4.2, initial_phi=np.radians(80.0), initial_theta=np.radians(25.0))

        # Volume state
        self.volume_tex = None
        self.volume_buf = None
        self.sampler = self.device.create_sampler(min_filter=spy.TextureFilteringMode.linear,
                                                  mag_filter=spy.TextureFilteringMode.linear,
                                                  mip_filter=spy.TextureFilteringMode.linear,
                                                  max_anisotropy=16)
        # Small helper to create structured buffers across slangpy versions
        def create_structured_buffer(elem_count: int, elem_bytes: int):
            for kwargs in (
                dict(element_count=elem_count, element_size=elem_bytes, usage=spy.BufferUsage.shader_resource),
                dict(element_count=elem_count, struct_size=elem_bytes, usage=spy.BufferUsage.shader_resource),
                dict(element_count=elem_count, usage=spy.BufferUsage.shader_resource),
            ):
                try:
                    return self.device.create_buffer(**kwargs)
                except TypeError:
                    continue
            try:
                return self.device.create_buffer(elem_count)
            except Exception as e:
                raise RuntimeError(f"Failed to create buffer with any signature: {e}")
        self._create_structured_buffer = create_structured_buffer
        # Volume dataset options
        self.volume_width = 180
        self.volume_height = 216
        self.volume_depth = 180
        self.format_options = {
            "r8unorm": HERE / "assets/volume/t1_icbm_normal_1mm_pn0_rf0_180x216x180_uint8_1x1.bin-gz",
            # Compressed ASTC/BC4 files are copied but not loaded here (SlangPy upload expects unpacked data).
        }
        self.selected_format_key = "r8unorm"
        # Load default r8 volume before UI setup
        self._load_volume_r8(self.format_options[self.selected_format_key])

        # UI elements
        self._setup_ui()
        # Show immediate status for visibility on start
        try:
            self.status.text = getattr(self, 'status_text', 'Ready')
        except Exception:
            pass

        # Events
        self.window.on_keyboard_event = self.on_keyboard
        self.window.on_mouse_event = self.on_mouse_event
        self.window.on_resize = self.on_resize

        # Orbit drag state (match mesh_rt)
        self._lmb = False
        self._last = None
        self._press_pos = None
        self._orbit_dragging = False
        self.drag_deadzone_px = 6.0

    def _setup_ui(self):
        screen = self.ui.screen
        w = spy.ui.Window(screen, "Volume Controls", size=spy.float2(420, 260))
        self.status = spy.ui.Text(w, "Ready")
        # Match WebGPU sample: rotate toggle, near/far, and texture format selector
        self.rotate_checkbox = spy.ui.CheckBox(w, "Rotate Camera", value=True)
        # Distance (camera radius)
        self.radius_slider = spy.ui.SliderFloat(w, "Distance", value=float(self.camera.radius), min=1.0, max=10.0)
        # Render UI
        self.step_slider = spy.ui.SliderInt(w, "Steps", value=64, min=16, max=512)
        self.near_slider = spy.ui.SliderFloat(w, "Near", value=4.3, min=1.0, max=7.0)
        self.far_slider = spy.ui.SliderFloat(w, "Far", value=4.4, min=1.0, max=7.0)

        # Texture format selection (use ComboBox as in SlangPy UI)
        self._formats = [
            ("r8unorm", None, self.format_options["r8unorm"]),
            ("bc4-r-unorm", "texture-compression-bc-sliced-3d", HERE / "assets/volume/t1_icbm_normal_1mm_pn0_rf0_180x216x180_bc4_4x4.bin-gz"),
            ("astc-12x12-unorm", "texture-compression-astc-sliced-3d", HERE / "assets/volume/t1_icbm_normal_1mm_pn0_rf0_180x216x180_astc_12x12.bin-gz"),
        ]
        self._format_index = 0
        def on_format_change(new_index: int):
            if new_index < 0 or new_index >= len(self._formats):
                return
            self._format_index = int(new_index)
            name, feature, path = self._formats[self._format_index]
            self._apply_format_selection(name, feature, path)
        items = [name for (name, _, __) in self._formats]
        self.format_combo = spy.ui.ComboBox(w, label="Texture Format", value=self._format_index, callback=on_format_change, items=items)

    def _apply_format_selection(self, name: str, feature: str | None, path: Path):
        # Implement r8 and BC4 via software (ASTC not implemented)
        try:
            if name == "r8unorm":
                self._load_volume_r8(path)
            elif name.startswith("bc4"):
                self._load_volume_bc4(path)
            else:
                # Mirror WebGPU sample message when a compression feature is missing.
                msg = f"{feature or name} not supported"
                self.status_text = msg
                self.volume_buf = None
        except Exception as e:
            self.status_text = f"Volume load failed: {e}"
            self.volume_buf = None

    def _upload_u8_volume_from_array(self, voxels_u8: np.ndarray):
        expected = self.volume_width * self.volume_height * self.volume_depth
        if voxels_u8.size != expected:
            raise RuntimeError(f"Unexpected data size: {voxels_u8.size} vs {expected}")
        arr = voxels_u8
        pad = (-arr.size) % 4
        if pad:
            arr = np.pad(arr, (0, pad), mode='constant')
        arr_u32 = arr.astype(np.uint32, copy=False).reshape((-1, 4))
        try:
            self.volume_buf = spy.Buffer.from_numpy(self.device, arr_u32)
        except Exception:
            self.volume_buf = self._create_structured_buffer(arr_u32.shape[0], 16)
            self.volume_buf.copy_from_numpy(arr_u32)

    def _load_volume_r8(self, path: Path):
        with gzip.open(path, "rb") as f:
            raw = f.read()
        arr = np.frombuffer(raw, dtype=np.uint8)
        self._upload_u8_volume_from_array(arr)
        self.status_text = f"Loaded: {path.name} (r8unorm)"

    def _load_volume_bc4(self, path: Path):
        # Decompress gzip to BC4 block stream, decode all tiles with NumPy
        with gzip.open(path, "rb") as f:
            bc = f.read()
        W, H, D = self.volume_width, self.volume_height, self.volume_depth
        bw = (W + 3) // 4
        bh = (H + 3) // 4
        nb = bw * bh
        block_size = 8  # BC4
        expected_bytes = D * nb * block_size
        if len(bc) != expected_bytes:
            raise RuntimeError(f"BC4 data size mismatch: {len(bc)} vs {expected_bytes}")

        blocks = np.frombuffer(bc, dtype=np.uint8).reshape(D, nb, 8)
        r0 = blocks[:, :, 0].astype(np.int32)
        r1 = blocks[:, :, 1].astype(np.int32)
        idxb = blocks[:, :, 2:8].astype(np.uint64)
        # Reconstruct 48-bit index per block
        shifts8 = np.array([0, 8, 16, 24, 32, 40], dtype=np.uint64)
        idx = np.bitwise_or.reduce(idxb << shifts8, axis=2)

        # Build palettes
        palette = np.empty((D, nb, 8), dtype=np.int32)
        palette[:, :, 0] = r0
        palette[:, :, 1] = r1
        gt = r0 > r1
        # r0 > r1: 6 interpolated entries
        for i in range(1, 7):
            val = (((7 - i) * r0 + i * r1) + 3) // 7
            palette[:, :, i + 1] = np.where(gt, val, 0)
        # r0 <= r1: 4 interpolated, then 0 and 255
        for i in range(1, 5):
            val = (((5 - i) * r0 + i * r1) + 2) // 5
            palette[:, :, i + 1] = np.where(gt, palette[:, :, i + 1], val)
        palette[:, :, 6] = np.where(gt, palette[:, :, 6], 0)
        palette[:, :, 7] = np.where(gt, palette[:, :, 7], 255)
        palette = palette.astype(np.uint8)

        # Extract 16 3-bit codes per block in parallel
        shifts3 = (np.arange(16, dtype=np.uint64) * 3).reshape(1, 1, 16)
        codes = ((idx[:, :, None] >> shifts3) & 0x7).astype(np.uint8)
        # Gather palette values per code
        vals = np.take_along_axis(palette, codes, axis=2).astype(np.uint8)  # (D, nb, 16)
        tiles = vals.reshape(D, bh, bw, 4, 4)
        # Tile into full slices
        slices = tiles.transpose(0, 1, 3, 2, 4).reshape(D, bh * 4, bw * 4)
        out = slices[:, :H, :W]

        vox = out.reshape(D * H * W)
        self._upload_u8_volume_from_array(vox)
        self.status_text = f"Loaded (CPU decode, optimized): {path.name} (bc4-r-unorm)"

    def on_keyboard(self, ev: spy.KeyboardEvent):
        if self.ui.handle_keyboard_event(ev):
            return
        if ev.type == spy.KeyboardEventType.key_press and ev.key == spy.KeyCode.escape:
            self.window.close()

    def on_mouse_event(self, ev: spy.MouseEvent):
        if self.ui.handle_mouse_event(ev):
            return
        if ev.type == spy.MouseEventType.button_down and ev.button == spy.MouseButton.left:
            self._lmb = True
            self._last = ev.pos
            self._press_pos = ev.pos
            self._orbit_dragging = False
        elif ev.type == spy.MouseEventType.button_up and ev.button == spy.MouseButton.left:
            self._lmb = False
            self._orbit_dragging = False
        elif ev.type == spy.MouseEventType.move and self._lmb:
            if not self._orbit_dragging:
                pdx = float(ev.pos.x - self._press_pos.x)
                pdy = float(ev.pos.y - self._press_pos.y)
                if (pdx*pdx + pdy*pdy) >= (self.drag_deadzone_px * self.drag_deadzone_px):
                    self._orbit_dragging = True
                    self._last = ev.pos
            else:
                dx = float(ev.pos.x - self._last.x)
                dy = float(ev.pos.y - self._last.y)
                self.camera.orbit(dx * 0.01, dy * 0.01)
                self._last = ev.pos
        elif hasattr(ev, "is_scroll") and ev.is_scroll():
            sy = float(ev.scroll.y)
            self.camera.zoom(pow(1.15, -sy))
            # Keep UI in sync with camera radius
            try:
                self.radius_slider.value = float(self.camera.radius)
            except Exception:
                pass

    def on_resize(self, w: int, h: int):
        self.device.wait()
        if w > 0 and h > 0:
            self.surface.configure(width=w, height=h)
        else:
            self.surface.unconfigure()

    def run(self):
        timer = spy.Timer()
        fps_avg = 0.0
        while not self.window.should_close():
            self.window.process_events()

            elapsed = max(1e-6, timer.elapsed_s()); timer.reset()
            fps_avg = 0.95 * fps_avg + 0.05 * (1.0 / elapsed)

            if not self.surface.config:
                continue
            surface_tex = self.surface.acquire_next_image()
            if not surface_tex:
                continue

            # Resize output if needed
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
                    label="volume_output",
                )

            # Apply distance from slider
            try:
                self.camera.radius = max(self.camera.min_radius, min(self.camera.max_radius, float(self.radius_slider.value)))
            except Exception:
                pass
            eye, gU, gV, gW = self.camera.get_basis()

            # Use FOV similar to WebGPU sample (72 deg)
            params = {
                "imageSize": (np.uint32(self.output_texture.width), np.uint32(self.output_texture.height)),
                "fovY": np.float32(np.radians(72.0)),
                "stepCount": np.float32(float(self.step_slider.value)),
                "nearPlane": np.float32(float(self.near_slider.value)),
                "farPlane": np.float32(float(self.far_slider.value)),
                "eye": eye.astype(np.float32),
                "U": gU.astype(np.float32),
                "V": gV.astype(np.float32),
                "W": gW.astype(np.float32),
                "volDim": (np.uint32(self.volume_width), np.uint32(self.volume_height), np.uint32(self.volume_depth)),
            }

            cmd = self.device.create_command_encoder()
            # Dispatch compute raymarch if volume is ready
            if self.volume_buf is not None:
                self.kernel.dispatch(
                    thread_count=[self.output_texture.width, self.output_texture.height, 1],
                    vars={
                        "gOutput": self.output_texture,
                        "gParams": params,
                        "gVolumeU8": self.volume_buf,
                    },
                    command_encoder=cmd,
                )

            # Blit compute output first
            cmd.blit(surface_tex, self.output_texture)

            # UI after blit so it overlays
            self.ui.new_frame(surface_tex.width, surface_tex.height)
            try:
                self.status.text = f"FPS: {fps_avg:.2f} | {self.status_text}"
            except Exception:
                pass
            self.ui.render(surface_tex, cmd)

            # Present
            self.device.submit_command_buffer(cmd.finish())
            del surface_tex
            self.surface.present()

            # Auto-rotate to mirror sample
            try:
                if bool(self.rotate_checkbox.value):
                    # Slow horizontal orbit
                    self.camera.orbit(0.6 * elapsed, 0.0)
                    # Keep UI radius synced if any drift
                    self.radius_slider.value = float(self.camera.radius)
            except Exception:
                pass


if __name__ == "__main__":
    App().run()
