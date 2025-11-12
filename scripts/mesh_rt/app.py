from __future__ import annotations

import math
from pathlib import Path
import sys
import numpy as np
import slangpy as spy

# Allow importing the shared orbital camera from scripts/raymarch
HERE = Path(__file__).parent
sys.path.append(str(HERE.parent / "raymarch"))
from camera import OrbitalCamera  # type: ignore
from ply_loader import load_ply_ascii
from bvh import build_bvh

HERE = Path(__file__).parent


class App:
    def __init__(self):
        self.window = spy.Window(title="Mesh RT (Slang/Metal)", resizable=True)
        self.device = spy.Device(enable_debug_layers=True, compiler_options={"include_paths": [HERE]})
        self.surface = self.device.create_surface(self.window)
        self.surface.configure(width=self.window.width, height=self.window.height)

        self.ui = spy.ui.Context(self.device)

        # Output + resources (populated after user loads a mesh)
        self.output_texture = None
        self.buf_nodes = None
        self.buf_tris = None
        self.buf_verts = None
        self.bvh = None

        # Shader program
        program = self.device.load_program("mesh_rt.slang", ["compute_main"])
        self.kernel = self.device.create_compute_kernel(program)

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

        # Camera
        self.camera = OrbitalCamera(initial_radius=2.2, initial_phi=math.radians(60), initial_theta=math.radians(25))

        # Params
        self.fov_deg = 45.0

        # UI
        self.setup_ui()

        # Events
        self.window.on_keyboard_event = self.on_keyboard_event
        self.window.on_mouse_event = self.on_mouse_event
        self.window.on_resize = self.on_resize

        self.fps_avg = 0.0
        # Orbit drag state
        self._lmb = False
        self._last = None
        self._press_pos = None
        self._orbit_dragging = False
        self.drag_deadzone_px = 6.0

    def load_mesh(self, path: Path):
        try:
            self.progress.fraction = 0.05
            self.mesh_info.text = f"Mesh: Loading… ({path.name})"
            verts, tris = load_ply_ascii(path, max_faces=None)
            self.progress.fraction = 0.3
            # Normalize to unit box and center
            vmin = verts.min(axis=0); vmax = verts.max(axis=0)
            center = 0.5 * (vmin + vmax)
            scale = 1.0 / float(np.max(vmax - vmin) + 1e-8)
            verts = (verts - center) * scale * 1.8

            self.progress.fraction = 0.5
            self.bvh = build_bvh(verts, tris, max_leaf_tris=4)

            # Upload
            nodes_packed = self.bvh.nodes.astype(np.float32, copy=False)
            nodes4 = nodes_packed.reshape((-1, 4)).astype(np.float32, copy=False)
            self.buf_nodes = self._create_structured_buffer(nodes4.shape[0], 16)
            self.buf_nodes.copy_from_numpy(nodes4)

            tris_u32 = self.bvh.tris.astype(np.uint32, copy=False)
            if tris_u32.shape[1] == 3:
                pad = np.zeros((tris_u32.shape[0], 1), dtype=np.uint32)
                tris_u32 = np.concatenate([tris_u32, pad], axis=1)
            self.buf_tris = self._create_structured_buffer(tris_u32.shape[0], 16)
            self.buf_tris.copy_from_numpy(tris_u32)

            v = self.bvh.vert_pos.astype(np.float32, copy=False)
            if v.shape[1] == 3:
                vw = np.ones((v.shape[0], 1), dtype=np.float32)
                v = np.concatenate([v, vw], axis=1)
            self.buf_verts = self._create_structured_buffer(v.shape[0], 16)
            self.buf_verts.copy_from_numpy(v)

            self.progress.fraction = 1.0
            self.mesh_info.text = f"Mesh: {path.name} | V:{len(self.bvh.vert_pos)} T:{len(self.bvh.tris)}"
        except Exception as e:
            self.mesh_info.text = f"Mesh: Error — {e}"
            self.progress.fraction = 0.0

    def setup_ui(self):
        screen = self.ui.screen
        window = spy.ui.Window(screen, "Mesh RT Settings", size=spy.float2(360, 180))
        self.fps_text = spy.ui.Text(window, "FPS: 0")
        self.fov_slider = spy.ui.SliderFloat(window, "FOV Y (deg)", value=self.fov_deg, min=20.0, max=90.0)
        spy.ui.Text(window, "Orbit: LMB drag, Zoom: wheel")
        # Mesh info + progress (used during startup load and future loads)
        self.mesh_info = spy.ui.Text(window, "Mesh: (none)")
        self.progress = spy.ui.ProgressBar(window, fraction=0.0)
        spy.ui.Button(window, "Load Mesh...", callback=self.on_click_load)

    def on_click_load(self):
        try:
            # Call without filters to avoid backend issues on macOS
            path = spy.platform.open_file_dialog()
            print("No Error!")
        except Exception as e:
            self.mesh_info.text = f"Open dialog error: {e}"
            return
        if path:
            self.load_mesh(Path(path))
        else:
            self.mesh_info.text = "Mesh: (cancelled)"

    def on_keyboard_event(self, event: spy.KeyboardEvent):
        if self.ui.handle_keyboard_event(event):
            return
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape:
                self.window.close()

    # LMB orbit with drag deadzone + wheel zoom
    def on_mouse_event(self, event: spy.MouseEvent):
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
        elif event.type == spy.MouseEventType.move and getattr(self, "_lmb", False):
            # Enforce a small deadzone before starting orbit to avoid accidental micro-drags
            if not self._orbit_dragging and self._press_pos is not None:
                pdx = float(event.pos.x - self._press_pos.x)
                pdy = float(event.pos.y - self._press_pos.y)
                if (pdx * pdx + pdy * pdy) >= (self.drag_deadzone_px * self.drag_deadzone_px):
                    self._orbit_dragging = True
                    # Reset last reference to current to avoid initial jump
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
        frame = 0
        while not self.window.should_close():
            self.window.process_events()
            # self.ui.process_events()

            elapsed = max(1e-6, timer.elapsed_s())
            timer.reset()
            self.fps_avg = 0.95 * self.fps_avg + 0.05 * (1.0 / elapsed)
            if self.bvh is not None:
                self.fps_text.text = f"FPS: {self.fps_avg:.2f} | V:{len(self.bvh.vert_pos)} T:{len(self.bvh.tris)} N:{self.bvh.nodes.shape[0]}"
            else:
                self.fps_text.text = f"FPS: {self.fps_avg:.2f} | Mesh: (none)"

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
                    label="mesh_rt_output",
                )

            eye, gU, gV, gW = self.camera.get_basis()
            command_encoder = self.device.create_command_encoder()
            if self.buf_nodes is not None and self.buf_tris is not None and self.buf_verts is not None:
                params = {
                    "imageSize": (np.uint32(self.output_texture.width), np.uint32(self.output_texture.height)),
                    "fovY": np.float32(math.radians(float(self.fov_slider.value))),
                    "maxBounces": np.uint32(1),
                    "eye": eye.astype(np.float32),
                    "U": gU.astype(np.float32),
                    "V": gV.astype(np.float32),
                    "W": gW.astype(np.float32),
                }
                self.kernel.dispatch(
                    thread_count=[self.output_texture.width, self.output_texture.height, 1],
                    vars={
                        "gOutput": self.output_texture,
                        "gBVHNodes": self.buf_nodes,
                        "gTris": self.buf_tris,
                        "gVerts": self.buf_verts,
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
            frame += 1


if __name__ == "__main__":
    App().run()

    
    
