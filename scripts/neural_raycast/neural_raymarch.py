"""
Minimal windowed SDF ray marcher that renders a UV-colored sphere.
All mouse camera controls removed per request.
"""

import math
from pathlib import Path
import numpy as np
import slangpy as spy

HERE = Path(__file__).parent


def look_at_view_to_world(yaw_deg: float, pitch_deg: float, radius: float, target: np.ndarray) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    # Camera position on a sphere around the given target (Y-up)
    pos = np.array([
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
        math.cos(pitch) * math.cos(yaw),
    ], dtype=np.float32) * float(radius) + target.astype(np.float32)
    forward = target.astype(np.float32) - pos
    forward /= max(1e-6, np.linalg.norm(forward))
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right /= max(1e-6, np.linalg.norm(right))
    up = np.cross(right, forward)
    M = np.eye(4, dtype=np.float32)
    M[0, 0:3] = right
    M[1, 0:3] = up
    M[2, 0:3] = forward
    M[3, 0:3] = pos
    return M


class App:
    def __init__(self):
        self.window = spy.Window(title="Neural Raymarch", resizable=True)
        self.device = spy.Device(enable_debug_layers=True, compiler_options={"include_paths": [HERE]})
        self.surface = self.device.create_surface(self.window)
        self.surface.configure(width=self.window.width, height=self.window.height)

        self.ui = spy.ui.Context(self.device)
        self.output_texture = None

        program = self.device.load_program("neural_raymarch.slang", ["neural_raymarch_cs"])
        self.kernel = self.device.create_compute_kernel(program)

        # State
        self.frame = 0
        self.fps_avg = 0.0

        # Camera
        self.yaw_deg = 25.0
        self.pitch_deg = -10.0
        self.radius = 2.0
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Shader params
        self.fov_deg = 45.0
        self.max_steps = 192
        self.max_distance = 5.0
        self.hit_threshold = 1e-3
        self.normal_eps = 1e-3

        # UI
        self.setup_ui()

        # Events
        self.window.on_keyboard_event = self.on_keyboard_event
        self.window.on_mouse_event = self.on_mouse_event
        self.window.on_resize = self.on_resize

    def setup_ui(self):
        screen = self.ui.screen
        window = spy.ui.Window(screen, "Neural Raymarch Settings", size=spy.float2(360, 220))
        self.fps_text = spy.ui.Text(window, "FPS: 0")
        self.radius_slider = spy.ui.SliderFloat(window, "Distance", value=self.radius, min=0.5, max=10.0)
        spy.ui.Text(window, "Shader")
        self.fov_slider = spy.ui.SliderFloat(window, "FOV Y (deg)", value=self.fov_deg, min=20.0, max=90.0)
        self.steps_slider = spy.ui.SliderInt(window, "Max Steps", value=self.max_steps, min=16, max=512)
        self.maxdist_slider = spy.ui.SliderFloat(window, "Max Distance", value=self.max_distance, min=1.0, max=50.0)
        self.hit_slider = spy.ui.SliderFloat(window, "Hit Threshold", value=self.hit_threshold, min=1e-5, max=1e-1)
        self.normeps_slider = spy.ui.SliderFloat(window, "Normal Eps", value=self.normal_eps, min=1e-5, max=5e-2)
        # No MLP/volume controls in analytic SDF mode

    def on_keyboard_event(self, event: spy.KeyboardEvent):
        if self.ui.handle_keyboard_event(event):
            return
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape:
                self.window.close()
            elif event.key == spy.KeyCode.f2 and self.output_texture is not None:
                bitmap = self.output_texture.to_bitmap()
                bitmap.convert(spy.Bitmap.PixelFormat.rgb, spy.Bitmap.ComponentType.uint8, srgb_gamma=True).write_async("neural_raymarch.png")

    def on_mouse_event(self, event: spy.MouseEvent):
        # Forward mouse events to UI only (no camera controls)
        self.ui.handle_mouse_event(event)

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
            self.ui.process_events()
            elapsed = max(1e-6, timer.elapsed_s())
            timer.reset()
            self.fps_avg = 0.95 * self.fps_avg + 0.05 * (1.0 / elapsed)
            self.fps_text.text = f"FPS: {self.fps_avg:.2f}"

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
                    label="neural_output",
                )

            # Update from UI (only distance)
            self.radius = self.radius_slider.value
            self.fov_deg = self.fov_slider.value
            self.max_steps = int(self.steps_slider.value)
            self.max_distance = self.maxdist_slider.value
            self.hit_threshold = self.hit_slider.value
            self.normal_eps = self.normeps_slider.value
            # analytic SDF only: no hidden/output act

            # Build camera basis like your TS code
            yaw = math.radians(self.yaw_deg)
            pitch = math.radians(self.pitch_deg)
            cp = math.cos(pitch); sp = math.sin(pitch)
            cy = math.cos(yaw);   sy = math.sin(yaw)
            ex = self.radius * cp * cy
            ey = self.radius * cp * sy
            ez = self.radius * sp
            eye = np.array([self.target[0] + ex, self.target[1] + ey, self.target[2] + ez], dtype=np.float32)
            f = (self.target - eye); f = f / max(1e-6, np.linalg.norm(f))
            s = np.cross(f, np.array([0.0,1.0,0.0], dtype=np.float32))
            if np.dot(s, s) < 1e-12:
                alt = np.array([0,1,0], dtype=np.float32) if abs(self.target[2]) > 0.5 else np.array([0,0,1], dtype=np.float32)
                s = np.cross(f, alt)
            s = s / max(1e-6, np.linalg.norm(s))
            u = np.cross(s, f)
            params = {
                "imageSize": (np.uint32(self.output_texture.width), np.uint32(self.output_texture.height)),
                "fovY": np.float32(math.radians(self.fov_deg)),
                "maxSteps": np.uint32(self.max_steps),
                "maxDistance": np.float32(self.max_distance),
                "hitThreshold": np.float32(self.hit_threshold),
                "normalEps": np.float32(self.normal_eps),
            }

            command_encoder = self.device.create_command_encoder()
            self.kernel.dispatch(
                thread_count=[self.output_texture.width, self.output_texture.height, 1],
                vars={
                    "render_texture": self.output_texture,
                    "gParams": params,
                    "gEye": eye,
                    "gU": s,
                    "gV": u,
                    "gW": f,
                },
                command_encoder=command_encoder,
            )
            command_encoder.blit(surface_texture, self.output_texture)
            self.ui.new_frame(surface_texture.width, surface_texture.height)
            self.ui.render(surface_texture, command_encoder)
            self.device.submit_command_buffer(command_encoder.finish())
            del surface_texture
            self.surface.present()
            self.frame += 1


if __name__ == "__main__":
    App().run()
