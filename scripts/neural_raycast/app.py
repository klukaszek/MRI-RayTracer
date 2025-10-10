# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
from pathlib import Path
import numpy as np
import slangpy as spy

HERE = Path(__file__).parent


# No MLP needed for analytic SDF sphere


def look_at_view_to_world(yaw_deg: float, pitch_deg: float, radius: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    # Forward points from eye to target (origin)
    fx = -math.cos(pitch) * math.sin(yaw)
    fy = math.sin(pitch)
    fz = -math.cos(pitch) * math.cos(yaw)
    forward = np.array([fx, fy, fz], dtype=np.float32)
    forward /= max(1e-6, np.linalg.norm(forward))
    # Eye position
    eye = -forward * radius
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(world_up, forward)
    rn = np.linalg.norm(right)
    if rn < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        right /= rn
    up = np.cross(forward, right).astype(np.float32)
    # Row-major view-to-world: rows are camera basis (right, up, forward), row3 is translation (eye)
    M = np.eye(4, dtype=np.float32)
    M[0, 0:3] = right
    M[1, 0:3] = up
    M[2, 0:3] = forward
    M[3, 0:3] = eye
    return M


class App:
    def __init__(self):
        self.window = spy.Window(title="Neural Raymarch", resizable=True)
        self.device = spy.Device(enable_debug_layers=True, compiler_options={"include_paths": [HERE]})
        self.surface = self.device.create_surface(self.window)
        self.surface.configure(width=self.window.width, height=self.window.height)

        self.ui = spy.ui.Context(self.device)

        # Output
        self.output_texture = None

        # Load shader
        program = self.device.load_program("neural_raymarch.slang", ["neural_raymarch_cs"])
        self.kernel = self.device.create_compute_kernel(program)

        # No MLP buffers; shader renders analytic sphere

        # State
        self.frame = 0
        self.fps_avg = 0.0

        # Camera params
        self.yaw_deg = 25.0
        self.pitch_deg = -10.0
        self.radius = 2.0

        # Shader params
        self.fov_deg = 45.0
        self.max_steps = 128
        self.max_distance = 10.0
        self.hit_threshold = 1e-3
        self.normal_eps = 1e-3
        # No MLP activations

        # UI controls
        self.setup_ui()

        # Events
        self.window.on_keyboard_event = self.on_keyboard_event
        # Forward mouse to UI (no camera controls)
        self.window.on_mouse_event = self.on_mouse_event
        self.window.on_resize = self.on_resize

    def setup_ui(self):
        screen = self.ui.screen
        window = spy.ui.Window(screen, "Neural Raymarch Settings", size=spy.float2(360, 220))
        self.fps_text = spy.ui.Text(window, "FPS: 0")

        spy.ui.Text(window, "Camera")
        self.yaw_slider = spy.ui.SliderFloat(window, "Yaw (deg)", value=self.yaw_deg, min=-180.0, max=180.0)
        self.pitch_slider = spy.ui.SliderFloat(window, "Pitch (deg)", value=self.pitch_deg, min=-80.0, max=80.0)
        self.radius_slider = spy.ui.SliderFloat(window, "Distance", value=self.radius, min=0.5, max=10.0)

        spy.ui.Text(window, "Shader")
        self.fov_slider = spy.ui.SliderFloat(window, "FOV Y (deg)", value=self.fov_deg, min=20.0, max=90.0)
        self.steps_slider = spy.ui.SliderInt(window, "Max Steps", value=self.max_steps, min=16, max=512)
        self.maxdist_slider = spy.ui.SliderFloat(window, "Max Distance", value=self.max_distance, min=1.0, max=50.0)
        self.hit_slider = spy.ui.SliderFloat(window, "Hit Threshold", value=self.hit_threshold, min=1e-5, max=1e-1)
        self.normeps_slider = spy.ui.SliderFloat(window, "Normal Eps", value=self.normal_eps, min=1e-5, max=5e-2)

    def on_keyboard_event(self, event: spy.KeyboardEvent):
        if self.ui.handle_keyboard_event(event):
            return
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape:
                self.window.close()

    # Mouse orbit/zoom removed; forward to UI only
    def on_mouse_event(self, event: spy.MouseEvent):
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

            # Resize output if needed
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

            # Update UI-driven params
            self.yaw_deg = self.yaw_slider.value
            self.pitch_deg = self.pitch_slider.value
            self.radius = self.radius_slider.value
            self.fov_deg = self.fov_slider.value
            self.max_steps = int(self.steps_slider.value)
            self.max_distance = self.maxdist_slider.value
            self.hit_threshold = self.hit_slider.value
            self.normal_eps = self.normeps_slider.value
            # No MLP params

            view_to_world = look_at_view_to_world(self.yaw_deg, self.pitch_deg, self.radius)
            eye = view_to_world[3, 0:3]
            gU = view_to_world[0, 0:3]
            gV = view_to_world[1, 0:3]
            gW = view_to_world[2, 0:3]

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
                    "gU": gU,
                    "gV": gV,
                    "gW": gW,
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
