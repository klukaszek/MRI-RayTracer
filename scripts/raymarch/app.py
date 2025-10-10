# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
from pathlib import Path
import numpy as np
import slangpy as spy
from camera import OrbitalCamera

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
        program = self.device.load_program("raymarch.slang", ["raymarch_cs"])
        self.kernel = self.device.create_compute_kernel(program)

        # No MLP buffers; shader renders analytic sphere

        # State
        self.frame = 0
        self.fps_avg = 0.0

        # Initial camera (yaw/pitch/radius mapped to orbital theta/phi)
        self.yaw_deg = 25.0
        self.pitch_deg = -10.0
        self.radius = 2.0
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        initial_phi = max(0.01, min(math.pi - 0.01, math.pi * 0.5 - math.radians(self.pitch_deg)))
        initial_theta = math.radians(self.yaw_deg)
        self.camera = OrbitalCamera(
            initial_target=self.target,
            initial_radius=self.radius,
            initial_phi=initial_phi,
            initial_theta=initial_theta,
            min_radius=0.1,
            max_radius=50.0,
            min_phi=0.01,
            max_phi=math.pi - 0.01,
        )

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
        self.window.on_mouse_event = self.on_mouse_event
        self.window.on_resize = self.on_resize

        # Mouse/keyboard interaction state (for optional pan/zoom)
        self.shift_down = False
        self.lmb_pressed = False
        self.rmb_pressed = False
        self.last_mouse_pos = spy.float2(0.0, 0.0)
        # Pan drag state
        self.pan_dragging = False
        self.pan_press_pos = spy.float2(0.0, 0.0)
        self.drag_deadzone_px = 6.0
        # Build a set of shift key codes that exist in this build
        self._shift_keys = set()
        for _name in ("shift", "left_shift", "right_shift"):
            kc = getattr(spy.KeyCode, _name, None)
            if kc is not None:
                self._shift_keys.add(kc)

    def setup_ui(self):
        screen = self.ui.screen
        window = spy.ui.Window(screen, "Neural Raymarch Settings", size=spy.float2(380, 260))
        self.fps_text = spy.ui.Text(window, "FPS: 0")
        spy.ui.Text(window, "Camera (Yaw/Pitch/Radius)")
        self.yaw_slider = spy.ui.SliderFloat(window, "Yaw (deg)", value=self.yaw_deg, min=-180.0, max=180.0)
        self.pitch_slider = spy.ui.SliderFloat(window, "Pitch (deg)", value=self.pitch_deg, min=-80.0, max=80.0)
        self.radius_slider = spy.ui.SliderFloat(window, "Distance", value=self.radius, min=0.5, max=10.0)
        spy.ui.Text(window, "Controls: Enable Pan, then Shift+LMB drag.\nWheel/Right-drag = Zoom")
        self.pan_enable = spy.ui.CheckBox(window, "Enable Pan", value=False)

        # No animation/effects controls per request

        spy.ui.Text(window, "Shader")
        self.fov_slider = spy.ui.SliderFloat(window, "FOV Y (deg)", value=self.fov_deg, min=20.0, max=90.0)
        self.steps_slider = spy.ui.SliderInt(window, "Max Steps", value=self.max_steps, min=16, max=512)
        self.maxdist_slider = spy.ui.SliderFloat(window, "Max Distance", value=self.max_distance, min=1.0, max=50.0)
        self.hit_slider = spy.ui.SliderFloat(window, "Hit Threshold", value=self.hit_threshold, min=1e-5, max=1e-1)
        self.normeps_slider = spy.ui.SliderFloat(window, "Normal Eps", value=self.normal_eps, min=1e-5, max=5e-2)

    def on_keyboard_event(self, event: spy.KeyboardEvent):
        # Track Shift state regardless of whether UI consumes the event
        if event.type == spy.KeyboardEventType.key_press:
            if event.key in self._shift_keys:
                self.shift_down = True
        elif event.type == spy.KeyboardEventType.key_release:
            if event.key in self._shift_keys:
                self.shift_down = False

        if self.ui.handle_keyboard_event(event):
            return

        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape:
                self.window.close()

    def on_mouse_event(self, event: spy.MouseEvent):
        if self.ui.handle_mouse_event(event):
            return

        # Scroll wheel zoom using slangpy MouseEvent.is_scroll() and .scroll
        try:
            if event.is_scroll():
                sy = float(event.scroll.y)
                # Use exponential scaling for smooth zoom
                zoom_factor = pow(1.15, -sy)
                zoom_factor = max(0.5, min(2.0, zoom_factor))
                self.camera.zoom(zoom_factor)
                self.radius_slider.value = float(self.camera.radius)
                return
        except Exception:
            pass

        # Button presses
        if event.type == spy.MouseEventType.button_down:
            if event.button == spy.MouseButton.left:
                self.lmb_pressed = True
                self.last_mouse_pos = event.pos
                self.pan_press_pos = event.pos
                self.pan_dragging = False
            elif event.button == spy.MouseButton.right:
                self.rmb_pressed = True
                self.last_mouse_pos = event.pos
        elif event.type == spy.MouseEventType.button_up:
            if event.button == spy.MouseButton.left:
                self.lmb_pressed = False
                self.pan_dragging = False
            elif event.button == spy.MouseButton.right:
                self.rmb_pressed = False
        elif event.type == spy.MouseEventType.move:
            # Pan when enabled (Shift+LMB drag)
            if self.lmb_pressed and self.pan_enable.value and self.shift_down:
                if not self.pan_dragging:
                    pdx = float(event.pos.x - self.pan_press_pos.x)
                    pdy = float(event.pos.y - self.pan_press_pos.y)
                    if (pdx*pdx + pdy*pdy) >= (self.drag_deadzone_px * self.drag_deadzone_px):
                        self.pan_dragging = True
                        # Reset last reference to current to avoid a jump on activation
                        self.last_mouse_pos = event.pos
                else:
                    dx = float(event.pos.x - self.last_mouse_pos.x)
                    dy = float(event.pos.y - self.last_mouse_pos.y)
                    self.camera.pan(dx, dy)
            # Zoom with right-drag vertical
            if self.rmb_pressed:
                dy = float(event.pos.y - self.last_mouse_pos.y)
                h = max(1.0, float(self.window.height))
                zoom_factor = 1.0 + (-dy / h) * 2.0
                zoom_factor = max(0.5, min(2.0, zoom_factor))
                self.camera.zoom(zoom_factor)
                self.radius_slider.value = float(self.camera.radius)
            self.last_mouse_pos = event.pos

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
            # Camera from yaw/pitch/radius via orbital mapping
            self.yaw_deg = float(self.yaw_slider.value)
            self.pitch_deg = float(self.pitch_slider.value)
            self.radius = max(0.1, float(self.radius_slider.value))
            self.camera.theta = math.radians(self.yaw_deg)
            self.camera.phi = max(self.camera.min_phi, min(self.camera.max_phi, math.pi * 0.5 - math.radians(self.pitch_deg)))
            self.camera.radius = max(self.camera.min_radius, min(self.camera.max_radius, self.radius))
            self.fov_deg = self.fov_slider.value
            self.max_steps = int(self.steps_slider.value)
            self.max_distance = self.maxdist_slider.value
            self.hit_threshold = self.hit_slider.value
            self.normal_eps = self.normeps_slider.value
            # No MLP params

            # Build basis from orbital camera
            eye, gU, gV, gW = self.camera.get_basis()
            # Keep UI sliders in sync with camera radius after zoom/pan
            self.radius_slider.value = float(self.camera.radius)

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
