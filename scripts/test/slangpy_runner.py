"""
SlangPy runner for neural raymarching image renderer.

Loads exported NPZ weights (feature grids + MLP), uploads as GPU resources,
dispatches the Slang compute shader, and displays/saves the output image.

Requirements:
- pip install slangpy numpy imageio

Usage:
  python slangpy_runner.py \
    --shader neural_raymarch_image.slang \
    --npz outputs/BraTS-GLI-00000-000_model.npz \
    --width 640 --height 480 --step 0.02 --max_steps 128
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

try:
    import slangpy as sp
except Exception as e:
    sp = None
    _slangpy_import_error = e


def _require_slangpy():
    if sp is None:
        raise RuntimeError(
            f"slangpy not available. Install with `pip install slangpy`.\nError: {_slangpy_import_error}"
        )


def _to_tex3d_array(dev, grid: np.ndarray):
    """
    Pack a [1, C, D, H, W] grid (C=32) into 8 Texture3D<float4> slices.
    Returns: list[Texture] of length 8.
    """
    assert grid.ndim == 5 and grid.shape[0] == 1, f"Expected [1,C,D,H,W], got {grid.shape}"
    _, C, D, H, W = grid.shape
    assert C == 32, f"Expected 32 channels per modality, got {C}"

    # Normalize channels to a reasonable range for visualization/inference if needed
    data = grid.astype(np.float32)[0]

    # Create 8 3D textures (RGBA32F)
    tex_list = []
    for s in range(8):
        rgba = np.stack([
            data[s*4 + 0], data[s*4 + 1], data[s*4 + 2], data[s*4 + 3]
        ], axis=-1)
        desc = sp.TextureDesc()
        desc.type = sp.TextureType.texture_3d
        desc.width = int(W)
        desc.height = int(H)
        desc.depth = int(D)
        desc.format = sp.Format.rgba32_float
        desc.usage = sp.TextureUsage.shader_resource | sp.TextureUsage.copy_destination | sp.TextureUsage.copy_source
        tex = dev.create_texture(desc)
        tex.copy_from_numpy(rgba)
        tex_list.append(tex)
    return tex_list


def run(args):
    _require_slangpy()

    # Create device (CPU for portability here). Switch to vulkan/metal/d3d12 if available.
    dev = sp.create_device(sp.DeviceType.cpu, enable_hot_reload=False)
    # Load slang source into a module bound to this device
    module = sp.Module.load_from_file(dev, args.shader)
    # Get compute kernel
    render = module.require_function('renderVolumeImage')

    # Load NPZ
    npz = np.load(args.npz)

    # Feature grids: expect keys per export from neural_shader_model.py
    # For multi-modal, keys are 'feature_grid_t1', 'feature_grid_t2', 'feature_grid_flair'
    if 'feature_grid_t1' in npz:
        t1 = npz['feature_grid_t1']
        t2 = npz['feature_grid_t2']
        fl = npz['feature_grid_flair']
    else:
        # Fallback: single fused grid under 'feature_grid'
        fused = npz['feature_grid']
        # Split into three equal parts along channel dimension C
        assert fused.shape[1] % 3 == 0, "Fused feature_grid must be divisible into 3 modalities"
        c_each = fused.shape[1] // 3
        t1 = fused[:, :c_each]
        t2 = fused[:, c_each:2*c_each]
        fl = fused[:, 2*c_each:]

    tex_t1 = _to_tex3d_array(dev, t1)
    tex_t2 = _to_tex3d_array(dev, t2)
    tex_fl = _to_tex3d_array(dev, fl)

    # MLP weights
    if 'mlp_weights' in npz:
        mlp = npz['mlp_weights'].astype(np.float32)
    elif 'density_mlp' in npz:
        mlp = npz['density_mlp'].astype(np.float32)
    else:
        raise KeyError("Could not find MLP weights in NPZ (mlp_weights or density_mlp)")

    # Metadata
    md = npz.get('metadata', {})
    if isinstance(md, np.ndarray):
        # np.savez may store dict as 0-d object array
        md = md.item()

    feature_dim = int(md.get('feature_dim', 32))
    grid_res = int(md.get('grid_resolution', t1.shape[-1]))
    hidden = int(md.get('mlp_hidden_dim', 64))
    num_classes = int(md.get('num_classes', 4))
    use_pe = bool(md.get('use_pos_encoding', True))

    # Try to infer exact input dim from flattened weights to avoid PE-dim mismatch
    mlp_output_dim = 1 + num_classes
    H = int(hidden)
    L = int(mlp.size)
    # Solve: L = H*(Din + 1) + H*(H + 1) + (mlp_output_dim)*(H + 1)
    known = H*(H + 1) + mlp_output_dim*(H + 1)
    Din = int(round(L / H - 1 - (H + 1) - (mlp_output_dim * (H + 1)) / H))
    # Fallback if above numeric quirks: compute directly
    denom = H
    num = L - (H*(H + 1) + mlp_output_dim*(H + 1))
    if denom > 0:
        Din2 = int(round(num / denom - 1))
    else:
        Din2 = feature_dim * 3 + (39 if use_pe else 3)
    # Choose plausible
    if Din2 > 0:
        mlp_input_dim = Din2
    else:
        mlp_input_dim = feature_dim * 3 + (39 if use_pe else 3)

    # Rays
    W, H = args.width, args.height
    # Build simple pinhole camera rays toward origin from z=+4
    i = np.arange(W, dtype=np.float32)
    j = np.arange(H, dtype=np.float32)
    grid_i, grid_j = np.meshgrid(i, j, indexing='xy')
    uvx = (grid_i + 0.5) / max(1, W)
    uvy = (grid_j + 0.5) / max(1, H)
    ndc_x = uvx * 2 - 1
    ndc_y = 1 - uvy * 2
    fov_y = np.deg2rad(args.fov_y)
    tan_half = np.tan(0.5 * fov_y)
    aspect = W / max(1, H)
    dx = ndc_x * aspect * tan_half
    dy = ndc_y * tan_half
    dz = np.ones_like(dx)

    dirs = np.stack([dx, dy, dz], axis=-1)
    norms = np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-8
    dirs = dirs / norms
    origins = np.zeros_like(dirs)
    origins[..., 2] = 4.0

    # Rays in [-1,1] cube range
    tMin = np.zeros((H, W), dtype=np.float32)
    tMax = np.full((H, W), 2.2, dtype=np.float32)

    rays_struct = np.zeros((H*W, 8), dtype=np.float32)
    rays_struct[:, 0:3] = origins.reshape(-1, 3)
    rays_struct[:, 3:6] = dirs.reshape(-1, 3)
    rays_struct[:, 6] = tMin.reshape(-1)
    rays_struct[:, 7] = tMax.reshape(-1)

    # Create GPU resources
    rays_buf = dev.create_buffer(sp.BufferDesc(size=rays_struct.nbytes))
    rays_buf.copy_from_numpy(rays_struct)
    # Create output 2D texture
    out_desc = sp.TextureDesc()
    out_desc.type = sp.TextureType.texture_2d
    out_desc.width = int(W)
    out_desc.height = int(H)
    out_desc.format = sp.Format.rgba32_float
    out_desc.usage = sp.TextureUsage.unordered_access | sp.TextureUsage.copy_source | sp.TextureUsage.shader_resource
    out_tex = dev.create_texture(out_desc)

    # Bindings
    # Create MLP buffer
    mlp_buf = dev.create_buffer(sp.BufferDesc(size=mlp.nbytes))
    mlp_buf.copy_from_numpy(mlp)

    # Create sampler
    samp = dev.create_sampler(sp.SamplerDesc())

    # Build compute kernel and bind resources
    kernel = dev.create_compute_kernel(sp.ComputeKernelDesc(function=render))
    params = kernel.shader_object
    # Bind arrays of textures
    for i in range(8):
        params[f'gFeatureGridT1[{i}]'] = tex_t1[i]
        params[f'gFeatureGridT2[{i}]'] = tex_t2[i]
        params[f'gFeatureGridFLAIR[{i}]'] = tex_fl[i]
    params['gLinearSampler'] = samp
    params['gMLPWeights'] = mlp_buf
    params['gMLPInputDim'] = int(mlp_input_dim)
    params['gMLPHiddenDim'] = int(hidden)
    params['gMLPOutputDim'] = int(mlp_output_dim)
    params['gBaseStepSize'] = float(args.step)
    params['gMaxSteps'] = int(args.max_steps)
    params['gImageWidth'] = int(W)
    params['gImageHeight'] = int(H)
    params['gRays'] = rays_buf
    params['gOutImage'] = out_tex

    # Dispatch
    gx = (W + 7) // 8
    gy = (H + 7) // 8
    # Encode and dispatch
    enc = dev.create_command_encoder()
    enc.dispatch_computes(kernel, gx, gy, 1)
    dev.submit(enc.finish())
    dev.wait_for_idle()

    # Download and save
    img = out_tex.to_numpy()  # shape [H, W, 4]
    rgb = np.clip(img[..., :3], 0.0, 1.0)
    if args.save:
        import imageio
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(out_path, (rgb * 255).astype(np.uint8))
        print(f"Saved image: {out_path}")

    # Optionally, show using slangpy simple viewer if available
    if args.show:
        try:
            sp.imshow(rgb)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="Run Slang neural raymarching renderer")
    ap.add_argument('--shader', type=str, default='neural_raymarch_image.slang')
    ap.add_argument('--npz', type=str, required=True)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--fov_y', type=float, default=60.0)
    ap.add_argument('--step', type=float, default=0.02)
    ap.add_argument('--max_steps', type=int, default=128)
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--save', type=str, default='outputs/slangpy_render.png')
    args = ap.parse_args()

    run(args)


if __name__ == '__main__':
    main()
