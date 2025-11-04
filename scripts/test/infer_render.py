"""
Simple neural renderer inference: renders a 2D image using the trained
NeuralShaderSegmentation model by casting camera rays through the learned
field, then saves as a PNG.

Usage example:
  ../../.venv/bin/python infer_render.py \
    --data_root data/BraTS-2023 \
    --patient_id BraTS-GLI-00000-000 \
    --ckpt outputs/BraTS-GLI-00000-000_best.pth \
    --width 640 --height 480 --samples_per_ray 64 --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from train import BraTSPatient
from neural_shader_model import NeuralShaderSegmentation, RenderConfig, RayBatch


def look_at(eye: np.ndarray, target: np.ndarray = None, up: np.ndarray = None) -> torch.Tensor:
    """Build a simple camera-to-world (c2w) matrix as torch.Tensor [4,4]."""
    if target is None:
        target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    if up is None:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    eye = np.asarray(eye, dtype=np.float32)
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-8)
    u = up / (np.linalg.norm(up) + 1e-8)
    s = np.cross(f, u)
    s = s / (np.linalg.norm(s) + 1e-8)
    u = np.cross(s, f)
    # Camera basis: columns are right(s), up(u), forward(f)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = s
    c2w[:3, 1] = u
    c2w[:3, 2] = f
    c2w[:3, 3] = eye
    return torch.from_numpy(c2w)


def generate_primary_rays(width: int, height: int, fov_y_deg: float, c2w: torch.Tensor, device: torch.device) -> RayBatch:
    # Pixel grid
    i = torch.arange(width, device=device)
    j = torch.arange(height, device=device)
    grid_i, grid_j = torch.meshgrid(i, j, indexing='xy')
    # NDC
    uv = torch.stack([(grid_i + 0.5) / width, (grid_j + 0.5) / height], dim=-1)
    ndc = torch.stack([uv[..., 0] * 2 - 1, 1 - uv[..., 1] * 2], dim=-1)
    # Directions in camera space
    fov_y = np.deg2rad(fov_y_deg)
    tan_half = np.tan(0.5 * fov_y)
    aspect = width / max(1, height)
    dirs_cam = torch.stack([
        ndc[..., 0] * aspect * tan_half,
        ndc[..., 1] * tan_half,
        torch.ones_like(ndc[..., 0])
    ], dim=-1)
    # Transform to world space using c2w basis
    R = c2w[:3, :3].to(device)
    t = c2w[:3, 3].to(device)
    dirs_world = F.normalize(dirs_cam @ R.T, dim=-1)
    origins = t.expand_as(dirs_world)
    # Flatten rays
    origins = origins.reshape(-1, 3)
    dirs_world = dirs_world.reshape(-1, 3)
    return RayBatch(origins=origins, directions=dirs_world, viewdirs=dirs_world)


def main():
    ap = argparse.ArgumentParser(description="Neural shader inference render")
    ap.add_argument('--data_root', required=True, type=str, help='BraTS data root (NIfTI)')
    ap.add_argument('--patient_id', required=True, type=str, help='Patient directory name')
    ap.add_argument('--ckpt', required=True, type=str, help='Path to best checkpoint .pth')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--samples_per_ray', type=int, default=64)
    ap.add_argument('--fov_y', type=float, default=60.0)
    ap.add_argument('--device', type=str, default='cpu')
    args = ap.parse_args()

    device = torch.device(args.device)

    # Load patient (for shape and potential ground-truth comparisons later)
    patient = BraTSPatient(Path(args.data_root) / args.patient_id)

    # Load model from checkpoint (reuse config saved inside)
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get('config', {})
    model = NeuralShaderSegmentation(
        feature_grid_res=cfg.get('grid_resolution', 64),
        feature_dim=cfg.get('feature_dim', 32),
        mlp_hidden_dim=cfg.get('mlp_hidden', 64),
        num_classes=4,
        use_positional_encoding=True,
        pos_encoding_freqs=6,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Camera setup: simple look-at from positive Z looking toward origin
    c2w = look_at(eye=np.array([0.0, 0.0, 4.0], dtype=np.float32))
    rays = generate_primary_rays(args.width, args.height, args.fov_y, c2w, device)

    # Render
    render_cfg = RenderConfig(
        num_samples=args.samples_per_ray,
        step_size=0.02,
        near=0.0,
        far=2.2,
        density_noise=0.0,
        white_background=True,
    )

    with torch.no_grad():
        outputs = model.render_rays(rays, render_cfg)
    rgb = outputs['rgb'].clamp(0, 1).cpu().numpy().reshape(args.height, args.width, 3)

    out_dir = Path('outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.patient_id}_inference.png"
    plt.imsave(out_path, rgb)
    print(f"Saved inference render: {out_path}")


if __name__ == '__main__':
    main()

