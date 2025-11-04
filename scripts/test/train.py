"""
train.py - Final Training Script for Neural Shader Medical Segmentation
Graduate Course Project

This is the MAIN training script you should use.
Works with neural_shader_model.py (the clean implementation).

Usage:
    python train.py --data_root ./data/BraTS2021 --patient_id BraTS2021_00000
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Tuple, Optional
import argparse
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# Import the neural shader model
# Make sure neural_shader_model.py is in the same directory or in your Python path
from neural_shader_model import (
    NeuralShaderSegmentation,
    RenderConfig,
    RayBatch,
    SegmentationLoss
)


class BraTSPatient:
    """Load and preprocess a single BraTS patient"""
    
    def __init__(self, patient_dir: Path):
        self.patient_dir = patient_dir
        self.patient_id = patient_dir.name
        
        print(f"Loading patient: {self.patient_id}")
        
        # Try different naming conventions (BraTS20/21/23 NIfTI)
        # We use T1 (native), T2 (weighted), and FLAIR (T2-FLAIR). T1CE is unused.
        self.t1 = self._load_modality(['t1n', 't1', 'T1', 't1w', 'T1w'])
        self.t2 = self._load_modality(['t2w', 't2', 'T2'])
        self.flair = self._load_modality(['t2f', 'flair', 'FLAIR'])
        self.seg = self._load_modality(['seg', 'SEG', 'segmentation', 'mask'])
        
        # Normalize
        self.t1 = self._normalize(self.t1)
        self.t2 = self._normalize(self.t2)
        self.flair = self._normalize(self.flair)
        
        # Remap segmentation: BraTS labels → our labels
        self.seg = self._remap_labels(self.seg)
        
        print(f"  Shape: {self.t1.shape}")
        print(f"  Tumor voxels: {(self.seg > 0).sum()}")
    
    def _load_modality(self, possible_names: list) -> np.ndarray:
        """Try multiple possible filenames and suffixes (.nii.gz, .nii).

        Supports both underscore and dash separators used in BraTS datasets,
        e.g., '<patient>_t1n.nii.gz' or '<patient>-t2f.nii.gz'.
        """
        def try_load(stem: str) -> Optional[np.ndarray]:
            for suf in ('.nii.gz', '.nii'):
                p = self.patient_dir / f"{stem}{suf}"
                if p.exists():
                    img = nib.load(str(p)).get_fdata()
                    return img.astype(np.float32)
            return None

        # 1) Try explicit patterns with patient_id prefix using '_' and '-'
        for name in possible_names:
            for sep in ('_', '-'):
                arr = try_load(f"{self.patient_id}{sep}{name}")
                if arr is not None:
                    return arr

        # 2) Try bare names in folder
        for name in possible_names:
            arr = try_load(name)
            if arr is not None:
                return arr

        # 3) Fallback: scan directory and match tokens with '_' or '-' before extension
        nii_files = list(self.patient_dir.glob('*.nii*'))
        for name in possible_names:
            token_underscore = f"_{name}"
            token_dash = f"-{name}"
            for p in nii_files:
                n = p.name
                if token_underscore in n or token_dash in n:
                    img = nib.load(str(p)).get_fdata()
                    return img.astype(np.float32)

        raise FileNotFoundError(f"Missing modality {possible_names} in {self.patient_dir}")
    
    @staticmethod
    def _normalize(volume: np.ndarray) -> np.ndarray:
        """Z-score normalization on brain tissue (non-zero voxels)"""
        mask = volume > 0
        if mask.sum() > 0:
            volume = volume.copy()
            volume[mask] = (volume[mask] - volume[mask].mean()) / (volume[mask].std() + 1e-8)
        return volume
    
    @staticmethod
    def _remap_labels(seg: np.ndarray) -> np.ndarray:
        """
        Standardize to: 0=background, 1=edema (ED), 2=core (NET/NEC), 3=enhancing (ET)

        Handles two common schemes:
        - BraTS 2020/2021: {0,1,2,4} where 1=core, 2=edema, 4=enhancing
        - BraTS 2023 GLI: {0,1,2,3} where 1=enhancing, 2=edema, 3=core
        """
        uniq = set(np.unique(seg).tolist())
        seg_new = np.zeros_like(seg, dtype=np.int64)
        if 4 in uniq:
            # BraTS 2020/2021 style
            seg_new[seg == 2] = 1  # Edema
            seg_new[seg == 1] = 2  # Core
            seg_new[seg == 4] = 3  # Enhancing
        else:
            # BraTS 2023 GLI style (already 0..3 but with different mapping)
            seg_new[seg == 2] = 1  # Edema
            seg_new[seg == 3] = 2  # Core
            seg_new[seg == 1] = 3  # Enhancing
        return seg_new


class RaySampler:
    """Generate rays for volume rendering"""
    
    def __init__(self, volume_shape: Tuple[int, int, int], device: str = 'cuda'):
        self.volume_shape = np.array(volume_shape)
        self.device = device
        
        # Compute normalization (volume fits in [-1, 1] cube)
        max_dim = self.volume_shape.max()
        self.scale = 2.0 / max_dim
        self.offset = -self.volume_shape * self.scale / 2.0
    
    def sample_rays(
        self,
        num_rays: int,
        segmentation: Optional[torch.Tensor] = None,
        tumor_bias: float = 0.7
    ) -> RayBatch:
        """
        Sample rays through the volume
        
        Args:
            num_rays: Number of rays to sample
            segmentation: Optional tumor mask for biased sampling
            tumor_bias: Fraction of rays that target tumor (if seg provided)
        """
        device = self.device
        
        if segmentation is not None and torch.rand(1).item() < tumor_bias:
            # Sample rays toward tumor
            return self._sample_tumor_rays(num_rays, segmentation)
        else:
            # Uniform sampling
            return self._sample_uniform_rays(num_rays)
    
    def _sample_uniform_rays(self, num_rays: int) -> RayBatch:
        """Sample rays uniformly from random camera positions"""
        device = self.device
        
        # Random points on sphere (camera positions)
        theta = torch.rand(num_rays, device=device) * 2 * np.pi
        phi = torch.rand(num_rays, device=device) * np.pi
        
        radius = 2.5  # Outside the [-1,1] volume
        origins = torch.stack([
            radius * torch.sin(phi) * torch.cos(theta),
            radius * torch.sin(phi) * torch.sin(theta),
            radius * torch.cos(phi)
        ], dim=-1)
        
        # Point toward origin
        directions = F.normalize(-origins, dim=-1)
        
        return RayBatch(
            origins=origins,
            directions=directions,
            viewdirs=directions
        )
    
    def _sample_tumor_rays(
        self,
        num_rays: int,
        segmentation: torch.Tensor
    ) -> RayBatch:
        """Sample rays biased toward tumor regions"""
        device = self.device
        
        # Find tumor voxels
        tumor_coords = torch.nonzero(segmentation > 0, as_tuple=False).float()
        
        if len(tumor_coords) == 0:
            return self._sample_uniform_rays(num_rays)
        
        # Sample tumor points as ray targets
        indices = torch.randint(0, len(tumor_coords), (num_rays,), device=device)
        target_voxels = tumor_coords[indices]
        
        # Convert to normalized coordinates
        target_points = (
            target_voxels * self.scale + 
            torch.tensor(self.offset, device=device, dtype=torch.float32)
        )
        
        # Random camera positions
        theta = torch.rand(num_rays, device=device) * 2 * np.pi
        phi = torch.rand(num_rays, device=device) * np.pi / 2  # Hemisphere
        
        radius = 2.5
        origins = torch.stack([
            radius * torch.sin(phi) * torch.cos(theta),
            radius * torch.sin(phi) * torch.sin(theta),
            radius * torch.cos(phi)
        ], dim=-1)
        
        # Point toward tumor
        directions = F.normalize(target_points - origins, dim=-1)
        
        return RayBatch(
            origins=origins,
            directions=directions,
            viewdirs=directions
        )
    
    def sample_ground_truth(
        self,
        rays: RayBatch,
        segmentation: torch.Tensor,
        num_samples: int = 64
    ) -> torch.Tensor:
        """
        Sample ground truth segmentation along rays
        Returns the most common label along each ray
        """
        device = rays.origins.device
        
        # Sample points along rays
        t_vals = torch.linspace(0.0, 2.0, num_samples, device=device)
        points = (
            rays.origins[:, None, :] + 
            rays.directions[:, None, :] * t_vals[None, :, None]
        )
        
        # Convert to voxel coordinates
        voxel_coords = (
            (points - torch.tensor(self.offset, device=device)) / self.scale
        ).round().long()
        
        # Clamp to volume bounds
        for i in range(3):
            voxel_coords[..., i] = torch.clamp(
                voxel_coords[..., i], 0, self.volume_shape[i] - 1
            )
        
        # Sample segmentation
        sampled = segmentation[
            voxel_coords[..., 0],
            voxel_coords[..., 1],
            voxel_coords[..., 2]
        ]
        
        # Most common label per ray (mode)
        gt_labels = torch.mode(sampled, dim=-1)[0]
        
        return gt_labels


def train_patient(
    patient: BraTSPatient,
    config: Dict,
    device: str = 'cuda',
    resume_checkpoint: Optional[str] = None
) -> NeuralShaderSegmentation:
    """
    Train neural shader model on a single patient
    
    This is the main training function you'll use.
    """
    print(f"\n{'='*70}")
    print(f"Training Neural Shader Model")
    print(f"Patient: {patient.patient_id}")
    print(f"{'='*70}\n")
    
    # Initialize model
    model = NeuralShaderSegmentation(
        feature_grid_res=config['grid_resolution'],
        feature_dim=config['feature_dim'],
        mlp_hidden_dim=config['mlp_hidden'],
        num_classes=4,
        use_positional_encoding=True,
        pos_encoding_freqs=6
    ).to(device)
    
    # Initialize feature grids with downsampled volumes
    print("Initializing feature grids from MRI volumes...")
    with torch.no_grad():
        for i, volume in enumerate([patient.t1, patient.t2, patient.flair]):
            # Convert to tensor
            vol_torch = torch.from_numpy(volume).float()
            vol_torch = vol_torch.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
            
            # Downsample to grid resolution
            grid_vol = F.interpolate(
                vol_torch,
                size=[config['grid_resolution']] * 3,
                mode='trilinear',
                align_corners=True
            )
            
            # Initialize grid (broadcast to all feature channels)
            model.feature_grids.grids[i].data = grid_vol.expand(
                -1, config['feature_dim'], -1, -1, -1
            ) * 0.1
    
    # Loss function
    criterion = SegmentationLoss(
        num_classes=4,
        dice_weight=0.5,
        ce_weight=0.5,
        density_reg=0.01
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate']
    )
    
    # Resume from checkpoint if provided
    best_loss = float('inf')
    start_epoch_offset = 0
    last_epoch_for_scheduler = -1
    if resume_checkpoint:
        ckpt_path = Path(resume_checkpoint)
        if ckpt_path.exists():
            print(f"Resuming from checkpoint: {ckpt_path}")
            ckpt = torch.load(str(ckpt_path), map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except Exception as e:
                    print(f"Warning: could not load optimizer state: {e}")
            if 'loss' in ckpt:
                best_loss = float(ckpt['loss'])
            if 'epoch' in ckpt:
                start_epoch_offset = int(ckpt['epoch']) + 1
                last_epoch_for_scheduler = int(ckpt['epoch'])
        else:
            print(f"Warning: resume checkpoint not found at {ckpt_path}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        last_epoch=last_epoch_for_scheduler
    )
    
    # Ray sampler
    sampler = RaySampler(patient.t1.shape, device=device)
    seg_torch = torch.from_numpy(patient.seg).long().to(device)
    
    # Render configuration
    render_config = RenderConfig(
        num_samples=config['samples_per_ray'],
        step_size=0.02,
        near=0.0,
        far=2.0,
        density_noise=0.0  # Set via CLI in future if needed
    )
    
    # Training loop
    history = {'loss': [], 'dice': []}
    
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        
        num_batches = config['rays_per_epoch'] // config['batch_size']
        
        pbar = tqdm(
            range(num_batches),
            desc=f"Epoch {start_epoch_offset+epoch+1}/{start_epoch_offset+config['num_epochs']}"
        )
        
        for batch_idx in pbar:
            # Sample rays
            rays = sampler.sample_rays(
                config['batch_size'],
                segmentation=seg_torch,
                tumor_bias=0.7
            )
            
            # Render
            outputs = model.render_rays(rays, render_config)
            
            # Ground truth
            gt_labels = sampler.sample_ground_truth(
                rays,
                seg_torch,
                num_samples=config['samples_per_ray']
            )
            
            # Compute loss
            losses = criterion(outputs, gt_labels)
            loss = losses['total']
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            epoch_dice += (1.0 - losses['dice'].item())  # Convert back to dice
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{1.0 - losses['dice'].item():.3f}"
            })
        
        scheduler.step()
        
        # Epoch summary
        avg_loss = epoch_loss / num_batches
        avg_dice = epoch_dice / num_batches
        history['loss'].append(avg_loss)
        history['dice'].append(avg_dice)
        
        print(f"\nEpoch {start_epoch_offset+epoch+1} Summary:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Dice: {avg_dice:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = Path(config['output_dir']) / f"{patient.patient_id}_best.pth"
            torch.save({
                'epoch': start_epoch_offset + epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'dice': avg_dice,
                'config': config
            }, save_path)
            print(f"  ✓ Saved best model: {save_path}")
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs_arr = np.arange(1, len(history['loss']) + 1)
    
    # Use markers so single-epoch runs still show a visible point
    ax1.plot(epochs_arr, history['loss'], marker='o', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    if len(epochs_arr) == 1:
        ax1.set_xlim(0.5, 1.5)
    
    ax2.plot(epochs_arr, history['dice'], marker='o', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Dice Score')
    ax2.grid(True)
    if len(epochs_arr) == 1:
        ax2.set_xlim(0.5, 1.5)
    
    plt.tight_layout()
    plot_path = Path(config['output_dir']) / f"{patient.patient_id}_training.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved training plot: {plot_path}")
    
    return model


def evaluate_model(
    model: NeuralShaderSegmentation,
    patient: BraTSPatient,
    device: str = 'cuda',
    num_test_rays: int = 8192
):
    """Quick evaluation of trained model"""
    print(f"\n{'='*70}")
    print("Evaluating Model")
    print(f"{'='*70}\n")
    
    model.eval()
    
    sampler = RaySampler(patient.t1.shape, device=device)
    seg_torch = torch.from_numpy(patient.seg).long().to(device)
    
    render_config = RenderConfig(
        num_samples=128,  # More samples for better quality
        step_size=0.01,
        near=0.0,
        far=2.0,
        density_noise=0.0
    )
    
    with torch.no_grad():
        # Sample rays
        rays = sampler.sample_rays(num_test_rays, segmentation=seg_torch)
        
        # Render
        outputs = model.render_rays(rays, render_config)
        
        # Ground truth
        gt_labels = sampler.sample_ground_truth(rays, seg_torch, num_samples=128)
        
        # Compute Dice per class
        pred_classes = outputs['segmentation'].argmax(dim=-1)
        
        dice_scores = []
        class_names = ['Background', 'Edema', 'Core', 'Enhancing']
        
        print("Per-Class Dice Scores:")
        for c in range(4):
            pred_mask = (pred_classes == c)
            gt_mask = (gt_labels == c)
            
            intersection = (pred_mask & gt_mask).sum().float()
            union = pred_mask.sum() + gt_mask.sum()
            
            if union > 0:
                dice = (2.0 * intersection / union).item()
                dice_scores.append(dice)
                print(f"  {class_names[c]:12s}: {dice:.4f}")
        
        mean_dice = np.mean(dice_scores[1:])  # Exclude background
        print(f"\nMean Dice (tumor classes): {mean_dice:.4f}")
    
    return mean_dice


def main():
    parser = argparse.ArgumentParser(
        description='Train Neural Shader for Medical Segmentation'
    )
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to BraTS dataset root (NIfTI)')
    parser.add_argument('--patient_id', type=str, default=None,
                       help='Specific patient (default: first found)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=2048,
                       help='Rays per batch')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--rays_per_epoch', type=int, default=20000,
                       help='Total rays sampled per epoch')
    parser.add_argument('--samples_per_ray', type=int, default=64,
                       help='Samples per ray during training')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint .pth to resume training')
    parser.add_argument('--threads', type=int, default=0,
                       help='Set torch CPU threads (0=leave default)')
    
    args = parser.parse_args()

    # Optional CPU threading tuning
    if args.device == 'cpu' and args.threads and args.threads > 0:
        try:
            import torch
            torch.set_num_threads(args.threads)
            torch.set_num_interop_threads(max(1, args.threads // 2))
        except Exception as e:
            print(f"Warning: could not set torch threads: {e}")
        # Also set common BLAS/vecLib env vars for consistency
        os.environ.setdefault('OMP_NUM_THREADS', str(args.threads))
        os.environ.setdefault('MKL_NUM_THREADS', str(args.threads))
        os.environ.setdefault('VECLIB_MAXIMUM_THREADS', str(args.threads))
    
    # Configuration
    config = {
        # Model
        'grid_resolution': 64,
        'feature_dim': 32,
        'mlp_hidden': 64,
        
        # Training
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'rays_per_epoch': args.rays_per_epoch,
        'samples_per_ray': args.samples_per_ray,
        'learning_rate': 1e-3,
        
        # Output
        'output_dir': args.output_dir
    }
    
    # Create output directory
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(Path(config['output_dir']) / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load patient
    data_root = Path(args.data_root)

    if args.patient_id:
        patient_dir = data_root / args.patient_id
    else:
        # Find first patient directory with NIfTI files
        patient_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
        if not patient_dirs:
            raise ValueError(f"No patient directories in {data_root}")
        patient_dir = patient_dirs[0]
    patient = BraTSPatient(patient_dir)
    
    # Train
    start_time = time.time()
    model = train_patient(patient, config, device=args.device, resume_checkpoint=args.resume)
    train_time = time.time() - start_time
    
    # Evaluate
    dice_score = evaluate_model(model, patient, device=args.device)
    
    # Export
    export_path = Path(config['output_dir']) / f"{patient.patient_id}_model.npz"
    model.export_for_shader(str(export_path))
    
    # Summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Patient: {patient.patient_id}")
    print(f"Training time: {train_time/60:.1f} minutes")
    print(f"Final Dice score: {dice_score:.4f}")
    print(f"Model size: {export_path.stat().st_size / (1024*1024):.1f} MB")
    print(f"Saved to: {export_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
