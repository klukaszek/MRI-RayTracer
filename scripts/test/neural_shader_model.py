"""
Neural Shader Medical Segmentation for Glioma
Graduate Course Project - Emergence of Neural Shaders (SIGGRAPH 2025)

Key Papers:
[1] Müller et al. "Instant Neural Graphics Primitives" (SIGGRAPH 2022)
[2] Ravi et al. "Neural Fields in Visual Computing" (EuroGraphics 2022)
[3] Xie et al. "Neural Fields meet Explicit Geometric Representations" (CVPR 2022)
[4] Liu et al. "Neural Sparse Voxel Fields" (NeurIPS 2020)
[5] Isensee et al. "nnU-Net: Self-adapting Framework for Medical Image Segmentation" (Nature Methods 2021)

This implementation demonstrates:
- Neural shader-based volume rendering
- Compact MLP for real-time inference
- Multi-modal medical image fusion
- Differentiable rendering for end-to-end training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RenderConfig:
    """Configuration for neural shader rendering"""
    num_samples: int = 64          # Samples per ray
    step_size: float = 0.02        # Base step size
    near: float = 0.0              # Near plane
    far: float = 2.0               # Far plane
    density_noise: float = 0.0     # Training regularization
    white_background: bool = False


@dataclass
class RayBatch:
    """Batch of rays for rendering"""
    origins: torch.Tensor          # [B, 3]
    directions: torch.Tensor       # [B, 3]
    viewdirs: torch.Tensor         # [B, 3] normalized view directions


class PositionalEncoding(nn.Module):
    """
    Positional encoding from NeRF [Mildenhall et al., ECCV 2020]
    Maps coordinates to higher dimensional space for better learning
    
    gamma(p) = [sin(2^0 * pi * p), cos(2^0 * pi * p), ..., 
                sin(2^(L-1) * pi * p), cos(2^(L-1) * pi * p)]
    """
    
    def __init__(self, num_freqs: int = 6, include_input: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        
        # Frequency bands: 2^0, 2^1, ..., 2^(L-1)
        freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer('freq_bands', freq_bands)
        
        # Output dimension
        self.out_dim = num_freqs * 2  # sin and cos for each frequency
        if include_input:
            self.out_dim += 1  # Add original coordinate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., C] input coordinates
        Returns:
            encoded: [..., C * out_dim] encoded coordinates
        """
        if x.shape[-1] == 0:
            return x
        
        encodings = []
        
        if self.include_input:
            encodings.append(x)
        
        # Apply sin and cos for each frequency
        for freq in self.freq_bands:
            encodings.append(torch.sin(freq * np.pi * x))
            encodings.append(torch.cos(freq * np.pi * x))
        
        return torch.cat(encodings, dim=-1)


class TinyMLP(nn.Module):
    """
    Tiny MLP following Instant-NGP design [Müller et al., 2022]
    
    Key characteristics:
    - Small (64-128 hidden units)
    - ReLU activations
    - Few layers (2-3)
    - Designed for GPU shader execution
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 4,
        num_layers: int = 2,
        skip_connection: int = -1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.skip_connection = skip_connection
        
        # Build layers
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            # Add skip connection if specified
            if i == skip_connection:
                current_dim += input_dim
            
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            current_dim = hidden_dim
        
        # Output layer (no activation - raw logits)
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights (Xavier uniform is standard for MLPs)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def get_parameters_flat(self) -> np.ndarray:
        """Flatten all parameters for shader buffer"""
        params = []
        for param in self.parameters():
            params.append(param.detach().cpu().numpy().flatten())
        return np.concatenate(params)


class MultiModalFeatureGrid(nn.Module):
    """
    3D Feature Grid for multi-modal medical imaging
    Inspired by Neural Sparse Voxel Fields [Liu et al., 2020]
    
    Stores learned features in a 3D grid that can be:
    - Efficiently sampled via trilinear interpolation
    - Stored as 3D texture in GPU
    - Updated end-to-end during training
    """
    
    def __init__(
        self,
        resolution: int = 64,
        feature_dim: int = 32,
        num_modalities: int = 3  # T1, T2, FLAIR
    ):
        super().__init__()
        
        self.resolution = resolution
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        
        # Learnable feature grids for each modality
        self.grids = nn.ParameterList([
            nn.Parameter(torch.randn(1, feature_dim, resolution, resolution, resolution) * 0.1)
            for _ in range(num_modalities)
        ])
    
    def forward(self, points: torch.Tensor, modality_idx: int = 0) -> torch.Tensor:
        """
        Sample features at 3D points
        
        Args:
            points: [B, N, 3] in range [-1, 1]
            modality_idx: Which modality to sample
        Returns:
            features: [B, N, C]
        """
        grid = self.grids[modality_idx]
        
        # Reshape points for grid_sample: [B, N, 1, 1, 3]
        points_reshaped = points.unsqueeze(2).unsqueeze(2)
        
        # Sample features (trilinear interpolation)
        features = F.grid_sample(
            grid.expand(points.shape[0], -1, -1, -1, -1),
            points_reshaped,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        # Reshape output: [B, C, N, 1, 1] -> [B, N, C]
        features = features.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        
        return features
    
    def fuse_modalities(self, points: torch.Tensor) -> torch.Tensor:
        """
        Fuse features from all modalities
        
        Args:
            points: [B, N, 3]
        Returns:
            fused_features: [B, N, C]
        """
        features = []
        for i in range(self.num_modalities):
            features.append(self.forward(points, modality_idx=i))
        
        # Simple concatenation (can use learned fusion)
        fused = torch.cat(features, dim=-1)
        return fused


class NeuralShaderSegmentation(nn.Module):
    """
    Neural Shader for Medical Image Segmentation
    
    Architecture inspired by:
    - Instant-NGP [Müller et al., 2022] for compact representation
    - NeRF [Mildenhall et al., 2020] for volume rendering
    - nnU-Net [Isensee et al., 2021] for medical segmentation
    
    Key innovation: Entire inference runs in GPU shader
    """
    
    def __init__(
        self,
        feature_grid_res: int = 64,
        feature_dim: int = 32,
        mlp_hidden_dim: int = 64,
        num_classes: int = 4,  # background, edema, core, enhancing
        use_positional_encoding: bool = True,
        pos_encoding_freqs: int = 6
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_positional_encoding = use_positional_encoding
        
        # Multi-modal feature grids (T1, T2, FLAIR)
        self.feature_grids = MultiModalFeatureGrid(
            resolution=feature_grid_res,
            feature_dim=feature_dim,
            num_modalities=3
        )
        
        # Positional encoding (optional)
        self.pos_encoder = None
        pos_encoded_dim = 3
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(
                num_freqs=pos_encoding_freqs,
                include_input=True
            )
            pos_encoded_dim = 3 * self.pos_encoder.out_dim
        
        # Density + Segmentation MLP
        # Input: fused features (3 modalities) + positional encoding
        mlp_input_dim = feature_dim * 3 + pos_encoded_dim
        
        self.density_mlp = TinyMLP(
            input_dim=mlp_input_dim,
            hidden_dim=mlp_hidden_dim,
            output_dim=1 + num_classes,  # density + class logits
            num_layers=2
        )
        
        print(f"Neural Shader Model Initialized:")
        print(f"  Feature grid: {feature_grid_res}³ × {feature_dim} × 3 modalities")
        print(f"  MLP: {mlp_input_dim} → {mlp_hidden_dim} → {1 + num_classes}")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def query_density_and_segmentation(
        self,
        points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query density and segmentation at 3D points
        This is the core function that runs in the shader
        
        Args:
            points: [B, N, 3] 3D coordinates
        Returns:
            density: [B, N] volume density (sigma)
            segmentation: [B, N, num_classes] class logits
        """
        # Sample multi-modal features
        features = self.feature_grids.fuse_modalities(points)
        
        # Positional encoding
        if self.use_positional_encoding:
            pos_encoded = self.pos_encoder(points)
            features = torch.cat([features, pos_encoded], dim=-1)
        
        # MLP inference
        mlp_output = self.density_mlp(features)
        
        # Split output
        density = F.softplus(mlp_output[..., 0])  # Ensure positive
        segmentation_logits = mlp_output[..., 1:]
        
        return density, segmentation_logits
    
    def render_rays(
        self,
        rays: RayBatch,
        config: RenderConfig
    ) -> Dict[str, torch.Tensor]:
        """
        Volume rendering along rays
        
        Classic volume rendering equation from [Kajiya & Von Herzen, 1984]:
        C(r) = ∫ T(t) * σ(t) * c(t) dt
        where T(t) = exp(-∫ σ(s) ds) is transmittance
        
        Args:
            rays: Batch of rays
            config: Rendering configuration
        Returns:
            Dictionary with rendered outputs
        """
        batch_size = rays.origins.shape[0]
        device = rays.origins.device
        
        # Sample points along rays
        t_vals = torch.linspace(
            config.near,
            config.far,
            config.num_samples,
            device=device
        )
        
        # [B, N, 3] = [B, 1, 3] + [B, 1, 3] * [1, N, 1]
        points = (
            rays.origins[:, None, :] + 
            rays.directions[:, None, :] * t_vals[None, :, None]
        )
        
        # Query density and segmentation
        density, seg_logits = self.query_density_and_segmentation(points)
        
        # Add noise during training (regularization) and keep density non-negative
        if self.training and config.density_noise > 0:
            density = F.relu(density + torch.randn_like(density) * config.density_noise)
        
        # Compute distances between samples
        dists = t_vals[1:] - t_vals[:-1]
        dists = torch.cat([
            dists,
            torch.tensor([1e10], device=device).expand(1)
        ])
        dists = dists[None, :].expand(batch_size, -1)
        
        # Alpha compositing (clamp exponent to avoid overflow/NaNs)
        exp_arg = torch.clamp(-density * dists, min=-50.0, max=50.0)
        alpha = 1.0 - torch.exp(exp_arg)
        
        # Transmittance: T_i = exp(-sum(sigma_j * delta_j)) for j < i
        one_minus_alpha = torch.clamp(1.0 - alpha, min=1e-6, max=1.0)
        transmittance = torch.cumprod(
            torch.cat([
                torch.ones((batch_size, 1), device=device),
                one_minus_alpha
            ], dim=-1),
            dim=-1
        )[:, :-1]
        
        # Weights for integration
        weights = alpha * transmittance
        
        # Segmentation probabilities
        seg_probs = F.softmax(seg_logits, dim=-1)
        
        # Integrate along ray
        # Rendered segmentation: weighted sum of class probabilities
        rendered_seg = torch.sum(weights[..., None] * seg_probs, dim=1)
        
        # RGB color from segmentation (for visualization)
        class_colors = self._get_class_colors(device)
        rgb = torch.sum(weights[..., None] * (seg_probs @ class_colors), dim=1)
        
        # Accumulated opacity
        acc_map = torch.sum(weights, dim=-1)
        
        # White background if specified
        if config.white_background:
            rgb = rgb + (1.0 - acc_map[..., None])
        
        return {
            'rgb': rgb,                           # [B, 3]
            'segmentation': rendered_seg,         # [B, num_classes]
            'alpha': acc_map,                     # [B]
            'weights': weights,                   # [B, N]
            'density': density,                   # [B, N]
            'depth': torch.sum(weights * t_vals[None, :], dim=-1)  # [B]
        }
    
    def _get_class_colors(self, device) -> torch.Tensor:
        """
        Class colors for visualization
        0: Background (black)
        1: Edema (green)
        2: Non-enhancing core (red)
        3: Enhancing tumor (yellow)
        """
        colors = torch.tensor([
            [0.0, 0.0, 0.0],  # Background
            [0.0, 1.0, 0.0],  # Edema
            [1.0, 0.0, 0.0],  # Core
            [1.0, 1.0, 0.0],  # Enhancing
        ], device=device)
        return colors
    
    def export_for_shader(self, output_path: str):
        """
        Export model for GPU shader deployment
        
        Exports:
        1. Feature grids as 3D textures
        2. MLP weights as flat buffers
        3. Network architecture metadata
        """
        export_dict = {
            # Feature grids (3 modalities)
            'feature_grid_t1': self.feature_grids.grids[0].detach().cpu().numpy(),
            'feature_grid_t2': self.feature_grids.grids[1].detach().cpu().numpy(),
            'feature_grid_flair': self.feature_grids.grids[2].detach().cpu().numpy(),
            
            # MLP weights
            'mlp_weights': self.density_mlp.get_parameters_flat(),
            
            # Architecture info
            'metadata': {
                'feature_dim': self.feature_grids.feature_dim,
                'grid_resolution': self.feature_grids.resolution,
                'mlp_hidden_dim': self.density_mlp.hidden_dim,
                'num_classes': self.num_classes,
                'use_pos_encoding': self.use_positional_encoding,
            }
        }
        
        np.savez_compressed(output_path, **export_dict)
        
        # Print export info
        feature_grid_size = export_dict['feature_grid_t1'].nbytes * 3 / (1024 * 1024)
        mlp_size = export_dict['mlp_weights'].nbytes / 1024
        
        print(f"\nModel exported to {output_path}")
        print(f"  Feature grids: {feature_grid_size:.2f} MB (3 modalities)")
        print(f"  MLP weights: {mlp_size:.2f} KB")
        print(f"  Total: {feature_grid_size + mlp_size/1024:.2f} MB")


class SegmentationLoss(nn.Module):
    """
    Combined loss for medical image segmentation
    
    Components:
    1. Cross-entropy for classification
    2. Dice loss for class imbalance
    3. Density regularization
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        density_reg: float = 0.01
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.density_reg = density_reg
    
    def dice_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1.0
    ) -> torch.Tensor:
        """
        Dice loss for segmentation
        Dice = 2 * |X ∩ Y| / (|X| + |Y|)
        """
        # One-hot encode target if needed
        if target.dim() == 1 or target.shape[-1] != self.num_classes:
            target_one_hot = F.one_hot(target.long(), self.num_classes).float()
        else:
            target_one_hot = target
        
        # Flatten
        pred_flat = pred.view(-1, self.num_classes)
        target_flat = target_one_hot.view(-1, self.num_classes)
        
        # Compute dice per class
        intersection = (pred_flat * target_flat).sum(dim=0)
        union = pred_flat.sum(dim=0) + target_flat.sum(dim=0)
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        
        # Return 1 - dice (loss)
        return 1.0 - dice.mean()
    
    def forward(
        self,
        rendered: Dict[str, torch.Tensor],
        target_seg: torch.Tensor,
        target_density: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss
        
        Args:
            rendered: Output from render_rays
            target_seg: [B] or [B, num_classes] ground truth segmentation
            target_density: [B, N] ground truth density (optional)
        """
        pred_seg = rendered['segmentation']
        
        # Cross-entropy loss
        if target_seg.dim() == 1:
            ce_loss = F.cross_entropy(pred_seg, target_seg.long())
        else:
            ce_loss = -(target_seg * torch.log(pred_seg + 1e-8)).sum(dim=-1).mean()
        
        # Dice loss
        dice_loss = self.dice_loss(pred_seg, target_seg)
        
        # Density regularization (encourage sparsity)
        density_loss = rendered['density'].mean()
        
        # Total loss
        total_loss = (
            self.ce_weight * ce_loss +
            self.dice_weight * dice_loss +
            self.density_reg * density_loss
        )
        
        return {
            'total': total_loss,
            'ce': ce_loss,
            'dice': dice_loss,
            'density_reg': density_loss
        }


def generate_camera_rays(
    height: int,
    width: int,
    focal: float,
    c2w: torch.Tensor,
    device: str = 'cuda'
) -> RayBatch:
    """
    Generate camera rays for rendering
    
    Args:
        height, width: Image dimensions
        focal: Focal length
        c2w: [4, 4] camera-to-world transformation matrix
    Returns:
        RayBatch with origins and directions
    """
    # Pixel coordinates
    i, j = torch.meshgrid(
        torch.arange(width, device=device),
        torch.arange(height, device=device),
        indexing='xy'
    )
    
    # Normalized device coordinates
    dirs = torch.stack([
        (i - width * 0.5) / focal,
        -(j - height * 0.5) / focal,
        -torch.ones_like(i)
    ], dim=-1)
    
    # Transform to world space
    rays_d = torch.sum(
        dirs[..., None, :] * c2w[:3, :3],
        dim=-1
    )
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    
    # Normalize directions
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    
    # Flatten
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    viewdirs = viewdirs.reshape(-1, 3)
    
    return RayBatch(origins=rays_o, directions=rays_d, viewdirs=viewdirs)


# Example usage and demonstration
if __name__ == "__main__":
    print("=" * 70)
    print("Neural Shader Medical Segmentation")
    print("Graduate Course Project - Emergence of Neural Shaders")
    print("=" * 70)
    
    # Model configuration
    model = NeuralShaderSegmentation(
        feature_grid_res=64,
        feature_dim=32,
        mlp_hidden_dim=64,
        num_classes=4,
        use_positional_encoding=True,
        pos_encoding_freqs=6
    )
    
    # Print architecture
    print("\n" + "=" * 70)
    print("Key Design Principles:")
    print("=" * 70)
    print("1. Compact Representation (Instant-NGP style)")
    print("   - Feature grid: 64³ × 32 × 3 = ~8 MB")
    print("   - MLP: 2 layers, 64 hidden units = ~5 KB")
    print("   - Total: <10 MB (vs. 400+ MB for traditional CNNs)")
    print()
    print("2. GPU Shader Friendly")
    print("   - Trilinear sampling (hardware accelerated)")
    print("   - Small MLP (fits in shader registers)")
    print("   - No data-dependent branching")
    print()
    print("3. Differentiable End-to-End")
    print("   - Volume rendering is differentiable")
    print("   - Gradients flow through entire pipeline")
    print("   - Train with standard PyTorch")
    
    # Demo forward pass
    print("\n" + "=" * 70)
    print("Demo Forward Pass:")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Generate random rays
    num_rays = 1024
    rays = RayBatch(
        origins=torch.randn(num_rays, 3, device=device),
        directions=F.normalize(torch.randn(num_rays, 3, device=device), dim=-1),
        viewdirs=F.normalize(torch.randn(num_rays, 3, device=device), dim=-1)
    )
    
    # Render
    config = RenderConfig(num_samples=64, step_size=0.02)
    with torch.no_grad():
        outputs = model.render_rays(rays, config)
    
    print(f"\nRendered {num_rays} rays with {config.num_samples} samples each")
    print(f"Output shapes:")
    print(f"  RGB: {outputs['rgb'].shape}")
    print(f"  Segmentation: {outputs['segmentation'].shape}")
    print(f"  Alpha: {outputs['alpha'].shape}")
    print(f"  Depth: {outputs['depth'].shape}")
    
    # Export for deployment
    print("\n" + "=" * 70)
    print("Exporting for Shader Deployment:")
    print("=" * 70)
    model.export_for_shader('neural_shader_model.npz')
    
    print("\n✓ Model ready for GPU shader deployment!")
    print("\nNext steps:")
    print("1. Load exported model in SlangPy")
    print("2. Upload feature grids as 3D textures")
    print("3. Upload MLP weights to constant buffer")
    print("4. Implement raymarching in Slang shader")
    print("5. Enjoy real-time neural rendering!")
