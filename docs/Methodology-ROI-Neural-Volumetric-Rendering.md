# Methodology: ROI‑Aware Neural Volumetric Rendering with SlangPy

This document details how we integrate neural rendering policies with neuroimaging outputs from UCSF‑PDGM into an interactive SlangPy viewer.

## Roles and Ownership
- Kyle (Rendering & Policy MLP)
  - SlangPy viewer and shader integration, volume ray marcher, in‑shader MLP inference
  - PyTorch training/export of policy, parity tests, performance/quality evaluation
- Kasra (Neuroimaging & ML Data Products)
  - UCSF‑PDGM preprocessing; tumor probability P(x), boundary distance B(x), optional uncertainty U(x)
  - Biomarker baselines (IDH/MGMT) via radiomics and lightweight CNN; overlays and reports

## Runtime Stack
- SlangPy: compile and call Slang shaders from Python; create 3D textures from NumPy arrays; simple cross‑platform viewer.
- PyTorch: train tiny policy MLP; export FP16 weights/biases + JSON meta; verify parity vs. Slang inference on probes.
- Python I/O: nibabel/SimpleITK for NIfTI, numpy/scipy for preprocessing, skimage for distance transforms.

## Data Ingestion and Preprocessing
- Load co‑registered NIfTI volumes (FLAIR, T1/T1c, T2, SWI, DTI maps such as FA/MD). Respect qform/sform and spacing.
- Normalize per modality (e.g., percentile clip 1–99 then scale to [0,1]); optional bias‑field correction.
- Crop to brain/tumor AABB to reduce empty space. Save transforms for consistent camera framing.
- ROI maps:
  - Use provided expert‑corrected segmentations (enhancing, non‑enhancing/necrotic, FLAIR abnormality).
  - Build a tumor probability P(x): one‑hot labels → soft probabilities via light smoothing; optional refinement via pretrained BraTS U‑Net (offline).
  - Boundary map B(x): signed/unsigned distance transform of tumor mask(s).
  - Optional uncertainty U(x): Monte‑Carlo dropout or test‑time augmentation over a lightweight voxel head (if time); otherwise U(x)=0.

## Volume Packing for SlangPy
- Export volumes as NumPy arrays: intensities as float16, masks/probabilities as uint8 or float16.
- Create SlangPy textures for each 3D array (e.g., `r16float` for intensities; `r8unorm`/`r16float` for P/B/U).
- Keep modalities separate or in a texture array; keep ROI maps as single‑channel textures.

## Shader Design (Slang) — Kyle
- Core ray marcher: emission–absorption integration with early termination and optional occupancy grid.
- On‑the‑fly features per step:
  - Density and gradient magnitude (via finite differences or mip‑based gradients)
  - Accumulated transmittance T and depth t
  - View–gradient alignment metric
  - ROI features: P(x), B(x) (distance), optional U(x)
- Policy MLP (tiny):
  - Inputs: 8–16 dims (features above)
  - Hidden: 16–32 units, ReLU
  - Outputs: delta‑t scale in [0.25, 2.0], skip probability (thresholded), optional LOD index/scale
  - Weight/bias buffers uploaded from Python (FP16); matvec ops implemented in Slang.
- ROI‑aware controls:
  - If P(x) high or near boundary (|B(x)| small), clamp delta‑t to a minimum and disable skipping.
  - Else, apply policy outputs to increase step size or skip low‑value regions.

## Training the Policy (Offline) — Kyle
- Data: a small set of volumes (medical + non‑medical) with rendered references (very small delta‑t, no skip).
- Loss: `J = image_error + α · sample_cost`, with voxel/pixel weights higher inside/near ROI boundaries.
- Optimizers: start with AdamW (1e‑4 to 3e‑4), cosine decay with warmup; gradient clip 1.0; EMA optional.
- Selection: early stop on validation PSNR; choose checkpoint by best ROI‑weighted PSNR at target speed.
- Export: row‑major FP16 weights/biases, dims, activation flags to JSON; verify output parity vs. Slang on probe inputs.

## Evaluation — Joint
- Performance: GPU time (SlangPy timestamps if available), steps/ray, rays/s; speedup vs. tuned fixed‑step baseline.
- Quality: PSNR/SSIM vs. fine reference; Pareto curves (quality vs. time) over α settings.
- ROI fidelity: ROI‑PSNR and Weighted‑PSNR `w(x)=1+β·P(x)+γ·boundary` with 95% CI across frames.
- Generalization: held‑out views and unseen subjects; report ΔPSNR/ΔSSIM.

## Neuro Track — Kasra (Owner)
- Segmentation outputs: curate and package expert masks; optional probability refinement via a pretrained BraTS U‑Net.
- Biomarker baselines: radiomics (PyRadiomics + Logistic Regression/Random Forest) and, time permitting, a lightweight 3D CNN for IDH/MGMT with stratified 5‑fold CV.
- Exports to renderer: per‑voxel P(x), boundary distance, optional U(x); per‑subject biomarker predictions and saliency overlays.
- Viewer overlays: toggle masks/probabilities, boundary heatmaps, and biomarker predictions to enrich the demo and paper.

## SlangPy Harness (Sketch)
```python
import slangpy as spy
from slangpy import App
import numpy as np

app = App(width=1280, height=720, title="ROI-Aware Volume Viewer")
module = spy.Module.load_from_file(app.device, "shaders/volume_raymarch.slang")

# Load volumes prepared offline
flair = spy.Tensor.from_numpy(app.device, flair_np.astype(np.float16), format="r16float")
tumor_p = spy.Tensor.from_numpy(app.device, p_np.astype(np.float16), format="r16float")
boundary = spy.Tensor.from_numpy(app.device, b_np.astype(np.float16), format="r16float")

# Upload policy weights (FP16) as buffers
policy = {
    "w1": spy.Buffer.from_numpy(app.device, w1), "b1": spy.Buffer.from_numpy(app.device, b1),
    "w2": spy.Buffer.from_numpy(app.device, w2), "b2": spy.Buffer.from_numpy(app.device, b2),
}

while app.process_events():
    out = spy.Tensor.empty(app.device, (app.height, app.width, 4), format="rgba16float")
    module.render(
        pixel = spy.call_id(),
        volume = flair, roi_p = tumor_p, roi_b = boundary,
        policy = policy, params = {"alpha": 0.2, "dt_base": 1.0},
        _result = out,
    )
    app.blit(out); app.present()
```

## Reproducibility
- Fixed seeds; versioned configs; cached preprocessed subject subset for the demo.
- Document modality choices and normalization; include dataset citation requirements.

## Milestones
- MVP: single‑modality render + reference PSNR.
- Policy: Δt‑only + ROI clamps; then add skip.
- Full: overlays, Pareto curves, ROI metrics, ablations.
