# Neural Methods for Interactive Brain Glioma Identification

- Kyle Lukaszek and Kasra Fard
### What we’re doing (in plain terms)
We’re building an interactive viewer for brain MRI that stays fast while keeping tumor detail crisp. We do this by combining:
- A tiny neural shader inside a volume ray‑marcher (runs in the shader) that learns when to take bigger or smaller steps along each ray, based on a policy that will be trained externally on volumetric data. (Kyle)
- Computing neuro-imaging signals (tumor regions and related maps) offline from the UCSF‑PDGM dataset for glioma identification. (Kasra)

We compromise on our ideas by keeping the neural‑shader contribution at the core, and we add meaningful neuro-imaging + deep-learning work that supplies data the renderer can use for better, clinically relevant visuals.
### How it will work (lightweight)
- SlangPy portable app: cross‑platform viewer that renders multi‑modal MRI (e.g., FLAIR/T1/T2/DTI maps) and overlays tumor regions.
- Tiny policy network: a small MLP embedded in the shader (weights trained in PyTorch and exported) adjusts step size/skip during ray marching.
- Neuro add‑ons: offline models produce tumor probability maps and boundary/uncertainty cues from UCSF‑PDGM. We then pass those to the shader to protect ROI (region of interest) quality while accelerating elsewhere.
### What ML we’re using
- In‑shader: a compact custom policy MLP for adaptive raycast sampling (no heavy CNNs on the GPU during rendering).
- Offline neuro track: segmentation/probability maps (from provided masks or a pretrained BraTS U‑Net), plus optional biomarker baselines for IDH/MGMT using pyradiomics and a small CNN. These outputs appear as overlays and help the viewer identify potential areas of concern in the renderer, or protect sensitive areas of our image.
### How we’ll measure success
- Speed vs. quality: aim for ≥1.8× faster with ≤0.5 dB PSNR (peak (final image) signal-to-noise ratio) loss overall, and ≤0.3 dB inside tumor ROIs.
- Usability: smooth viewer with toggles (policy on/off, ROI overlays) and clear metrics onscreen.
- Evidence: side‑by‑sides, Pareto curves (quality vs. time), and ROI‑focused quality numbers.
### What we’ll deliver
- SlangPy viewer and shader code, plus training/export scripts for the tiny policy.
- Preprocessing and neuro baselines that output tumor probability/boundary maps and optional IDH/MGMT predictions for overlays.
- A concise write‑up with visuals, including before/after frames, error maps, speed/quality plots, and a short discussion.
- A presentation outlining what we wanted to research, why we wanted to research it, a brief intro to the required topics at hand, our methodology, and finally our results. We can even include a live demo of some test files if everything goes well.
### Datasets:
- UCSF‑PDGM (DOI: 10.7937/tcia.bdgf-8v37). We’ll cite per policy, load co‑registered NIfTI volumes, normalize them, and pack them as 3D textures for the viewer.
	- https://www.cancerimagingarchive.net/collection/ucsf-pdgm/
- Free OpenVDB sample files from EmberGen. There are loads of static (and dynamic) volumetric clouds, dust storms, fluids, and other cool things. We can cite them per download via the URL and Corp name (JangaFX).
	- https://jangafx.com/software/embergen/download/free-vdb-animations
### Who does what
- Kyle: Rendering & policy MLP
  - SlangPy viewer and shader integration
  - In‑shader MLP inference, PyTorch training/export, parity checks
  - Loading of volumetric data into the renderer using appropriate pipelines.
  - Performance/quality measurement and Pareto plots
- Kasra: Neuro-imaging & ML data 
  - UCSF‑PDGM preprocessing: curate/export tumor probability and boundary/uncertainty maps
  - Biomarker baselines (IDH/MGMT) via pyradiomics and a small CNN. We use this information to generate overlays for renderer tumor identification.
  - Collaborate on ROI‑focused metrics and viewer toggles for our renderer. 
### Extra Resources
- For more info on Slang, see:
	- https://github.com/shader-slang/slang
	- https://github.com/shader-slang/slangpy
- For info on neural shaders using Slang, see Nvidia's SIGGRAPH2025 Workshop:
	- https://github.com/shader-slang/neural-shading-s25/
