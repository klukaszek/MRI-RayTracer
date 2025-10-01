# Showcase Plan: What We Will Demonstrate

This plan lists concrete artifacts (live demos, figures, videos, and tables) we will deliver to convincingly show our method’s benefits at 1080p on Bonsai, OpenVDB clouds, and procedural noise volumes, with an optional point‑cloud extension.

## Live Demo (Metal + Slang)
- Toggle view: baseline (B1) ↔ learned policy (Δt/skip/LOD)
- On‑screen overlays: ms/frame (GPU timestamps), PSNR vs. reference, steps/ray (avg, p90), skip rate, Δt stats
- Scenes: Bonsai, Clouds, Noise; 2 camera paths per scene (1 train, 1 held‑out)
- Budget sweep: hotkey to switch α values (e.g., 0.0, 0.1, 0.2, 0.4) and show Pareto movement in real time
- Optional: Point cloud splat scene with policy‑controlled step/LOD

## Figures (Paper/Slides)
- Side‑by‑side images (1080p): Reference, B1 baseline, Ours at iso‑quality
- Error heatmaps (abs error, log scale) for the same frames
- Pareto curves: PSNR/SSIM vs. ms/frame for each scene (B1 vs. ours), with key operating points labeled
- Steps‑per‑ray histograms (mean/p90) and skip‑rate distributions
- Spatial heatmaps: per‑pixel mean steps; show where savings occur
- Feature ablations: bar charts of speedup and PSNR for Δt‑only vs. Δt+skip vs. Δt+skip+LOD
- Generalization matrix: in‑distribution vs. held‑out views for each scene (ΔPSNR, speedup)
- Occupancy visualization: slices of the multi‑res occupancy grid and LOD usage under the policy
- GPU profiling: timeline chart of kernel times (march, feature, policy) with % of frame time

## Tables (Paper/Appendix)
- Per‑scene results at the selected operating point (iso‑quality): PSNR/SSIM, ms/frame, speedup, steps/ray (avg/p90)
- Aggregated means (±95% CI) across scenes/views
- Success criteria table: SC1/SC2/SC3 pass/fail per scene
- Ablation table: feature importance (drop in PSNR/speedup when removing a feature)
- Optimizer comparison: Adam/SGD/RMSProp/Muon with best hyperparameters, showing validation PSNR and iso‑quality speedup

## Videos
- 60‑second sizzle: split‑screen baseline vs. ours with overlays and a moving camera
- 2–3 minute walkthrough: explains method, shows Pareto sweep, ablations, and held‑out generalization

## Portability Story (Slang)
- Shaders authored once in Slang; compiled to Metal for our demo
- Artifact: build script that also emits SPIR‑V (for portability evidence); include identical shader hash in report
- Note: cross‑engine run is out‑of‑scope for the semester demo but the shader provenance enables it

## Reporting Protocol (for all showcased results)
- Resolution: 1920×1080 fixed; record GPU model
- References: very fine Δt_ref, no skipping; per‑scene stored images
- Determinism: fixed RNG seeds; log commit IDs and configs alongside outputs
- Timing: GPU timestamps only (exclude CPU); report median over ≥30 frames per view
- Confidence: report 95% CI across frames for PSNR/SSIM and ms/frame
- Quality metric details: compute PSNR per the protocol in docs/Neural-Volumetric-Shading.md

## Target Outcomes (to claim success)
- ≥1.8× speedup vs. B1 at ≤0.5 dB PSNR loss on ≥3/4 scenes
- Pareto dominance on ≥2 scenes; ≤0.7 dB extra PSNR loss on held‑out views
- Clear visual parity (no objectionable artifacts) in side‑by‑sides

## Checklist for Advisor Demo
- [ ] Live app with toggle and overlays at 1080p
- [ ] Bonsai, Clouds, Noise scenes each with train and held‑out views
- [ ] Reference, baseline, and ours frames exported for key views
- [ ] Pareto plots and ablation figures generated from docs/experiments.csv
- [ ] One profiling capture (Xcode GPU) per scene with annotated breakdown
- [ ] Optional: point‑cloud scene demo and results table
