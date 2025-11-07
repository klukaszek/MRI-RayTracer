# Neumours: Neural Implicit Multi-Modal Tumour Representations
**Kyle Lukaszek, Kasra Fard, and Zack**

## What We're Doing (In Plain Terms)

We're building a system that bridges the gap between offline AI brain tumor segmentation and real-time interactive visualization. Instead of having doctors look at static 3D masks that AI produces separately, we're embedding the AI's learned knowledge directly into the rendering engine itself—making the entire pipeline differentiable and interactive.

We do this by:
1. Training a proven segmentation CNN (nnU-Net) on multi-modal brain MRI data to learn what tumors look like
2. **Distilling** that knowledge into a tiny neural network that lives inside a GPU shader
3. Using this shader-based network to render tumor segmentation in real-time as a **continuous implicit function** f(x,y,z) → tumor_class
4. Leveraging Slang's automatic differentiation (SLANG.D) to make the entire rendering pipeline differentiable—gradients flow from pixels back to the neural representation

The key innovation: the neural network **is** the renderer. No offline processing, no separate visualization step, no manual gradient coding.

## How It Works (System Architecture)

### Phase 1: Offline Learning (Kasra & Zack lead)
- Train nnU-Net on BraTS 2023 adult glioma dataset (multi-modal: T1, T1-ce, T2, FLAIR)
- Network learns robust tumor segmentation: MRI volumes → tumor masks
- Generate predictions on training set for use in Phase 2
- Validate segmentation quality (Dice scores, visual inspection)

### Phase 2: Knowledge Distillation (Joint)
- Extract nnU-Net's learned function and compress it into a compact MLP
- MLP architecture: (x, y, z) coordinates → tumor class probabilities (background, NCR/NET, edema, enhancing)
- Train this implicit representation to match nnU-Net's outputs
- Export MLP weights in format compatible with Slang shader

### Phase 3: Shader Implementation (Kyle leads)
- Implement MLP evaluation directly in Slang shader code
- Integrate into existing raymarching volume renderer
- Network inference happens **during** ray traversal—no pre-computed volumes
- Add toggle to switch between ground truth rendering and implicit network rendering

### Phase 4: Differentiable Pipeline (Joint)
- Wire up SLANG.D automatic differentiation through the renderer
- Prove gradients flow: rendered pixels → raytracer → MLP weights
- Demonstrate simple optimization example (e.g., refine boundary based on visual loss)

### Phase 5: Evaluation & Analysis (Joint)
- Visual comparison: ground truth vs. nnU-Net vs. implicit representation
- Quantitative metrics: Dice scores, rendering quality (PSNR), frame rates
- ROI-focused analysis: how well does implicit rep preserve tumor detail?
- Compression analysis: voxel grid size vs. MLP parameter count

## What ML We're Using

### Offline (Training Phase)
- **nnU-Net**: State-of-the-art medical segmentation CNN
  - Proven architecture, well-tuned for BraTS
  - Learns from multi-modal MRI → tumor masks
  - ~80M parameters (stays on CPU/GPU during training)

### Implicit Representation (Distillation Phase)
- **Compact MLP**: Tiny neural network (~10K-100K parameters)
  - Input: (x, y, z) world coordinates
  - Output: 4-class probabilities (background, NCR/NET, edema, enhancing)
  - Trained to approximate nnU-Net's function
  - Architecture: 3-4 hidden layers, ReLU activations, suitable for shader implementation

### In-Shader (Runtime)
- **GPU Shader Network**: MLP evaluation in Slang
  - Forward pass only during rendering (no backprop at runtime... yet)
  - Evaluated per-sample during raymarching
  - Target: <1ms per frame overhead for network evaluation

## How We'll Measure Success

### Performance Metrics
- **Frame rate**: Maintain 30+ FPS for interactive visualization
- **Rendering quality**: Ground truth vs. implicit representation (visual similarity, PSNR)
- **Compression ratio**: Voxel grid size (MB) vs. MLP weights (KB)

### Segmentation Quality

**On BraTS (Internal Validation Set):**
- **Dice coefficient**: nnU-Net predictions vs. ground truth
- **Baseline performance**: Establish upper bound for implicit representation

**On MU-GLIOMA-POST (External Test Set):**
- **Generalization**: nnU-Net trained on BraTS, tested on MU-GLIOMA-POST
- **Implicit rep fidelity**: How well does the distilled MLP match nnU-Net on novel data?
- **Boundary accuracy**: Tumor edge preservation on post-treatment cases
- **ROI preservation**: Tumor region detail vs. background regions
- **Cross-dataset robustness**: Performance drop from BraTS to MU-GLIOMA-POST

### Differentiability Proof
- **Gradient flow**: Successfully compute ∂loss/∂MLP_weights through renderer
- **Optimization demo**: Show simple refinement task using gradients
- **Convergence**: Demonstrate that gradient-based updates improve metrics

### Usability
- **Toggle modes**: Switch between ground truth, nnU-Net, and implicit rendering
- **Interactive controls**: Real-time parameter adjustment (window/level, modality weights)
- **Visual quality**: Side-by-side comparisons, error maps, difference visualizations
- **Dataset compatibility**: Seamlessly load both BraTS and MU-GLIOMA-POST cases

## What We'll Deliver

### Code & Implementation
1. **SlangPy Viewer**: Interactive multi-modal MRI renderer
   - Ground truth rendering (working)
   - Implicit representation toggle (in progress)
   - Differentiable optimization demo (planned)

2. **Training Pipeline**: 
   - nnU-Net training scripts for BraTS 2023
   - Implicit representation distillation code
   - Weight export utilities (PyTorch → Slang)

3. **Shader Code**:
   - MLP evaluation in Slang
   - Integration with raymarching pipeline
   - SLANG.D gradient computation setup

### Documentation & Analysis
1. **Jupyter Notebooks**: 
   - Data loading and visualization (working)
   - Training visualization and metrics (planned)
   - Evaluation and comparison (planned)

2. **Paper/Report**:
   - Introduction explaining the clinical disconnect and our solution
   - Related work: medical segmentation, neural implicit representations, differentiable rendering
   - Methods: architecture, distillation approach, shader implementation
   - Results: quantitative metrics, visual comparisons, ablation studies
   - Discussion: limitations, future work, clinical applications

3. **Presentation**:
   - Problem motivation (why this matters clinically)
   - Technical background (CNNs, implicit representations, differentiable rendering)
   - Our approach (the bridge between offline and interactive)
   - Live demo of viewer with toggles
   - Results and future directions

## Datasets

### Training: BraTS 2023 Adult Glioma
- **URL**: https://www.synapse.org/brats2023/
- **Contents**: Multi-modal MRI (T1, T1-ce, T2, FLAIR) + ground truth segmentation
- **Training Set**: 1251 cases (with masks available)
- **Why BraTS**: 
  - Well-established benchmark with high-quality annotations
  - Multi-institutional data with diverse scanner protocols
  - Standardized preprocessing and evaluation metrics
  - Validation/test sets are privatized (require submission to organizers)
- **Usage**: 
  - Train nnU-Net on the 1251 training cases
  - Use internal train/val split for hyperparameter tuning
  - Generate predictions on training set for implicit representation distillation
  - Reserve portion of training set for internal validation

### Testing: MU-GLIOMA-POST
- **URL**: https://www.cancerimagingarchive.net/collection/mu-glioma-post/
- **DOI**: 10.7937/TCIA.9YTJ-5Q73
- **Contents**: Post-treatment glioma MRI scans with segmentation masks
- **Why This Dataset**: 
  - Publicly available with ground truth masks (no submission required!)
  - Independent test set from different institution than BraTS training data
  - Tests generalization to real clinical data
  - Post-treatment cases add complexity (recurrence, treatment effects)
- **Usage**:
  - Final evaluation of nnU-Net generalization
  - Test implicit representation on unseen data
  - Validate rendering quality on novel cases
  - Demonstrate real-world applicability

### Optional: UCSF-PDGM
- **URL**: https://www.cancerimagingarchive.net/collection/ucsf-pdgm/
- **DOI**: 10.7937/tcia.bdgf-8v37
- **Usage**: Additional test cases if time permits for extended generalization experiments

## Task Breakdown & Assignment

## Task Breakdown & Assignment

### Phase 1: Foundation
**Kasra & Zack:**
- [ ] Implement nnU-Net training pipeline
- [ ] Create internal train/validation split from BraTS training set (e.g., 1100/151 split)
- [ ] Train nnU-Net on BraTS training subset
- [ ] Generate validation metrics on BraTS internal validation set (Dice scores)
- [ ] Export predictions on BraTS training set for distillation
- [x] Verify data format compatibility between BraTS and MU-GLIOMA-POST

**Kyle:**
- [x] ✅ BraTS 2023 dataset setup and preprocessing
- [x] ✅ MU-GLIOMA-POST dataset setup and preprocessing
- [x] ✅ Development environment configuration
- [x] ✅ Ground truth rendering pipeline
- [x] ✅ Multi-modal volume rendering (T1, T2, FLAIR)
- [x] ✅ Segmentation overlay with LUT coloring
- [x] ✅ Jupyter notebook for data exploration
- [x] Research Slang MLP implementation patterns
- [x] Design MLP architecture suitable for shader
- [x] Test renderer with MU-GLIOMA-POST cases

**Joint:**
- [x] Document current system architecture
- [ ] Plan distillation approach (sampling strategy, loss functions)

### Phase 2: Distillation
**Kasra & Zack:**
- [ ] Implement coordinate sampling from volumes
- [ ] Create training dataset: (x,y,z, nnU-Net_prediction) tuples
- [ ] Set up MLP training in PyTorch
- [ ] Train implicit representation to match nnU-Net
- [ ] Validate reconstruction quality (Dice, visual)

**Kyle:**
- [x] Implement MLP forward pass in Slang
- [x] Create weight loading mechanism (PyTorch → Slang)
- [ ] Test shader network evaluation (correctness, performance)
- [ ] Profile rendering overhead

**Joint:**
- [ ] Iterate on MLP architecture (size vs. accuracy tradeoff)
- [ ] Compare implicit representation rendering to ground truth
- [ ] Debug any quality issues

### Phase 3: Integration & Differentiability 
**Kyle:**
- [ ] Integrate shader MLP into main rendering pipeline
- [ ] Add UI toggle for ground truth vs. implicit rendering
- [ ] Wire up SLANG.D automatic differentiation
- [ ] Implement simple gradient-based optimization demo

**Kasra & Zack:**
- [ ] Run nnU-Net inference on MU-GLIOMA-POST test set
- [ ] Generate implicit representation predictions on test cases
- [ ] Create evaluation metrics pipeline (Dice, boundary accuracy)
- [ ] Analyze segmentation quality across test cases
- [ ] Document cases where implicit rep struggles vs. excels

**Joint:**
- [ ] Design and implement gradient-driven refinement experiment
- [ ] Measure end-to-end performance (frame rates, quality)
- [ ] Create comparison visualizations (side-by-side, error maps)
- [ ] Test generalization: BraTS-trained model on MU-GLIOMA-POST data

### Phase 4: Evaluation & Documentation
**Both:**
- [ ] Run comprehensive evaluation suite
- [ ] Generate all figures for paper/presentation
- [ ] Write methods section (architecture, training, implementation)
- [ ] Write results section (metrics, comparisons, ablations)
- [ ] Prepare live demo for presentation
- [ ] Practice presentation with Q&A

## Success Criteria (Must-Haves)

1. **Working viewer** with toggle between ground truth and implicit representation rendering
2. **Trained implicit representation** that visually matches nnU-Net output
3. **Real-time performance**: 30+ FPS on clinical-resolution volumes
4. **Differentiability proof**: Successfully compute gradients through renderer
5. **Quantitative validation**: Dice scores, PSNR, frame rate measurements
6. **Documentation**: Clear write-up explaining approach and results

## Stretch Goals (Nice-to-Haves)

1. **Interactive refinement**: Clinician paints corrections → gradient updates → instant visual feedback
2. **Uncertainty visualization**: Perturb MLP weights, show confidence regions
3. **Multi-resolution rendering**: Leverage continuous representation for arbitrary zoom levels
4. **Comparison to other compression methods**: Implicit rep vs. octrees, sparse voxels
5. **Generalization test**: Train on BraTS, test on UCSF-PDGM

## Resources & References

### Slang & Rendering
- Slang Language: https://github.com/shader-slang/slang
- SlangPy: https://github.com/shader-slang/slangpy
- Neural Shading Workshop (SIGGRAPH 2025): https://github.com/shader-slang/neural-shading-s25/

### Medical Segmentation
- nnU-Net paper: Isensee et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation"
- BraTS Challenge: https://www.synapse.org/brats2023/

### Neural Implicit Representations
- NeRF: Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
- Instant NGP: Müller et al., "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"
- SIREN: Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions"

### Differentiable Rendering
- Slang Differentiable Programming: Official Slang documentation on SLANG.D
- Differentiable rendering survey papers (cite as needed)

---

**Last Updated**: [Current Date]  
**Status**: Phase 1 (Foundation) - Ground truth rendering complete, nnU-Net training in progress
