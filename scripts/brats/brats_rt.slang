// Combined BraTS viewer shader: multi-intensity volumes + segmentation overlay

RWTexture2D<float4> gOutput;
// Optional INR overlay target (bound by host UI; unused in compute but kept for compatibility)
RWTexture2D<float4> gINROverlay;
// Precomputed INR probability volume (xyz voxel space):
// Layout: .x = NCR/NET (necrosis+non-enhancing), .y = Edema, .z = Enhancing, .w = TumorMass (sum of xyz)
Texture3D<float4> gINRVolume;
// Sampler for volume sampling (trilinear clamp)
SamplerState trilinearSampler;

// Up to 4 intensity volumes (float) in the same grid
StructuredBuffer<float> gIntensity0;
StructuredBuffer<float> gIntensity1;
StructuredBuffer<float> gIntensity2;
StructuredBuffer<float> gIntensity3;

// Optional segmentation labels (uint)
StructuredBuffer<uint> gLabels;

// INR weights: dimensions are provided by host via gParams to support small models

// Flattened row-major matrices W_i shaped (in, out), and biases b_i shaped (out)
StructuredBuffer<float> gW0; StructuredBuffer<float> gB0;
StructuredBuffer<float> gW1; StructuredBuffer<float> gB1;
StructuredBuffer<float> gW2; StructuredBuffer<float> gB2;
StructuredBuffer<float> gW3; StructuredBuffer<float> gB3;
StructuredBuffer<float> gW4; StructuredBuffer<float> gB4;

// -----------------------------------------------------------------------------
// Differentiable volume access (experimental)
// Can be disabled with: #define ENABLE_DIFF_PATH 0 before including this file.
// -----------------------------------------------------------------------------

#ifndef ENABLE_DIFF_PATH
#define ENABLE_DIFF_PATH 1
#endif

#if ENABLE_DIFF_PATH

// Toggle atomic accumulation for gradient writes. Some targets with SM 6.0
// may not support float atomics; keep off by default.
#ifndef ATOMIC_FLOAT_GRAD
#define ATOMIC_FLOAT_GRAD 0
#endif

inline void atomicAddFloatRW(inout RWStructuredBuffer<float> buf, uint index, float v)
{
#if ATOMIC_FLOAT_GRAD
    InterlockedAdd(buf[index], v);
#else
    buf[index] += v;
#endif
}

struct VolumePtr : IDifferentiablePtrType
{
    // The differential part is another VolumePtr, which is expected to carry a
    // matching gradient buffer to accumulate dL/dVox.
    typealias Differential = VolumePtr;

    StructuredBuffer<float> buf;         // primal samples
    RWStructuredBuffer<float> gradBuf;   // gradient accumulation target
    uint dimX; uint dimY; uint dimZ;     // dimensions for addressing
};

// Trilinear sampling with a custom backward derivative that splats gradients
// to the 8 neighboring voxels according to interpolation weights.
[BackwardDerivative(sampleTriLinear_bwd)]
float sampleTriLinear(VolumePtr p, float3 pIdx)
{
    float3 q = clamp(pIdx, float3(0.0,0.0,0.0), float3(p.dimX, p.dimY, p.dimZ) - float3(1.001,1.001,1.001));
    uint3 i0 = uint3(floor(q));
    float3 f = q - float3(i0);
    uint strideY = p.dimX;
    uint strideZ = p.dimX * p.dimY;
    uint base = i0.x + i0.y * strideY + i0.z * strideZ;
    float c000 = p.buf[base];
    float c100 = p.buf[base + 1];
    float c010 = p.buf[base + strideY];
    float c110 = p.buf[base + strideY + 1];
    float c001 = p.buf[base + strideZ];
    float c101 = p.buf[base + strideZ + 1];
    float c011 = p.buf[base + strideZ + strideY];
    float c111 = p.buf[base + strideZ + strideY + 1];
    float c00 = lerp(c000, c100, f.x);
    float c10 = lerp(c010, c110, f.x);
    float c01 = lerp(c001, c101, f.x);
    float c11 = lerp(c011, c111, f.x);
    float c0 = lerp(c00, c10, f.y);
    float c1 = lerp(c01, c11, f.y);
    return lerp(c0, c1, f.z);
}

void sampleTriLinear_bwd(DifferentialPtrPair<VolumePtr> dp, inout DifferentialPair<float3> dpIdx, float dOut)
{
    // Primal values come from dp.p; gradients accumulate to dp.d
    VolumePtr p = dp.p;
    VolumePtr g = dp.d;
    float3 q = clamp(dpIdx.p, float3(0.0,0.0,0.0), float3(p.dimX, p.dimY, p.dimZ) - float3(1.001,1.001,1.001));
    uint3 i0 = uint3(floor(q));
    float3 f = q - float3(i0);
    float fx = f.x, fy = f.y, fz = f.z;
    float wx0 = 1.0 - fx, wy0 = 1.0 - fy, wz0 = 1.0 - fz;
    float wx1 = fx,       wy1 = fy,       wz1 = fz;

    uint strideY = p.dimX;
    uint strideZ = p.dimX * p.dimY;
    uint base = i0.x + i0.y * strideY + i0.z * strideZ;

    // 8 weights
    float w000 = wx0 * wy0 * wz0;
    float w100 = wx1 * wy0 * wz0;
    float w010 = wx0 * wy1 * wz0;
    float w110 = wx1 * wy1 * wz0;
    float w001 = wx0 * wy0 * wz1;
    float w101 = wx1 * wy0 * wz1;
    float w011 = wx0 * wy1 * wz1;
    float w111 = wx1 * wy1 * wz1;

    // Accumulate voxel gradients via helper to allow opting into atomics.
    atomicAddFloatRW(g.gradBuf, base,                         dOut * w000);
    atomicAddFloatRW(g.gradBuf, base + 1,                     dOut * w100);
    atomicAddFloatRW(g.gradBuf, base + strideY,               dOut * w010);
    atomicAddFloatRW(g.gradBuf, base + strideY + 1,           dOut * w110);
    atomicAddFloatRW(g.gradBuf, base + strideZ,               dOut * w001);
    atomicAddFloatRW(g.gradBuf, base + strideZ + 1,           dOut * w101);
    atomicAddFloatRW(g.gradBuf, base + strideZ + strideY,     dOut * w011);
    atomicAddFloatRW(g.gradBuf, base + strideZ + strideY + 1, dOut * w111);

    // Derivatives w.r.t. pIdx (treating floor/clamp as constant almost
    // everywhere). Compute partials using the corner samples.
    float c000 = p.buf[base];
    float c100 = p.buf[base + 1];
    float c010 = p.buf[base + strideY];
    float c110 = p.buf[base + strideY + 1];
    float c001 = p.buf[base + strideZ];
    float c101 = p.buf[base + strideZ + 1];
    float c011 = p.buf[base + strideZ + strideY];
    float c111 = p.buf[base + strideZ + strideY + 1];

    float dVal_dfx = (wy0 * wz0) * (c100 - c000)
                   + (wy1 * wz0) * (c110 - c010)
                   + (wy0 * wz1) * (c101 - c001)
                   + (wy1 * wz1) * (c111 - c011);

    float dVal_dfy = (wx0 * wz0) * (c010 - c000)
                   + (wx1 * wz0) * (c110 - c100)
                   + (wx0 * wz1) * (c011 - c001)
                   + (wx1 * wz1) * (c111 - c101);

    float dVal_dfz = (wx0 * wy0) * (c001 - c000)
                   + (wx1 * wy0) * (c101 - c100)
                   + (wx0 * wy1) * (c011 - c010)
                   + (wx1 * wy1) * (c111 - c110);

    float3 dIdx = float3(dOut * dVal_dfx, dOut * dVal_dfy, dOut * dVal_dfz);
    dpIdx = DifferentialPair<float3>(dpIdx.p, dpIdx.d + dIdx);
}

struct Params
{
    uint2 imageSize; float fovY; float pad0;
    float3 eye; float pad1;
    float3 U; float pad2;
    float3 V; float pad3;
    float3 W; float pad4;
    float3 volMin; float pad5;
    float3 voxelSize; float pad6;
    uint3 dims; uint pad7;
    // raymarch controls
    float stepSize; float nearT; float farT; float pad8c;
    float3 bgColor; float pad9;

    // Intensity compositing controls
    uint4 volEnabled;     // 0/1 flags for 4 volumes
    float4 volWeight;     // weights for 4 volumes
    float ww;             // window width
    float wl;             // window level
    float intensityAlpha; // alpha scale for intensity
    float padInt[1];

    // Segmentation overlay
    uint showSeg; uint padSeg[3];
    float4 lutColorAlpha[8];

    // Tone/opacity shaping
    float gamma;        // intensity gamma for TF in [0.2, 3]
    float gradBoost;    // scales gradient magnitude contribution
    float gradScale;    // normalizes gradient magnitude (units ~ per-voxel)
    // INR overlay toggle
    uint showINR; uint padINR0[3];
    // INR dims (host-provided to avoid hard-coding gigantic networks)
    uint inrKFF;      // K_FF used when building FFs (for info)
    uint inrInDim;    // input dimension
    uint inrH0;
    uint inrH1;
    uint inrH2;
    uint inrH3;
    uint inrOutDim;   // number of output classes
    float inrAlpha;   // alpha for INR overlay
    uint inrStepStride; // evaluate INR every N steps along ray
    float inrMinIntensity; // only evaluate INR if base intensity >= this
    uint useDiffINR; // 1 => use differentiable modality-only INR path
    // Overlay texture dimensions (UI passes these; shader may ignore)
    uint inrOverlayWidth;
    uint inrOverlayHeight;
    uint padINRExtra[1];
    // Precomputed INR usage flag
    uint precomputedINR; uint padINR2[3];
};
ConstantBuffer<Params> gParams;
// Placeholder differentiable modality-only INR to satisfy references.
// -----------------------------------------------------------------------------
// Added: Differentiable INR utilities closer to reference example
// -----------------------------------------------------------------------------

// Positional encoding (Fourier features) for better spatial frequency capture.
[Differentiable]
float3 positionalEncoding(float3 pos, int numFreqs)
{
    float3 encoded = pos;
    [MaxIters(16)] // cap positional encoding frequencies
    for (int i = 0; i < numFreqs; i++)
    {
        float freq = pow(2.0, float(i));
        encoded += sin(pos * freq) * 0.1;
        encoded += cos(pos * freq) * 0.1;
    }
    return encoded;
}

[Differentiable]
float leakyRelu(float x, float alpha)
{
    return x > 0.0 ? x : alpha * x;
}

[Differentiable]
void softmax(inout float output[5])
{
    float maxVal = output[0];
    [ForceUnroll]
    for (int i = 1; i < 5; i++) maxVal = max(maxVal, output[i]);
    float sum = 0.0;
    [ForceUnroll]
    for (int i = 0; i < 5; i++) { output[i] = exp(output[i] - maxVal); sum += output[i]; }
    [ForceUnroll]
    for (int i = 0; i < 5; i++) output[i] /= max(1e-6, sum);
}

// Forward pass using existing flexible weight buffers when dims match a canonical
// 4->64->128->64->5 architecture. Falls back to heuristic distribution if dims differ.
[Differentiable]
void queryINR_modality_only(float4 mriValues, out float probs[5])
{
    // Canonical target dims
    bool archOK = (gParams.inrInDim == 4) && (gParams.inrH0 == 64) && (gParams.inrH1 == 128) && (gParams.inrH2 == 64) && (gParams.inrOutDim == 5);
    if (!archOK)
    {
        // Fallback heuristic using reordered semantic channels
        float T1   = mriValues.x;
        float T1CE = mriValues.y;
        float T2   = mriValues.z;
        float FLAIR= mriValues.w;
        float s = T1 + T2 + FLAIR + T1CE + 1e-6;
        probs[0] = clamp(1.0 - (T1CE + FLAIR) * 0.5, 0.0, 1.0); // background
        probs[1] = T1CE / s;  // necrosis proxy
        probs[2] = FLAIR / s; // edema proxy
        probs[3] = T1CE / s;  // enhancing proxy
        probs[4] = T2   / s;  // non-enhancing proxy
        float sum = probs[0]+probs[1]+probs[2]+probs[3]+probs[4];
        [ForceUnroll]
        for(int i=0;i<5;i++) probs[i] = probs[i] / max(1e-6, sum);
        return;
    }

    // Input modalities expected by reference: [T1, T2, FLAIR, T1CE].
    // Host binding order is: x=T1n, y=T1c, z=T2w, w=FLAIR. Reorder here.
    float in[4];
    float T1   = mriValues.x;
    float T1CE = mriValues.y;
    float T2   = mriValues.z;
    float FLAIR= mriValues.w;
    in[0] = T1;
    in[1] = T2;
    in[2] = FLAIR;
    in[3] = T1CE;

    // Layer 1: 4 -> 64 (gW0 shaped (4,64))
    float h1[64];
    [ForceUnroll]
    for (int o=0;o<64;o++)
    {
        float acc = gB0[o];
        [ForceUnroll]
        for (int i=0;i<4;i++) acc += in[i] * gW0[i*64 + o];
        h1[o] = leakyRelu(acc, 0.01);
    }
    // Layer 2: 64 -> 128 (gW1 shaped (64,128))
    float h2[128];
    [MaxIters(128)]
    for (int o=0;o<128;o++)
    {
        float acc = gB1[o];
        [ForceUnroll]
        for (int i=0;i<64;i++) acc += h1[i] * gW1[i*128 + o];
        h2[o] = leakyRelu(acc, 0.01);
    }
    // Layer 3: 128 -> 64 (gW2 shaped (128,64))
    float h3[64];
    [ForceUnroll]
    for (int o=0;o<64;o++)
    {
        float acc = gB2[o];
        [MaxIters(128)]
        for (int i=0;i<128;i++) acc += h2[i] * gW2[i*64 + o];
        h3[o] = leakyRelu(acc, 0.01);
    }
    // (Optional extra hidden if provided: 64 -> 64 using gW3/gB3 with identity fallback)
    float h4[64];
    bool extraHidden = (gParams.inrH3 == 64);
    if (extraHidden)
    {
        [ForceUnroll]
        for (int o=0;o<64;o++)
        {
            float acc = gB3[o];
            [ForceUnroll]
            for (int i=0;i<64;i++) acc += h3[i] * gW3[i*64 + o];
            h4[o] = leakyRelu(acc, 0.01);
        }
    }
    else
    {
        for (int i=0;i<64;i++) h4[i] = h3[i];
    }
    // Output: 64 -> 5 (gW4 shaped (64,5))
    [ForceUnroll]
    for (int o=0;o<5;o++)
    {
        float acc = gB4[o];
        [ForceUnroll]
        for (int i=0;i<64;i++) acc += h4[i] * gW4[i*5 + o];
        probs[o] = acc;
    }
    softmax(probs);
}

struct Ray { float3 o; float3 d; };

Ray makePrimary(uint2 pixel)
{
    float2 dim = float2(gParams.imageSize);
    float2 ndc = (float2(pixel) + 0.5) / dim;
    float2 uv = ndc * 2.0 - 1.0;
    float f = 1.0 / tan(0.5 * gParams.fovY);
    float aspect = dim.x / max(1.0, dim.y);
    float3 rdCam = normalize(float3(uv.x * aspect / f, -uv.y / f, 1.0));
    float3 ro = gParams.eye;
    float3 rd = normalize(rdCam.x * gParams.U + rdCam.y * gParams.V + rdCam.z * gParams.W);
    Ray r; r.o = ro; r.d = rd; return r;
}

bool aabbHit(float3 ro, float3 rcpDir, float3 bmin, float3 bmax, out float tmin, out float tmax)
{
    float3 t0 = (bmin - ro) * rcpDir;
    float3 t1 = (bmax - ro) * rcpDir;
    float3 tsm = min(t0, t1);
    float3 tbg = max(t0, t1);
    tmin = max(max(tsm.x, tsm.y), tsm.z);
    tmax = min(min(tbg.x, tbg.y), tbg.z);
    return tmax >= max(tmin, 0.0);
}

float sampleLinearBuffer(StructuredBuffer<float> buf, float3 pIdx)
{
    float3 q = clamp(pIdx, float3(0.0,0.0,0.0), float3(gParams.dims) - float3(1.001,1.001,1.001));
    uint3 i0 = uint3(floor(q));
    float3 f = q - float3(i0);
    uint dimX = gParams.dims.x;
    uint dimY = gParams.dims.y;
    uint strideY = dimX;
    uint strideZ = dimX * dimY;
    uint base = i0.x + i0.y * strideY + i0.z * strideZ;
    float c000 = buf[base];
    float c100 = buf[base + 1];
    float c010 = buf[base + strideY];
    float c110 = buf[base + strideY + 1];
    float c001 = buf[base + strideZ];
    float c101 = buf[base + strideZ + 1];
    float c011 = buf[base + strideZ + strideY];
    float c111 = buf[base + strideZ + strideY + 1];
    float c00 = lerp(c000, c100, f.x);
    float c10 = lerp(c010, c110, f.x);
    float c01 = lerp(c001, c101, f.x);
    float c11 = lerp(c011, c111, f.x);
    float c0 = lerp(c00, c10, f.y);
    float c1 = lerp(c01, c11, f.y);
    return lerp(c0, c1, f.z);
}

// Sample each modality at pIdx (trilinear)
float4 sampleModalities(float3 pIdx)
{
    float i0 = sampleLinearBuffer(gIntensity0, pIdx);
    float i1 = sampleLinearBuffer(gIntensity1, pIdx);
    float i2 = sampleLinearBuffer(gIntensity2, pIdx);
    float i3 = sampleLinearBuffer(gIntensity3, pIdx);
    return float4(i0, i1, i2, i3);
}

// Compute intensity TF from already-sampled modalities (avoids re-sampling)
float computeIntensityFromModalities(float4 mods)
{
    float acc = 0.0;
    float wsum = 0.0;
    if (gParams.volEnabled.x != 0) { acc += mods.x * gParams.volWeight.x; wsum += gParams.volWeight.x; }
    if (gParams.volEnabled.y != 0) { acc += mods.y * gParams.volWeight.y; wsum += gParams.volWeight.y; }
    if (gParams.volEnabled.z != 0) { acc += mods.z * gParams.volWeight.z; wsum += gParams.volWeight.z; }
    if (gParams.volEnabled.w != 0) { acc += mods.w * gParams.volWeight.w; wsum += gParams.volWeight.w; }
    float v = (wsum > 0.0) ? (acc / wsum) : 0.0;
    float low = gParams.wl - 0.5 * gParams.ww;
    float hi  = gParams.wl + 0.5 * gParams.ww;
    float tf = saturate((v - low) / max(1e-6, (hi - low)));
    tf = pow(saturate(tf), max(0.01, gParams.gamma));
    return tf;
}

// Non-differentiable path (uses global buffers)
float sampleIntensity(float3 pIdx)
{
    float acc = 0.0;
    float wsum = 0.0;
    if (gParams.volEnabled.x != 0) { float v = sampleLinearBuffer(gIntensity0, pIdx); acc += v * gParams.volWeight.x; wsum += gParams.volWeight.x; }
    if (gParams.volEnabled.y != 0) { float v = sampleLinearBuffer(gIntensity1, pIdx); acc += v * gParams.volWeight.y; wsum += gParams.volWeight.y; }
    if (gParams.volEnabled.z != 0) { float v = sampleLinearBuffer(gIntensity2, pIdx); acc += v * gParams.volWeight.z; wsum += gParams.volWeight.z; }
    if (gParams.volEnabled.w != 0) { float v = sampleLinearBuffer(gIntensity3, pIdx); acc += v * gParams.volWeight.w; wsum += gParams.volWeight.w; }
    float v = (wsum > 0.0) ? (acc / wsum) : 0.0;
    // Window/level to [0,1]
    float low = gParams.wl - 0.5 * gParams.ww;
    float hi  = gParams.wl + 0.5 * gParams.ww;
    float tf = saturate((v - low) / max(1e-6, (hi - low)));
    // gamma shaping
    tf = pow(saturate(tf), max(0.01, gParams.gamma));
    return tf;
}

// Differentiable intensity sampling using VolumePtr inputs. The layout matches
// the non-diff version but uses sampleTriLinear so gradients can propagate to
// the underlying volume(s).
[Differentiable]
float sampleIntensityDiff(float3 pIdx, VolumePtr vol0, VolumePtr vol1, VolumePtr vol2, VolumePtr vol3)
{
    float acc = 0.0;
    float wsum = 0.0;
    if (gParams.volEnabled.x != 0) { float v0 = sampleTriLinear(vol0, pIdx); acc += v0 * gParams.volWeight.x; wsum += gParams.volWeight.x; }
    if (gParams.volEnabled.y != 0) { float v1 = sampleTriLinear(vol1, pIdx); acc += v1 * gParams.volWeight.y; wsum += gParams.volWeight.y; }
    if (gParams.volEnabled.z != 0) { float v2 = sampleTriLinear(vol2, pIdx); acc += v2 * gParams.volWeight.z; wsum += gParams.volWeight.z; }
    if (gParams.volEnabled.w != 0) { float v3 = sampleTriLinear(vol3, pIdx); acc += v3 * gParams.volWeight.w; wsum += gParams.volWeight.w; }
    float v = (wsum > 0.0) ? (acc / wsum) : 0.0;
    float low = gParams.wl - 0.5 * gParams.ww;
    float hi  = gParams.wl + 0.5 * gParams.ww;
    float tf = saturate((v - low) / max(1e-6, (hi - low)));
    tf = pow(saturate(tf), max(0.01, gParams.gamma));
    return tf;
}

uint sampleNearestLabel(float3 pIdx)
{
    float3 q = clamp(pIdx, float3(0.0,0.0,0.0), float3(gParams.dims) - float3(1.0,1.0,1.0));
    uint3 i = uint3(round(q));
    i = min(i, gParams.dims - uint3(1,1,1));
    uint dimX = gParams.dims.x;
    uint dimY = gParams.dims.y;
    uint strideY = dimX;
    uint strideZ = dimX * dimY;
    uint idx = i.x + i.y * strideY + i.z * strideZ;
    return gLabels[idx];
}

// Approximate gradient magnitude with forward differences to reduce fetches.
// Accepts the center value v0 to avoid an extra sample at pIdx.
float gradientMagForward(float3 pIdx, float v0)
{
    float3 px = float3(1.0, 0.0, 0.0);
    float3 py = float3(0.0, 1.0, 0.0);
    float3 pz = float3(0.0, 0.0, 1.0);
    float vx1 = sampleIntensity(pIdx + px);
    float vy1 = sampleIntensity(pIdx + py);
    float vz1 = sampleIntensity(pIdx + pz);
    float dx = (vx1 - v0);
    float dy = (vy1 - v0);
    float dz = (vz1 - v0);
    float g = sqrt(dx*dx + dy*dy + dz*dz);
    // Normalize by provided scale, clamp to [0,1]
    float gs = g * max(0.0, gParams.gradScale);
    return saturate(gs);
}

// Differentiable counterpart that uses the differentiable intensity path and
// reuses the center value v0.
[Differentiable]
float gradientMagForwardDiff(float3 pIdx, float v0, VolumePtr vol0, VolumePtr vol1, VolumePtr vol2, VolumePtr vol3)
{
    float3 px = float3(1.0, 0.0, 0.0);
    float3 py = float3(0.0, 1.0, 0.0);
    float3 pz = float3(0.0, 0.0, 1.0);
    float vx1 = sampleIntensityDiff(pIdx + px, vol0, vol1, vol2, vol3);
    float vy1 = sampleIntensityDiff(pIdx + py, vol0, vol1, vol2, vol3);
    float vz1 = sampleIntensityDiff(pIdx + pz, vol0, vol1, vol2, vol3);
    float dx = (vx1 - v0);
    float dy = (vy1 - v0);
    float dz = (vz1 - v0);
    float g = sqrt(dx*dx + dy*dy + dz*dz);
    float gs = g * max(0.0, gParams.gradScale);
    return saturate(gs);
}

// ---------------------------------------------------------------------------------
// INR forward: Fourier features + MLP (fixed dims defined above)
// ---------------------------------------------------------------------------------

float2 sincos(float x) { return float2(sin(x), cos(x)); }

// Build input vector x of size INR_IN_DIM from normalized coords in [-1,1]
void buildInrInput(float3 coordsNorm, float4 intens, out float x[])
{
    // coords first
    uint inDim = max(1u, gParams.inrInDim);
    // Initialize to zero to avoid uninitialized warnings when early returning
    [loop] for (uint i = 0; i < inDim && i < 64; ++i) x[i] = 0.0;
    if (inDim < 7) return; // requires at least coords(3)+intens(4)
    x[0] = coordsNorm.x; x[1] = coordsNorm.y; x[2] = coordsNorm.z;
    // Fourier features per-axis, harmonics 1..K_FF with pi scale to match training
    uint idx = 3;
    [unroll]
    for (uint axis = 0; axis < 3; ++axis)
    {
        float c = (axis == 0) ? coordsNorm.x : (axis == 1 ? coordsNorm.y : coordsNorm.z);
        [unroll]
        for (uint k = 1; k <= gParams.inrKFF; ++k)
        {
            float a = c * float(k) * 3.14159265359;
            float2 sc = sincos(a);
            if (idx + 1 < gParams.inrInDim) x[idx++] = sc.x; // sin
            if (idx + 0 < gParams.inrInDim) x[idx++] = sc.y; // cos
        }
    }
    // modalities
    if (idx + 3 < gParams.inrInDim) {
        x[idx++] = intens.x; x[idx++] = intens.y; x[idx++] = intens.z; x[idx++] = intens.w;
    }
}

// y = x @ W + b, where W is (in,out) row-major flattened
void dense(const float xIn[], uint inDim, StructuredBuffer<float> W, StructuredBuffer<float> b, out float yOut[], uint outDim)
{
    [loop]
    for (uint o = 0; o < outDim; ++o)
    {
        float acc = b[o];
        [loop]
        for (uint i = 0; i < inDim; ++i)
        {
            acc += xIn[i] * W[i * outDim + o];
        }
        yOut[o] = acc;
    }
}

void relu_inplace(inout float v[], uint n)
{
    [loop]
    for (uint i = 0; i < n; ++i) v[i] = max(0.0, v[i]);
}

// Compute logits for current sample
void inr_logits(float3 pIdx, out float logits[])
{
    // Normalize coordinates to [-1,1] using dims
    float3 norm = (pIdx / (float3(gParams.dims) - float3(1.0,1.0,1.0))) * 2.0 - 1.0;
    float4 intens = sampleModalities(pIdx);
    // Allocate with upper bounds based on provided dims. Slang requires static sizes for local arrays,
    // so pick conservative maxima that are still small for our mac targets.
    static const uint MAX_IN = 64;   // covers inrInDim up to 64 (31 for KFF=4)
    static const uint MAX_H = 256;   // covers small models (e.g., 128)
    static const uint MAX_OUT = 8;   // up to 8 classes
    uint inDim = min(MAX_IN, max(1u, gParams.inrInDim));
    uint h0Dim = min(MAX_H, max(1u, gParams.inrH0));
    uint h1Dim = min(MAX_H, max(1u, gParams.inrH1));
    uint h2Dim = min(MAX_H, max(1u, gParams.inrH2));
    uint h3Dim = min(MAX_H, max(1u, gParams.inrH3));
    uint outDim = min(MAX_OUT, max(1u, gParams.inrOutDim));

    float x[MAX_IN];
    buildInrInput(norm, intens, x);

    float h0[MAX_H];
    dense(x, inDim, gW0, gB0, h0, h0Dim);
    relu_inplace(h0, h0Dim);

    float h1[MAX_H];
    dense(h0, h0Dim, gW1, gB1, h1, h1Dim);
    relu_inplace(h1, h1Dim);

    float h2[MAX_H];
    dense(h1, h1Dim, gW2, gB2, h2, h2Dim);
    relu_inplace(h2, h2Dim);

    float h3[MAX_H];
    dense(h2, h2Dim, gW3, gB3, h3, h3Dim);
    relu_inplace(h3, h3Dim);

    dense(h3, h3Dim, gW4, gB4, logits, outDim);
}

uint argmax4(const float v[4])
{
    uint mi = 0; float mv = v[0];
    if (v[1] > mv) { mv = v[1]; mi = 1; }
    if (v[2] > mv) { mv = v[2]; mi = 2; }
    if (v[3] > mv) { /*mv = v[3];*/ mi = 3; }
    return mi;
}

// Generic argmax for variable-sized arrays (n <= 8 in our use)
uint argmaxN(const float v[], uint n)
{
    uint mi = 0; float mv = v[0];
    for (uint i = 1; i < n; ++i)
    {
        if (v[i] > mv) { mv = v[i]; mi = i; }
    }
    return mi;
}

// Compute top class and its softmax probability
void sampleINRTop(float3 pIdx, out uint lbl, out float prob)
{
    float logits[8];
    inr_logits(pIdx, logits);
    uint n = min(8u, max(1u, gParams.inrOutDim));
    // argmax
    uint mi = 0; float mv = logits[0];
    for (uint i = 1; i < n; ++i) { if (logits[i] > mv) { mv = logits[i]; mi = i; } }
    // softmax prob of top (numerically stable)
    float maxv = mv;
    float sum = 0.0;
    for (uint i = 0; i < n; ++i) sum += exp(logits[i] - maxv);
    float p = 1.0 / max(1e-6, sum);
    lbl = mi;
    prob = p;
}

[shader("compute")]
[numthreads(8,8,1)]
void brats_main(uint3 tid : SV_DispatchThreadID)
{
    uint2 dims2D = gParams.imageSize;
    if (any(tid.xy >= dims2D)) return;

    Ray ray = makePrimary(tid.xy);
    float3 bmin = gParams.volMin;
    float3 bmax = gParams.volMin + gParams.voxelSize * float3(gParams.dims);
    float3 d = ray.d;
    d = float3(
        abs(d.x) < 1e-7 ? (d.x >= 0 ? 1e-7 : -1e-7) : d.x,
        abs(d.y) < 1e-7 ? (d.y >= 0 ? 1e-7 : -1e-7) : d.y,
        abs(d.z) < 1e-7 ? (d.z >= 0 ? 1e-7 : -1e-7) : d.z);
    float3 rcpDir = 1.0 / d;
    float t0, t1;
    float3 C = gParams.bgColor;
    if (!aabbHit(ray.o, rcpDir, bmin, bmax, t0, t1))
    {
        gOutput[tid.xy] = float4(C, 1.0);
        return;
    }
    t0 = max(t0, max(0.0, gParams.nearT));
    t1 = min(t1, (gParams.farT > 0.0) ? gParams.farT : t1);
    if (t1 <= t0) { gOutput[tid.xy] = float4(C, 1.0); return; }

    float T = 1.0;
    float step = max(1e-5, gParams.stepSize);
    float t = t0;
    // Volume diagonal for distance normalization (eye-space heuristic)
    float volDiag = length(gParams.voxelSize * float3(gParams.dims));
    int lastInrStep = -1024;
    float prevConfidence = 0.0;
    // Derive a safe max steps based on segment length
    int maxSteps = min(1024, int(ceil((t1 - t0) / step))); // cap hard at 1024
    int steps = 0;
    [MaxIters(1024)]
    while (t < t1 && T > 1e-3 && steps < maxSteps)
    {
        float3 P = ray.o + t * ray.d;
        float3 pIdx = (P - gParams.volMin) / gParams.voxelSize;

        // Intensity base; reuse modalities for INR gating
        float4 intens4 = sampleModalities(pIdx);
        float v = computeIntensityFromModalities(intens4);
        if (v > 0.0)
        {
            float edge = 1.0;
            if (gParams.gradBoost > 0.001)
            {
                float gmag = gradientMagForward(pIdx, v);
                edge = 1.0 + max(0.0, gParams.gradBoost) * gmag;
            }
            float aI = saturate(v) * edge * gParams.intensityAlpha;
            float3 colI = v.xxx;
            float alphaI = 1.0 - exp(-aI * step);
            float3 contribI = alphaI * T * colI;
            C = C + contribI;
            T *= (1.0 - alphaI);
        }

        // Segmentation overlay
        if (gParams.showSeg != 0 && T > 1e-3)
        {
            // Labels are non-differentiable; plain call in non-diff shader
            uint lbl = sampleNearestLabel(pIdx) & 0xFFu;
            uint idx = min(lbl, 7u);
            float4 ca = gParams.lutColorAlpha[idx];
            float alphaS = 1.0 - exp(-ca.w * step);
            float3 contribS = alphaS * T * ca.xyz;
            C = C + contribS;
            T *= (1.0 - alphaS);
        }

        // INR per-step (adaptive): interior + signal + adaptive stride based on distance & confidence
        bool interior = all(pIdx > float3(1.0,1.0,1.0)) && all(pIdx < (float3(gParams.dims) - float3(2.0,2.0,2.0)));
        float sumMods = intens4.x + intens4.y + intens4.z + intens4.w;
        float dCam = length((ray.o + t * ray.d) - gParams.eye);
        float dNorm = saturate(dCam / max(1e-6, volDiag));
        float strideDist = lerp(1.0, 8.0, dNorm);          // farther => larger stride
        float strideCert = lerp(1.0, 4.0, saturate(1.0 - prevConfidence)); // low confidence => larger stride
        float adaptiveStrideF = max(1.0, float(gParams.inrStepStride) * strideDist * strideCert);
        int adaptiveStride = int(clamp(round(adaptiveStrideF), 1.0, 128.0));
        bool timeToEval = (steps - lastInrStep) >= adaptiveStride;
        if (gParams.showINR != 0 && interior && (sumMods > 1e-6) && v >= gParams.inrMinIntensity && timeToEval && T > 1e-3)
        {
            if (gParams.precomputedINR != 0)
            {
                float3 uvw = (pIdx + float3(0.5,0.5,0.5)) / float3(gParams.dims);
                float4 pre = gINRVolume.SampleLevel(trilinearSampler, uvw, 0);
                float pNCR = pre.x;
                float pED  = pre.y;
                float pET  = pre.z;
                float pTumor = saturate(pre.w);
                float3 colNCR = gParams.lutColorAlpha[1].xyz;
                float3 colED  = gParams.lutColorAlpha[2].xyz;
                float3 colET  = gParams.lutColorAlpha[4].xyz;
                float3 inrColor = colNCR * pNCR + colED * pED + colET * pET;
                const float INREmissionAbsorption = 10.0;
                float a = gParams.inrAlpha * pTumor * INREmissionAbsorption;
                float alphaS = 1.0 - exp(-a * step);
                float3 contribS = alphaS * T * inrColor;
                C += contribS;
                T *= (1.0 - alphaS);
                prevConfidence = max(pNCR, max(pED, pET));
                lastInrStep = steps;
            }
            else if (gParams.useDiffINR != 0)
            {
                // Use the same input and network as JAX viewer: coords+FF+modalities
                // and mix colors via the ground-truth LUT mapping.
                float logits[8];
                inr_logits(pIdx, logits);
                uint n = min(8u, max(1u, gParams.inrOutDim));
                // Softmax over first n logits
                float maxv = logits[0];
                for (uint i = 1; i < n; ++i) maxv = max(maxv, logits[i]);
                float sum = 0.0;
                float probs[8];
                for (uint i = 0; i < n; ++i) { probs[i] = exp(logits[i] - maxv); sum += probs[i]; }
                sum = max(sum, 1e-6);
                for (uint i = 0; i < n; ++i) probs[i] /= sum;

                // Map to BraTS LUT colors
                float3 inrColor = float3(0.0, 0.0, 0.0);
                float pTumor = 0.0;
                if (n >= 4u)
                {
                    // 0=bg, 1=NCR/NET, 2=Edema, 3=Enhancing
                    float pNCR = probs[1];
                    float pED  = probs[2];
                    float pET  = probs[3];
                    float3 colNCR = gParams.lutColorAlpha[1].xyz;
                    float3 colED  = gParams.lutColorAlpha[2].xyz;
                    float3 colET  = gParams.lutColorAlpha[4].xyz;
                    inrColor = colNCR * pNCR + colED * pED + colET * pET;
                    pTumor = saturate(pNCR + pED + pET);
                }
                else if (n >= 5u)
                {
                    // 0=bg,1=necrosis,2=edema,3=enhancing,4=non-enhancing
                    float pNCR = probs[1] + probs[4];
                    float pED  = probs[2];
                    float pET  = probs[3];
                    float3 colNCR = gParams.lutColorAlpha[1].xyz;
                    float3 colED  = gParams.lutColorAlpha[2].xyz;
                    float3 colET  = gParams.lutColorAlpha[4].xyz;
                    inrColor = colNCR * pNCR + colED * pED + colET * pET;
                    pTumor = saturate(pNCR + pED + pET);
                }

                const float INREmissionAbsorption = 10.0;
                float a = gParams.inrAlpha * pTumor * INREmissionAbsorption;
                float alphaS = 1.0 - exp(-a * step);
                float3 contribS = alphaS * T * inrColor;
                C += contribS;
                T *= (1.0 - alphaS);
                // Update confidence and mark last eval step
                float conf = 0.0;
                if (n >= 4u)
                {
                    conf = max(probs[1], max(probs[2], probs[3]));
                }
                else if (n >= 5u)
                {
                    conf = max(max(probs[1], probs[2]), max(probs[3], probs[4]));
                }
                prevConfidence = saturate(conf);
                lastInrStep = steps;
            }
            else
            {
                // Legacy top-class overlay
                uint lbl; float p;
                sampleINRTop(pIdx, lbl, p);
                if (lbl == 3u) lbl = 4u;
                if (lbl != 0u)
                {
                    uint lutIdx = min(lbl & 0xFFu, 7u);
                    float4 ca = gParams.lutColorAlpha[lutIdx];
                    float a = saturate(gParams.inrAlpha) * saturate(p);
                    float alphaS = 1.0 - exp(-a * step);
                    float3 contribS = alphaS * T * ca.xyz;
                    C += contribS;
                    T *= (1.0 - alphaS);
                }
            }
        }

        t += step;
        steps++;
    }
    // Final composite after intensity + optional INR
    C = lerp(gParams.bgColor, C, 1.0);
    gOutput[tid.xy] = float4(C, 1.0);
}

// Precompute INR probability volume (voxel resolution). Non-differentiable pass.
RWTexture3D<float4> gINRVolumeOut; // same layout as gINRVolume

[shader("compute")]
[numthreads(8,8,4)]
void inr_precompute(uint3 tid : SV_DispatchThreadID)
{
    if (any(tid >= gParams.dims)) return;
    float3 pIdx = float3(tid);
    // Direct nearest (no filtering) for modalities
    uint dimX = gParams.dims.x;
    uint dimY = gParams.dims.y;
    uint strideY = dimX;
    uint strideZ = dimX * dimY;
    uint base = tid.x + tid.y * strideY + tid.z * strideZ;
    float4 intens4 = float4(gIntensity0[base], gIntensity1[base], gIntensity2[base], gIntensity3[base]);
    float logits[8];
    inr_logits(pIdx, logits);
    uint n = min(8u, max(1u, gParams.inrOutDim));
    float maxv = logits[0];
    for (uint i = 1; i < n; ++i) maxv = max(maxv, logits[i]);
    float sum = 0.0; float probs[8];
    for (uint i = 0; i < n; ++i) { probs[i] = exp(logits[i] - maxv); sum += probs[i]; }
    sum = max(sum, 1e-6);
    for (uint i = 0; i < n; ++i) probs[i] /= sum;
    float pNCR=0.0, pED=0.0, pET=0.0;
    if (n >= 5u)
    {
        pNCR = probs[1] + probs[4]; pED = probs[2]; pET = probs[3];
    }
    else if (n >= 4u)
    {
        pNCR = probs[1]; pED = probs[2]; pET = probs[3];
    }
    float pTumor = saturate(pNCR + pED + pET);
    gINRVolumeOut[tid] = float4(pNCR, pED, pET, pTumor);
}

// -----------------------------------------------------------------------------
// Differentiable entry (library-style, not a shader entry point): computes the
// color for a pixel while enabling gradients to flow back to volume data via
// VolumePtr.gradBuf using the custom derivative of sampleTriLinear.
// This function intentionally does not write to global outputs.
// -----------------------------------------------------------------------------

[Differentiable]
float3 renderPixelColorDiff(uint2 pixel, VolumePtr vol0, VolumePtr vol1, VolumePtr vol2, VolumePtr vol3)
{
    Ray ray = makePrimary(pixel);
    float3 bmin = gParams.volMin;
    float3 bmax = gParams.volMin + gParams.voxelSize * float3(gParams.dims);
    float3 d = ray.d;
    d = float3(
        abs(d.x) < 1e-7 ? (d.x >= 0 ? 1e-7 : -1e-7) : d.x,
        abs(d.y) < 1e-7 ? (d.y >= 0 ? 1e-7 : -1e-7) : d.y,
        abs(d.z) < 1e-7 ? (d.z >= 0 ? 1e-7 : -1e-7) : d.z);
    float3 rcpDir = 1.0 / d;
    float t0, t1;
    float3 C = gParams.bgColor;
    if (!aabbHit(ray.o, rcpDir, bmin, bmax, t0, t1))
        return C;

    t0 = max(t0, max(0.0, gParams.nearT));
    t1 = min(t1, (gParams.farT > 0.0) ? gParams.farT : t1);
    if (t1 <= t0) return C;

    float T = 1.0;
    float step = max(1e-5, gParams.stepSize);
    float t = t0;
    const int MAX_STEPS = 4096;
    int steps = 0;
    [MaxIters(4096)]
    while (t < t1 && T > 1e-3 && steps < MAX_STEPS)
    {
        float3 P = ray.o + t * ray.d;
        float3 pIdx = (P - gParams.volMin) / gParams.voxelSize;

        float v = sampleIntensityDiff(pIdx, vol0, vol1, vol2, vol3);
        if (v > 0.0)
        {
            float gmag = gradientMagForwardDiff(pIdx, v, vol0, vol1, vol2, vol3);
            float edge = 1.0 + max(0.0, gParams.gradBoost) * gmag;
            float aI = saturate(v) * edge * gParams.intensityAlpha;
            float3 colI = v.xxx;
            float alphaI = 1.0 - exp(-aI * step);
            float3 contribI = alphaI * T * colI;
            C = C + contribI;
            T *= (1.0 - alphaI);
        }

        // Segmentation overlay is non-differentiable — skip here deliberately.

        t += step;
        steps++;
    }
    return lerp(gParams.bgColor, C, 1.0);
}

#endif // ENABLE_DIFF_PATH

// (overlay compute entry removed)
