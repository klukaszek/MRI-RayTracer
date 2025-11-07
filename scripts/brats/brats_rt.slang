// Combined BraTS viewer shader: multi-intensity volumes + segmentation overlay

RWTexture2D<float4> gOutput;

// Up to 4 intensity volumes (float) in the same grid
StructuredBuffer<float> gIntensity0;
StructuredBuffer<float> gIntensity1;
StructuredBuffer<float> gIntensity2;
StructuredBuffer<float> gIntensity3;

// Optional segmentation labels (uint)
StructuredBuffer<uint> gLabels;

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
};
ConstantBuffer<Params> gParams;

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
    const int MAX_STEPS = 4096;
    int steps = 0;
    [MaxIters(4096)]
    while (t < t1 && T > 1e-3 && steps < MAX_STEPS)
    {
        float3 P = ray.o + t * ray.d;
        float3 pIdx = (P - gParams.volMin) / gParams.voxelSize;

        // Intensity base
        float v = sampleIntensity(pIdx);
        if (v > 0.0)
        {
            // Edge-enhanced opacity using gradient magnitude
            float gmag = gradientMagForward(pIdx, v);
            float edge = 1.0 + max(0.0, gParams.gradBoost) * gmag;
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

        t += step;
        steps++;
    }
    C = lerp(gParams.bgColor, C, 1.0);
    gOutput[tid.xy] = float4(C, 1.0);
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
