// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Segmentation volume renderer (nearest sampling), front-to-back alpha

RWTexture2D<float4> gOutput;
StructuredBuffer<uint> gLabels; // one 32-bit label per voxel (x + y*dimX + z*dimX*dimY)

struct Params
{
    uint2 imageSize; float fovY; float pad0;
    float3 eye; float pad1;
    float3 U; float pad2;
    float3 V; float pad3;
    float3 W; float pad4;
    float3 volMin; float pad5;       // world-space min corner
    float3 voxelSize; float pad6;    // world voxel size per axis
    uint3 dims; uint pad7;           // (dimX, dimY, dimZ)
    float stepSize; float pad8a; float pad8b; float pad8c;
    float3 bgColor; float pad9;
    float4 lutColorAlpha[8];         // RGBA for label indices 0..7 (alpha in .w)
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

[shader("compute")]
[numthreads(8,8,1)]
void seg_main(uint3 tid : SV_DispatchThreadID)
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
    t0 = max(t0, 0.0);
    if (t1 <= t0)
    {
        gOutput[tid.xy] = float4(C, 1.0);
        return;
    }

    float T = 1.0; // transmittance
    float step = max(1e-5, gParams.stepSize);
    float t = t0;
    const int MAX_STEPS = 4096;
    int steps = 0;
    while (t < t1 && T > 1e-3 && steps < MAX_STEPS)
    {
        float3 P = ray.o + t * ray.d;                         // world
        float3 pIdx = (P - gParams.volMin) / gParams.voxelSize; // index
        uint lbl = sampleNearestLabel(pIdx) & 0xFFu;
        uint lutIdx = min(lbl, 7u);
        float4 ca = gParams.lutColorAlpha[lutIdx];
        float a = ca.w; // opacity per step for this label
        float3 col = ca.xyz;
        float alpha = 1.0 - exp(-a * step); // convert density-like alpha to per-step transparency
        float3 contrib = alpha * T * col;
        C = C + contrib;
        T *= (1.0 - alpha);
        t += step;
        steps++;
    }
    C = lerp(gParams.bgColor, C, 1.0);
    gOutput[tid.xy] = float4(C, 1.0);
}

