// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Simple volume ray marcher using a 3D texture, modeled after
// webgpu-samples/sample/volumeRenderingTexture3D but implemented as a
// compute shader writing to a 2D render target.

RWTexture2D<float4> gOutput;

// Scene/camera params
struct Params
{
    uint2 imageSize;
    float fovY;
    float stepCount;   // number of steps (as float for easy division)
    float nearPlane;
    float farPlane;
    float3 eye; float padEye;
    float3 U;   float padU;
    float3 V;   float padV;
    float3 W;   float padW;
    uint3 volDim; uint padDim;
};
ConstantBuffer<Params> gParams;

// Volume as structured buffer of uint4 (0..255 per component). We'll sample manually.
StructuredBuffer<uint4> gVolumeU8;

// Helpers for sampling (declare at global scope in Slang/HLSL)
float sampleU8(uint3 p, uint3 d)
{
    p.x = min(max(p.x, 0), d.x - 1);
    p.y = min(max(p.y, 0), d.y - 1);
    p.z = min(max(p.z, 0), d.z - 1);
    uint idx = p.x + p.y * d.x + p.z * d.x * d.y;
    uint elem = idx >> 2; // 4 voxels per element
    uint lane = idx & 3;
    uint4 v4 = gVolumeU8[elem];
    uint v = (lane == 0) ? v4.x : (lane == 1) ? v4.y : (lane == 2) ? v4.z : v4.w;
    return float(v & 0xffu) / 255.0;
}

float sampleTrilinear(float3 uvw, uint3 d)
{
    float x = saturate(uvw.x) * (float(d.x) - 1.0);
    float y = saturate(uvw.y) * (float(d.y) - 1.0);
    float z = saturate(uvw.z) * (float(d.z) - 1.0);
    uint3 p0 = uint3((uint)floor(x), (uint)floor(y), (uint)floor(z));
    uint3 d1 = uint3(d.x - 1, d.y - 1, d.z - 1);
    uint3 p1 = min(p0 + 1, d1);
    float3 t = float3(x - p0.x, y - p0.y, z - p0.z);
    float c000 = sampleU8(p0, d);
    float c100 = sampleU8(uint3(p1.x, p0.y, p0.z), d);
    float c010 = sampleU8(uint3(p0.x, p1.y, p0.z), d);
    float c110 = sampleU8(uint3(p1.x, p1.y, p0.z), d);
    float c001 = sampleU8(uint3(p0.x, p0.y, p1.z), d);
    float c101 = sampleU8(uint3(p1.x, p0.y, p1.z), d);
    float c011 = sampleU8(uint3(p0.x, p1.y, p1.z), d);
    float c111 = sampleU8(uint3(p1.x, p1.y, p1.z), d);
    float c00 = lerp(c000, c100, t.x);
    float c01 = lerp(c001, c101, t.x);
    float c10 = lerp(c010, c110, t.x);
    float c11 = lerp(c011, c111, t.x);
    float c0 = lerp(c00, c10, t.y);
    float c1 = lerp(c01, c11, t.y);
    return lerp(c0, c1, t.z);
}

// Construct primary ray for pixel coordinates in [0,width) x [0,height)
struct Ray { float3 o; float3 d; };

Ray makePrimary(uint2 pix, float2 invDims, float fovY)
{
    // Map pixel center to NDC [-1,1]
    float2 uv = (float2(pix) + 0.5) * invDims;
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    // Build ray direction using camera basis; assume square pixels
    float tanHalfFovY = tan(0.5 * fovY);
    float aspect = 1.0 / max(1e-6, invDims.y / invDims.x); // width/height
    float3 dir = normalize(ndc.x * aspect * tanHalfFovY * gParams.U + ndc.y * tanHalfFovY * gParams.V + gParams.W);
    Ray r; r.o = gParams.eye; r.d = dir; return r;
}

// Intersect ray with axis-aligned box in [-1,1]^3
bool aabbIntersect(Ray r, out float t0, out float t1)
{
    float3 bmin = float3(-1.0, -1.0, -1.0);
    float3 bmax = float3( 1.0,  1.0,  1.0);
    float3 invD = 1.0 / max(abs(r.d), float3(1e-6,1e-6,1e-6)) * sign(r.d);
    float3 tA = (bmin - r.o) * invD;
    float3 tB = (bmax - r.o) * invD;
    float3 tsm = min(tA, tB);
    float3 tbg = max(tA, tB);
    t0 = max(max(tsm.x, tsm.y), tsm.z);
    t1 = min(min(tbg.x, tbg.y), tbg.z);
    return t1 >= max(t0, 0.0);
}

// Front-to-back compositing of a single-channel volume mapped to grayscale
float3 shade(float density)
{
    return float3(density, density, density);
}

[shader("compute")]
[numthreads(8,8,1)]
void volume_cs(uint3 tid : SV_DispatchThreadID)
{
    uint2 dims = gParams.imageSize;
    if (any(tid.xy >= dims)) return;

    float2 invDims = 1.0 / float2(dims);
    float fovY = gParams.fovY;

    // Compute per-pixel near/far world positions matching WGSL inverse-MVP approach
    float2 uv = (float2(tid.xy) + 0.5) * invDims;
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float tanHalfFovY = tan(0.5 * fovY);
    float aspect = (float)dims.x / max(1.0, (float)dims.y);
    float3 v = float3(ndc.x * aspect * tanHalfFovY, ndc.y * tanHalfFovY, 1.0);
    float n = max(0.0, gParams.nearPlane);
    float f = max(n, gParams.farPlane);
    float3 worldNear = gParams.eye + gParams.U * (v.x * n) + gParams.V * (v.y * n) + gParams.W * (v.z * n);
    float3 worldFar  = gParams.eye + gParams.U * (v.x * f) + gParams.V * (v.y * f) + gParams.W * (v.z * f);
    float3 stepVec = (worldFar - worldNear) / max(1.0, gParams.stepCount);

    // Volume dims
    uint3 dim = gParams.volDim;

    // Match WebGPU sample stepping: fixed steps between near/far, accumulate only inside cube
    float steps = max(1.0, gParams.stepCount);
    float accum = 0.0;
    float3 rayPos = worldNear;
    [loop]
    for (uint i = 0; i < (uint)steps; ++i)
    {
        bool inside = all(rayPos < 1.0.xxx) && all(rayPos > (-1.0).xxx);
        if (inside && accum < 1.0)
        {
            float3 uvw = 0.5 * (rayPos + 1.0);
            float s = sampleTrilinear(uvw, dim) * (4.0 / steps);
            accum += (1.0 - accum) * s;
        }
        rayPos += stepVec;
        if (accum > 0.995) break;
    }

    gOutput[tid.xy] = float4(accum.xxx, 1.0);
}
