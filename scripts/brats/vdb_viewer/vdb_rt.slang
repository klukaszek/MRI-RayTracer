// Simple volume ray marcher for density volumes in a linear buffer

RWTexture2D<float4> gOutput;
StructuredBuffer<float> gVolume; // linearized as x + y*dimX + z*dimX*dimY (x-fastest)

struct Params
{
    uint2 imageSize; float fovY; float pad0;
    float3 eye; float pad1;
    float3 U; float pad2;
    float3 V; float pad3;
    float3 W; float pad4;
    float3 volMin; float pad5;  // world-space min corner of volume AABB
    float3 voxelSize; float pad6; // size of a voxel in world units (xyz)
    uint3 dims; uint pad7;      // (dimX, dimY, dimZ)
    float stepSize; float sigmaScale; float pad8a; float pad8b;
    float3 albedo; float pad9;
    float3 bgColor; float pad10;
};
ConstantBuffer<Params> gParams;

struct Ray { float3 o; float3 d; };

Ray makePrimary(uint2 pixel)
{
    float2 dim = float2(gParams.imageSize);
    float2 ndc = (float2(pixel) + 0.5) / dim;
    float2 uv = ndc * 2.0 - 1.0;
    float f = 1.0 / tan(0.5 * gParams.fovY);
    float aspect = dim.x / dim.y;
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

float sampleVolumeLinear(float3 pIdx)
{
    // pIdx in index space [0, dims)
    float3 q = clamp(pIdx, float3(0.0,0.0,0.0), float3(gParams.dims) - float3(1.001, 1.001, 1.001));
    uint3 i0 = uint3(floor(q));
    float3 f = q - float3(i0);
    uint dimX = gParams.dims.x;
    uint dimY = gParams.dims.y;
    uint dimZ = gParams.dims.z;
    uint strideX = 1;
    uint strideY = dimX;
    uint strideZ = dimX * dimY;
    uint base = i0.x + i0.y * strideY + i0.z * strideZ;
    float c000 = gVolume[base];
    float c100 = gVolume[base + strideX];
    float c010 = gVolume[base + strideY];
    float c110 = gVolume[base + strideY + strideX];
    float c001 = gVolume[base + strideZ];
    float c101 = gVolume[base + strideZ + strideX];
    float c011 = gVolume[base + strideZ + strideY];
    float c111 = gVolume[base + strideZ + strideY + strideX];
    float c00 = lerp(c000, c100, f.x);
    float c10 = lerp(c010, c110, f.x);
    float c01 = lerp(c001, c101, f.x);
    float c11 = lerp(c011, c111, f.x);
    float c0 = lerp(c00, c10, f.y);
    float c1 = lerp(c01, c11, f.y);
    return lerp(c0, c1, f.z);
}

[shader("compute")]
[numthreads(8,8,1)]
void volume_main(uint3 tid : SV_DispatchThreadID)
{
    uint2 dims2D = gParams.imageSize;
    if (any(tid.xy >= dims2D)) return;
    Ray ray = makePrimary(tid.xy);

    // Intersect with volume AABB in world space
    float3 bmin = gParams.volMin;
    float3 bmax = gParams.volMin + gParams.voxelSize * float3(gParams.dims);
    float3 dd = ray.d;
    dd = float3(
        abs(dd.x) < 1e-7 ? (dd.x >= 0 ? 1e-7 : -1e-7) : dd.x,
        abs(dd.y) < 1e-7 ? (dd.y >= 0 ? 1e-7 : -1e-7) : dd.y,
        abs(dd.z) < 1e-7 ? (dd.z >= 0 ? 1e-7 : -1e-7) : dd.z);
    float3 rcpDir = 1.0 / dd;
    float t0, t1;
    float3 color = gParams.bgColor;
    if (!aabbHit(ray.o, rcpDir, bmin, bmax, t0, t1))
    {
        gOutput[tid.xy] = float4(color, 1.0);
        return;
    }
    t0 = max(t0, 0.0);
    if (t1 <= t0)
    {
        gOutput[tid.xy] = float4(color, 1.0);
        return;
    }

    // Front-to-back ray marching
    float T = 1.0;          // transmittance
    float3 C = color;       // start at background
    float step = max(1e-4, gParams.stepSize);
    float t = t0;
    const int MAX_STEPS = 4096;
    int steps = 0;
    while (t < t1 && T > 1e-3 && steps < MAX_STEPS)
    {
        float3 P = ray.o + t * ray.d;               // world space
        float3 pIdx = (P - gParams.volMin) / gParams.voxelSize; // index space
        float sigma = max(0.0, sampleVolumeLinear(pIdx)) * gParams.sigmaScale;
        float alpha = 1.0 - exp(-sigma * step);     // absorption-only
        float3 contrib = alpha * T * gParams.albedo; // simple white scattering
        C = lerp(color, C + contrib, 1.0);          // composite over bg
        T *= (1.0 - alpha);
        t += step;
        steps++;
    }
    gOutput[tid.xy] = float4(C, 1.0);
}

