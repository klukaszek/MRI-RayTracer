// Combined BraTS viewer shader: multi-intensity volumes + segmentation overlay

RWTexture2D<float4> gOutput;

// Up to 4 intensity volumes (float) in the same grid
StructuredBuffer<float> gIntensity0;
StructuredBuffer<float> gIntensity1;
StructuredBuffer<float> gIntensity2;
StructuredBuffer<float> gIntensity3;

// Optional segmentation labels (uint)
StructuredBuffer<uint> gLabels;

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
    return saturate((v - low) / max(1e-6, (hi - low)));
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
    while (t < t1 && T > 1e-3 && steps < MAX_STEPS)
    {
        float3 P = ray.o + t * ray.d;
        float3 pIdx = (P - gParams.volMin) / gParams.voxelSize;

        // Intensity base
        float v = sampleIntensity(pIdx);
        if (v > 0.0)
        {
            float aI = saturate(v) * gParams.intensityAlpha;
            float3 colI = v.xxx;
            float alphaI = 1.0 - exp(-aI * step);
            float3 contribI = alphaI * T * colI;
            C = C + contribI;
            T *= (1.0 - alphaI);
        }

        // Segmentation overlay
        if (gParams.showSeg != 0 && T > 1e-3)
        {
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
