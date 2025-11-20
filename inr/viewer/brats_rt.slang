RWTexture2D<float4> gOutput;

// Data
StructuredBuffer<float> gIntensity0;
StructuredBuffer<float> gIntensity1;
StructuredBuffer<float> gIntensity2;
StructuredBuffer<float> gIntensity3;

StructuredBuffer<uint> gLabels; // Ground Truth
StructuredBuffer<uint> gPreds;  // INR Prediction

struct Params
{
    uint2 imageSize; float fovY; float pad0;
    float3 eye; float pad1;
    float3 U; float pad2; float3 V; float pad3; float3 W; float pad4;
    float3 volMin; float pad5; float3 voxelSize; float pad6; uint3 dims; uint pad7;
    float stepSize; float nearT; float farT; float pad8;
    float3 bgColor; float pad9;
    
    uint4 volEnabled;
    float4 volWeight;
    float ww; float wl; float intensityAlpha; float padInt;
    float gamma; float gradBoost; float gradScale; float padTone;
    
    uint showSeg; 
    uint showPred; 
    uint2 padFlags;
    
    float4 lutColorAlpha[8];
};
ConstantBuffer<Params> gParams;

// Ray Logic
struct Ray { float3 o; float3 d; };
Ray makePrimary(uint2 pixel)
{
    float2 dim = float2(gParams.imageSize);
    float2 ndc = (float2(pixel) + 0.5) / dim;
    float2 uv = ndc * 2.0 - 1.0;
    float f = 1.0 / tan(0.5 * gParams.fovY);
    float aspect = dim.x / max(1.0, dim.y);
    float3 rdCam = normalize(float3(uv.x * aspect / f, -uv.y / f, 1.0));
    float3 rd = normalize(rdCam.x * gParams.U + rdCam.y * gParams.V + rdCam.z * gParams.W);
    return { gParams.eye, rd };
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

// Sampling
float sampleLinear(StructuredBuffer<float> buf, float3 pIdx)
{
    float3 q = clamp(pIdx, 0.0, float3(gParams.dims) - 1.001);
    uint3 i = uint3(floor(q));
    float3 f = q - float3(i);
    uint dimX = gParams.dims.x; uint dimY = gParams.dims.y;
    uint strideY = dimX; uint strideZ = dimX * dimY;
    uint b = i.x + i.y * strideY + i.z * strideZ;
    
    float c000 = buf[b]; float c100 = buf[b+1];
    float c010 = buf[b+strideY]; float c110 = buf[b+strideY+1];
    float c001 = buf[b+strideZ]; float c101 = buf[b+strideZ+1];
    float c011 = buf[b+strideZ+strideY]; float c111 = buf[b+strideZ+strideY+1];
    
    return lerp(lerp(lerp(c000,c100,f.x), lerp(c010,c110,f.x), f.y),
                lerp(lerp(c001,c101,f.x), lerp(c011,c111,f.x), f.y), f.z);
}

uint sampleLabel(StructuredBuffer<uint> buf, float3 pIdx)
{
    uint3 i = uint3(round(clamp(pIdx, 0.0, float3(gParams.dims) - 1.0)));
    uint idx = i.x + i.y * gParams.dims.x + i.z * gParams.dims.x * gParams.dims.y;
    return buf[idx];
}

[shader("compute")]
[numthreads(8,8,1)]
void brats_main(uint3 tid : SV_DispatchThreadID)
{
    if (any(tid.xy >= gParams.imageSize)) return;
    
    Ray r = makePrimary(tid.xy);
    float3 bmin = gParams.volMin;
    float3 bmax = gParams.volMin + gParams.voxelSize * float3(gParams.dims);
    
    float3 d = r.d;
    if (abs(d.x) < 1e-6) d.x = 1e-6;
    if (abs(d.y) < 1e-6) d.y = 1e-6;
    if (abs(d.z) < 1e-6) d.z = 1e-6;
    float3 rcpDir = 1.0 / d;
    
    float t0, t1;
    if (!aabbHit(r.o, rcpDir, bmin, bmax, t0, t1)) {
        gOutput[tid.xy] = float4(gParams.bgColor, 1.0);
        return;
    }
    
    t0 = max(t0, max(0.0, gParams.nearT));
    t1 = min(t1, (gParams.farT > 0.0) ? gParams.farT : t1);
    if (t1 <= t0) { gOutput[tid.xy] = float4(gParams.bgColor, 1.0); return; }
    
    float3 C = gParams.bgColor;
    float T = 1.0;
    float t = t0;
    
    // Raymarch
    [MaxIters(1024)]
    while (t < t1 && T > 0.01)
    {
        float3 p = r.o + t * r.d;
        float3 pIdx = (p - bmin) / gParams.voxelSize;
        
        // 1. Accumulate Intensity
        float v = 0.0; 
        float wSum = 0.0;
        if (gParams.volEnabled.x != 0) { v += sampleLinear(gIntensity0, pIdx) * gParams.volWeight.x; wSum += gParams.volWeight.x; }
        if (gParams.volEnabled.y != 0) { v += sampleLinear(gIntensity1, pIdx) * gParams.volWeight.y; wSum += gParams.volWeight.y; }
        if (gParams.volEnabled.z != 0) { v += sampleLinear(gIntensity2, pIdx) * gParams.volWeight.z; wSum += gParams.volWeight.z; }
        if (gParams.volEnabled.w != 0) { v += sampleLinear(gIntensity3, pIdx) * gParams.volWeight.w; wSum += gParams.volWeight.w; }
        
        if (wSum > 0.0) v /= wSum;
        
        float val = saturate((v - (gParams.wl - gParams.ww*0.5)) / gParams.ww);
        val = pow(val, gParams.gamma);
        
        if (val > 0.0) {
            float a = val * gParams.intensityAlpha;
            float alpha = 1.0 - exp(-a * gParams.stepSize);
            C += alpha * T * val; 
            T *= (1.0 - alpha);
        }
        
        // 2. Accumulate GT Seg
        if (gParams.showSeg != 0) {
            uint l = sampleLabel(gLabels, pIdx);
            if (l > 0 && l < 8) {
                float4 col = gParams.lutColorAlpha[l];
                float alpha = 1.0 - exp(-col.w * gParams.stepSize);
                C += alpha * T * col.xyz;
                T *= (1.0 - alpha);
            }
        }
        
        // 3. Accumulate Prediction
        if (gParams.showPred != 0) {
            uint l = sampleLabel(gPreds, pIdx);
            if (l > 0 && l < 8) {
                float4 col = gParams.lutColorAlpha[l];
                float alpha = 1.0 - exp(-col.w * gParams.stepSize * 1.5); 
                C += alpha * T * col.xyz; 
                T *= (1.0 - alpha);
            }
        }
        
        t += gParams.stepSize;
    }
    
    gOutput[tid.xy] = float4(C, 1.0);
}