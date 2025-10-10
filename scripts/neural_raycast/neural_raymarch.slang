// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Minimal analytic SDF ray marcher (Metal-friendly)

RWTexture2D<float4> render_texture;

struct Params
{
    uint2 imageSize;    // width, height (unused for rays; we query texture size)
    float fovY;         // radians
    uint maxSteps;      // sphere tracing steps
    float maxDistance;  // far plane / max march distance
    float hitThreshold; // SDF convergence threshold
    float normalEps;    // epsilon for finite-diff normals
};

ConstantBuffer<Params> gParams;
float3 gEye;
float3 gU;
float3 gV;
float3 gW;

// Analytic SDFs
float sdSphere(float3 p, float r) { return length(p) - r; }
float sdPlane(float3 p, float3 n, float h) { return dot(p, normalize(n)) + h; }

// Scene distance: single sphere at origin (radius 0.6)
float sceneSDF(float3 p)
{
    return sdSphere(p, 0.6);
}

float3 estimateNormal(float3 p)
{
    float e = gParams.normalEps;
    float d1 = sceneSDF(p + e * float3( 1, -1, -1));
    float d2 = sceneSDF(p + e * float3(-1, -1,  1));
    float d3 = sceneSDF(p + e * float3(-1,  1, -1));
    float d4 = sceneSDF(p + e * float3( 1,  1,  1));
    return normalize(float3(d1 - d2 - d3 + d4, -d1 - d2 + d3 + d4, -d1 + d2 - d3 + d4));
}

struct Ray { float3 o; float3 d; };

Ray makePrimary(uint2 pixel, uint2 dims)
{
    float2 dim = float2(dims);
    float2 ndc = (float2(pixel) + 0.5) / dim; // [0,1]
    float2 uv = ndc * 2.0 - 1.0;              // [-1,1]

    float f = 1.0 / tan(0.5 * gParams.fovY);
    float aspect = dim.x / dim.y;
    // Camera looks along +Z in camera space; use +1 here and feed world gW accordingly
    float3 rdCam = normalize(float3(uv.x * aspect / f, -uv.y / f, 1.0));
    float3 ro = gEye;
    float3 rd = normalize(rdCam.x * gU + rdCam.y * gV + rdCam.z * gW);
    Ray r; r.o = ro; r.d = rd; return r;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void neural_raymarch_cs(uint3 tid : SV_DispatchThreadID)
{
    uint2 dims; render_texture.GetDimensions(dims.x, dims.y);
    if (any(tid.xy >= dims)) return;

    Ray ray = makePrimary(tid.xy, dims);
    float t = 0.0;
    float3 p = ray.o;
    bool hit = false;

    [loop]
    for (uint i = 0; i < gParams.maxSteps; ++i)
    {
        p = ray.o + t * ray.d;
        float d = sceneSDF(p);
        if (d < gParams.hitThreshold) { hit = true; break; }
        t += clamp(d, 0.01, 0.25);
        if (t > gParams.maxDistance) break;
    }

    float3 color;
    if (hit)
    {
        // UV coloring on sphere using spherical coordinates
        float3 n = normalize(p); // sphere normal
        float u = atan2(n.z, n.x) / (2.0 * 3.14159265) + 0.5;
        float v = n.y * 0.5 + 0.5;
        color = float3(u, v, 1.0 - u);
    }
    else
    {
        // Simple background
        float tbg = 0.5 * (normalize(ray.d).y + 1.0);
        color = lerp(float3(0.05,0.06,0.08), float3(0.2,0.25,0.3), tbg);
    }

    render_texture[tid.xy] = float4(color, 1.0);
}
