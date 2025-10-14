// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Simple triangle BVH ray tracing on compute (Metal-compatible via Slang)

RWTexture2D<float4> gOutput;

// Node layout: float4 a (min.xyz, max.x), float4 b (max.yz, leftFirst, triCountOrRight)
StructuredBuffer<float4> gBVHNodes; // two float4s per node
StructuredBuffer<uint4> gTris;      // compacted triangle index list (matching leaf ordering), xyz used
StructuredBuffer<float4> gVerts;    // vertex positions, xyz used

struct Params
{
    uint2 imageSize;
    float fovY;
    uint maxBounces;
    float3 eye;
    float3 U; float padU;
    float3 V; float padV;
    float3 W; float padW;
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

bool aabbHit(float3 ro, float3 rcpDir, float3 bmin, float3 bmax, out float tmin)
{
    float3 t0 = (bmin - ro) * rcpDir;
    float3 t1 = (bmax - ro) * rcpDir;
    float3 tsm = min(t0, t1);
    float3 tbg = max(t0, t1);
    float tN = max(max(tsm.x, tsm.y), tsm.z);
    float tF = min(min(tbg.x, tbg.y), tbg.z);
    tmin = tN;
    return tF >= max(tN, 0.0);
}

bool triHit(Ray r, float3 a, float3 b, float3 c, out float t, out float3 n)
{
    t = 0.0; n = float3(0,0,0);
    float3 ab = b - a;
    float3 ac = c - a;
    float3 p = cross(r.d, ac);
    float det = dot(ab, p);
    if (abs(det) < 1e-8) return false;
    float invDet = 1.0 / det;
    float3 s = r.o - a;
    float u = dot(s, p) * invDet;
    if (u < 0.0 || u > 1.0) return false;
    float3 q = cross(s, ab);
    float v = dot(r.d, q) * invDet;
    if (v < 0.0 || u + v > 1.0) return false;
    float th = dot(ac, q) * invDet;
    if (th <= 1e-5) return false;
    t = th;
    n = normalize(cross(ab, ac));
    return true;
}

struct Hit { float t; float3 n; uint triIndex; };

bool bvhTrace(Ray ray, out Hit hit)
{
    hit.t = 1e30; hit.n = float3(0,0,0); hit.triIndex = 0xffffffff;
    float3 dd = ray.d;
    if (abs(dd.x) < 1e-8) dd.x = (dd.x >= 0.0) ? 1e-8 : -1e-8;
    if (abs(dd.y) < 1e-8) dd.y = (dd.y >= 0.0) ? 1e-8 : -1e-8;
    if (abs(dd.z) < 1e-8) dd.z = (dd.z >= 0.0) ? 1e-8 : -1e-8;
    float3 rcpDir = 1.0 / dd;

    uint stack[64]; uint sp = 0; stack[sp++] = 0; // root at 0

    while (sp > 0)
    {
        uint ni = stack[--sp];
        float4 a = gBVHNodes[ni*2+0];
        float4 b = gBVHNodes[ni*2+1];
        float3 bmin = a.xyz;
        float3 bmax = float3(a.w, b.x, b.y);
        float tmin;
        if (!aabbHit(ray.o, rcpDir, bmin, bmax, tmin) || tmin > hit.t)
            continue;
        int leftFirst = int(b.z + 0.5);
        int triCountOrRight = int(b.w + (b.w >= 0 ? 0.5 : -0.5));
        if (triCountOrRight > 0)
        {
            // leaf: leftFirst=start, triCount=triCountOrRight (encoded as positive)
            uint start = uint(leftFirst);
            uint count = uint(triCountOrRight);
            [unroll] for (uint i = 0; i < count; ++i)
            {
                uint ti = start + i;
                uint3 idx = gTris[ti].xyz;
                float t; float3 n;
                if (triHit(ray, gVerts[idx.x].xyz, gVerts[idx.y].xyz, gVerts[idx.z].xyz, t, n))
                {
                    if (t < hit.t) { hit.t = t; hit.n = n; hit.triIndex = ti; }
                }
            }
        }
        else
        {
            // inner: leftFirst=left, triCountOrRight=-(right+1)
            uint l = uint(leftFirst);
            // triCountOrRight is negative in inner nodes
            uint r = uint(-triCountOrRight - 1);
            // Near-first traversal: compute child AABB tmin
            float4 al = gBVHNodes[l*2+0]; float4 bl = gBVHNodes[l*2+1];
            float4 ar = gBVHNodes[r*2+0]; float4 br = gBVHNodes[r*2+1];
            float tl, tr;
            bool hl = aabbHit(ray.o, rcpDir, al.xyz, float3(al.w, bl.x, bl.y), tl);
            bool hr = aabbHit(ray.o, rcpDir, ar.xyz, float3(ar.w, br.x, br.y), tr);
            if (hl && hr)
            {
                if (tl < tr) { stack[sp++] = r; stack[sp++] = l; }
                else         { stack[sp++] = l; stack[sp++] = r; }
            }
            else if (hl) { stack[sp++] = l; }
            else if (hr) { stack[sp++] = r; }
        }
    }
    return hit.t < 1e29;
}

[shader("compute")]
[numthreads(8,8,1)]
void compute_main(uint3 tid : SV_DispatchThreadID)
{
    uint2 dims = gParams.imageSize;
    if (any(tid.xy >= dims)) return;
    Ray ray = makePrimary(tid.xy);
    Hit h;
    float3 color;
    if (bvhTrace(ray, h))
    {
        float3 p = ray.o + h.t * ray.d;
        float3 n = normalize(h.n);
        // Two-sided normal so we don't lose backfaces
        if (dot(n, ray.d) > 0.0) n = -n;
        float3 l = normalize(float3(0.3, 0.8, 0.5));
        float ndotl = max(0.0, dot(n, l));
        float ao = 0.3 + 0.7 * saturate(1.0 - 0.05 * h.t);
        color = (0.15 + ndotl) * ao * float3(0.8, 0.7, 0.6);
    }
    else
    {
        float tbg = 0.5 * (normalize(ray.d).y + 1.0);
        color = lerp(float3(0.05,0.06,0.08), float3(0.2,0.25,0.3), tbg);
    }
    gOutput[tid.xy] = float4(color, 1.0);
}
