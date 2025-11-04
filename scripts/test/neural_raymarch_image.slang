/*
 * Neural Raymarching Image Renderer (Slang)
 * - Fused multi-modal features (T1/T2/FLAIR), 32 channels each
 * - Tiny MLP for density + segmentation logits (1 + 3 logits; background implicit)
 * - Writes colorized segmentation to RWTexture2D
 */

// ============================================================================
// Structures & Globals
// ============================================================================

struct Ray {
    float3 origin;
    float3 direction;
    float tMin;
    float tMax;
};

struct VolumeRenderResult {
    float3 color;
    float4 segmentation;
    float density;
    float numSamples;
    float importanceMap;
    float transmittance;
};

// Feature grids as arrays of Texture3D<float4>
// 8 slices × RGBA = 32 channels per modality
Texture3D<float4> gFeatureGridT1[8];
Texture3D<float4> gFeatureGridT2[8];
Texture3D<float4> gFeatureGridFLAIR[8];
SamplerState gLinearSampler;

// Flattened MLP weights (PyTorch order: W[row-major], b per layer)
StructuredBuffer<float> gMLPWeights;
int gMLPInputDim;   // expected 96 + 39 = 135 if PE enabled
int gMLPHiddenDim;  // e.g., 64
int gMLPOutputDim;  // 1 + 4 = 5 (density + 4 class logits)

// Render params
float gBaseStepSize; // e.g., 0.02
int gMaxSteps;       // e.g., 128

// Image IO
uint gImageWidth;
uint gImageHeight;
StructuredBuffer<Ray> gRays;
RWTexture2D<float4> gOutImage;

// ============================================================================
// Utilities
// ============================================================================

float relu(float x) { return max(0.0, x); }
float softplus(float x) { return log(1.0 + exp(x)); }

float4 softmax(float4 logits)
{
    float m = max(max(logits.x, logits.y), max(logits.z, logits.w));
    float4 e = exp(logits - m);
    float s = e.x + e.y + e.z + e.w;
    return e / max(s, 1e-8);
}

void positionalEncode(float3 p, out float pe[39])
{
    // Normalize to [-1,1]
    float3 x = saturate((p + 1.0) * 0.5) * 2.0 - 1.0;
    int idx = 0;
    pe[idx++] = x.x; pe[idx++] = x.y; pe[idx++] = x.z;
    float freqs[6] = {1.0, 2.0, 4.0, 8.0, 16.0, 32.0};
    [unroll]
    for (int k = 0; k < 6; ++k)
    {
        float f = freqs[k] * 3.1415926535;
        pe[idx++] = sin(f * x.x); pe[idx++] = cos(f * x.x);
        pe[idx++] = sin(f * x.y); pe[idx++] = cos(f * x.y);
        pe[idx++] = sin(f * x.z); pe[idx++] = cos(f * x.z);
    }
}

float3 getSegmentationColor(float4 p)
{
    // 0: Background (black)
    // 1: Edema (green)
    // 2: Core  (red)
    // 3: Enhancing (yellow)
    const float3 colors[4] = {
        float3(0.0, 0.0, 0.0),
        float3(0.0, 1.0, 0.0),
        float3(1.0, 0.0, 0.0),
        float3(1.0, 1.0, 0.0)
    };
    float3 c = float3(0,0,0);
    [unroll]
    for (int i = 0; i < 4; ++i) c += p[i] * colors[i];
    return c;
}

// ============================================================================
// Feature Sampling (3 × 32 → 96 fused)
// ============================================================================

void sampleModality32(Texture3D<float4> gridArr[8], SamplerState samp, float3 uvw, out float outFeat[32])
{
    [unroll]
    for (int s = 0; s < 8; ++s)
    {
        float4 v = gridArr[s].SampleLevel(samp, uvw, 0);
        outFeat[s*4 + 0] = v.x;
        outFeat[s*4 + 1] = v.y;
        outFeat[s*4 + 2] = v.z;
        outFeat[s*4 + 3] = v.w;
    }
}

void fuseFeatures96(float3 uvw, out float fused[96])
{
    float t1[32]; float t2[32]; float fl[32];
    sampleModality32(gFeatureGridT1, gLinearSampler, uvw, t1);
    sampleModality32(gFeatureGridT2, gLinearSampler, uvw, t2);
    sampleModality32(gFeatureGridFLAIR, gLinearSampler, uvw, fl);
    [unroll] for (int i = 0; i < 32; ++i) { fused[i] = t1[i]; fused[32+i] = t2[i]; fused[64+i] = fl[i]; }
}

// ============================================================================
// Tiny MLP (2 hidden layers) using flattened weights
// ============================================================================

void evaluateTinyMLP(
    StructuredBuffer<float> weights,
    int inputDim,
    int hiddenDim,
    int outputDim,
    float inputVec[],
    out float density,
    out float4 classLogits
)
{
    // Layer 1 offsets
    int w1Size = hiddenDim * inputDim;
    int b1Size = hiddenDim;
    int w1Start = 0;
    int b1Start = w1Start + w1Size;

    // Layer 2 offsets
    int w2Size = hiddenDim * hiddenDim;
    int b2Size = hiddenDim;
    int w2Start = b1Start + b1Size;
    int b2Start = w2Start + w2Size;

    // Output layer offsets
    int w3Size = outputDim * hiddenDim;
    int b3Size = outputDim;
    int w3Start = b2Start + b2Size;
    int b3Start = w3Start + w3Size;

    float h1[128];
    [unroll]
    for (int h = 0; h < hiddenDim; ++h)
    {
        float sum = 0.0;
        int rowStart = w1Start + h * inputDim;
        [loop]
        for (int i = 0; i < inputDim; ++i)
        {
            sum += weights[rowStart + i] * inputVec[i];
        }
        float b = weights[b1Start + h];
        h1[h] = relu(sum + b);
    }

    float h2[128];
    [unroll]
    for (int h = 0; h < hiddenDim; ++h)
    {
        float sum = 0.0;
        int rowStart = w2Start + h * hiddenDim;
        [loop]
        for (int i = 0; i < hiddenDim; ++i)
        {
            sum += weights[rowStart + i] * h1[i];
        }
        float b = weights[b2Start + h];
        h2[h] = relu(sum + b);
    }

    float outv[8];
    [unroll]
    for (int o = 0; o < outputDim; ++o)
    {
        float sum = 0.0;
        int rowStart = w3Start + o * hiddenDim;
        [loop]
        for (int i = 0; i < hiddenDim; ++i)
        {
            sum += weights[rowStart + i] * h2[i];
        }
        float b = weights[b3Start + o];
        outv[o] = sum + b;
    }

    density = softplus(outv[0]);
    classLogits = float4(outv[1], outv[2], outv[3], outv[4]);
}

// ============================================================================
// Raymarcher
// ============================================================================

VolumeRenderResult marchRay(Ray ray)
{
    VolumeRenderResult r;
    r.color = float3(0,0,0);
    r.segmentation = float4(0,0,0,0);
    r.density = 0.0;
    r.numSamples = 0.0;
    r.importanceMap = 0.0;
    r.transmittance = 1.0;

    float t = ray.tMin;
    float T = 1.0;

    [loop]
    for (int step = 0; step < gMaxSteps; ++step)
    {
        if (T < 0.01 || t >= ray.tMax) break;

        float3 pos = ray.origin + t * ray.direction;
        float3 uvw = (pos + 1.0) * 0.5; // [-1,1] → [0,1]
        if (any(uvw < 0.0) || any(uvw > 1.0)) { t += gBaseStepSize; continue; }

        // Features + PE
        float fused[96]; fuseFeatures96(uvw, fused);
        float pe[39]; positionalEncode(pos, pe);
        float inputVec[180];
        [unroll] for (int i = 0; i < 96; ++i) inputVec[i] = fused[i];
        [unroll] for (int i = 0; i < 39; ++i) inputVec[96+i] = pe[i];

    float density; float4 logits;
    evaluateTinyMLP(gMLPWeights, gMLPInputDim, gMLPHiddenDim, gMLPOutputDim, inputVec, density, logits);

    float4 probs = softmax(logits);

        float alpha = 1.0 - exp(-density * gBaseStepSize);
        float w = T * alpha;
        r.color += w * getSegmentationColor(probs);
        r.segmentation += w * probs;
        r.density += w * density;
        T *= (1.0 - alpha);

        r.importanceMap += saturate(density);
        t += gBaseStepSize;
        r.numSamples += 1.0;
    }

    r.importanceMap /= max(r.numSamples, 1.0);
    r.transmittance = T;
    return r;
}

// ============================================================================
// Entry Point
// ============================================================================

[shader("compute")]
[numthreads(8,8,1)]
void renderVolumeImage(uint3 tid : SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    if (pixel.x >= gImageWidth || pixel.y >= gImageHeight) return;
    uint idx = pixel.x + pixel.y * gImageWidth;
    Ray ray = gRays[idx];
    VolumeRenderResult r = marchRay(ray);
    float3 c = getSegmentationColor(r.segmentation);
    gOutImage[pixel] = float4(c, 1.0);
}
