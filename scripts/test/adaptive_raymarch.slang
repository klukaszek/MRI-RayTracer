/*
 * Adaptive Neural Volume Rendering Shader
 * For Glioma Segmentation with T1, T2, FLAIR
 * 
 * Features:
 * - Differentiable raymarching (SlangPy AutoDiff)
 * - Adaptive sampling with learned step sizes
 * - Lightweight MLP inference
 * - Multi-class segmentation
 */

// ============================================================================
// Structures
// ============================================================================

struct Ray {
    float3 origin;
    float3 direction;
    float tMin;
    float tMax;
};

struct VolumeRenderResult {
    float3 color;
    float4 segmentation;      // Class probabilities [background, edema, core, enhancing]
    float density;
    float numSamples;
    float importanceMap;
    float transmittance;
};

struct MLPWeights {
    StructuredBuffer<float> data;
    int inputDim;
    int hiddenDim;
    int outputDim;
};

// ============================================================================
// Utility Functions
// ============================================================================

float relu(float x) {
    return max(0.0, x);
}

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float softplus(float x) {
    return log(1.0 + exp(x));
}

float4 softmax(float4 logits) {
    float4 expVals = exp(logits - max(max(logits.x, logits.y), max(logits.z, logits.w)));
    float sum = expVals.x + expVals.y + expVals.z + expVals.w;
    return expVals / sum;
}

// ============================================================================
// MLP Evaluation (Tiny 2-Layer Network)
// ============================================================================

float4 evaluateDensityMLP(
    float3 features,
    StructuredBuffer<float> weights,
    int inputDim,    // 32
    int hiddenDim,   // 64
    int outputDim    // 5 (density + 4 classes)
) {
    // Layer 1: input -> hidden
    float hidden[64];
    int weightIdx = 0;
    
    for (int h = 0; h < hiddenDim; h++) {
        float sum = 0.0;
        for (int i = 0; i < inputDim; i++) {
            sum += features[i] * weights[weightIdx++];
        }
        sum += weights[weightIdx++];  // bias
        hidden[h] = relu(sum);
    }
    
    // Layer 2: hidden -> output
    float4 output = float4(0, 0, 0, 0);
    for (int o = 0; o < 4; o++) {  // Only compute first 4 outputs for now
        float sum = 0.0;
        for (int h = 0; h < hiddenDim; h++) {
            sum += hidden[h] * weights[weightIdx++];
        }
        sum += weights[weightIdx++];  // bias
        output[o] = sum;
    }
    
    // Fifth output (if needed)
    float density = 0.0;
    for (int h = 0; h < hiddenDim; h++) {
        density += hidden[h] * weights[weightIdx++];
    }
    density += weights[weightIdx++];  // bias
    
    return float4(density, output.x, output.y, output.z);
}

float2 evaluateSamplingMLP(
    float3 features,
    float3 rayDir,
    StructuredBuffer<float> weights,
    int inputDim,    // 35 (32 features + 3 ray dir)
    int hiddenDim,   // 32
    int outputDim    // 2 (step_multiplier, importance)
) {
    // Concatenate input
    float input[35];
    for (int i = 0; i < 32; i++) {
        input[i] = features[i];
    }
    input[32] = rayDir.x;
    input[33] = rayDir.y;
    input[34] = rayDir.z;
    
    // Layer 1: input -> hidden
    float hidden[32];
    int weightIdx = 0;
    
    for (int h = 0; h < hiddenDim; h++) {
        float sum = 0.0;
        for (int i = 0; i < inputDim; i++) {
            sum += input[i] * weights[weightIdx++];
        }
        sum += weights[weightIdx++];  // bias
        hidden[h] = relu(sum);
    }
    
    // Layer 2: hidden -> output
    float2 output;
    for (int o = 0; o < 2; o++) {
        float sum = 0.0;
        for (int h = 0; h < hiddenDim; h++) {
            sum += hidden[h] * weights[weightIdx++];
        }
        sum += weights[weightIdx++];  // bias
        output[o] = sigmoid(sum);
    }
    
    return output;
}

// ============================================================================
// Segmentation Color Mapping
// ============================================================================

float3 getSegmentationColor(float4 classProbabilities) {
    // Class colors:
    // 0: Background (black)
    // 1: Edema (green)
    // 2: Non-enhancing core (red)
    // 3: Enhancing tumor (yellow)
    
    const float3 colors[4] = {
        float3(0.0, 0.0, 0.0),    // Background
        float3(0.0, 1.0, 0.0),    // Edema
        float3(1.0, 0.0, 0.0),    // Core
        float3(1.0, 1.0, 0.0)     // Enhancing
    };
    
    float3 finalColor = float3(0, 0, 0);
    for (int i = 0; i < 4; i++) {
        finalColor += classProbabilities[i] * colors[i];
    }
    
    return finalColor;
}

// ============================================================================
// Adaptive Volume Raymarching (Main Function)
// ============================================================================

VolumeRenderResult adaptiveVolumeMarch(
    Ray ray,
    Texture3D<float4> featureGrid,
    SamplerState linearSampler,
    StructuredBuffer<float> densityMLPWeights,
    StructuredBuffer<float> samplingMLPWeights,
    float baseStepSize,
    int maxSteps
) {
    VolumeRenderResult result;
    result.color = float3(0, 0, 0);
    result.segmentation = float4(0, 0, 0, 0);
    result.density = 0.0;
    result.numSamples = 0.0;
    result.importanceMap = 0.0;
    result.transmittance = 1.0;
    
    float t = ray.tMin;
    float transmittance = 1.0;
    
    // Adaptive raymarching loop
    for (int step = 0; step < maxSteps; step++) {
        // Early termination
        if (transmittance < 0.01 || t >= ray.tMax) {
            break;
        }
        
        // Current sample position
        float3 pos = ray.origin + t * ray.direction;
        
        // Normalize position to [0, 1] for texture sampling
        float3 texCoord = (pos + 1.0) * 0.5;  // Assuming volume in [-1, 1]
        
        // Check bounds
        if (any(texCoord < 0.0) || any(texCoord > 1.0)) {
            t += baseStepSize;
            continue;
        }
        
        // Sample features from 3D texture (trilinear interpolation)
        float4 featureSample = featureGrid.SampleLevel(linearSampler, texCoord, 0);
        float3 features = featureSample.xyz;  // 32 channels (reduced for demo)
        
        // ========================================
        // Adaptive Sampling Decision
        // ========================================
        float2 samplingOutput = evaluateSamplingMLP(
            features,
            ray.direction,
            samplingMLPWeights,
            35,  // inputDim
            32,  // hiddenDim
            2    // outputDim
        );
        
        float stepMultiplier = samplingOutput.x * 1.9 + 0.1;  // Range: [0.1, 2.0]
        float importance = samplingOutput.y;
        
        float adaptiveStepSize = baseStepSize * stepMultiplier;
        
        // ========================================
        // Density & Segmentation Prediction
        // ========================================
        float4 mlpOutput = evaluateDensityMLP(
            features,
            densityMLPWeights,
            32,  // inputDim
            64,  // hiddenDim
            5    // outputDim
        );
        
        float density = softplus(mlpOutput.x);  // Ensure positive density
        float3 classLogits = mlpOutput.yzw;
        
        // Softmax for class probabilities
        float4 classProbabilities = softmax(float4(0.0, classLogits.x, classLogits.y, classLogits.z));
        
        // ========================================
        // Volume Rendering Integration
        // ========================================
        
        // Skip if very low importance and density
        if (importance > 0.3 || density > 0.1) {
            // Opacity from density
            float alpha = 1.0 - exp(-density * adaptiveStepSize);
            
            // Get color from segmentation
            float3 segColor = getSegmentationColor(classProbabilities);
            
            // Accumulate color and segmentation
            float weight = transmittance * alpha;
            result.color += weight * segColor;
            result.segmentation += weight * classProbabilities;
            result.density += weight * density;
            
            // Update transmittance
            transmittance *= (1.0 - alpha);
        }
        
        // Accumulate importance for monitoring
        result.importanceMap += importance;
        
        // Advance along ray with adaptive step
        t += adaptiveStepSize;
        result.numSamples += 1.0;
    }
    
    // Normalize importance map
    result.importanceMap /= max(result.numSamples, 1.0);
    result.transmittance = transmittance;
    
    return result;
}

// ============================================================================
// Entry Points for SlangPy
// ============================================================================

[shader("compute")]
[numthreads(8, 8, 1)]
void renderVolume(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform Texture3D<float4> featureGrid,
    uniform SamplerState linearSampler,
    uniform StructuredBuffer<float> densityMLPWeights,
    uniform StructuredBuffer<float> samplingMLPWeights,
    uniform StructuredBuffer<Ray> rays,
    uniform RWStructuredBuffer<VolumeRenderResult> results,
    uniform float baseStepSize,
    uniform int maxSteps
) {
    uint rayIdx = dispatchThreadID.x + dispatchThreadID.y * 1024;
    
    // Load ray
    Ray ray = rays[rayIdx];
    
    // Perform adaptive raymarching
    VolumeRenderResult result = adaptiveVolumeMarch(
        ray,
        featureGrid,
        linearSampler,
        densityMLPWeights,
        samplingMLPWeights,
        baseStepSize,
        maxSteps
    );
    
    // Write result
    results[rayIdx] = result;
}

// ============================================================================
// Pre/Post-Op Comparison Shader
// ============================================================================

struct ChangeDetectionResult {
    float3 color;
    float4 preOpSeg;
    float4 postOpSeg;
    float4 changeMap;
    float resectionQuality;
};

ChangeDetectionResult detectChanges(
    Ray ray,
    Texture3D<float4> preOpFeatures,
    Texture3D<float4> postOpFeatures,
    SamplerState linearSampler,
    StructuredBuffer<float> densityMLPWeights,
    float baseStepSize,
    int maxSteps
) {
    ChangeDetectionResult result;
    result.color = float3(0, 0, 0);
    result.preOpSeg = float4(0, 0, 0, 0);
    result.postOpSeg = float4(0, 0, 0, 0);
    result.changeMap = float4(0, 0, 0, 0);
    result.resectionQuality = 0.0;
    
    float t = ray.tMin;
    float transmittance = 1.0;
    float tumorRemoved = 0.0;
    float totalTumor = 0.0;
    
    for (int step = 0; step < maxSteps; step++) {
        if (transmittance < 0.01 || t >= ray.tMax) break;
        
        float3 pos = ray.origin + t * ray.direction;
        float3 texCoord = (pos + 1.0) * 0.5;
        
        if (any(texCoord < 0.0) || any(texCoord > 1.0)) {
            t += baseStepSize;
            continue;
        }
        
        // Sample both pre and post-op features
        float3 preFeat = preOpFeatures.SampleLevel(linearSampler, texCoord, 0).xyz;
        float3 postFeat = postOpFeatures.SampleLevel(linearSampler, texCoord, 0).xyz;
        
        // Predict segmentation for both
    float4 preOut = evaluateDensityMLP(float3(preFeat.x, preFeat.y, preFeat.z), densityMLPWeights, 32, 64, 5);
    float4 postOut = evaluateDensityMLP(float3(postFeat.x, postFeat.y, postFeat.z), densityMLPWeights, 32, 64, 5);
        
        float preDensity = softplus(preOut.x);
        float postDensity = softplus(postOut.x);
        
        float4 preClass = softmax(float4(0.0, preOut.yzw));
        float4 postClass = softmax(float4(0.0, postOut.yzw));
        
        // Detect changes
        float4 change = abs(preClass - postClass);
        
        // Check if tumor was present pre-op
        float tumorPresent = preClass.z + preClass.w;  // Core + enhancing
        totalTumor += tumorPresent;
        
        // Check if tumor removed post-op
        float tumorAfter = postClass.z + postClass.w;
        tumorRemoved += (tumorPresent - tumorAfter) * (tumorPresent > 0.5 ? 1.0 : 0.0);
        
        // Volume rendering
        float alpha = 1.0 - exp(-max(preDensity, postDensity) * baseStepSize);
        float weight = transmittance * alpha;
        
        // Color code: red=removed, yellow=remains, green=normal
        float3 changeColor;
        if (tumorPresent > 0.5 && tumorAfter < 0.3) {
            changeColor = float3(0.0, 1.0, 0.0);  // Successfully removed (green)
        } else if (tumorPresent > 0.5 && tumorAfter > 0.5) {
            changeColor = float3(1.0, 0.0, 0.0);  // Tumor remains (red)
        } else {
            changeColor = float3(0.5, 0.5, 0.5);  // No significant change
        }
        
        result.color += weight * changeColor;
        result.preOpSeg += weight * preClass;
        result.postOpSeg += weight * postClass;
        result.changeMap += weight * change;
        
        transmittance *= (1.0 - alpha);
        t += baseStepSize;
    }
    
    // Compute resection quality metric
    result.resectionQuality = totalTumor > 0.0 ? tumorRemoved / totalTumor : 0.0;
    
    return result;
}

// ============================================================================
// Compute Shader for Change Detection
// ============================================================================

[shader("compute")]
[numthreads(8, 8, 1)]
void detectPrePostChanges(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform Texture3D<float4> preOpFeatures,
    uniform Texture3D<float4> postOpFeatures,
    uniform SamplerState linearSampler,
    uniform StructuredBuffer<float> densityMLPWeights,
    uniform StructuredBuffer<Ray> rays,
    uniform RWStructuredBuffer<ChangeDetectionResult> results,
    uniform float baseStepSize,
    uniform int maxSteps
) {
    uint rayIdx = dispatchThreadID.x + dispatchThreadID.y * 1024;
    
    Ray ray = rays[rayIdx];
    
    ChangeDetectionResult result = detectChanges(
        ray,
        preOpFeatures,
        postOpFeatures,
        linearSampler,
        densityMLPWeights,
        baseStepSize,
        maxSteps
    );
    
    results[rayIdx] = result;
}
