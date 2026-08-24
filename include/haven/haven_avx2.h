#pragma once

#include "haven_types.h"
#include <immintrin.h>
#include <vector>
#include <cmath>

namespace haven {

class Avx2Math {
public:
    // Convert FP16 uint16_t to float
    static inline float fp16_to_fp32(uint16_t h) {
        __m128i hv = _mm_cvtsi32_si128(h);
        __m128 fv = _mm_cvtph_ps(hv);
        return _mm_cvtss_f32(fv);
    }

    // Dot product of quantized Q8_0 weights with FP32 vector using AVX2 and FMA
    static float vec_dot_q8_0(int n, const block_q8_0* vx, const float* vy);

    // Dot product of quantized Q4_K weights with FP32 vector using AVX2 and FMA
    static float vec_dot_q4_K(int n, const block_q4_K* vx, const float* vy);

    // In-place RMS Normalization: y = x / sqrt(mean(x^2) + eps) * weight
    static void rms_norm(float* out, const float* x, const float* weight, int size, float eps);

    // In-place Rotary Position Embeddings (RoPE) for 131k context window
    static void apply_rope(float* q, float* k, int head_dim, int num_heads, int num_kv_heads, int pos, float freq_base, float freq_scale);

    // Fast In-Place Softmax over an array of floats
    static void softmax(float* x, int size);

    // Fast In-Place SiLU Activation: x * sigmoid(x)
    static void silu(float* x, int size);

    // Matrix-Vector Multiplication: y = W * x
    static void gemv(float* y, const TensorDesc& weight, const float* x, int rows, int cols);
};

} // namespace haven