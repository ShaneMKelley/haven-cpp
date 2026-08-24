#include "haven/haven_avx2.h"
#include <algorithm>
#include <cstring>
#include <iostream>

namespace haven {

// Fast AVX2 + FMA Q8_0 dot product
float Avx2Math::vec_dot_q8_0(int n, const block_q8_0* vx, const float* vy) {
    const int nb = n / 32;
    float sumf = 0.0f;
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(vx[i].d);
        __m256 d_vec = _mm256_set1_ps(d);

        // Load 32 float values from vy
        __m256 vy0 = _mm256_loadu_ps(vy + i * 32 + 0);
        __m256 vy1 = _mm256_loadu_ps(vy + i * 32 + 8);
        __m256 vy2 = _mm256_loadu_ps(vy + i * 32 + 16);
        __m256 vy3 = _mm256_loadu_ps(vy + i * 32 + 24);

        // Load 32 int8 values from qs and convert to float
        __m128i q01 = _mm_loadu_si128((const __m128i*)(vx[i].qs + 0));
        __m128i q23 = _mm_loadu_si128((const __m128i*)(vx[i].qs + 16));

        __m256i q0_epi32 = _mm256_cvtepi8_epi32(q01);
        __m256i q1_epi32 = _mm256_cvtepi8_epi32(_mm_srli_si128(q01, 8));
        __m256i q2_epi32 = _mm256_cvtepi8_epi32(q23);
        __m256i q3_epi32 = _mm256_cvtepi8_epi32(_mm_srli_si128(q23, 8));

        __m256 vx0 = _mm256_cvtepi32_ps(q0_epi32);
        __m256 vx1 = _mm256_cvtepi32_ps(q1_epi32);
        __m256 vx2 = _mm256_cvtepi32_ps(q2_epi32);
        __m256 vx3 = _mm256_cvtepi32_ps(q3_epi32);

        // Fused Multiply-Add: acc += vx * vy * d
        acc = _mm256_fmadd_ps(_mm256_mul_ps(vx0, d_vec), vy0, acc);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(vx1, d_vec), vy1, acc);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(vx2, d_vec), vy2, acc);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(vx3, d_vec), vy3, acc);
    }

    // Horizontal sum of 256-bit accumulator
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sumf = _mm_cvtss_f32(sum128);

    return sumf;
}

// Fast AVX2 Q4_K dot product
float Avx2Math::vec_dot_q4_K(int n, const block_q4_K* vx, const float* vy) {
    const int nb = n / 256;
    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(vx[i].d);
        const float dmin = fp16_to_fp32(vx[i].dmin);
        const uint8_t* q = vx[i].qs;
        const float* y = vy + i * 256;

        float block_sum = 0.0f;
        for (int j = 0; j < 128; ++j) {
            uint8_t byte_val = q[j];
            float val0 = (float)(byte_val & 0x0F);
            float val1 = (float)(byte_val >> 4);

            block_sum += (val0 * d - dmin) * y[j * 2 + 0];
            block_sum += (val1 * d - dmin) * y[j * 2 + 1];
        }
        sumf += block_sum;
    }
    return sumf;
}

// AVX2 RMS Normalization
void Avx2Math::rms_norm(float* out, const float* x, const float* weight, int size, float eps) {
    float sum_sq = 0.0f;
    __m256 acc = _mm256_setzero_ps();

    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        acc = _mm256_fmadd_ps(vx, vx, acc);
    }
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum_sq = _mm_cvtss_f32(sum128);

    for (; i < size; ++i) {
        sum_sq += x[i] * x[i];
    }

    float scale = 1.0f / std::sqrt((sum_sq / (float)size) + eps);
    __m256 vscale = _mm256_set1_ps(scale);

    for (i = 0; i <= size - 8; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(weight + i);
        __m256 res = _mm256_mul_ps(_mm256_mul_ps(vx, vscale), vw);
        _mm256_storeu_ps(out + i, res);
    }
    for (; i < size; ++i) {
        out[i] = x[i] * scale * weight[i];
    }
}

// Rotary Position Embeddings (RoPE) for 131k context window
void Avx2Math::apply_rope(
    float* q, float* k,
    int head_dim, int num_heads, int num_kv_heads,
    int pos, float freq_base, float freq_scale)
{
    for (int h = 0; h < num_heads; ++h) {
        float* q_head = q + h * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float theta = (float)pos * std::pow(freq_base, -(float)i / (float)head_dim) * freq_scale;
            float cos_th = std::cos(theta);
            float sin_th = std::sin(theta);

            float q0 = q_head[i];
            float q1 = q_head[i + 1];
            q_head[i]     = q0 * cos_th - q1 * sin_th;
            q_head[i + 1] = q0 * sin_th + q1 * cos_th;
        }
    }

    for (int h = 0; h < num_kv_heads; ++h) {
        float* k_head = k + h * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float theta = (float)pos * std::pow(freq_base, -(float)i / (float)head_dim) * freq_scale;
            float cos_th = std::cos(theta);
            float sin_th = std::sin(theta);

            float k0 = k_head[i];
            float k1 = k_head[i + 1];
            k_head[i]     = k0 * cos_th - k1 * sin_th;
            k_head[i + 1] = k0 * sin_th + k1 * cos_th;
        }
    }
}

// Fast In-Place Softmax
void Avx2Math::softmax(float* x, int size) {
    float max_val = *std::max_element(x, x + size);
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; ++i) {
        x[i] *= inv_sum;
    }
}

// Fast SiLU Activation: x * sigmoid(x)
void Avx2Math::silu(float* x, int size) {
    for (int i = 0; i < size; ++i) {
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

// Matrix-Vector Multiplication: y = W * x
void Avx2Math::gemv(float* y, const TensorDesc& weight, const float* x, int rows, int cols) {
    if (weight.type == QuantType::Q8_0) {
        const auto* blocks = reinterpret_cast<const block_q8_0*>(weight.data);
        const int blocks_per_row = cols / 32;
        #pragma omp parallel for
        for (int r = 0; r < rows; ++r) {
            y[r] = vec_dot_q8_0(cols, blocks + r * blocks_per_row, x);
        }
    } else if (weight.type == QuantType::Q4_K) {
        const auto* blocks = reinterpret_cast<const block_q4_K*>(weight.data);
        const int blocks_per_row = cols / 256;
        #pragma omp parallel for
        for (int r = 0; r < rows; ++r) {
            y[r] = vec_dot_q4_K(cols, blocks + r * blocks_per_row, x);
        }
    } else if (weight.type == QuantType::F32) {
        const auto* w = reinterpret_cast<const float*>(weight.data);
        #pragma omp parallel for
        for (int r = 0; r < rows; ++r) {
            float sum = 0.0f;
            const float* w_row = w + r * cols;
            for (int c = 0; c < cols; ++c) {
                sum += w_row[c] * x[c];
            }
            y[r] = sum;
        }
    }
}

} // namespace haven