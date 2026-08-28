#include "haven/haven_flash_attention.h"
#include "haven/haven_avx2.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace haven {

FlashAttentionCpu::FlashAttentionCpu(uint32_t head_dim)
    : head_dim_(head_dim)
{
    scores_scratch_.resize(FLASH_BLOCK_BC, 0.0f);
}

FlashAttentionCpu::~FlashAttentionCpu() = default;

void FlashAttentionCpu::compute_tiled_attention(
    const float* q,
    const float* k_cache,
    const float* v_cache,
    uint32_t num_tokens,
    uint32_t head_dim,
    float scale,
    float* out_attn)
{
    if (num_tokens == 0) return;
    std::memset(out_attn, 0, head_dim * sizeof(float));

    float max_score = -1e9f;
    float sum_exp = 0.0f;

    // Tile across Key/Value cache blocks (fitting in L1/L2 CPU cache)
    for (uint32_t block_start = 0; block_start < num_tokens; block_start += FLASH_BLOCK_BC) {
        uint32_t block_end = std::min(num_tokens, (uint32_t)(block_start + FLASH_BLOCK_BC));
        uint32_t current_block_size = block_end - block_start;

        float block_max = -1e9f;

        // Compute Q · K^T dot products for this block
        for (uint32_t i = 0; i < current_block_size; ++i) {
            uint32_t tok_idx = block_start + i;
            const float* k_vec = k_cache + (tok_idx * head_dim);

            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                dot += q[d] * k_vec[d];
            }
            float score = dot * scale;
            scores_scratch_[i] = score;
            if (score > block_max) block_max = score;
        }

        // Online softmax update: Rescale previous accumulator
        float new_max = std::max(max_score, block_max);
        float rescale_prev = (max_score > -1e8f) ? std::exp(max_score - new_max) : 0.0f;
        sum_exp *= rescale_prev;

        for (uint32_t d = 0; d < head_dim; ++d) {
            out_attn[d] *= rescale_prev;
        }

        // Accumulate new weighted values from this block
        for (uint32_t i = 0; i < current_block_size; ++i) {
            uint32_t tok_idx = block_start + i;
            const float* v_vec = v_cache + (tok_idx * head_dim);

            float exp_val = std::exp(scores_scratch_[i] - new_max);
            sum_exp += exp_val;

            for (uint32_t d = 0; d < head_dim; ++d) {
                out_attn[d] += exp_val * v_vec[d];
            }
        }

        max_score = new_max;
    }

    // Final normalization
    if (sum_exp > 1e-6f) {
        float inv_sum = 1.0f / sum_exp;
        for (uint32_t d = 0; d < head_dim; ++d) {
            out_attn[d] *= inv_sum;
        }
    }
}

} // namespace haven
