#pragma once

#include "haven_types.h"
#include <vector>
#include <cstdint>

namespace haven {

constexpr size_t FLASH_BLOCK_BR = 32; // Query block size
constexpr size_t FLASH_BLOCK_BC = 64; // Key/Value block size

class FlashAttentionCpu {
public:
    FlashAttentionCpu(uint32_t head_dim = 128);
    ~FlashAttentionCpu();

    // Executes FlashAttention tiled online-softmax kernel across Query, Key, Value vectors
    void compute_tiled_attention(
        const float* q,
        const float* k_cache,
        const float* v_cache,
        uint32_t num_tokens,
        uint32_t head_dim,
        float scale,
        float* out_attn
    );

private:
    uint32_t head_dim_;
    std::vector<float> scores_scratch_;
};

} // namespace haven
