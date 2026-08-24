#pragma once

#include "haven_types.h"
#include <vector>
#include <string>

namespace haven {

struct MemoryAnchor {
    std::string concept_name;
    float weight = 1.0f;
    std::vector<float> embedding; // Dimension matches head_dim / embedding_dim
};

class MemoryAttentionEngine {
public:
    MemoryAttentionEngine(float alpha = 0.35f);

    // Register a memory anchor into the active DMA cache
    void inject_memory(const std::string& concept_name, float weight, const std::vector<float>& embedding);
    void clear_memories();

    size_t get_memory_count() const { return memories_.size(); }
    float get_alpha() const { return alpha_; }
    void set_alpha(float alpha) { alpha_ = alpha; }

    // Computes augmented attention scores: QK^T / sqrt(d_k) + alpha * (Q @ M^T)
    void apply_memory_attention(
        float* attention_scores,
        const float* query,
        int seq_len,
        int head_dim,
        int head_idx
    );

private:
    float alpha_;
    std::vector<MemoryAnchor> memories_;
};

} // namespace haven