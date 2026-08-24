#include "haven/haven_memory.h"
#include <cmath>
#include <iostream>

namespace haven {

MemoryAttentionEngine::MemoryAttentionEngine(float alpha)
    : alpha_(alpha)
{
}

void MemoryAttentionEngine::inject_memory(const std::string& concept_name, float weight, const std::vector<float>& embedding) {
    memories_.push_back(MemoryAnchor{concept_name, weight, embedding});
}

void MemoryAttentionEngine::clear_memories() {
    memories_.clear();
}

// In-Attention Direct Memory Access (DMA) Kernel:
// Attention_Scores = (Q @ K^T) / sqrt(d_k) + alpha * (Q @ M^T) / sqrt(d_k)
void MemoryAttentionEngine::apply_memory_attention(
    float* attention_scores,
    const float* query,
    int seq_len,
    int head_dim,
    int head_idx)
{
    if (memories_.empty() || alpha_ <= 0.0f) {
        return;
    }

    const float scale = 1.0f / std::sqrt((float)head_dim);

    // Compute direct memory projection scores for each injected memory anchor
    for (size_t m = 0; m < memories_.size(); ++m) {
        const auto& mem = memories_[m];
        if (mem.embedding.size() < (size_t)head_dim) continue;

        float q_dot_m = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            q_dot_m += query[i] * mem.embedding[i];
        }

        float memory_boost = alpha_ * (q_dot_m * scale) * mem.weight;

        // Modulate attention scores across current sequence context
        for (int pos = 0; pos < seq_len; ++pos) {
            attention_scores[pos] += memory_boost * (1.0f / (float)(seq_len - pos + 1));
        }
    }
}

} // namespace haven