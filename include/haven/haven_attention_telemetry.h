#pragma once

#include "haven_types.h"
#include <vector>
#include <string>
#include <cmath>

namespace haven {

struct AttentionLayerTelemetry {
    uint32_t layer_index = 0;
    float mean_attention_entropy = 0.0f; // Measure of cognitive focus (low = sharp focus, high = diffuse contemplation)
    int max_attended_position = 0;
    float peak_attention_weight = 0.0f;
    std::vector<float> head_entropies;   // Entropy per attention head
};

class AttentionTelemetryEngine {
public:
    AttentionTelemetryEngine(uint32_t num_layers = 32, uint32_t num_heads = 32);

    // Records attention pattern weights for a specific layer and head
    void record_head_attention(
        uint32_t layer,
        uint32_t head,
        const float* attention_weights,
        int seq_len
    );

    // Finalizes telemetry for the current forward step
    void finalize_step();

    // Computes Shannon Entropy: H = -sum(p * log2(p))
    static float compute_entropy(const float* probs, int size);

    const std::vector<AttentionLayerTelemetry>& get_latest_telemetry() const { return telemetry_history_; }
    float get_global_attention_entropy() const { return global_entropy_; }

    // Formats telemetry packet into JSON for WebSocket broadcast to Gemmi WebGL visualizer
    std::string to_json_telemetry(int current_pos, uint32_t active_token) const;

private:
    uint32_t num_layers_;
    uint32_t num_heads_;
    float global_entropy_ = 0.0f;
    std::vector<AttentionLayerTelemetry> telemetry_history_;
};

} // namespace haven