#pragma once

#include "haven_types.h"
#include <vector>
#include <cmath>

namespace haven {

struct YaRNConfig {
    float scale = 4.0f;           // Context scaling factor (e.g. 4.0x -> 32k context)
    float original_context = 8192.0f;
    float alpha = 1.0f;           // Low-frequency threshold
    float beta = 32.0f;           // High-frequency threshold
    float extrapolation_factor = 1.0f;
    float attn_factor = 1.0f;     // Attention temperature multiplier
};

class YaRNScaler {
public:
    YaRNScaler(const YaRNConfig& config = YaRNConfig());
    ~YaRNScaler();

    // Computes dynamic YaRN scaled rotary frequencies for head dimension
    void compute_frequencies(uint32_t head_dim, float base_freq, std::vector<float>& inv_freqs);

    // Applies YaRN RoPE to query and key vectors at given position
    void apply_rope(
        float* q,
        float* k,
        uint32_t num_heads,
        uint32_t num_kv_heads,
        uint32_t head_dim,
        int pos,
        const std::vector<float>& inv_freqs
    );

    float get_scale() const { return config_.scale; }
    void set_scale(float scale) { 
        config_.scale = scale; 
        update_temperature();
    }

private:
    YaRNConfig config_;
    float mscale_;

    void update_temperature();
};

} // namespace haven
