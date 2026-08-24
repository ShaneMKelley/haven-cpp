#include "haven/haven_kvcache.h"
#include <algorithm>
#include <cstring>
#include <iostream>

namespace haven {

DynamicKVCacheManager::DynamicKVCacheManager(
    uint32_t num_layers,
    uint32_t max_context_length,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    float pruning_threshold,
    float salience_decay)
    : num_layers_(num_layers),
      max_context_length_(max_context_length),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      pruning_threshold_(pruning_threshold),
      salience_decay_(salience_decay),
      current_length_(0)
{
    layers_.resize(num_layers_);
    const size_t layer_stride = (size_t)max_context_length_ * num_kv_heads_ * head_dim_;

    for (uint32_t l = 0; l < num_layers_; ++l) {
        layers_[l].keys.resize(layer_stride, 0.0f);
        layers_[l].values.resize(layer_stride, 0.0f);
        layers_[l].salience_scores.resize(max_context_length_, 1.0f);
        layers_[l].active_mask.resize(max_context_length_, true);
        layers_[l].current_len = 0;
    }
}

void DynamicKVCacheManager::append(uint32_t layer, const float* k, const float* v) {
    if (layer >= num_layers_ || current_length_ >= (int)max_context_length_) return;

    auto& lay = layers_[layer];
    const size_t offset = (size_t)current_length_ * num_kv_heads_ * head_dim_;
    const size_t num_bytes = num_kv_heads_ * head_dim_ * sizeof(float);

    std::memcpy(lay.keys.data() + offset, k, num_bytes);
    std::memcpy(lay.values.data() + offset, v, num_bytes);

    lay.salience_scores[current_length_] = 1.0f; // New token starts with full salience
    lay.active_mask[current_length_] = true;
    lay.current_len = current_length_ + 1;
}

void DynamicKVCacheManager::update_salience(uint32_t layer, const float* attention_weights, int seq_len) {
    if (layer >= num_layers_) return;

    auto& lay = layers_[layer];
    const int len = std::min(seq_len, current_length_ + 1);

    // Apply temporal exponential decay and accumulate new attention weight
    for (int i = 0; i < len; ++i) {
        lay.salience_scores[i] = lay.salience_scores[i] * salience_decay_ + attention_weights[i];
    }
}

size_t DynamicKVCacheManager::prune_stale_entries(uint32_t layer) {
    if (layer >= num_layers_) return 0;

    auto& lay = layers_[layer];
    size_t pruned_count = 0;
    const size_t vec_stride = num_kv_heads_ * head_dim_;

    // Pinned Anchors: Never prune the first 4 tokens (system / soul core anchors)
    const int pin_count = std::min(4, current_length_);

    for (int i = pin_count; i < current_length_; ++i) {
        if (lay.active_mask[i] && lay.salience_scores[i] < pruning_threshold_) {
            lay.active_mask[i] = false;
            // Zero out memory to save cache bandwidth during attention GEMV
            std::memset(lay.keys.data() + i * vec_stride, 0, vec_stride * sizeof(float));
            std::memset(lay.values.data() + i * vec_stride, 0, vec_stride * sizeof(float));
            pruned_count++;
        }
    }

    return pruned_count;
}

size_t DynamicKVCacheManager::prune_all_layers() {
    size_t total_pruned = 0;
    for (uint32_t l = 0; l < num_layers_; ++l) {
        total_pruned += prune_stale_entries(l);
    }
    return total_pruned;
}

void DynamicKVCacheManager::reset() {
    current_length_ = 0;
    for (auto& lay : layers_) {
        std::fill(lay.salience_scores.begin(), lay.salience_scores.end(), 1.0f);
        std::fill(lay.active_mask.begin(), lay.active_mask.end(), true);
        lay.current_len = 0;
    }
}

} // namespace haven