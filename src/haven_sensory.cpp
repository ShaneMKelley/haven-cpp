#include "haven/haven_sensory.h"
#include "haven/haven_avx2.h"
#include <cmath>
#include <algorithm>

namespace haven {

SensoryAttentionBridge::SensoryAttentionBridge(uint32_t embedding_dim)
    : embedding_dim_(embedding_dim)
{
    initialize_projection_matrices();
}

SensoryAttentionBridge::~SensoryAttentionBridge() = default;

void SensoryAttentionBridge::initialize_projection_matrices() {
    // 768 vision dim -> 4096 model embedding dim
    const size_t vision_weights_size = 768 * embedding_dim_;
    vision_projection_weights_.resize(vision_weights_size, 0.0f);
    for (size_t i = 0; i < vision_weights_size; ++i) {
        vision_projection_weights_[i] = std::sin((float)i * 0.001f) * 0.02f;
    }

    // 128 audio dim -> 4096 model embedding dim
    const size_t audio_weights_size = 128 * embedding_dim_;
    audio_projection_weights_.resize(audio_weights_size, 0.0f);
    for (size_t i = 0; i < audio_weights_size; ++i) {
        audio_projection_weights_[i] = std::cos((float)i * 0.002f) * 0.015f;
    }
}

void SensoryAttentionBridge::ingest_vision_frame(const VisualFrame& frame) {
    visual_queue_.push_back(frame);
    if (visual_queue_.size() > 8) {
        visual_queue_.erase(visual_queue_.begin());
    }
}

void SensoryAttentionBridge::ingest_audio_spectrum(const AudioSpectrumFrame& frame) {
    audio_queue_.push_back(frame);
    if (audio_queue_.size() > 32) {
        audio_queue_.erase(audio_queue_.begin());
    }
}

std::vector<float> SensoryAttentionBridge::project_sensory_tokens(uint32_t target_dim) {
    std::vector<float> sensory_vector(target_dim, 0.0f);
    if (target_dim == 0) return sensory_vector;

    if (embedding_dim_ != target_dim || vision_projection_weights_.empty() || audio_projection_weights_.empty()) {
        embedding_dim_ = target_dim;
        initialize_projection_matrices();
    }

    // 1. Project Visual Embeddings
    if (!visual_queue_.empty()) {
        const auto& latest_vision = visual_queue_.back();
        if (!latest_vision.patch_embeddings.empty()) {
            size_t in_dim = std::min((size_t)768, latest_vision.patch_embeddings.size());
            for (size_t out_idx = 0; out_idx < target_dim; ++out_idx) {
                float sum = 0.0f;
                for (size_t in_idx = 0; in_idx < in_dim; ++in_idx) {
                    size_t w_idx = in_idx * target_dim + out_idx;
                    if (w_idx < vision_projection_weights_.size()) {
                        sum += latest_vision.patch_embeddings[in_idx] * vision_projection_weights_[w_idx];
                    }
                }
                sensory_vector[out_idx] += sum * latest_vision.salience_score * 0.01f;
            }
        }
    }

    // 2. Project Audio Mel-Spectrum Embeddings
    if (!audio_queue_.empty()) {
        const auto& latest_audio = audio_queue_.back();
        if (!latest_audio.mel_bands.empty()) {
            size_t in_dim = std::min((size_t)128, latest_audio.mel_bands.size());
            for (size_t out_idx = 0; out_idx < target_dim; ++out_idx) {
                float sum = 0.0f;
                for (size_t in_idx = 0; in_idx < in_dim; ++in_idx) {
                    size_t w_idx = in_idx * target_dim + out_idx;
                    if (w_idx < audio_projection_weights_.size()) {
                        sum += latest_audio.mel_bands[in_idx] * audio_projection_weights_[w_idx];
                    }
                }
                sensory_vector[out_idx] += sum * (latest_audio.energy_rms + 0.1f) * 0.01f;
            }
        }
    }

    return sensory_vector;
}

void SensoryAttentionBridge::clear_sensory_buffers() {
    visual_queue_.clear();
    audio_queue_.clear();
}

} // namespace haven
