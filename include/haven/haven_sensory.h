#pragma once

#include "haven_types.h"
#include <vector>
#include <string>
#include <cstdint>

namespace haven {

struct VisualFrame {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t channels = 0;
    std::vector<float> patch_embeddings; // e.g. 768-dim patch vectors
    float salience_score = 1.0f;
    uint64_t timestamp_ms = 0;
};

struct AudioSpectrumFrame {
    std::vector<float> mel_bands; // 64 or 128 mel frequency bands
    float energy_rms = 0.0f;
    float spectral_centroid = 0.0f;
    bool is_speech_active = false;
    uint64_t timestamp_ms = 0;
};

class SensoryAttentionBridge {
public:
    SensoryAttentionBridge(uint32_t embedding_dim = 4096);
    ~SensoryAttentionBridge();

    // Ingests screen or camera vision embeddings and projects them to model embedding dimension
    void ingest_vision_frame(const VisualFrame& frame);

    // Ingests real-time microphone mel-spectrogram frame
    void ingest_audio_spectrum(const AudioSpectrumFrame& frame);

    // Projects sensory buffer into prompt token prefill stream
    std::vector<float> project_sensory_tokens(uint32_t target_dim);

    // Clears buffered sensory frames
    void clear_sensory_buffers();

    size_t get_visual_frame_count() const { return visual_queue_.size(); }
    size_t get_audio_frame_count() const { return audio_queue_.size(); }

private:
    uint32_t embedding_dim_;
    std::vector<VisualFrame> visual_queue_;
    std::vector<AudioSpectrumFrame> audio_queue_;
    std::vector<float> vision_projection_weights_;
    std::vector<float> audio_projection_weights_;

    void initialize_projection_matrices();
};

} // namespace haven
