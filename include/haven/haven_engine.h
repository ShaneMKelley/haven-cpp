#pragma once

#include "haven_types.h"
#include "haven_gguf.h"
#include "haven_avx2.h"
#include "haven_memory.h"
#include "haven_sampler.h"
#include <string>
#include <vector>
#include <functional>

namespace haven {

struct KeyValueCache {
    std::vector<float> k_cache; // [num_layers, max_seq, num_kv_heads, head_dim]
    std::vector<float> v_cache; // [num_layers, max_seq, num_kv_heads, head_dim]
    int current_pos = 0;
};

class HavenEngine {
public:
    HavenEngine();
    ~HavenEngine();

    bool load_model(const std::string& gguf_filepath);
    
    // Injects episodic memory anchors directly into DMA memory attention cache
    void inject_memory(const std::string& concept_name, float weight, const std::vector<float>& embedding);
    void clear_memories();

    // Sets soul persona vector for logit steering
    void set_persona_embedding(const std::vector<float>& persona_vector);

    // Executes forward pass for a single token at sequence position pos
    void forward(uint32_t token, int pos, float* out_logits);

    // Generates completion stream for a sequence of prompt tokens
    void generate(
        const std::vector<uint32_t>& prompt_tokens,
        int max_new_tokens,
        const std::function<bool(uint32_t token, const std::string& text)>& on_token_callback
    );

    const ModelConfig& get_config() const { return loader_.get_config(); }
    size_t get_memory_count() const { return memory_engine_.get_memory_count(); }

private:
    GgufLoader loader_;
    MemoryAttentionEngine memory_engine_;
    PersonaSampler sampler_;
    KeyValueCache kv_cache_;
    
    // Scratch execution buffers
    std::vector<float> embedding_scratch_;
    std::vector<float> norm_scratch_;
    std::vector<float> q_scratch_;
    std::vector<float> k_scratch_;
    std::vector<float> v_scratch_;
    std::vector<float> attn_out_scratch_;
    std::vector<float> ffn_gate_scratch_;
    std::vector<float> ffn_up_scratch_;
    std::vector<float> ffn_down_scratch_;
    std::vector<float> logits_scratch_;
};

} // namespace haven

// ============================================================================
// C ABI Export Interface for P/Invoke (C# Gemmi & Android Kotlin Native)
// ============================================================================
#ifdef _WIN32
#define HAVEN_API extern "C" __declspec(dllexport)
#else
#define HAVEN_API extern "C" __attribute__((visibility("default")))
#endif

HAVEN_API void* haven_create_engine();
HAVEN_API bool  haven_load_model(void* engine, const char* filepath);
HAVEN_API void  haven_inject_memory(void* engine, const char* concept_name, float weight, const float* embedding, int dim);
HAVEN_API void  haven_set_persona(void* engine, const float* persona_vector, int dim);
HAVEN_API void  haven_forward(void* engine, uint32_t token, int pos, float* out_logits);
HAVEN_API uint32_t haven_sample_token(void* engine, float* logits, uint32_t vocab_size);
HAVEN_API void  haven_destroy_engine(void* engine);