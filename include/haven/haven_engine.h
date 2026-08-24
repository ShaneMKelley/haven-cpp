#pragma once

#include "haven_types.h"
#include "haven_gguf.h"
#include "haven_avx2.h"
#include "haven_memory.h"
#include "haven_sampler.h"
#include "haven_kvcache.h"
#include "haven_knowledge_store.h"
#include "haven_attention_telemetry.h"
#include <string>
#include <vector>
#include <functional>

namespace haven {

class HavenEngine {
public:
    HavenEngine();
    ~HavenEngine();

    bool load_model(const std::string& gguf_filepath);
    
    // Injects episodic memory anchors directly into DMA memory attention cache
    void inject_memory(const std::string& concept_name, float weight, const std::vector<float>& embedding);
    void clear_memories();

    // Ingests structured relational knowledge with composite key boosts
    void add_knowledge_relation(uint32_t token_id, uint32_t cluster_id, float boost, const std::string& label, const std::string& target);

    // Sets soul persona vector for logit steering
    void set_persona_embedding(const std::vector<float>& persona_vector);

    // Executes forward pass for a single token with dynamic KV cache, knowledge injection, and telemetry
    void forward(uint32_t token, int pos, float* out_logits, uint32_t active_cluster = 0);

    // Prunes stale cache entries across all layers and returns total pruned tokens
    size_t prune_kv_cache();

    // Generates completion stream for a sequence of prompt tokens
    void generate(
        const std::vector<uint32_t>& prompt_tokens,
        int max_new_tokens,
        const std::function<bool(uint32_t token, const std::string& text)>& on_token_callback
    );

    const ModelConfig& get_config() const { return loader_.get_config(); }
    size_t get_memory_count() const { return memory_engine_.get_memory_count(); }
    size_t get_knowledge_count() const { return knowledge_store_.get_relation_count(); }
    float get_current_attention_entropy() const { return telemetry_engine_.get_global_attention_entropy(); }
    std::string get_json_telemetry(int pos, uint32_t token) const { return telemetry_engine_.to_json_telemetry(pos, token); }

private:
    GgufLoader loader_;
    MemoryAttentionEngine memory_engine_;
    StructuredKnowledgeStore knowledge_store_;
    AttentionTelemetryEngine telemetry_engine_;
    DynamicKVCacheManager kv_cache_;
    PersonaSampler sampler_;
    
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
HAVEN_API void  haven_add_knowledge(void* engine, uint32_t token_id, uint32_t cluster_id, float boost, const char* label, const char* target);
HAVEN_API size_t haven_prune_kv_cache(void* engine);
HAVEN_API float  haven_get_attention_entropy(void* engine);
HAVEN_API void  haven_set_persona(void* engine, const float* persona_vector, int dim);
HAVEN_API void  haven_forward(void* engine, uint32_t token, int pos, float* out_logits, uint32_t active_cluster);
HAVEN_API uint32_t haven_sample_token(void* engine, float* logits, uint32_t vocab_size);
HAVEN_API void  haven_destroy_engine(void* engine);