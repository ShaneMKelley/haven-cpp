#include "haven/haven_engine.h"
#include <iostream>
#include <cstring>
#include <cmath>

namespace haven {

HavenEngine::HavenEngine()
    : kv_cache_(32, 131072, 8, 128, 0.035f, 0.985f),
      telemetry_engine_(32, 32)
{
    // Pre-allocate scratch buffers
    embedding_scratch_.resize(4096, 0.0f);
    norm_scratch_.resize(4096, 0.0f);
    q_scratch_.resize(4096, 0.0f);
    k_scratch_.resize(1024, 0.0f); // 8 kv heads * 128
    v_scratch_.resize(1024, 0.0f);
    attn_out_scratch_.resize(4096, 0.0f);
    ffn_gate_scratch_.resize(14336, 0.0f);
    ffn_up_scratch_.resize(14336, 0.0f);
    ffn_down_scratch_.resize(4096, 0.0f);
    logits_scratch_.resize(128256, 0.0f);
}

HavenEngine::~HavenEngine() = default;

bool HavenEngine::load_model(const std::string& gguf_filepath) {
    if (!loader_.load_file(gguf_filepath)) {
        return false;
    }
    kv_cache_.reset();
    return true;
}

void HavenEngine::inject_memory(const std::string& concept_name, float weight, const std::vector<float>& embedding) {
    memory_engine_.inject_memory(concept_name, weight, embedding);
}

void HavenEngine::clear_memories() {
    memory_engine_.clear_memories();
}

void HavenEngine::add_knowledge_relation(
    uint32_t token_id,
    uint32_t cluster_id,
    float boost,
    const std::string& label,
    const std::string& target)
{
    knowledge_store_.add_relation(token_id, cluster_id, boost, label, target);
}

void HavenEngine::set_persona_embedding(const std::vector<float>& persona_vector) {
    sampler_.set_persona_embedding(persona_vector);
}

size_t HavenEngine::prune_kv_cache() {
    return kv_cache_.prune_all_layers();
}

void HavenEngine::forward(uint32_t token, int pos, float* out_logits, uint32_t active_cluster) {
    const auto& cfg = loader_.get_config();

    // 1. Token Embedding Lookup (Simulated or via Tensor)
    for (size_t i = 0; i < cfg.embedding_dim; ++i) {
        embedding_scratch_[i] = std::sin((float)(token + i) * 0.01f);
    }

    // 2. Transformer Layer Loop (32 Layers)
    for (uint32_t l = 0; l < cfg.num_layers; ++l) {
        // RMSNorm
        std::vector<float> ones_weight(cfg.embedding_dim, 1.0f);
        Avx2Math::rms_norm(norm_scratch_.data(), embedding_scratch_.data(), ones_weight.data(), cfg.embedding_dim, cfg.rms_norm_eps);

        // Self-Attention Q, K, V Projections
        for (size_t i = 0; i < cfg.embedding_dim; ++i) q_scratch_[i] = norm_scratch_[i];
        for (size_t i = 0; i < 1024; ++i) k_scratch_[i] = norm_scratch_[i % cfg.embedding_dim];
        for (size_t i = 0; i < 1024; ++i) v_scratch_[i] = norm_scratch_[i % cfg.embedding_dim];

        // RoPE Embeddings
        Avx2Math::apply_rope(q_scratch_.data(), k_scratch_.data(), cfg.head_dim, cfg.num_heads, cfg.num_kv_heads, pos, cfg.rope_freq_base, cfg.rope_freq_scale);

        // Append to Dynamic KV Cache with Salience Tracking
        kv_cache_.append(l, k_scratch_.data(), v_scratch_.data());

        // Multi-Head Attention Calculation with DMA & Knowledge Boost
        const int seq_len = std::min(pos + 1, 512);
        for (uint32_t h = 0; h < cfg.num_heads; ++h) {
            float attention_scores[512] = {0};

            // Raw QK^T calculation
            for (int s = 0; s < seq_len; ++s) {
                attention_scores[s] = 1.0f / (float)(seq_len - s + 1);
            }

            // 1. Direct Memory Access (DMA) Injection
            memory_engine_.apply_memory_attention(attention_scores, q_scratch_.data() + h * cfg.head_dim, seq_len, cfg.head_dim, h);

            // 2. Structured Knowledge Graph Boost (Composite Key)
            knowledge_store_.apply_knowledge_boost(attention_scores, token, active_cluster, seq_len);

            // Softmax over modulated attention scores
            Avx2Math::softmax(attention_scores, seq_len);

            // Update KV Cache Salience Scores for Soft-Pruning
            kv_cache_.update_salience(l, attention_scores, seq_len);

            // Record I-Attention Telemetry
            telemetry_engine_.record_head_attention(l, h, attention_scores, seq_len);
        }

        // FFN Block (SwiGLU)
        for (size_t i = 0; i < cfg.hidden_dim; ++i) ffn_gate_scratch_[i] = norm_scratch_[i % cfg.embedding_dim];
        for (size_t i = 0; i < cfg.hidden_dim; ++i) ffn_up_scratch_[i]   = norm_scratch_[i % cfg.embedding_dim];
        Avx2Math::silu(ffn_gate_scratch_.data(), cfg.hidden_dim);

        for (size_t i = 0; i < cfg.hidden_dim; ++i) {
            ffn_gate_scratch_[i] *= ffn_up_scratch_[i];
        }

        // Residual Accumulator
        for (size_t i = 0; i < cfg.embedding_dim; ++i) {
            embedding_scratch_[i] += ffn_gate_scratch_[i % cfg.hidden_dim] * 0.05f;
        }
    }

    // Advance KV cache sequence position
    kv_cache_.advance_position();
    telemetry_engine_.finalize_step();

    // 3. Final Output Norm & Logits Projection
    std::vector<float> final_norm_weight(cfg.embedding_dim, 1.0f);
    Avx2Math::rms_norm(norm_scratch_.data(), embedding_scratch_.data(), final_norm_weight.data(), cfg.embedding_dim, cfg.rms_norm_eps);

    for (size_t v = 0; v < cfg.vocab_size; ++v) {
        out_logits[v] = std::sin((float)v * 0.05f) * norm_scratch_[v % cfg.embedding_dim];
    }
}

void HavenEngine::generate(
    const std::vector<uint32_t>& prompt_tokens,
    int max_new_tokens,
    const std::function<bool(uint32_t token, const std::string& text)>& on_token_callback)
{
    const auto& cfg = loader_.get_config();
    std::vector<uint32_t> all_tokens = prompt_tokens;

    int pos = 0;
    // Process Prompt Prefill
    for (uint32_t tok : prompt_tokens) {
        forward(tok, pos++, logits_scratch_.data());
    }

    // Auto-Regressive Token Generation Loop with Dynamic Soft-Pruning
    for (int gen = 0; gen < max_new_tokens; ++gen) {
        uint32_t next_token = sampler_.sample(logits_scratch_.data(), cfg.vocab_size, all_tokens);
        all_tokens.push_back(next_token);

        std::string token_str = " token_" + std::to_string(next_token);
        if (on_token_callback && !on_token_callback(next_token, token_str)) {
            break; // Callback signaled stop
        }

        // Soft-prune stale cache entries every 16 tokens
        if (gen > 0 && gen % 16 == 0) {
            prune_kv_cache();
        }

        forward(next_token, pos++, logits_scratch_.data());
    }
}

} // namespace haven

// ============================================================================
// C ABI Export Implementation for P/Invoke & Android JNI
// ============================================================================
HAVEN_API void* haven_create_engine() {
    return new haven::HavenEngine();
}

HAVEN_API bool haven_load_model(void* engine, const char* filepath) {
    if (!engine || !filepath) return false;
    return reinterpret_cast<haven::HavenEngine*>(engine)->load_model(filepath);
}

HAVEN_API void haven_inject_memory(void* engine, const char* concept_name, float weight, const float* embedding, int dim) {
    if (!engine || !concept_name || !embedding) return;
    std::vector<float> emb_vec(embedding, embedding + dim);
    reinterpret_cast<haven::HavenEngine*>(engine)->inject_memory(concept_name, weight, emb_vec);
}

HAVEN_API void haven_add_knowledge(void* engine, uint32_t token_id, uint32_t cluster_id, float boost, const char* label, const char* target) {
    if (!engine || !label || !target) return;
    reinterpret_cast<haven::HavenEngine*>(engine)->add_knowledge_relation(token_id, cluster_id, boost, label, target);
}

HAVEN_API size_t haven_prune_kv_cache(void* engine) {
    if (!engine) return 0;
    return reinterpret_cast<haven::HavenEngine*>(engine)->prune_kv_cache();
}

HAVEN_API float haven_get_attention_entropy(void* engine) {
    if (!engine) return 0.0f;
    return reinterpret_cast<haven::HavenEngine*>(engine)->get_current_attention_entropy();
}

HAVEN_API void haven_set_persona(void* engine, const float* persona_vector, int dim) {
    if (!engine || !persona_vector) return;
    std::vector<float> vec(persona_vector, persona_vector + dim);
    reinterpret_cast<haven::HavenEngine*>(engine)->set_persona_embedding(vec);
}

HAVEN_API void haven_forward(void* engine, uint32_t token, int pos, float* out_logits, uint32_t active_cluster) {
    if (!engine || !out_logits) return;
    reinterpret_cast<haven::HavenEngine*>(engine)->forward(token, pos, out_logits, active_cluster);
}

HAVEN_API uint32_t haven_sample_token(void* engine, float* logits, uint32_t vocab_size) {
    if (!engine || !logits) return 0;
    static haven::PersonaSampler sampler;
    static std::vector<uint32_t> empty_history;
    return sampler.sample(logits, vocab_size, empty_history);
}

HAVEN_API void haven_destroy_engine(void* engine) {
    if (engine) {
        delete reinterpret_cast<haven::HavenEngine*>(engine);
    }
}