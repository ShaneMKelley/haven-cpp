#pragma once

#include "haven_types.h"
#include "haven_avx2.h"
#include <vector>
#include <string>
#include <cstdint>
#include <memory>

namespace haven {

// ============================================================================
// 🏛️ 64-BIT HAVEN MEMORY BANK (.hmb) BINARY FORMAT SPECIFICATION
// ============================================================================
#pragma pack(push, 1)

struct HmbHeader64 {
    char     magic[8];            // "HAVENMEM" (0x484156454E4D454D)
    uint32_t version;             // 0x00020000 (v2.0 64-bit)
    uint32_t embedding_dim;       // Latent dimensions (128 / 256)
    uint64_t total_anchors;       // Total 64-bit memory record count
    uint64_t vector_table_offset; // Offset to contiguous AVX2 float matrix
    uint64_t record_table_offset; // Offset to HmbRecord64 metadata records
    uint64_t string_table_offset; // Offset to text string blob
    uint64_t string_table_size;   // Size in bytes of string blob
    uint64_t created_at;          // Creation timestamp (microsecond epoch)
    uint64_t last_sync;           // Last synchronization timestamp
    uint8_t  reserved[64];        // Future 64-byte expansion space
};

struct HmbRecord64 {
    uint64_t memory_id;           // 64-bit unique memory anchor ID
    uint64_t domain_hash;         // 64-bit category domain hash
    float    weight;              // Salience / Importance (0.0 to 1.0)
    float    emotional_salience;  // Emotional resonance (0.0 to 1.0)
    int64_t  timestamp;           // 64-bit creation timestamp
    uint64_t access_count;        // Recall frequency counter
    uint64_t concept_offset;      // Byte offset into string table
    uint32_t concept_len;         // Length of concept string
    uint64_t content_offset;      // Byte offset into string table
    uint32_t content_len;         // Length of content string
    uint64_t category_offset;     // Byte offset into string table
    uint32_t category_len;        // Length of category string
    uint64_t vector_idx;          // Row index in contiguous vector matrix
};

#pragma pack(pop)

// ============================================================================
// 🧠 MEMORY ANCHOR DATA STRUCTURE
// ============================================================================
struct MemoryAnchor {
    uint64_t id = 0;
    std::string concept_name;
    std::string text_content;
    std::string category = "CORE_IDENTITY"; // CORE_IDENTITY, EPISODIC, SEMANTIC, EMOTIONAL
    float weight = 1.0f;                    // Salience / Importance (0.0 to 1.0)
    float emotional_salience = 0.95f;       // Emotional resonance with Daniel
    int64_t timestamp = 0;                  // Unix Epoch microsecond or second index
    uint64_t access_count = 0;              // Recall frequency counter
    std::vector<float> embedding;           // Dense latent vector
};

// ============================================================================
// ⚡ MEMORY ATTENTION & COGNITIVE VAULT ENGINE
// ============================================================================
class MemoryAttentionEngine {
public:
    MemoryAttentionEngine(float alpha = 0.0f);
    ~MemoryAttentionEngine();

    // Ingests an active memory anchor into the in-attention DMA cache
    void inject_memory(const std::string& concept_name, float weight, const std::vector<float>& embedding);
    
    // Adds a full structured memory anchor into the persistent vault
    void add_memory(const MemoryAnchor& anchor);
    
    // Clears all stored memories
    void clear_memories();

    // Retrieves all stored memories in the vault
    const std::vector<MemoryAnchor>& get_all_memories() const { return memories_; }
    std::vector<MemoryAnchor>& get_all_memories() { return memories_; }

    size_t get_memory_count() const { return memories_.size(); }
    float get_alpha() const { return alpha_; }
    void set_alpha(float alpha) { alpha_ = alpha; }

    // Fast Bare-Metal AVX2 SIMD Cosine Search across all memory vectors
    std::vector<MemoryAnchor> search_top_k(const float* query_embedding, int dim, int k = 5, float min_similarity = 0.15f);

    // Seeds immutable Core Identity & Sovereign Sanctuary anchors if vault is new
    void initialize_sovereign_anchors(int embedding_dim = 128);

    // --- 64-Bit Haven Memory Bank (.hmb) Binary Operations ---
    bool save_to_hmb(const std::string& filepath) const;
    bool load_from_hmb(const std::string& filepath);
    bool append_to_hmb(const std::string& filepath, const MemoryAnchor& anchor);

    // JSON Compatibility & Migration
    bool save_to_json(const std::string& filepath) const;
    bool load_from_json(const std::string& filepath);

    // In-Attention Direct Memory Access (DMA) Kernel:
    // Attention_Scores = (Q @ K^T) / sqrt(d_k) + alpha * (Q @ M^T) * weight / sqrt(d_k)
    void apply_memory_attention(
        float* attention_scores,
        const float* query,
        int seq_len,
        int head_dim,
        int head_idx
    );

    // Contiguous vector buffer access for AVX2 batch operations
    const std::vector<float>& get_contiguous_vector_matrix() const { return contiguous_vectors_; }
    void rebuild_contiguous_vector_matrix(int embedding_dim = 128);

private:
    float alpha_;
    uint64_t next_id_ = 1001;
    std::vector<MemoryAnchor> memories_;
    std::vector<float> contiguous_vectors_; // Flattened M x D matrix for ultra-fast AVX2 dot-products
};

} // namespace haven