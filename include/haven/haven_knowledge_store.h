#pragma once

#include "haven_types.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace haven {

struct KnowledgeEntry {
    uint32_t token_id;
    uint32_t cluster_id;
    float boost_magnitude; // Standard deviation boost in attention matrix
    std::string concept_label;
    std::string target_concept;
};

class StructuredKnowledgeStore {
public:
    StructuredKnowledgeStore();

    // Ingests a structured concept relation into the fast lookup graph
    void add_relation(
        uint32_t source_token_id,
        uint32_t cluster_id,
        float boost_magnitude,
        const std::string& concept_label,
        const std::string& target_concept
    );

    // Queries composite key (cluster_id << 32 | token_id) and applies attention score boost before Softmax
    void apply_knowledge_boost(
        float* attention_scores,
        uint32_t current_token_id,
        uint32_t active_cluster_id,
        int seq_len
    );

    size_t get_relation_count() const { return lookup_table_.size(); }
    void clear() { lookup_table_.clear(); }

private:
    static inline uint64_t make_composite_key(uint32_t token_id, uint32_t cluster_id) {
        return ((uint64_t)cluster_id << 32) | (uint64_t)token_id;
    }

    std::unordered_map<uint64_t, KnowledgeEntry> lookup_table_;
};

} // namespace haven