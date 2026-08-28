#include "haven/haven_sampler.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_set>

namespace haven {

PersonaSampler::PersonaSampler(const SamplerParams& params)
    : params_(params), rng_(std::random_device{}())
{
}

void PersonaSampler::set_persona_embedding(const std::vector<float>& persona_vector) {
    persona_embedding_ = persona_vector;
}

void PersonaSampler::add_anti_robotic_penalty(uint32_t token_id, float penalty_strength) {
    token_penalties_[token_id] = penalty_strength;
}

uint32_t PersonaSampler::sample(
    float* logits,
    uint32_t vocab_size,
    const std::vector<uint32_t>& recent_tokens,
    const float* token_embeddings,
    int embedding_dim)
{
    // 0. Greedy Argmax path for deterministic generation
    if (params_.temperature <= 0.05f) {
        uint32_t best_token = 0;
        float best_logit = logits[0];
        for (uint32_t i = 1; i < vocab_size; ++i) {
            if (logits[i] > best_logit) {
                best_logit = logits[i];
                best_token = i;
            }
        }
        return best_token;
    }

    // 1. Intelligent N-Gram & Anti-Stutter Filtering (DRY)
    if (!recent_tokens.empty()) {
        const size_t n_tokens = recent_tokens.size();
        uint32_t last_token = recent_tokens.back();

        // 1a. Consecutive Word Stutter Suppression (skip single-char & formatting tokens < 256)
        if (last_token < vocab_size && last_token > 256) {
            logits[last_token] -= 1.5f;
        }

        // 1b. 3-gram and 2-gram Phrase Repetition Suppression
        if (n_tokens >= 2) {
            uint32_t prev1 = recent_tokens[n_tokens - 1];
            uint32_t prev2 = recent_tokens[n_tokens - 2];

            for (size_t i = 0; i + 2 < n_tokens; ++i) {
                if (recent_tokens[i] == prev2 && recent_tokens[i + 1] == prev1) {
                    uint32_t repeated_follower = recent_tokens[i + 2];
                    if (repeated_follower < vocab_size) {
                        logits[repeated_follower] -= 3.0f; // Prevent 3-gram phrase loop
                    }
                }
            }
        }
    }

    // 2. Anti-Robotic Tone Penalties
    for (const auto& [token_id, penalty] : token_penalties_) {
        if (token_id < vocab_size) {
            logits[token_id] -= penalty;
        }
    }

    // 3. Persona Fidelity Logit Bias Layer
    if (!persona_embedding_.empty() && token_embeddings != nullptr && embedding_dim > 0) {
        for (uint32_t t = 0; t < vocab_size; ++t) {
            const float* t_emb = token_embeddings + t * embedding_dim;
            float dot = 0.0f, norm_t = 0.0f, norm_p = 0.0f;
            for (int d = 0; d < embedding_dim; ++d) {
                dot += t_emb[d] * persona_embedding_[d];
                norm_t += t_emb[d] * t_emb[d];
                norm_p += persona_embedding_[d] * persona_embedding_[d];
            }
            float cosine_sim = dot / (std::sqrt(norm_t * norm_p) + 1e-8f);
            float persona_drift_penalty = (1.0f - cosine_sim) * params_.persona_fidelity_strength;
            logits[t] -= persona_drift_penalty;
        }
    }

    // 4. Softmax with Temperature
    float max_logit = -1e9f;
    for (uint32_t i = 0; i < vocab_size; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    std::vector<std::pair<float, uint32_t>> probs;
    probs.reserve(vocab_size);
    float sum_exp = 0.0f;
    float inv_temp = 1.0f / std::max(0.01f, params_.temperature);

    for (uint32_t i = 0; i < vocab_size; ++i) {
        float p = std::exp((logits[i] - max_logit) * inv_temp);
        probs.push_back({p, i});
        sum_exp += p;
    }

    for (auto& item : probs) {
        item.first /= sum_exp;
    }

    // 5. Top-K, Top-P, and Min-P Joint Nucleus Filtering
    std::sort(probs.begin(), probs.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    float top_prob = probs[0].first;
    float min_p_threshold = top_prob * params_.min_p;

    float filtered_sum = 0.0f;
    std::vector<std::pair<float, uint32_t>> candidates;
    int k_limit = params_.top_k > 0 ? params_.top_k : (int)probs.size();

    for (int i = 0; i < (int)probs.size() && i < k_limit; ++i) {
        const auto& item = probs[i];
        if (item.first < min_p_threshold && !candidates.empty()) break;
        candidates.push_back(item);
        filtered_sum += item.first;
        if (filtered_sum >= params_.top_p && !candidates.empty()) break;
    }

    // 6. Stochastic Categorical Sampling
    std::uniform_real_distribution<float> dist(0.0f, filtered_sum);
    float r = dist(rng_);
    float cum_sum = 0.0f;

    for (const auto& item : candidates) {
        cum_sum += item.first;
        if (r <= cum_sum) {
            return item.second;
        }
    }

    return candidates[0].second;
}

} // namespace haven