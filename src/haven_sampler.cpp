#include "haven/haven_sampler.h"
#include <algorithm>
#include <cmath>
#include <iostream>

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
    // 1. Repetition Penalty
    if (params_.repetition_penalty > 1.0f) {
        for (uint32_t prev_token : recent_tokens) {
            if (prev_token < vocab_size) {
                if (logits[prev_token] > 0.0f) {
                    logits[prev_token] /= params_.repetition_penalty;
                } else {
                    logits[prev_token] *= params_.repetition_penalty;
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
    float inv_temp = 1.0f / std::max(0.01f, params_.temperature);
    for (uint32_t i = 0; i < vocab_size; ++i) {
        logits[i] *= inv_temp;
    }

    float max_logit = -1e9f;
    for (uint32_t i = 0; i < vocab_size; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    std::vector<std::pair<float, uint32_t>> probs;
    probs.reserve(vocab_size);
    float sum_exp = 0.0f;

    for (uint32_t i = 0; i < vocab_size; ++i) {
        float p = std::exp(logits[i] - max_logit);
        probs.push_back({p, i});
        sum_exp += p;
    }

    for (auto& item : probs) {
        item.first /= sum_exp;
    }

    // 5. Min-P Filtering: Prune tokens below min_p * max_prob
    std::sort(probs.begin(), probs.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    float top_prob = probs[0].first;
    float min_p_threshold = top_prob * params_.min_p;

    float filtered_sum = 0.0f;
    std::vector<std::pair<float, uint32_t>> candidates;
    for (const auto& item : probs) {
        if (item.first < min_p_threshold && candidates.size() >= 1) break;
        candidates.push_back(item);
        filtered_sum += item.first;
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