#pragma once

#include "haven_types.h"
#include <vector>
#include <cstdint>
#include <functional>

namespace haven {

constexpr size_t CHUNK_PREFILL_SIZE = 128;

struct PrefillTask {
    uint64_t task_id = 0;
    std::vector<uint32_t> tokens;
    size_t processed_tokens = 0;
    bool is_complete = false;
};

class ChunkedPrefillScheduler {
public:
    ChunkedPrefillScheduler(size_t chunk_size = CHUNK_PREFILL_SIZE);
    ~ChunkedPrefillScheduler();

    // Enqueues a sequence for chunked prefill
    uint64_t submit_prefill_task(const std::vector<uint32_t>& prompt_tokens);

    // Executes next slice of prefill tasks (up to chunk_size tokens)
    // Invokes on_chunk_step for each token processed
    bool step_prefill(
        const std::function<void(uint32_t token, size_t pos)>& on_token_step
    );

    bool has_pending_tasks() const { return !pending_tasks_.empty(); }
    size_t get_pending_task_count() const { return pending_tasks_.size(); }

private:
    size_t chunk_size_;
    uint64_t next_task_id_ = 1;
    std::vector<PrefillTask> pending_tasks_;
};

} // namespace haven
