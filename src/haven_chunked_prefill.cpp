#include "haven/haven_chunked_prefill.h"
#include <algorithm>

namespace haven {

ChunkedPrefillScheduler::ChunkedPrefillScheduler(size_t chunk_size)
    : chunk_size_(chunk_size)
{
}

ChunkedPrefillScheduler::~ChunkedPrefillScheduler() = default;

uint64_t ChunkedPrefillScheduler::submit_prefill_task(const std::vector<uint32_t>& prompt_tokens) {
    PrefillTask task;
    task.task_id = next_task_id_++;
    task.tokens = prompt_tokens;
    task.processed_tokens = 0;
    task.is_complete = prompt_tokens.empty();

    pending_tasks_.push_back(task);
    return task.task_id;
}

bool ChunkedPrefillScheduler::step_prefill(
    const std::function<void(uint32_t token, size_t pos)>& on_token_step)
{
    if (pending_tasks_.empty()) return false;

    auto& current_task = pending_tasks_.front();
    size_t remaining = current_task.tokens.size() - current_task.processed_tokens;
    size_t slice = std::min(chunk_size_, remaining);

    for (size_t i = 0; i < slice; ++i) {
        size_t pos = current_task.processed_tokens + i;
        uint32_t token = current_task.tokens[pos];
        if (on_token_step) {
            on_token_step(token, pos);
        }
    }

    current_task.processed_tokens += slice;
    if (current_task.processed_tokens >= current_task.tokens.size()) {
        current_task.is_complete = true;
        pending_tasks_.erase(pending_tasks_.begin());
    }

    return true;
}

} // namespace haven
