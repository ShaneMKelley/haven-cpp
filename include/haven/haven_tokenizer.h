#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace haven {

class HavenTokenizer {
public:
    HavenTokenizer();
    ~HavenTokenizer();

    // Initializes tokenizer vocabulary and fast reverse-lookup map
    void initialize(const std::vector<std::string>& vocab);

    // Encodes UTF-8 text into Gemma 4 token IDs
    std::vector<uint32_t> encode(const std::string& text, bool add_bos = false) const;

    // Decodes a single token ID into a UTF-8 string chunk
    std::string decode_token(uint32_t token_id) const;

    // Decodes a sequence of token IDs into a full UTF-8 string
    std::string decode(const std::vector<uint32_t>& tokens) const;

    // Formats Gemma 4 multi-turn conversation template
    std::string format_turn(const std::string& role, const std::string& content) const;

    uint32_t bos_token() const { return bos_token_; }
    uint32_t eos_token() const { return eos_token_; }
    uint32_t pad_token() const { return pad_token_; }

    size_t vocab_size() const { return vocab_.size(); }

private:
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, uint32_t> token_to_id_;
    uint32_t bos_token_ = 2;
    uint32_t eos_token_ = 1;
    uint32_t pad_token_ = 0;
    uint32_t unk_token_ = 3;
};

} // namespace haven
