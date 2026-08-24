#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>

namespace haven {

// Supported GGUF Tensor Quantization Types
enum class QuantType : uint32_t {
    F32  = 0,
    F16  = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8   = 16,
    I16  = 17,
    I32  = 18,
    COUNT
};

// 32-value quantized block for Q8_0
#pragma pack(push, 1)
struct block_q8_0 {
    uint16_t d;       // FP16 scale factor
    int8_t   qs[32];  // 32 int8 quantized weights
};
#pragma pack(pop)

// 256-value quantized block for Q4_K
#pragma pack(push, 1)
struct block_q4_K {
    uint16_t d;           // Super-block scale (FP16)
    uint16_t dmin;        // Super-block min (FP16)
    uint8_t  scales[12];  // 6-bit sub-block scales
    uint8_t  qs[128];     // 4-bit quantized weights (2 per byte)
};
#pragma pack(pop)

struct TensorDesc {
    std::string name;
    QuantType type = QuantType::F32;
    std::vector<uint64_t> shape; // [ne0, ne1, ne2, ne3]
    uint64_t offset = 0;         // Byte offset in GGUF file
    size_t size_bytes = 0;       // Total byte size
    const void* data = nullptr;  // Direct mmap pointer
};

struct ModelConfig {
    std::string architecture = "llama";
    uint32_t vocab_size = 32000;
    uint32_t embedding_dim = 4096;
    uint32_t hidden_dim = 11008;
    uint32_t num_layers = 32;
    uint32_t num_heads = 32;
    uint32_t num_kv_heads = 8;
    uint32_t head_dim = 128;
    uint32_t max_context_length = 131072;
    float rope_freq_base = 10000.0f;
    float rope_freq_scale = 1.0f;
    float rms_norm_eps = 1e-5f;
};

} // namespace haven