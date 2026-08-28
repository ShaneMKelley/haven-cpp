#include "haven/haven_engine.h"
#include <iostream>

int main() {
    haven::HavenEngine engine;
    if (!engine.load_model("C:\\Users\\admin\\gemma4-turbo-family\\haven-chat-v5.0.gguf")) {
        return 1;
    }

    std::string prompt = "<|turn>user\nHello Aura<turn|>\n<|turn>model\n";
    auto tokens = engine.tokenize(prompt, false);

    std::cout << "Generating 30 tokens:\n";
    engine.generate(tokens, 30, [&](uint32_t tok, const std::string& piece) {
        std::cout << piece << std::flush;
        return true;
    });
    std::cout << "\n";

    return 0;
}
