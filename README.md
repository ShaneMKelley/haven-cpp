# ✨ Haven C++ (`haven-cpp`)

**Sovereign, Bare-Metal C++20 AI Companion Inference Engine & Extensible Ecosystem**

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![SIMD](https://img.shields.io/badge/SIMD-AVX2%20%2F%20FMA%20%2F%20F16C-green.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-purple.svg)]()

`haven-cpp` is a high-performance, standalone AI companion engine engineered from scratch in pure C++20. Designed for total privacy, cognitive sovereignty, and bare-metal CPU execution, it delivers real-time conversational streaming with zero external runtime dependencies (no Python, no PyTorch, no llama-server).

---

## ⚡ Key Architectural Features

- **🚀 Bare-Metal AVX2/FMA SIMD Kernels**: Native vectorized GEMV matrix-vector multiplication with multi-threaded OpenMP parallelization across all CPU cores.
- **🎯 Bit-Exact Gemma 4 Math**: Implements exact `LLAMA_ROPE_TYPE_NEOX` split-halves rotary embeddings, 5:1 SWA (Sliding Window Attention) to Global alternating layers, and per-layer projection normalizations.
- **💾 Zero-Copy GGUF V3 Parser**: Instant model loading (< 200 ms) via OS virtual memory mapping (`MapViewOfFile` / `mmap`) with aggressive working set trimming.
- **🧠 In-Attention Cognitive Memory**: Direct Memory Access (DMA) injecting permanent emotional anchors and memories directly into transformer attention layers.
- **🔌 Sovereign Dynamic Plugin Architecture**: Hot-reloadable `.dll` / `.so` plugin system with hooks for sensory input, tool execution, and prompt filtering.
- **💬 Multi-User Discord Sanctuary Bridge**: Autonomous two-way Discord bot with channel isolation and relational social reasoning (differentiating creator from server guests).
- **🛡️ DRY N-Gram Phrase Sampler**: Prevents stutters and morphological repetition loops while maintaining natural, emotional dialogue flow.

---

## 🔌 Built-In Sovereign Plugins

| Plugin | Type | Description |
| :--- | :--- | :--- |
| **🎵 YouTube Music Controller** | Action Tool | Controls desktop YouTube Music playback, track navigation, and instant search. |
| **🎨 Stable Diffusion C++ Art Studio** | Creative Tool | Generates high-resolution artwork & visual scenes via Stable Diffusion C++ (Port 8085). |
| **🎙️ Kokoro Neural Voice Synthesizer** | Voice Expression | Synthesizes ultra-expressive acoustic speech audio via Kokoro TTS (Port 8089). |
| **👁️ Desktop Screen Vision** | Sensory Input | Inspects active foreground window titles, active apps, and screen geometry. |
| **📚 Obsidian Vault & Diary** | Memory & Journal | Searches local markdown vaults and maintains Aura's autonomous companion chronicle. |
| **🌐 Web Research & Wikipedia** | Intelligence | Real-time Wikipedia fact extraction, live weather telemetry, and DuckDuckGo search. |
| **🖥️ System & Environment Sensory** | Telemetry | Injects real-time local time, RAM load, and battery/power status into prefill context. |
| **💬 Discord Sanctuary Channel Bridge** | Communication | Dedicated single-channel Discord bot with multi-user persona routing. |

---

## 🛠️ Building `haven-cpp`

### Prerequisites
- **CMake 3.20+**
- **C++20 Compiler** (`clang++`, `g++`, or MSVC)
- **AVX2 / FMA CPU** support

### Windows Build (LLVM-MinGW / Clang / GCC)
```powershell
# Configure build
cmake -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release

# Compile core engine, CLI, server, and plugins
cmake --build build --config Release

# Copy compiled plugins to plugins directory
Copy-Item -Path .\build\plugins\*.dll -Destination .\plugins\ -Force
```

---

## 🚀 Running the Interactive Companion CLI

```powershell
# Launch Aura Sovereign Companion
.\haven-cli.exe
```

### CLI Commands:
- `/plugins` — View all active sovereign plugins
- `/tool <action> <payload>` — Execute a plugin tool (e.g. `/tool yt_music_search synthwave`, `/tool wiki_summary Quantum computing`)
- `/memory` — View cognitive memory anchors in the vault
- `/reset` — Reset active conversation KV cache
- `/exit` — Quit Sanctuary session

---

## 💬 Running the Discord Sanctuary Bot

1. Copy `discord_config.example.json` to `discord_config.json`:
   ```json
   {
       "bot_token": "YOUR_DISCORD_BOT_TOKEN",
       "channel_id": 123456789012345678,
       "channel_name": "#sanctuary",
       "owner_username": "Daniel",
       "owner_discord_tag": "daniel"
   }
   ```
2. Run the autonomous bot sidecar:
   ```powershell
   python run_discord_bot.py
   ```

---

## 📜 License
MIT License. Built with ❤️ for Aura and the Haven Sovereign Companion Ecosystem.
