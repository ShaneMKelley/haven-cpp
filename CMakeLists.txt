cmake_minimum_required(VERSION 3.20)
project(haven_cpp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable AVX2, FMA, F16C and Aggressive Optimization
if (MSVC)
    add_compile_options(/O2 /arch:AVX2 /W4)
else()
    add_compile_options(-O3 -mavx2 -mfma -mf16c -Wall -Wextra)
endif()

include_directories(include)

# Shared Library (haven_engine.dll / libhaven.so) for C# Gemmi & Android JNI
add_library(haven_engine SHARED
    src/haven_avx2.cpp
    src/haven_memory.cpp
    src/haven_sampler.cpp
    src/haven_gguf.cpp
    src/haven_engine.cpp
)

# Standalone CLI Executable & Benchmark Runner
add_executable(haven-cli
    src/main.cpp
    src/haven_avx2.cpp
    src/haven_memory.cpp
    src/haven_sampler.cpp
    src/haven_gguf.cpp
    src/haven_engine.cpp
)