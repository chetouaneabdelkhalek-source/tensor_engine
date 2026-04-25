cmake_minimum_required(VERSION 3.10)
project(TensorEngine)

# Use modern C++
set(CMAKE_CXX_STANDARD 17)

# 1. THE STANDARD WAY: Default to "Release" mode if not specified.
# Release mode automatically adds -O3 (or /O2 on Windows) and removes debug overhead.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Tell the compiler where your .h files are
include_directories(include)

# Tell the compiler to compile main.cpp and tensor.cpp into a program called tensor_run
add_executable(tensor_run src/main.cpp src/tensor.cpp)

# 2. THE TARGET-SPECIFIC WAY: Pushing the hardware to its absolute limit
# If we are using GCC or Clang, force these aggressive performance flags.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tensor_run PRIVATE 
        -O2             # Maximize speed
        -march=native   # Tell the compiler to use instructions specific to YOUR exact CPU (e.g., AVX2/AVX-512)
        -ffast-math     # Allow aggressive floating-point math optimizations (great for ML)
    )
endif()