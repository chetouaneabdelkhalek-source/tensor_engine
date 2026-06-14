cmake_minimum_required(VERSION 3.10)
project(TensorEngine)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(tensor_run src/main.cpp src/tensor.cpp)
target_include_directories(tensor_run PRIVATE include)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")

    # --- Release: full hardware optimization ---
    target_compile_options(tensor_run PRIVATE
        $<$<CONFIG:Release>:-march=native> 
        $<$<CONFIG:Release>:-ffast-math>
        $<$<CONFIG:Release>:-O3>
    )

    # --- Valgrind/Debug: no arch-specific instructions so Valgrind can execute them ---
    # Build with: cmake -DCMAKE_BUILD_TYPE=Debug ..
    target_compile_options(tensor_run PRIVATE
        $<$<CONFIG:Debug>:-O0>
        $<$<CONFIG:Debug>:-g>
    )

    # --- AddressSanitizer profile (preferred over Valgrind on Linux) ---
    # Build with: cmake -DCMAKE_BUILD_TYPE=Asan ..
    if(CMAKE_BUILD_TYPE STREQUAL "Asan")
        target_compile_options(tensor_run PRIVATE -O1 -g -fsanitize=address -fno-omit-frame-pointer)
        target_link_options(tensor_run PRIVATE -fsanitize=address)
    endif()

endif()