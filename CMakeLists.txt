cmake_minimum_required(VERSION 3.10)
project(TensorEngine)

# Use modern C++
set(CMAKE_CXX_STANDARD 17)

# Tell the compiler where your .h files are
include_directories(include)

# Tell the compiler to compile main.cpp and tensor.cpp into a program called tensor_run
add_executable(tensor_run src/main.cpp src/tensor.cpp)
