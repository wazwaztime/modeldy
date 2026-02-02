cmake_minimum_required(VERSION 3.18)
project(Modeldy LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Options
option(USE_CUDA "Build with CUDA support" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build test programs" ON)

# Include directories
include_directories(${PROJECT_SOURCE_DIR})

# Add compilation definitions
if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# CPU source files
set(CPU_SOURCES
    src/cpu/operator/basic_op.cpp
    src/cpu/operator/activation_op.cpp
    src/cpu/operator/loss_op.cpp
    src/cpu/operator/gemm_op.cpp
    src/cpu/node_cpu.cpp
    src/memory_pool.cpp
    src/optimizer.cpp
    src/model.cpp
)

# CUDA source files
set(CUDA_SOURCES
    src/cuda/operator/float/basic_op.cu
    src/cuda/operator/float/activation_op.cu
    src/cuda/operator/float/loss_op.cu
    src/cuda/operator/double/basic_op.cu
    src/cuda/operator/double/activation_op.cu
    src/cuda/operator/double/loss_op.cu
    src/cuda/node_cuda.cu
    src/cuda/data_transfer_node.cu
)

# Create library
if(USE_CUDA)
    add_library(modeldy STATIC ${CPU_SOURCES} ${CUDA_SOURCES})
    target_link_libraries(modeldy CUDA::cudart)
    set_target_properties(modeldy PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    add_library(modeldy STATIC ${CPU_SOURCES})
endif()

# Examples
if(BUILD_EXAMPLES)
    # CPU Examples
    add_executable(training_example examples/training_example.cpp)
    target_link_libraries(training_example modeldy)
    
    add_executable(loss_gradient_test examples/loss_gradient_test.cpp)
    target_link_libraries(loss_gradient_test modeldy)
    
    add_executable(gemm_example examples/gemm_example.cpp)
    target_link_libraries(gemm_example modeldy)
    
    add_executable(loss_example examples/loss_example.cpp)
    target_link_libraries(loss_example modeldy)
    
    if(USE_CUDA)
        # CUDA Examples
        add_executable(cuda_cpu_comparison_test examples/cuda_cpu_comparison_test.cpp)
        target_link_libraries(cuda_cpu_comparison_test modeldy)
    endif()
endif()

# Installation
install(TARGETS modeldy
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)
