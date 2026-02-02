@echo off
REM Build script for Modeldy with CUDA support

echo ========================================
echo Building Modeldy with CUDA support
echo ========================================

REM Create build directories
if not exist build mkdir build
if not exist build\obj mkdir build\obj
if not exist build\obj\cpu mkdir build\obj\cpu
if not exist build\obj\cuda mkdir build\obj\cuda

REM Set CUDA paths
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set CUDA_INC=%CUDA_PATH%\include
set CUDA_LIB=%CUDA_PATH%\lib\x64

echo.
echo [1/4] Compiling CPU source files with nvcc...
nvcc -std=c++17 -O2 -I. -DUSE_CUDA -c src/cpu/operator/basic_op.cpp -o build/obj/cpu/basic_op.o
nvcc -std=c++17 -O2 -I. -DUSE_CUDA -c src/cpu/operator/activation_op.cpp -o build/obj/cpu/activation_op.o
nvcc -std=c++17 -O2 -I. -DUSE_CUDA -c src/cpu/operator/loss_op.cpp -o build/obj/cpu/loss_op.o
nvcc -std=c++17 -O2 -I. -DUSE_CUDA -c src/cpu/operator/gemm_op.cpp -o build/obj/cpu/gemm_op.o
nvcc -std=c++17 -O2 -I. -DUSE_CUDA -c src/cpu/node_cpu.cpp -o build/obj/cpu/node_cpu.o
nvcc -std=c++17 -O2 -I. -DUSE_CUDA -c src/memory_pool.cpp -o build/obj/memory_pool.o
nvcc -std=c++17 -O2 -I. -DUSE_CUDA -c src/optimizer.cpp -o build/obj/optimizer.o
nvcc -std=c++17 -O2 -I. -DUSE_CUDA -c src/model.cpp -o build/obj/model.o

if %errorlevel% neq 0 (
    echo Error: CPU compilation failed
    exit /b 1
)

echo.
echo [2/4] Compiling CUDA source files...
nvcc -std=c++17 -O2 -I. -DUSE_CUDA -c src/cuda/operator/float/basic_op.cu -o build/obj/cuda/basic_op_float.o
nvcc -std=c++17 -O2 -I. -DUSE_CUDA -c src/cuda/operator/double/basic_op.cu -o build/obj/cuda/basic_op_double.o
nvcc -std=c++17 -O2 -I. -DUSE_CUDA -c src/cuda/operator/float/activation_op.cu -o build/obj/cuda/activation_op_float.o
nvcc -std=c++17 -O2 -I. -DUSE_CUDA -c src/cuda/operator/double/activation_op.cu -o build/obj/cuda/activation_op_double.o
nvcc -std=c++17 -O2 -I. -DUSE_CUDA -c src/cuda/operator/float/loss_op.cu -o build/obj/cuda/loss_op_float.o
nvcc -std=c++17 -O2 -I. -DUSE_CUDA -c src/cuda/operator/double/loss_op.cu -o build/obj/cuda/loss_op_double.o
nvcc -std=c++17 -O2 -I. -DUSE_CUDA -c src/cuda/node_cuda.cu -o build/obj/cuda/node_cuda.o
nvcc -std=c++17 -O2 -I. -DUSE_CUDA -c src/cuda/data_transfer_node.cu -o build/obj/cuda/data_transfer_node.o
nvcc -std=c++17 -O2 -I. -DUSE_CUDA -c src/cuda/optimizer_kernels.cu -o build/obj/cuda/optimizer_kernels.o

if %errorlevel% neq 0 (
    echo Error: CUDA compilation failed
    exit /b 1
)

echo.
echo [3/4] Compiling test program...
nvcc -std=c++17 -O2 -I. -DUSE_CUDA -c examples/cuda_cpu_comparison_test.cpp -o build/obj/cuda_cpu_comparison_test.o

if %errorlevel% neq 0 (
    echo Error: Test compilation failed
    exit /b 1
)

echo.
echo [4/4] Linking...
nvcc -o build/cuda_cpu_test.exe ^
    build/obj/cuda_cpu_comparison_test.o ^
    build/obj/cpu/basic_op.o ^
    build/obj/cpu/activation_op.o ^
    build/obj/cpu/loss_op.o ^
    build/obj/cpu/gemm_op.o ^
    build/obj/cpu/node_cpu.o ^
    build/obj/memory_pool.o ^
    build/obj/optimizer.o ^
    build/obj/model.o ^
    build/obj/cuda/basic_op_float.o ^
    build/obj/cuda/basic_op_double.o ^
    build/obj/cuda/activation_op_float.o ^
    build/obj/cuda/activation_op_double.o ^
    build/obj/cuda/loss_op_float.o ^
    build/obj/cuda/loss_op_double.o ^
    build/obj/cuda/node_cuda.o ^
    build/obj/cuda/data_transfer_node.o ^
    build/obj/cuda/optimizer_kernels.o ^
    -lcudart

if %errorlevel% neq 0 (
    echo Error: Linking failed
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo Executable: build\cuda_cpu_test.exe
echo ========================================
echo.
echo Run with: build\cuda_cpu_test.exe
