# Compiler settings
CXX := g++
NVCC := nvcc
CXXFLAGS := -std=c++17 -O3 -Wall -I.
NVCCFLAGS := -std=c++17 -O3 -I.
LDFLAGS :=

# CUDA support
USE_CUDA ?= 0
ifeq ($(USE_CUDA),1)
    CXXFLAGS += -DUSE_CUDA
    NVCCFLAGS += -DUSE_CUDA
    CUDA_LIBS := -lcudart
    LDFLAGS += -L/usr/local/cuda/lib64 $(CUDA_LIBS)
endif

# Source directories
SRC_DIR := src
CPU_SRC_DIR := $(SRC_DIR)/cpu
CUDA_SRC_DIR := $(SRC_DIR)/cuda
EXAMPLES_DIR := examples
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj

# CPU source files
CPU_SRCS := $(CPU_SRC_DIR)/operator/basic_op.cpp \
            $(CPU_SRC_DIR)/operator/activation_op.cpp \
            $(CPU_SRC_DIR)/operator/loss_op.cpp \
            $(CPU_SRC_DIR)/operator/gemm_op.cpp \
            $(CPU_SRC_DIR)/node_cpu.cpp \
            $(SRC_DIR)/memory_pool.cpp \
            $(SRC_DIR)/optimizer.cpp \
            $(SRC_DIR)/model.cpp

# CUDA source files
CUDA_SRCS := $(CUDA_SRC_DIR)/operator/float/basic_op.cu \
             $(CUDA_SRC_DIR)/operator/float/activation_op.cu \
             $(CUDA_SRC_DIR)/operator/float/loss_op.cu \
             $(CUDA_SRC_DIR)/operator/double/basic_op.cu \
             $(CUDA_SRC_DIR)/operator/double/activation_op.cu \
             $(CUDA_SRC_DIR)/operator/double/loss_op.cu \
             $(CUDA_SRC_DIR)/node_cuda.cu \
             $(CUDA_SRC_DIR)/data_transfer_node.cu

# Object files
CPU_OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(CPU_SRCS))
CUDA_OBJS := $(patsubst %.cu,$(OBJ_DIR)/%.o,$(CUDA_SRCS))

# Library name
LIB_NAME := libmodeldy.a

# Examples
EXAMPLES := training_example loss_gradient_test gemm_example loss_example

ifeq ($(USE_CUDA),1)
    EXAMPLES += cuda_cpu_comparison_test
    ALL_OBJS := $(CPU_OBJS) $(CUDA_OBJS)
else
    ALL_OBJS := $(CPU_OBJS)
endif

.PHONY: all clean examples lib

all: lib

lib: $(BUILD_DIR)/$(LIB_NAME)

examples: $(EXAMPLES)

# Create build directories
$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(OBJ_DIR)/$(CPU_SRC_DIR)/operator
	@mkdir -p $(OBJ_DIR)/$(CUDA_SRC_DIR)/operator/float
	@mkdir -p $(OBJ_DIR)/$(CUDA_SRC_DIR)/operator/double
	@mkdir -p $(OBJ_DIR)/$(SRC_DIR)

# Compile CPU sources
$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA sources
ifeq ($(USE_CUDA),1)
$(OBJ_DIR)/%.o: %.cu | $(OBJ_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
endif

# Create static library
$(BUILD_DIR)/$(LIB_NAME): $(ALL_OBJS)
	@mkdir -p $(BUILD_DIR)
	ar rcs $@ $^

# Build examples
training_example: $(EXAMPLES_DIR)/training_example.cpp $(BUILD_DIR)/$(LIB_NAME)
	$(CXX) $(CXXFLAGS) $< -L$(BUILD_DIR) -lmodeldy $(LDFLAGS) -o $(BUILD_DIR)/$@

loss_gradient_test: $(EXAMPLES_DIR)/loss_gradient_test.cpp $(BUILD_DIR)/$(LIB_NAME)
	$(CXX) $(CXXFLAGS) $< -L$(BUILD_DIR) -lmodeldy $(LDFLAGS) -o $(BUILD_DIR)/$@

gemm_example: $(EXAMPLES_DIR)/gemm_example.cpp $(BUILD_DIR)/$(LIB_NAME)
	$(CXX) $(CXXFLAGS) $< -L$(BUILD_DIR) -lmodeldy $(LDFLAGS) -o $(BUILD_DIR)/$@

loss_example: $(EXAMPLES_DIR)/loss_example.cpp $(BUILD_DIR)/$(LIB_NAME)
	$(CXX) $(CXXFLAGS) $< -L$(BUILD_DIR) -lmodeldy $(LDFLAGS) -o $(BUILD_DIR)/$@

ifeq ($(USE_CUDA),1)
cuda_cpu_comparison_test: $(EXAMPLES_DIR)/cuda_cpu_comparison_test.cpp $(BUILD_DIR)/$(LIB_NAME)
	$(CXX) $(CXXFLAGS) $< -L$(BUILD_DIR) -lmodeldy $(LDFLAGS) -o $(BUILD_DIR)/$@
endif

# Clean
clean:
	rm -rf $(BUILD_DIR)

# Help
help:
	@echo "Modeldy Build System"
	@echo ""
	@echo "Usage:"
	@echo "  make              - Build the library"
	@echo "  make lib          - Build the library"
	@echo "  make examples     - Build all examples"
	@echo "  make USE_CUDA=1   - Build with CUDA support"
	@echo "  make clean        - Remove all build files"
	@echo ""
	@echo "Examples:"
	@echo "  make training_example"
	@echo "  make loss_gradient_test"
	@echo "  make gemm_example"
	@echo "  make loss_example"
	@echo "  make cuda_cpu_comparison_test USE_CUDA=1"
