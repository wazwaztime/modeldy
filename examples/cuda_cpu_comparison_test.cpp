/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <include/model.h>
#include <include/optimizer.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

#ifdef USE_CUDA
#include <include/cuda/cuda_check.h>
#endif

using namespace modeldy;

// Helper function to compare floating point values
template <typename T>
bool isClose(T a, T b, T rtol = 1e-4, T atol = 1e-5) {
    return std::abs(a - b) <= (atol + rtol * std::abs(b));
}

// Test basic operations: Add and Mul
template <typename T>
bool testBasicOps() {
    std::cout << "\n=== Testing Basic Operations (Add, Mul) ===" << std::endl;
    
    const std::vector<size_t> shape = {2, 3};
    std::vector<T> data1 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<T> data2 = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5};
    
    // CPU Model
    Model<T> cpu_model;
    cpu_model.newDataNode("input1", shape, false, "cpu");
    cpu_model.newDataNode("input2", shape, false, "cpu");
    cpu_model.newDataNode("add_out", shape, false, "cpu");
    cpu_model.newDataNode("mul_out", shape, false, "cpu");
    
    cpu_model.setData("input1", data1);
    cpu_model.setData("input2", data2);
    
    cpu_model.newComputeNode("Add", "add_node", {"input1", "input2"}, {"add_out"}, "cpu");
    cpu_model.newComputeNode("Mul", "mul_node", {"input1", "input2"}, {"mul_out"}, "cpu");
    
    cpu_model.compile();
    cpu_model.predict();
    
#ifdef USE_CUDA
    // CUDA Model
    Model<T> cuda_model;
    cuda_model.newDataNode("input1", shape, false, "cuda");
    cuda_model.newDataNode("input2", shape, false, "cuda");
    cuda_model.newDataNode("add_out", shape, false, "cuda");
    cuda_model.newDataNode("mul_out", shape, false, "cuda");
    
    cuda_model.setData("input1", data1);
    cuda_model.setData("input2", data2);
    
    cuda_model.newComputeNode("Add", "add_node", {"input1", "input2"}, {"add_out"}, "cuda");
    cuda_model.newComputeNode("Mul", "mul_node", {"input1", "input2"}, {"mul_out"}, "cuda");
    
    cuda_model.compile();
    cuda_model.predict();
    
    // Compare results
    std::vector<T> cpu_add_result = cpu_model.getData("add_out");
    std::vector<T> cuda_add_result = cuda_model.getData("add_out");
    std::vector<T> cpu_mul_result = cpu_model.getData("mul_out");
    std::vector<T> cuda_mul_result = cuda_model.getData("mul_out");
    
    bool add_match = true;
    bool mul_match = true;
    
    std::cout << "Add Results:" << std::endl;
    for (size_t i = 0; i < cpu_add_result.size(); ++i) {
        if (!isClose(cpu_add_result[i], cuda_add_result[i])) {
            add_match = false;
            std::cout << "  Mismatch at index " << i << ": CPU=" << cpu_add_result[i] 
                      << " CUDA=" << cuda_add_result[i] << std::endl;
        }
    }
    
    std::cout << "Mul Results:" << std::endl;
    for (size_t i = 0; i < cpu_mul_result.size(); ++i) {
        if (!isClose(cpu_mul_result[i], cuda_mul_result[i])) {
            mul_match = false;
            std::cout << "  Mismatch at index " << i << ": CPU=" << cpu_mul_result[i] 
                      << " CUDA=" << cuda_mul_result[i] << std::endl;
        }
    }
    
    if (add_match && mul_match) {
        std::cout << "✓ Basic operations match!" << std::endl;
        return true;
    } else {
        std::cout << "✗ Basic operations mismatch!" << std::endl;
        return false;
    }
#else
    std::cout << "CUDA not enabled, skipping comparison" << std::endl;
    return true;
#endif
}

// Test activation functions
template <typename T>
bool testActivations() {
    std::cout << "\n=== Testing Activation Functions ===" << std::endl;
    
    const std::vector<size_t> shape = {2, 4};
    std::vector<T> data = {-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -0.5, 0.5};
    
    // CPU Model
    Model<T> cpu_model;
    cpu_model.newDataNode("input", shape, false, "cpu");
    cpu_model.newDataNode("relu_out", shape, false, "cpu");
    cpu_model.newDataNode("sigmoid_out", shape, false, "cpu");
    cpu_model.newDataNode("tanh_out", shape, false, "cpu");
    
    cpu_model.setData("input", data);
    
    cpu_model.newComputeNode("ReLU", "relu_node", {"input"}, {"relu_out"}, "cpu");
    cpu_model.newComputeNode("Sigmoid", "sigmoid_node", {"input"}, {"sigmoid_out"}, "cpu");
    cpu_model.newComputeNode("Tanh", "tanh_node", {"input"}, {"tanh_out"}, "cpu");
    
    cpu_model.compile();
    cpu_model.predict();
    
#ifdef USE_CUDA
    // CUDA Model
    Model<T> cuda_model;
    cuda_model.newDataNode("input", shape, false, "cuda");
    cuda_model.newDataNode("relu_out", shape, false, "cuda");
    cuda_model.newDataNode("sigmoid_out", shape, false, "cuda");
    cuda_model.newDataNode("tanh_out", shape, false, "cuda");
    
    cuda_model.setData("input", data);
    
    cuda_model.newComputeNode("ReLU", "relu_node", {"input"}, {"relu_out"}, "cuda");
    cuda_model.newComputeNode("Sigmoid", "sigmoid_node", {"input"}, {"sigmoid_out"}, "cuda");
    cuda_model.newComputeNode("Tanh", "tanh_node", {"input"}, {"tanh_out"}, "cuda");
    
    cuda_model.compile();
    cuda_model.predict();
    
    // Compare results
    auto cpu_relu = cpu_model.getData("relu_out");
    auto cuda_relu = cuda_model.getData("relu_out");
    auto cpu_sigmoid = cpu_model.getData("sigmoid_out");
    auto cuda_sigmoid = cuda_model.getData("sigmoid_out");
    auto cpu_tanh = cpu_model.getData("tanh_out");
    auto cuda_tanh = cuda_model.getData("tanh_out");
    
    bool all_match = true;
    
    std::cout << "ReLU Results:" << std::endl;
    for (size_t i = 0; i < cpu_relu.size(); ++i) {
        if (!isClose(cpu_relu[i], cuda_relu[i])) {
            all_match = false;
            std::cout << "  Mismatch at " << i << ": CPU=" << cpu_relu[i] 
                      << " CUDA=" << cuda_relu[i] << std::endl;
        }
    }
    
    std::cout << "Sigmoid Results:" << std::endl;
    for (size_t i = 0; i < cpu_sigmoid.size(); ++i) {
        if (!isClose(cpu_sigmoid[i], cuda_sigmoid[i])) {
            all_match = false;
            std::cout << "  Mismatch at " << i << ": CPU=" << cpu_sigmoid[i] 
                      << " CUDA=" << cuda_sigmoid[i] << std::endl;
        }
    }
    
    std::cout << "Tanh Results:" << std::endl;
    for (size_t i = 0; i < cpu_tanh.size(); ++i) {
        if (!isClose(cpu_tanh[i], cuda_tanh[i])) {
            all_match = false;
            std::cout << "  Mismatch at " << i << ": CPU=" << cpu_tanh[i] 
                      << " CUDA=" << cuda_tanh[i] << std::endl;
        }
    }
    
    if (all_match) {
        std::cout << "✓ Activation functions match!" << std::endl;
        return true;
    } else {
        std::cout << "✗ Activation functions mismatch!" << std::endl;
        return false;
    }
#else
    std::cout << "CUDA not enabled, skipping comparison" << std::endl;
    return true;
#endif
}

// Test loss functions and backward pass
template <typename T>
bool testLossAndBackward() {
    std::cout << "\n=== Testing Loss Functions and Backward Pass ===" << std::endl;
    
    const std::vector<size_t> shape = {4};
    std::vector<T> predictions = {0.7, 0.2, 0.8, 0.3};
    std::vector<T> targets = {1.0, 0.0, 1.0, 0.0};
    
    // CPU Model
    Model<T> cpu_model;
    cpu_model.newDataNode("pred", shape, true, "cpu");
    cpu_model.newDataNode("target", shape, false, "cpu");
    cpu_model.newDataNode("loss", {1}, true, "cpu");  // Loss needs grad for backward
    
    cpu_model.setData("pred", predictions);
    cpu_model.setData("target", targets);
    
    cpu_model.newComputeNode("MSELoss", "loss_node", {"pred", "target"}, {"loss"}, "cpu");
    
    cpu_model.compile();
    cpu_model.predict();
    cpu_model.initGrad("loss", 1.0);
    cpu_model.backward("loss");
    
#ifdef USE_CUDA
    // CUDA Model
    std::cout << "Creating CUDA model..." << std::endl;
    Model<T> cuda_model;
    std::cout << "Creating CUDA data nodes..." << std::endl;
    cuda_model.newDataNode("pred", shape, true, "cuda");
    cuda_model.newDataNode("target", shape, false, "cuda");
    cuda_model.newDataNode("loss", {1}, true, "cuda");  // Loss needs grad for backward
    
    std::cout << "Setting CUDA model data..." << std::endl;
    cuda_model.setData("pred", predictions);
    cuda_model.setData("target", targets);
    
    std::cout << "Creating CUDA compute node..." << std::endl;
    cuda_model.newComputeNode("MSELoss", "loss_node", {"pred", "target"}, {"loss"}, "cuda");
    
    std::cout << "Compiling CUDA model..." << std::endl;
    cuda_model.compile();
    std::cout << "Running CUDA predict..." << std::endl;
    cuda_model.predict();
    std::cout << "Initializing CUDA gradient..." << std::endl;
    cuda_model.initGrad("loss", 1.0);
    std::cout << "Running CUDA backward..." << std::endl;
    cuda_model.backward("loss");
    
    // Compare forward pass
    auto cpu_loss = cpu_model.getData("loss");
    auto cuda_loss = cuda_model.getData("loss");
    
    std::cout << "Loss: CPU=" << cpu_loss[0] << " CUDA=" << cuda_loss[0] << std::endl;
    
    bool loss_match = isClose(cpu_loss[0], cuda_loss[0]);
    
    // Compare gradients
    auto cpu_grad = cpu_model.getGrad("pred");
    auto cuda_grad = cuda_model.getGrad("pred");
    
    bool grad_match = true;
    std::cout << "Gradients:" << std::endl;
    for (size_t i = 0; i < cpu_grad.size(); ++i) {
        std::cout << "  [" << i << "] CPU=" << cpu_grad[i] 
                  << " CUDA=" << cuda_grad[i] << std::endl;
        if (!isClose(cpu_grad[i], cuda_grad[i])) {
            grad_match = false;
        }
    }
    
    if (loss_match && grad_match) {
        std::cout << "✓ Loss and gradients match!" << std::endl;
        return true;
    } else {
        std::cout << "✗ Loss or gradients mismatch!" << std::endl;
        return false;
    }
#else
    std::cout << "CUDA not enabled, skipping comparison" << std::endl;
    return true;
#endif
}

// Test simple training loop
template <typename T>
bool testTraining() {
    std::cout << "\n=== Testing Training Loop ===" << std::endl;
    
    const std::vector<size_t> input_shape = {4};
    const std::vector<size_t> hidden_shape = {3};
    const std::vector<size_t> output_shape = {1};
    
    // Initialize weights
    std::vector<T> w1_data = {0.5, -0.3, 0.2, 0.4, -0.1, 0.6, 0.3, -0.2, 0.1, -0.4, 0.5, -0.3};
    std::vector<T> w2_data = {0.7, -0.5, 0.3};
    
    // Training data
    std::vector<T> input_data = {1.0, 2.0, -1.0, 0.5};
    std::vector<T> target_data = {0.8};
    
    // CPU Model
    Model<T> cpu_model;
    cpu_model.newDataNode("input", input_shape, false, "cpu");
    cpu_model.newDataNode("w1", {hidden_shape[0], input_shape[0]}, true, "cpu");
    cpu_model.newDataNode("hidden", hidden_shape, true, "cpu");
    cpu_model.newDataNode("relu_out", hidden_shape, true, "cpu");
    cpu_model.newDataNode("w2", {output_shape[0], hidden_shape[0]}, true, "cpu");
    cpu_model.newDataNode("output", output_shape, true, "cpu");
    cpu_model.newDataNode("target", output_shape, false, "cpu");
    cpu_model.newDataNode("loss", {1}, false, "cpu");
    
    cpu_model.setData("input", input_data);
    cpu_model.setData("w1", w1_data);
    cpu_model.setData("w2", w2_data);
    cpu_model.setData("target", target_data);
    
    // Note: Without GemmOO, we'll use simple operations for now
    // This is a simplified test
    
#ifdef USE_CUDA
    // CUDA Model
    Model<T> cuda_model;
    cuda_model.newDataNode("input", input_shape, false, "cuda");
    cuda_model.newDataNode("w1", {hidden_shape[0], input_shape[0]}, true, "cuda");
    cuda_model.newDataNode("hidden", hidden_shape, true, "cuda");
    cuda_model.newDataNode("relu_out", hidden_shape, true, "cuda");
    cuda_model.newDataNode("w2", {output_shape[0], hidden_shape[0]}, true, "cuda");
    cuda_model.newDataNode("output", output_shape, true, "cuda");
    cuda_model.newDataNode("target", output_shape, false, "cuda");
    cuda_model.newDataNode("loss", {1}, false, "cuda");
    
    cuda_model.setData("input", input_data);
    cuda_model.setData("w1", w1_data);
    cuda_model.setData("w2", w2_data);
    cuda_model.setData("target", target_data);
    
    std::cout << "✓ Training setup complete (full test requires GemmOO)" << std::endl;
    return true;
#else
    std::cout << "CUDA not enabled, skipping comparison" << std::endl;
    return true;
#endif
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CUDA vs CPU Model Comparison Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    
#ifdef USE_CUDA
    std::cout << "CUDA support: ENABLED" << std::endl;
#else
    std::cout << "CUDA support: DISABLED" << std::endl;
#endif
    
    bool all_passed = true;
    
    // Run tests with float
    std::cout << "\n\n########## Testing with float ##########" << std::endl;
    all_passed &= testBasicOps<float>();
    all_passed &= testActivations<float>();
    all_passed &= testLossAndBackward<float>();
    all_passed &= testTraining<float>();
    
    // Run tests with double
    std::cout << "\n\n########## Testing with double ##########" << std::endl;
    all_passed &= testBasicOps<double>();
    all_passed &= testActivations<double>();
    all_passed &= testLossAndBackward<double>();
    all_passed &= testTraining<double>();
    
    std::cout << "\n========================================" << std::endl;
    if (all_passed) {
        std::cout << "✓ All tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "✗ Some tests FAILED!" << std::endl;
        return 1;
    }
}
