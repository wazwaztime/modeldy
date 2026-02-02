/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

/*
 * Training example with optimizers
 * 
 * This demonstrates how to train a simple linear regression model
 * using different optimizers (SGD, Adam, RMSprop).
 */

#include <modeldy/include/model.h>
#include <modeldy/include/optimizer.h>
#include <modeldy/include/operator_registry.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>

// Simple linear regression: y = 2*x + 3 + noise
void generate_data(std::vector<float>& x, std::vector<float>& y, size_t n_samples) {
  std::random_device rd;
  std::mt19937 gen(42);  // Fixed seed for reproducibility
  std::normal_distribution<float> noise(0.0f, 0.1f);
  
  x.resize(n_samples);
  y.resize(n_samples);
  
  for (size_t i = 0; i < n_samples; ++i) {
    x[i] = static_cast<float>(i) / static_cast<float>(n_samples);
    y[i] = 2.0f * x[i] + 3.0f + noise(gen);
  }
}

void train_linear_regression_sgd() {
  std::cout << "=== Training Linear Regression with SGD ===" << std::endl;
  
  // Generate training data
  size_t n_samples = 100;
  std::vector<float> x_data, y_data;
  generate_data(x_data, y_data, n_samples);
  
  // Build model: y = w * x + b
  modeldy::Model<float> model;
  
  // Create nodes
  model.newDataNode("x", {n_samples}, false, "cpu");
  model.newDataNode("w", {1}, true, "cpu");  // Weight (trainable)
  model.newDataNode("b", {1}, true, "cpu");  // Bias (trainable)
  model.newDataNode("wx", {n_samples}, false, "cpu");
  model.newDataNode("predictions", {n_samples}, false, "cpu");
  model.newDataNode("targets", {n_samples}, false, "cpu");
  model.newDataNode("loss", {1}, true, "cpu");
  
  // Initialize parameters
  model.setData("w", {0.0f});  // Initialize weight to 0
  model.setData("b", {0.0f});  // Initialize bias to 0
  model.setData("targets", y_data);
  
  // Mark trainable parameters
  model.add_parameter("w");
  model.add_parameter("b");
  
  // Build computation graph: predictions = x * w + b
  // Note: This is simplified - in real implementation you'd need broadcast operations
  // For now, we'll manually update for each sample or use a simpler approach
  
  std::cout << "Note: Full matrix operations not implemented in this example." << std::endl;
  std::cout << "For complete training, implement broadcast and batch operations." << std::endl;
  std::cout << std::endl;
}

void train_simple_network() {
  std::cout << "=== Training Simple Network with Different Optimizers ===" << std::endl;
  
  // Simple example: learn to predict a constant
  // Input: [1.0], Target: [5.0]
  // Network: input -> (learned weight) -> output
  
  modeldy::Model<float> model;
  
  // Create nodes
  model.newDataNode("input", {1}, false, "cpu");
  model.newDataNode("weight", {1}, true, "cpu");
  model.newDataNode("output", {1}, true, "cpu");  // Need grad for backprop
  model.newDataNode("target", {1}, false, "cpu");
  model.newDataNode("loss", {1}, true, "cpu");
  
  // Set data
  model.setData("input", {1.0f});
  model.setData("weight", {0.1f});  // Initialize weight to 0.1
  model.setData("target", {5.0f});  // Target value
  
  // Mark trainable parameter
  model.add_parameter("weight");
  
  // Build graph: output = input * weight, loss = MSE(output, target)
  model.newComputeNode("Mul", "multiply", {"input", "weight"}, {"output"}, "cpu");
  model.newComputeNode("MSELoss", "mse", {"output", "target"}, {"loss"}, "cpu");
  
  // Test with SGD
  {
    std::cout << "\n--- Using SGD ---" << std::endl;
    modeldy::Model<float> test_model = model;  // Copy model
    test_model.setData("weight", {0.1f});  // Reset weight
    
    modeldy::SGD<float> optimizer(0.1f);  // Learning rate 0.1
    test_model.setup_optimizer(optimizer);
    
    std::cout << "Initial weight: 0.1, Target: 5.0" << std::endl;
    
    auto losses = test_model.train(optimizer, "loss", 20, 5);
    
    const float* final_weight = test_model.data("weight");
    std::cout << "Final weight: " << final_weight[0] << std::endl;
    std::cout << "Expected: ~5.0" << std::endl;
  }
  
  // Test with Adam
  {
    std::cout << "\n--- Using Adam ---" << std::endl;
    model.setData("weight", {0.1f});  // Reset weight
    
    modeldy::Adam<float> optimizer(0.1f);
    model.setup_optimizer(optimizer);
    
    std::cout << "Initial weight: 0.1, Target: 5.0" << std::endl;
    
    auto losses = model.train(optimizer, "loss", 20, 5);
    
    const float* final_weight = model.data("weight");
    std::cout << "Final weight: " << final_weight[0] << std::endl;
    std::cout << "Expected: ~5.0" << std::endl;
  }
  
  std::cout << std::endl;
}

void train_with_momentum() {
  std::cout << "=== Comparing SGD with and without Momentum ===" << std::endl;
  
  modeldy::Model<float> model;
  
  // Create a simple network
  model.newDataNode("x", {2}, false, "cpu");
  model.newDataNode("w", {2}, true, "cpu");
  model.newDataNode("pred", {2}, true, "cpu");  // Need grad
  model.newDataNode("target", {2}, false, "cpu");
  model.newDataNode("loss", {1}, true, "cpu");
  
  // Set data: trying to learn w=[2, 3] to minimize loss
  model.setData("x", {1.0f, 1.0f});
  model.setData("w", {0.5f, 0.5f});
  model.setData("target", {2.0f, 3.0f});
  
  model.add_parameter("w");
  
  // Build graph
  model.newComputeNode("Mul", "multiply", {"x", "w"}, {"pred"}, "cpu");
  model.newComputeNode("MSELoss", "mse", {"pred", "target"}, {"loss"}, "cpu");
  
  // Test SGD without momentum
  {
    std::cout << "\n--- SGD (no momentum) ---" << std::endl;
    modeldy::Model<float> test_model = model;
    test_model.setData("w", {0.5f, 0.5f});
    
    modeldy::SGD<float> optimizer(0.1f, 0.0f);  // No momentum
    test_model.setup_optimizer(optimizer);
    
    auto losses = test_model.train(optimizer, "loss", 30, 10);
    
    const float* final_w = test_model.data("w");
    std::cout << "Final weights: [" << final_w[0] << ", " << final_w[1] << "]" << std::endl;
  }
  
  // Test SGD with momentum
  {
    std::cout << "\n--- SGD (momentum=0.9) ---" << std::endl;
    model.setData("w", {0.5f, 0.5f});
    
    modeldy::SGD<float> optimizer(0.1f, 0.9f);  // With momentum
    model.setup_optimizer(optimizer);
    
    auto losses = model.train(optimizer, "loss", 30, 10);
    
    const float* final_w = model.data("w");
    std::cout << "Final weights: [" << final_w[0] << ", " << final_w[1] << "]" << std::endl;
  }
  
  std::cout << std::endl;
}

void demonstrate_optimizer_features() {
  std::cout << "=== Demonstrating Optimizer Features ===" << std::endl;
  
  modeldy::Model<float> model;
  
  model.newDataNode("param", {3}, true, "cpu");
  model.newDataNode("output", {3}, true, "cpu");  // Need grad
  model.newDataNode("target", {3}, false, "cpu");
  model.newDataNode("loss", {1}, true, "cpu");
  
  model.setData("param", {1.0f, 2.0f, 3.0f});
  model.setData("target", {0.0f, 0.0f, 0.0f});
  
  model.add_parameter("param");
  
  // Simple identity: output = param (using ReLU as passthrough for positive values)
  model.newComputeNode("ReLU", "identity", {"param"}, {"output"}, "cpu");
  model.newComputeNode("MSELoss", "mse", {"output", "target"}, {"loss"}, "cpu");
  
  // Test weight decay
  {
    std::cout << "\n--- Testing Weight Decay (L2 Regularization) ---" << std::endl;
    model.setData("param", {1.0f, 2.0f, 3.0f});
    
    modeldy::SGD<float> optimizer(0.01f, 0.0f, 0.1f);  // weight_decay=0.1
    model.setup_optimizer(optimizer);
    
    std::cout << "Initial params: [1.0, 2.0, 3.0]" << std::endl;
    std::cout << "Weight decay should push params toward zero" << std::endl;
    
    model.train(optimizer, "loss", 50, 10);
    
    const float* final_params = model.data("param");
    std::cout << "Final params: [" << final_params[0] << ", " 
              << final_params[1] << ", " << final_params[2] << "]" << std::endl;
  }
  
  // Test learning rate adjustment
  {
    std::cout << "\n--- Testing Learning Rate Adjustment ---" << std::endl;
    model.setData("param", {1.0f, 1.0f, 1.0f});
    
    modeldy::Adam<float> optimizer(0.1f);
    model.setup_optimizer(optimizer);
    
    std::cout << "Initial learning rate: " << optimizer.learning_rate() << std::endl;
    
    model.train(optimizer, "loss", 10, 0);
    
    optimizer.set_learning_rate(0.01f);
    std::cout << "Reduced learning rate to: " << optimizer.learning_rate() << std::endl;
    
    model.train(optimizer, "loss", 10, 0);
  }
  
  std::cout << std::endl;
}

int main() {
  std::cout << std::fixed << std::setprecision(6);
  
  std::cout << "========================================" << std::endl;
  std::cout << "  Optimizer and Training Examples" << std::endl;
  std::cout << "========================================" << std::endl << std::endl;
  
  train_simple_network();
  train_with_momentum();
  demonstrate_optimizer_features();
  
  std::cout << "========================================" << std::endl;
  std::cout << "  Training examples completed!" << std::endl;
  std::cout << "========================================" << std::endl;
  
  return 0;
}
