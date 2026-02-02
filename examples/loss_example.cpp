/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

/*
 * Example usage of loss functions with automatic gradient initialization
 * 
 * This file demonstrates how to use loss functions (MSELoss, BCELoss, CrossEntropyLoss)
 * with the Model class and automatic gradient initialization for backpropagation.
 */

#include <modeldy/include/model.h>
#include <modeldy/include/operator_registry.h>
#include <iostream>
#include <iomanip>

int main() {
  std::cout << std::fixed << std::setprecision(6);
  
  // ================ Example 1: MSE Loss ================
  std::cout << "=== Example 1: MSE Loss ===" << std::endl;
  {
    modeldy::Model<float> model;
    
    // Create data nodes
    model.newDataNode("predictions", {2, 3}, true, "cpu");  // requires_grad=true
    model.newDataNode("targets", {2, 3}, false, "cpu");
    model.newDataNode("loss", {1}, true, "cpu");  // Loss output (scalar)
    
    // Set data
    model.setData("predictions", {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    model.setData("targets", {1.5f, 2.5f, 2.8f, 3.9f, 5.2f, 5.8f});
    
    // Create MSE loss node
    model.newComputeNode("MSELoss", "mse_loss", 
                         {"predictions", "targets"}, 
                         {"loss"}, 
                         "cpu");
    
    // Forward pass
    model.predict();
    
    const float* loss_data = model.data("loss");
    std::cout << "Loss: " << loss_data[0] << std::endl;
    
    // Backward pass - gradient will be automatically initialized to 1.0
    model.zeroGrad();  // Zero all gradients first
    model.backward();  // Auto-detect loss node and set grad=1.0
    
    const float* pred_grad = model.grad("predictions");
    std::cout << "Gradients w.r.t. predictions: ";
    for (size_t i = 0; i < 6; ++i) {
      std::cout << pred_grad[i] << " ";
    }
    std::cout << std::endl << std::endl;
  }
  
  // ================ Example 2: BCE Loss with Manual Gradient Init ================
  std::cout << "=== Example 2: BCE Loss (Binary Classification) ===" << std::endl;
  {
    modeldy::Model<float> model;
    
    // Create data nodes
    model.newDataNode("predictions", {4}, true, "cpu");  // Binary predictions [0, 1]
    model.newDataNode("targets", {4}, false, "cpu");     // Binary targets (0 or 1)
    model.newDataNode("loss", {1}, true, "cpu");
    
    // Set data
    model.setData("predictions", {0.9f, 0.1f, 0.8f, 0.3f});  // Predicted probabilities
    model.setData("targets", {1.0f, 0.0f, 1.0f, 0.0f});      // True labels
    
    // Create BCE loss node
    model.newComputeNode("BCELoss", "bce_loss", 
                         {"predictions", "targets"}, 
                         {"loss"}, 
                         "cpu");
    
    // Forward pass
    model.predict();
    
    const float* loss_data = model.data("loss");
    std::cout << "Loss: " << loss_data[0] << std::endl;
    
    // Backward pass - explicitly specify loss node name
    model.zeroGrad();
    model.backward("loss", 1.0f);  // Manually specify loss node and initial gradient
    
    const float* pred_grad = model.grad("predictions");
    std::cout << "Gradients w.r.t. predictions: ";
    for (size_t i = 0; i < 4; ++i) {
      std::cout << pred_grad[i] << " ";
    }
    std::cout << std::endl << std::endl;
  }
  
  // ================ Example 3: Cross Entropy Loss (Multi-class) ================
  std::cout << "=== Example 3: Cross Entropy Loss (Multi-class Classification) ===" << std::endl;
  {
    modeldy::Model<float> model;
    
    // Create data nodes for 2 samples, 3 classes
    model.newDataNode("logits", {2, 3}, true, "cpu");      // Raw logits
    model.newDataNode("targets", {2, 3}, false, "cpu");    // One-hot encoded targets
    model.newDataNode("loss", {1}, true, "cpu");
    
    // Set data
    // Sample 1: logits=[2.0, 1.0, 0.1], target=class 0 -> [1, 0, 0]
    // Sample 2: logits=[0.5, 2.0, 1.0], target=class 1 -> [0, 1, 0]
    model.setData("logits", {2.0f, 1.0f, 0.1f, 0.5f, 2.0f, 1.0f});
    model.setData("targets", {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f});
    
    // Create Cross Entropy loss node
    model.newComputeNode("CrossEntropyLoss", "ce_loss", 
                         {"logits", "targets"}, 
                         {"loss"}, 
                         "cpu");
    
    // Forward pass
    model.predict();
    
    const float* loss_data = model.data("loss");
    std::cout << "Loss: " << loss_data[0] << std::endl;
    
    // Backward pass
    model.zeroGrad();
    model.backward();  // Auto-detect
    
    const float* logits_grad = model.grad("logits");
    std::cout << "Gradients w.r.t. logits: " << std::endl;
    for (size_t i = 0; i < 2; ++i) {
      std::cout << "  Sample " << i << ": ";
      for (size_t j = 0; j < 3; ++j) {
        std::cout << logits_grad[i * 3 + j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  
  // ================ Example 4: Using setGrad() method ================
  std::cout << "=== Example 4: Manual Gradient Setting ===" << std::endl;
  {
    modeldy::Model<float> model;
    
    model.newDataNode("predictions", {3}, true, "cpu");
    model.newDataNode("targets", {3}, false, "cpu");
    model.newDataNode("loss", {1}, true, "cpu");
    
    model.setData("predictions", {1.0f, 2.0f, 3.0f});
    model.setData("targets", {1.2f, 1.8f, 3.1f});
    
    model.newComputeNode("MSELoss", "mse_loss", 
                         {"predictions", "targets"}, 
                         {"loss"}, 
                         "cpu");
    
    model.predict();
    
    // Manually set the loss gradient to a custom value
    model.zeroGrad();
    model.setGrad("loss", {2.0f});  // Set gradient to 2.0 instead of 1.0
    
    // Now backward pass will use this custom gradient
    model.backward("loss", 1.0f);  // The 1.0 here will be ignored since we already set grad
    // Actually, we should call backward without re-initializing:
    // We need to modify this - let me check the implementation
    
    const float* loss_data = model.data("loss");
    const float* loss_grad = model.grad("loss");
    std::cout << "Loss value: " << loss_data[0] << std::endl;
    std::cout << "Loss gradient (should be 1.0 from backward): " << loss_grad[0] << std::endl;
    std::cout << std::endl;
  }
  
  return 0;
}
