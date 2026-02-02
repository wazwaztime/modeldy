/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

/*
 * Simple gradient test for loss functions
 * 
 * This test verifies the correctness of backpropagation for loss functions
 * by comparing computed gradients with manually calculated expected values.
 */

#include <modeldy/include/model.h>
#include <modeldy/include/operator_registry.h>
#include <iostream>
#include <iomanip>
#include <cmath>

const float EPSILON = 1e-5f;

bool check_close(float a, float b, float tol = 1e-4f) {
  return std::abs(a - b) < tol;
}

void test_mse_loss() {
  std::cout << "=== Testing MSE Loss Gradient ===" << std::endl;
  
  modeldy::Model<float> model;
  
  // Create simple 2-element tensors for easy manual calculation
  model.newDataNode("predictions", {2}, true, "cpu");
  model.newDataNode("targets", {2}, false, "cpu");
  model.newDataNode("loss", {1}, true, "cpu");
  
  // Set simple values: pred=[3.0, 5.0], target=[1.0, 4.0]
  model.setData("predictions", {3.0f, 5.0f});
  model.setData("targets", {1.0f, 4.0f});
  
  model.newComputeNode("MSELoss", "mse_loss", 
                       {"predictions", "targets"}, 
                       {"loss"}, 
                       "cpu");
  
  // Forward pass
  model.predict();
  const float* loss_data = model.data("loss");
  
  // Expected loss: (1/2) * [(3-1)^2 + (5-4)^2] = (1/2) * [4 + 1] = 2.5
  float expected_loss = 2.5f;
  std::cout << "Computed loss: " << loss_data[0] << std::endl;
  std::cout << "Expected loss: " << expected_loss << std::endl;
  
  if (!check_close(loss_data[0], expected_loss)) {
    std::cout << "❌ Loss calculation FAILED!" << std::endl;
    return;
  }
  std::cout << "✓ Loss calculation passed" << std::endl;
  
  // Backward pass
  model.zeroGrad();
  model.backward("loss", 1.0f);
  
  const float* pred_grad = model.grad("predictions");
  
  // Expected gradient: dL/d(pred) = (2/N) * (pred - target)
  // For pred[0]: (2/2) * (3 - 1) = 1.0 * 2 = 2.0
  // For pred[1]: (2/2) * (5 - 4) = 1.0 * 1 = 1.0
  float expected_grad[2] = {2.0f, 1.0f};
  
  std::cout << "Computed gradients: [" << pred_grad[0] << ", " << pred_grad[1] << "]" << std::endl;
  std::cout << "Expected gradients: [" << expected_grad[0] << ", " << expected_grad[1] << "]" << std::endl;
  
  bool grad_correct = true;
  for (int i = 0; i < 2; ++i) {
    if (!check_close(pred_grad[i], expected_grad[i])) {
      std::cout << "❌ Gradient[" << i << "] mismatch!" << std::endl;
      grad_correct = false;
    }
  }
  
  if (grad_correct) {
    std::cout << "✓ MSE Loss gradient test PASSED!" << std::endl;
  } else {
    std::cout << "❌ MSE Loss gradient test FAILED!" << std::endl;
  }
  std::cout << std::endl;
}

void test_bce_loss() {
  std::cout << "=== Testing BCE Loss Gradient ===" << std::endl;
  
  modeldy::Model<float> model;
  
  model.newDataNode("predictions", {2}, true, "cpu");
  model.newDataNode("targets", {2}, false, "cpu");
  model.newDataNode("loss", {1}, true, "cpu");
  
  // Set values: pred=[0.7, 0.3], target=[1.0, 0.0]
  model.setData("predictions", {0.7f, 0.3f});
  model.setData("targets", {1.0f, 0.0f});
  
  model.newComputeNode("BCELoss", "bce_loss", 
                       {"predictions", "targets"}, 
                       {"loss"}, 
                       "cpu");
  
  // Forward pass
  model.predict();
  const float* loss_data = model.data("loss");
  
  // Expected loss: -(1/2) * [1*log(0.7) + (1-1)*log(1-0.7) + 0*log(0.3) + (1-0)*log(1-0.3)]
  //              = -(1/2) * [log(0.7) + log(0.7)]
  //              = -(1/2) * [log(0.7) + log(0.7)]
  float expected_loss = -(1.0f/2.0f) * (std::log(0.7f) + std::log(0.7f));
  std::cout << "Computed loss: " << loss_data[0] << std::endl;
  std::cout << "Expected loss: " << expected_loss << std::endl;
  
  if (!check_close(loss_data[0], expected_loss, 1e-3f)) {
    std::cout << "❌ Loss calculation FAILED!" << std::endl;
    return;
  }
  std::cout << "✓ Loss calculation passed" << std::endl;
  
  // Backward pass
  model.zeroGrad();
  model.backward("loss", 1.0f);
  
  const float* pred_grad = model.grad("predictions");
  
  // Expected gradient: dL/d(pred) = -(1/N) * (y/p - (1-y)/(1-p))
  // For pred[0]: -(1/2) * (1/0.7 - 0/0.3) = -(1/2) * (1.4286) = -0.7143
  // For pred[1]: -(1/2) * (0/0.3 - 1/0.7) = -(1/2) * (-1.4286) = 0.7143
  float expected_grad[2] = {
    -(1.0f/2.0f) * (1.0f/0.7f - 0.0f/0.3f),
    -(1.0f/2.0f) * (0.0f/0.3f - 1.0f/0.7f)
  };
  
  std::cout << "Computed gradients: [" << pred_grad[0] << ", " << pred_grad[1] << "]" << std::endl;
  std::cout << "Expected gradients: [" << expected_grad[0] << ", " << expected_grad[1] << "]" << std::endl;
  
  bool grad_correct = true;
  for (int i = 0; i < 2; ++i) {
    if (!check_close(pred_grad[i], expected_grad[i], 1e-3f)) {
      std::cout << "❌ Gradient[" << i << "] mismatch!" << std::endl;
      grad_correct = false;
    }
  }
  
  if (grad_correct) {
    std::cout << "✓ BCE Loss gradient test PASSED!" << std::endl;
  } else {
    std::cout << "❌ BCE Loss gradient test FAILED!" << std::endl;
  }
  std::cout << std::endl;
}

void test_cross_entropy_loss() {
  std::cout << "=== Testing Cross Entropy Loss Gradient ===" << std::endl;
  
  modeldy::Model<float> model;
  
  // Single sample with 3 classes
  model.newDataNode("logits", {3}, true, "cpu");
  model.newDataNode("targets", {3}, false, "cpu");
  model.newDataNode("loss", {1}, true, "cpu");
  
  // Set values: logits=[2.0, 1.0, 0.1], target=[1, 0, 0] (class 0)
  model.setData("logits", {2.0f, 1.0f, 0.1f});
  model.setData("targets", {1.0f, 0.0f, 0.0f});
  
  model.newComputeNode("CrossEntropyLoss", "ce_loss", 
                       {"logits", "targets"}, 
                       {"loss"}, 
                       "cpu");
  
  // Forward pass
  model.predict();
  const float* loss_data = model.data("loss");
  
  // Compute softmax manually
  float max_logit = 2.0f;
  float exp_sum = std::exp(2.0f - max_logit) + std::exp(1.0f - max_logit) + std::exp(0.1f - max_logit);
  float softmax[3] = {
    std::exp(2.0f - max_logit) / exp_sum,
    std::exp(1.0f - max_logit) / exp_sum,
    std::exp(0.1f - max_logit) / exp_sum
  };
  
  // Expected loss: -log(softmax[0]) since target is [1, 0, 0]
  float expected_loss = -std::log(softmax[0]);
  
  std::cout << "Softmax: [" << softmax[0] << ", " << softmax[1] << ", " << softmax[2] << "]" << std::endl;
  std::cout << "Computed loss: " << loss_data[0] << std::endl;
  std::cout << "Expected loss: " << expected_loss << std::endl;
  
  if (!check_close(loss_data[0], expected_loss, 1e-3f)) {
    std::cout << "❌ Loss calculation FAILED!" << std::endl;
    return;
  }
  std::cout << "✓ Loss calculation passed" << std::endl;
  
  // Backward pass
  model.zeroGrad();
  model.backward("loss", 1.0f);
  
  const float* logits_grad = model.grad("logits");
  
  // Expected gradient: dL/d(logits) = softmax - target
  float expected_grad[3] = {
    softmax[0] - 1.0f,  // softmax[0] - target[0]
    softmax[1] - 0.0f,  // softmax[1] - target[1]
    softmax[2] - 0.0f   // softmax[2] - target[2]
  };
  
  std::cout << "Computed gradients: [" << logits_grad[0] << ", " << logits_grad[1] << ", " << logits_grad[2] << "]" << std::endl;
  std::cout << "Expected gradients: [" << expected_grad[0] << ", " << expected_grad[1] << ", " << expected_grad[2] << "]" << std::endl;
  
  bool grad_correct = true;
  for (int i = 0; i < 3; ++i) {
    if (!check_close(logits_grad[i], expected_grad[i], 1e-3f)) {
      std::cout << "❌ Gradient[" << i << "] mismatch!" << std::endl;
      grad_correct = false;
    }
  }
  
  if (grad_correct) {
    std::cout << "✓ Cross Entropy Loss gradient test PASSED!" << std::endl;
  } else {
    std::cout << "❌ Cross Entropy Loss gradient test FAILED!" << std::endl;
  }
  std::cout << std::endl;
}

void test_loss_with_network() {
  std::cout << "=== Testing Loss with Simple Network ===" << std::endl;
  
  modeldy::Model<float> model;
  
  // Create a simple network: input -> relu -> output -> mse_loss
  model.newDataNode("input", {2}, true, "cpu");
  model.newDataNode("relu_out", {2}, true, "cpu");
  model.newDataNode("targets", {2}, false, "cpu");
  model.newDataNode("loss", {1}, true, "cpu");
  
  // Set input and targets
  model.setData("input", {-1.0f, 2.0f});
  model.setData("targets", {0.5f, 3.0f});
  
  // Create compute nodes
  model.newComputeNode("ReLU", "relu", {"input"}, {"relu_out"}, "cpu");
  model.newComputeNode("MSELoss", "mse_loss", {"relu_out", "targets"}, {"loss"}, "cpu");
  
  // Forward pass
  model.predict();
  
  const float* relu_data = model.data("relu_out");
  const float* loss_data = model.data("loss");
  
  // After ReLU: [-1, 2] -> [0, 2]
  // Loss: (1/2) * [(0-0.5)^2 + (2-3)^2] = (1/2) * [0.25 + 1] = 0.625
  float expected_relu[2] = {0.0f, 2.0f};
  float expected_loss = 0.625f;
  
  std::cout << "ReLU output: [" << relu_data[0] << ", " << relu_data[1] << "]" << std::endl;
  std::cout << "Expected: [" << expected_relu[0] << ", " << expected_relu[1] << "]" << std::endl;
  std::cout << "Loss: " << loss_data[0] << std::endl;
  std::cout << "Expected loss: " << expected_loss << std::endl;
  
  if (!check_close(loss_data[0], expected_loss)) {
    std::cout << "❌ Forward pass FAILED!" << std::endl;
    return;
  }
  std::cout << "✓ Forward pass passed" << std::endl;
  
  // Backward pass
  model.zeroGrad();
  model.backward();
  
  const float* input_grad = model.grad("input");
  const float* relu_grad = model.grad("relu_out");
  
  // Gradient at relu_out: dL/d(relu_out) = (2/N) * (relu_out - targets)
  //   = (2/2) * [(0-0.5), (2-3)] = [-0.5, -1.0]
  // Gradient at input: 
  //   For input[0]=-1: relu output is 0, so gradient is 0
  //   For input[1]=2: relu output is 2, so gradient passes through: -1.0
  float expected_relu_grad[2] = {-0.5f, -1.0f};
  float expected_input_grad[2] = {0.0f, -1.0f};  // ReLU blocks gradient for negative input
  
  std::cout << "ReLU gradients: [" << relu_grad[0] << ", " << relu_grad[1] << "]" << std::endl;
  std::cout << "Expected: [" << expected_relu_grad[0] << ", " << expected_relu_grad[1] << "]" << std::endl;
  std::cout << "Input gradients: [" << input_grad[0] << ", " << input_grad[1] << "]" << std::endl;
  std::cout << "Expected: [" << expected_input_grad[0] << ", " << expected_input_grad[1] << "]" << std::endl;
  
  bool grad_correct = true;
  for (int i = 0; i < 2; ++i) {
    if (!check_close(relu_grad[i], expected_relu_grad[i])) {
      std::cout << "❌ ReLU gradient[" << i << "] mismatch!" << std::endl;
      grad_correct = false;
    }
    if (!check_close(input_grad[i], expected_input_grad[i])) {
      std::cout << "❌ Input gradient[" << i << "] mismatch!" << std::endl;
      grad_correct = false;
    }
  }
  
  if (grad_correct) {
    std::cout << "✓ Network backpropagation test PASSED!" << std::endl;
  } else {
    std::cout << "❌ Network backpropagation test FAILED!" << std::endl;
  }
  std::cout << std::endl;
}

int main() {
  std::cout << std::fixed << std::setprecision(6);
  
  std::cout << "========================================" << std::endl;
  std::cout << "  Loss Function Gradient Tests" << std::endl;
  std::cout << "========================================" << std::endl << std::endl;
  
  test_mse_loss();
  test_bce_loss();
  test_cross_entropy_loss();
  test_loss_with_network();
  
  std::cout << "========================================" << std::endl;
  std::cout << "  All tests completed!" << std::endl;
  std::cout << "========================================" << std::endl;
  
  return 0;
}
