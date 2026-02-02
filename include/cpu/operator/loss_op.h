/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef MODELDY_INCLUDE_CPU_OPERATOR_LOSS_OP_H_
#define MODELDY_INCLUDE_CPU_OPERATOR_LOSS_OP_H_

#include <modeldy/include/cpu/node_cpu.h>
#include <modeldy/include/cpu/cpu_math.h>

#include <cassert>
#include <cmath>

namespace modeldy {

namespace cpu {

/*!
 * \brief Mean Squared Error (MSE) Loss Node
 * \details Computes MSE loss between predictions and targets
 *          Loss = (1/N) * sum((pred - target)^2)
 *          Two inputs: [0] predictions, [1] targets
 *          One output: scalar loss value
 */
template <typename T>
class cpuMSELoss : public cpuComputeNode<T> {
 public:
  explicit cpuMSELoss(const std::vector<NodePtr<T>>& input,
                      const std::vector<NodePtr<T>>& output,
                      const std::string& name = "")
      : cpuComputeNode<T>(input, output, name) {
    this->validate_shape();
  }

  ~cpuMSELoss() override = default;

  /*! \brief validate the shape of the input and output */
  void validate_shape() const override {
    assert(this->inputs().size() == 2 && "MSELoss requires exactly two inputs (predictions and targets)");
    assert(this->outputs().size() == 1 && "MSELoss requires exactly one output (loss scalar)");
    
    const auto& pred_shape = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[0])->shape();
    const auto& target_shape = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[1])->shape();
    const auto& output_shape = std::dynamic_pointer_cast<cpuDataNode<T>>(this->outputs()[0])->shape();
    
    assert(pred_shape == target_shape && "Predictions and targets must have the same shape");
    assert(output_shape.size() == 1 && output_shape[0] == 1 && "Output must be a scalar (shape [1])");
  }

  /*! \brief forward computation */
  void forward() override {
    auto predictions = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[0]);
    auto targets = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[1]);
    auto output = std::dynamic_pointer_cast<cpuDataNode<T>>(this->outputs()[0]);
    
    T* pred_data = predictions->data();
    T* target_data = targets->data();
    T* loss_data = output->data();
    
    // Calculate total number of elements
    size_t total_size = 1;
    for (const auto& dim : predictions->shape()) {
      total_size *= dim;
    }
    
    // Compute MSE: (1/N) * sum((pred - target)^2)
    T sum_squared_error = static_cast<T>(0);
    for (size_t i = 0; i < total_size; ++i) {
      T diff = pred_data[i] - target_data[i];
      sum_squared_error += diff * diff;
    }
    
    loss_data[0] = sum_squared_error / static_cast<T>(total_size);
  }

  /*! \brief backward computation */
  void backward() override {
    // 对于损失函数，output的梯度初始为1（损失对自身的梯度）
    auto predictions = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[0]);
    auto targets = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[1]);
    auto output = std::dynamic_pointer_cast<cpuDataNode<T>>(this->outputs()[0]);
    
    T* pred_data = predictions->data();
    T* target_data = targets->data();
    T* grad_output = output->grad();  // This is typically 1.0 for loss
    
    // Calculate total number of elements
    size_t total_size = 1;
    for (const auto& dim : predictions->shape()) {
      total_size *= dim;
    }
    
    // dL/d(pred) = (2/N) * (pred - target) * grad_output
    if (predictions->requires_grad()) {
      T* grad_pred = predictions->grad();
      T factor = static_cast<T>(2) * grad_output[0] / static_cast<T>(total_size);
      for (size_t i = 0; i < total_size; ++i) {
        grad_pred[i] += factor * (pred_data[i] - target_data[i]);
      }
    }
    
    // Usually we don't need gradients for targets, but if required:
    if (targets->requires_grad()) {
      T* grad_target = targets->grad();
      T factor = static_cast<T>(2) * grad_output[0] / static_cast<T>(total_size);
      for (size_t i = 0; i < total_size; ++i) {
        grad_target[i] += factor * (target_data[i] - pred_data[i]);
      }
    }
  }
};

/*!
 * \brief Binary Cross Entropy Loss Node
 * \details Computes binary cross-entropy loss between predictions and targets
 *          Loss = -(1/N) * sum(target * log(pred) + (1-target) * log(1-pred))
 *          Two inputs: [0] predictions (in range [0,1]), [1] targets (0 or 1)
 *          One output: scalar loss value
 */
template <typename T>
class cpuBCELoss : public cpuComputeNode<T> {
 public:
  explicit cpuBCELoss(const std::vector<NodePtr<T>>& input,
                      const std::vector<NodePtr<T>>& output,
                      const std::string& name = "",
                      T epsilon = static_cast<T>(1e-7))
      : cpuComputeNode<T>(input, output, name),
        epsilon_(epsilon) {
    this->validate_shape();
  }

  ~cpuBCELoss() override = default;

  /*! \brief validate the shape of the input and output */
  void validate_shape() const override {
    assert(this->inputs().size() == 2 && "BCELoss requires exactly two inputs (predictions and targets)");
    assert(this->outputs().size() == 1 && "BCELoss requires exactly one output (loss scalar)");
    
    const auto& pred_shape = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[0])->shape();
    const auto& target_shape = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[1])->shape();
    const auto& output_shape = std::dynamic_pointer_cast<cpuDataNode<T>>(this->outputs()[0])->shape();
    
    assert(pred_shape == target_shape && "Predictions and targets must have the same shape");
    assert(output_shape.size() == 1 && output_shape[0] == 1 && "Output must be a scalar (shape [1])");
  }

  /*! \brief forward computation */
  void forward() override {
    auto predictions = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[0]);
    auto targets = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[1]);
    auto output = std::dynamic_pointer_cast<cpuDataNode<T>>(this->outputs()[0]);
    
    T* pred_data = predictions->data();
    T* target_data = targets->data();
    T* loss_data = output->data();
    
    // Calculate total number of elements
    size_t total_size = 1;
    for (const auto& dim : predictions->shape()) {
      total_size *= dim;
    }
    
    // Compute BCE: -(1/N) * sum(y*log(p) + (1-y)*log(1-p))
    T sum_loss = static_cast<T>(0);
    for (size_t i = 0; i < total_size; ++i) {
      // Clip predictions to avoid log(0)
      T pred = std::max(epsilon_, std::min(static_cast<T>(1) - epsilon_, pred_data[i]));
      sum_loss += target_data[i] * std::log(pred) + 
                  (static_cast<T>(1) - target_data[i]) * std::log(static_cast<T>(1) - pred);
    }
    
    loss_data[0] = -sum_loss / static_cast<T>(total_size);
  }

  /*! \brief backward computation */
  void backward() override {
    auto predictions = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[0]);
    auto targets = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[1]);
    auto output = std::dynamic_pointer_cast<cpuDataNode<T>>(this->outputs()[0]);
    
    T* pred_data = predictions->data();
    T* target_data = targets->data();
    T* grad_output = output->grad();  // Typically 1.0 for loss
    
    // Calculate total number of elements
    size_t total_size = 1;
    for (const auto& dim : predictions->shape()) {
      total_size *= dim;
    }
    
    // dL/d(pred) = -(1/N) * (y/p - (1-y)/(1-p)) * grad_output
    if (predictions->requires_grad()) {
      T* grad_pred = predictions->grad();
      T factor = grad_output[0] / static_cast<T>(total_size);
      for (size_t i = 0; i < total_size; ++i) {
        T pred = std::max(epsilon_, std::min(static_cast<T>(1) - epsilon_, pred_data[i]));
        grad_pred[i] += factor * (-(target_data[i] / pred) + 
                                   (static_cast<T>(1) - target_data[i]) / (static_cast<T>(1) - pred));
      }
    }
    
    // Usually we don't need gradients for targets
    if (targets->requires_grad()) {
      T* grad_target = targets->grad();
      T factor = grad_output[0] / static_cast<T>(total_size);
      for (size_t i = 0; i < total_size; ++i) {
        T pred = std::max(epsilon_, std::min(static_cast<T>(1) - epsilon_, pred_data[i]));
        grad_target[i] += factor * (-std::log(pred) + std::log(static_cast<T>(1) - pred));
      }
    }
  }

 private:
  T epsilon_;  // Small value to prevent log(0)
};

/*!
 * \brief Cross Entropy Loss Node (for multi-class classification)
 * \details Computes cross-entropy loss between predictions and targets
 *          Loss = -(1/N) * sum(target * log(softmax(pred)))
 *          Two inputs: [0] logits (raw predictions), [1] targets (one-hot encoded)
 *          One output: scalar loss value
 */
template <typename T>
class cpuCrossEntropyLoss : public cpuComputeNode<T> {
 public:
  explicit cpuCrossEntropyLoss(const std::vector<NodePtr<T>>& input,
                               const std::vector<NodePtr<T>>& output,
                               const std::string& name = "",
                               T epsilon = static_cast<T>(1e-7))
      : cpuComputeNode<T>(input, output, name),
        epsilon_(epsilon) {
    this->validate_shape();
  }

  ~cpuCrossEntropyLoss() override = default;

  /*! \brief validate the shape of the input and output */
  void validate_shape() const override {
    assert(this->inputs().size() == 2 && "CrossEntropyLoss requires exactly two inputs (logits and targets)");
    assert(this->outputs().size() == 1 && "CrossEntropyLoss requires exactly one output (loss scalar)");
    
    const auto& logits_shape = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[0])->shape();
    const auto& target_shape = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[1])->shape();
    const auto& output_shape = std::dynamic_pointer_cast<cpuDataNode<T>>(this->outputs()[0])->shape();
    
    assert(logits_shape == target_shape && "Logits and targets must have the same shape");
    assert(logits_shape.size() >= 1 && "Logits must be at least 1D");
    assert(output_shape.size() == 1 && output_shape[0] == 1 && "Output must be a scalar (shape [1])");
  }

  /*! \brief forward computation */
  void forward() override {
    auto logits = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[0]);
    auto targets = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[1]);
    auto output = std::dynamic_pointer_cast<cpuDataNode<T>>(this->outputs()[0]);
    
    T* logits_data = logits->data();
    T* target_data = targets->data();
    T* loss_data = output->data();
    
    const auto& shape = logits->shape();
    size_t batch_size = 1;
    size_t num_classes = shape[shape.size() - 1];
    
    // Calculate batch size (all dimensions except last)
    for (size_t i = 0; i < shape.size() - 1; ++i) {
      batch_size *= shape[i];
    }
    
    // Compute cross-entropy with softmax
    T total_loss = static_cast<T>(0);
    
    // Temporary storage for softmax probabilities
    std::vector<T> softmax_probs(num_classes);
    
    for (size_t b = 0; b < batch_size; ++b) {
      T* batch_logits = logits_data + b * num_classes;
      T* batch_targets = target_data + b * num_classes;
      
      // Find max for numerical stability
      T max_logit = batch_logits[0];
      for (size_t c = 1; c < num_classes; ++c) {
        max_logit = std::max(max_logit, batch_logits[c]);
      }
      
      // Compute softmax
      T sum_exp = static_cast<T>(0);
      for (size_t c = 0; c < num_classes; ++c) {
        softmax_probs[c] = std::exp(batch_logits[c] - max_logit);
        sum_exp += softmax_probs[c];
      }
      
      for (size_t c = 0; c < num_classes; ++c) {
        softmax_probs[c] /= sum_exp;
        softmax_probs[c] = std::max(epsilon_, softmax_probs[c]); // Prevent log(0)
      }
      
      // Compute loss for this batch
      for (size_t c = 0; c < num_classes; ++c) {
        total_loss += -batch_targets[c] * std::log(softmax_probs[c]);
      }
    }
    
    loss_data[0] = total_loss / static_cast<T>(batch_size);
  }

  /*! \brief backward computation */
  void backward() override {
    auto logits = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[0]);
    auto targets = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[1]);
    auto output = std::dynamic_pointer_cast<cpuDataNode<T>>(this->outputs()[0]);
    
    T* logits_data = logits->data();
    T* target_data = targets->data();
    T* grad_output = output->grad();
    
    const auto& shape = logits->shape();
    size_t batch_size = 1;
    size_t num_classes = shape[shape.size() - 1];
    
    for (size_t i = 0; i < shape.size() - 1; ++i) {
      batch_size *= shape[i];
    }
    
    // dL/d(logits) = (softmax(logits) - targets) / batch_size
    if (logits->requires_grad()) {
      T* grad_logits = logits->grad();
      std::vector<T> softmax_probs(num_classes);
      T factor = grad_output[0] / static_cast<T>(batch_size);
      
      for (size_t b = 0; b < batch_size; ++b) {
        T* batch_logits = logits_data + b * num_classes;
        T* batch_targets = target_data + b * num_classes;
        T* batch_grad = grad_logits + b * num_classes;
        
        // Compute softmax
        T max_logit = batch_logits[0];
        for (size_t c = 1; c < num_classes; ++c) {
          max_logit = std::max(max_logit, batch_logits[c]);
        }
        
        T sum_exp = static_cast<T>(0);
        for (size_t c = 0; c < num_classes; ++c) {
          softmax_probs[c] = std::exp(batch_logits[c] - max_logit);
          sum_exp += softmax_probs[c];
        }
        
        for (size_t c = 0; c < num_classes; ++c) {
          softmax_probs[c] /= sum_exp;
        }
        
        // Gradient: (softmax - target) * factor
        for (size_t c = 0; c < num_classes; ++c) {
          batch_grad[c] += factor * (softmax_probs[c] - batch_targets[c]);
        }
      }
    }
  }

 private:
  T epsilon_;  // Small value to prevent log(0)
};

} // namespace cpu

} // namespace modeldy

#include <modeldy/include/operator_registry.h>

REGISTER_OPERATOR(float, MSELoss, cpu);
REGISTER_OPERATOR(double, MSELoss, cpu);
REGISTER_OPERATOR(float, BCELoss, cpu);
REGISTER_OPERATOR(double, BCELoss, cpu);
REGISTER_OPERATOR(float, CrossEntropyLoss, cpu);
REGISTER_OPERATOR(double, CrossEntropyLoss, cpu);

#endif // MODELDY_INCLUDE_CPU_OPERATOR_LOSS_OP_H_
