/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef MODELDY_INCLUDE_CUDA_OPERATOR_LOSS_OP_H_
#define MODELDY_INCLUDE_CUDA_OPERATOR_LOSS_OP_H_

#include <include/cuda/node_cuda.h>
#include <include/cuda/cuda_math.h>
#include <include/cuda/cuda_check.h>
#include <cassert>
#include <vector>

namespace modeldy {

namespace cuda {

/*!
 * \brief Mean Squared Error (MSE) Loss Node
 * \details Computes MSE loss between predictions and targets on CUDA
 *          Loss = (1/N) * sum((pred - target)^2)
 *          Two inputs: [0] predictions, [1] targets
 *          One output: scalar loss value
 */
template <typename T>
class cudaMSELoss : public cudaComputeNode<T> {
 public:
  explicit cudaMSELoss(const std::vector<NodePtr<T>>& input,
                       const std::vector<NodePtr<T>>& output,
                       const std::string& name = "")
      : cudaComputeNode<T>(input, output, name),
        d_partial_sums_(nullptr),
        num_blocks_(0) {
    this->validate_shape();
  }

  ~cudaMSELoss() override {
    if (d_partial_sums_) {
      cudaFree(d_partial_sums_);
    }
  }

  /*! \brief validate the shape of the input and output */
  void validate_shape() const override {
    assert(this->inputs().size() == 2 && "MSELoss requires exactly two inputs (predictions and targets)");
    assert(this->outputs().size() == 1 && "MSELoss requires exactly one output (loss scalar)");
    
    const auto& pred_shape = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[0])->shape();
    const auto& target_shape = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[1])->shape();
    const auto& output_shape = std::dynamic_pointer_cast<cudaDataNode<T>>(this->outputs()[0])->shape();
    
    assert(pred_shape == target_shape && "Predictions and targets must have the same shape");
    assert(output_shape.size() == 1 && output_shape[0] == 1 && "Output must be a scalar (shape [1])");
  }

  /*! \brief forward computation */
  void forward() override {
    auto predictions = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[0]);
    auto targets = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[1]);
    auto output = std::dynamic_pointer_cast<cudaDataNode<T>>(this->outputs()[0]);
    
    T* pred_data = predictions->data();
    T* target_data = targets->data();
    T* loss_data = output->data();
    
    // Calculate total number of elements
    size_t total_size = 1;
    for (const auto& dim : predictions->shape()) {
      total_size *= dim;
    }
    
    // Allocate device memory for partial sums if needed
    const int block_size = 256;
    num_blocks_ = (total_size + block_size - 1) / block_size;
    
    if (!d_partial_sums_) {
      CUDA_CHECK(cudaMalloc(&d_partial_sums_, num_blocks_ * sizeof(T)));
    }
    
    // Launch kernel to compute partial sums
    size_t shared_mem_size = block_size * sizeof(T);
    mse_forward_kernel<<<num_blocks_, block_size, shared_mem_size>>>(
        pred_data, target_data, d_partial_sums_, total_size);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy partial sums to host and compute final loss
    std::vector<T> h_partial_sums(num_blocks_);
    CUDA_CHECK(cudaMemcpy(h_partial_sums.data(), d_partial_sums_, 
                          num_blocks_ * sizeof(T), cudaMemcpyDeviceToHost));
    
    T sum = 0;
    for (int i = 0; i < num_blocks_; ++i) {
      sum += h_partial_sums[i];
    }
    
    T final_loss = sum / static_cast<T>(total_size);
    CUDA_CHECK(cudaMemcpy(loss_data, &final_loss, sizeof(T), cudaMemcpyHostToDevice));
  }

  /*! \brief backward computation */
  void backward() override {
    auto predictions = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[0]);
    auto targets = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[1]);
    auto output = std::dynamic_pointer_cast<cudaDataNode<T>>(this->outputs()[0]);
    
    T* pred_data = predictions->data();
    T* target_data = targets->data();
    T* grad_output = output->grad();
    
    // Calculate total number of elements
    size_t total_size = 1;
    for (const auto& dim : predictions->shape()) {
      total_size *= dim;
    }
    
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    T factor = static_cast<T>(2) / static_cast<T>(total_size);
    
    if (predictions->requires_grad()) {
      T* grad_pred = predictions->grad();
      mse_backward_kernel<<<num_blocks, block_size>>>(
          pred_data, target_data, grad_output, grad_pred, total_size, factor);
      CUDA_CHECK(cudaGetLastError());
    }
  }

 private:
  T* d_partial_sums_;  // Device memory for partial sums
  int num_blocks_;
};

/*!
 * \brief Binary Cross Entropy Loss Node
 * \details Computes binary cross-entropy loss between predictions and targets on CUDA
 *          Loss = -(1/N) * sum(target * log(pred) + (1-target) * log(1-pred))
 *          Two inputs: [0] predictions (in range [0,1]), [1] targets (0 or 1)
 *          One output: scalar loss value
 */
template <typename T>
class cudaBCELoss : public cudaComputeNode<T> {
 public:
  explicit cudaBCELoss(const std::vector<NodePtr<T>>& input,
                       const std::vector<NodePtr<T>>& output,
                       const std::string& name = "",
                       T epsilon = static_cast<T>(1e-7))
      : cudaComputeNode<T>(input, output, name),
        epsilon_(epsilon),
        d_partial_sums_(nullptr),
        num_blocks_(0) {
    this->validate_shape();
  }

  ~cudaBCELoss() override {
    if (d_partial_sums_) {
      cudaFree(d_partial_sums_);
    }
  }

  /*! \brief validate the shape of the input and output */
  void validate_shape() const override {
    assert(this->inputs().size() == 2 && "BCELoss requires exactly two inputs (predictions and targets)");
    assert(this->outputs().size() == 1 && "BCELoss requires exactly one output (loss scalar)");
    
    const auto& pred_shape = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[0])->shape();
    const auto& target_shape = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[1])->shape();
    const auto& output_shape = std::dynamic_pointer_cast<cudaDataNode<T>>(this->outputs()[0])->shape();
    
    assert(pred_shape == target_shape && "Predictions and targets must have the same shape");
    assert(output_shape.size() == 1 && output_shape[0] == 1 && "Output must be a scalar (shape [1])");
  }

  /*! \brief forward computation */
  void forward() override {
    auto predictions = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[0]);
    auto targets = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[1]);
    auto output = std::dynamic_pointer_cast<cudaDataNode<T>>(this->outputs()[0]);
    
    T* pred_data = predictions->data();
    T* target_data = targets->data();
    T* loss_data = output->data();
    
    // Calculate total number of elements
    size_t total_size = 1;
    for (const auto& dim : predictions->shape()) {
      total_size *= dim;
    }
    
    // Allocate device memory for partial sums if needed
    const int block_size = 256;
    num_blocks_ = (total_size + block_size - 1) / block_size;
    
    if (!d_partial_sums_) {
      CUDA_CHECK(cudaMalloc(&d_partial_sums_, num_blocks_ * sizeof(T)));
    }
    
    // Launch kernel to compute partial sums
    size_t shared_mem_size = block_size * sizeof(T);
    bce_forward_kernel<<<num_blocks_, block_size, shared_mem_size>>>(
        pred_data, target_data, d_partial_sums_, total_size, epsilon_);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy partial sums to host and compute final loss
    std::vector<T> h_partial_sums(num_blocks_);
    CUDA_CHECK(cudaMemcpy(h_partial_sums.data(), d_partial_sums_, 
                          num_blocks_ * sizeof(T), cudaMemcpyDeviceToHost));
    
    T sum = 0;
    for (int i = 0; i < num_blocks_; ++i) {
      sum += h_partial_sums[i];
    }
    
    T final_loss = -sum / static_cast<T>(total_size);
    CUDA_CHECK(cudaMemcpy(loss_data, &final_loss, sizeof(T), cudaMemcpyHostToDevice));
  }

  /*! \brief backward computation */
  void backward() override {
    auto predictions = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[0]);
    auto targets = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[1]);
    auto output = std::dynamic_pointer_cast<cudaDataNode<T>>(this->outputs()[0]);
    
    T* pred_data = predictions->data();
    T* target_data = targets->data();
    T* grad_output = output->grad();
    
    // Calculate total number of elements
    size_t total_size = 1;
    for (const auto& dim : predictions->shape()) {
      total_size *= dim;
    }
    
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    T factor = static_cast<T>(1) / static_cast<T>(total_size);
    
    if (predictions->requires_grad()) {
      T* grad_pred = predictions->grad();
      bce_backward_kernel<<<num_blocks, block_size>>>(
          pred_data, target_data, grad_output, grad_pred, total_size, epsilon_, factor);
      CUDA_CHECK(cudaGetLastError());
    }
  }

 private:
  T epsilon_;  // Small value to prevent log(0)
  T* d_partial_sums_;  // Device memory for partial sums
  int num_blocks_;
};

/*!
 * \brief Cross Entropy Loss Node (for multi-class classification)
 * \details Computes cross-entropy loss with softmax on CUDA
 *          Loss = -(1/N) * sum(target * log(softmax(pred)))
 *          Two inputs: [0] logits (raw predictions), [1] targets (one-hot encoded)
 *          One output: scalar loss value
 */
template <typename T>
class cudaCrossEntropyLoss : public cudaComputeNode<T> {
 public:
  explicit cudaCrossEntropyLoss(const std::vector<NodePtr<T>>& input,
                                const std::vector<NodePtr<T>>& output,
                                const std::string& name = "",
                                T epsilon = static_cast<T>(1e-7))
      : cudaComputeNode<T>(input, output, name),
        epsilon_(epsilon) {
    this->validate_shape();
  }

  ~cudaCrossEntropyLoss() override = default;

  /*! \brief validate the shape of the input and output */
  void validate_shape() const override {
    assert(this->inputs().size() == 2 && "CrossEntropyLoss requires exactly two inputs (logits and targets)");
    assert(this->outputs().size() == 1 && "CrossEntropyLoss requires exactly one output (loss scalar)");
    
    const auto& logits_shape = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[0])->shape();
    const auto& target_shape = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[1])->shape();
    const auto& output_shape = std::dynamic_pointer_cast<cudaDataNode<T>>(this->outputs()[0])->shape();
    
    assert(logits_shape == target_shape && "Logits and targets must have the same shape");
    assert(logits_shape.size() >= 1 && "Logits must be at least 1D");
    assert(output_shape.size() == 1 && output_shape[0] == 1 && "Output must be a scalar (shape [1])");
  }

  /*! \brief forward computation */
  void forward() override {
    auto logits = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[0]);
    auto targets = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[1]);
    auto output = std::dynamic_pointer_cast<cudaDataNode<T>>(this->outputs()[0]);
    
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
    
    // Initialize loss to zero
    T zero = 0;
    CUDA_CHECK(cudaMemcpy(loss_data, &zero, sizeof(T), cudaMemcpyHostToDevice));
    
    // Launch kernel (one block per batch)
    cross_entropy_softmax_forward_kernel<<<batch_size, 1>>>(
        logits_data, target_data, loss_data, batch_size, num_classes);
    CUDA_CHECK(cudaGetLastError());
    
    // Divide by batch_size
    T h_loss;
    CUDA_CHECK(cudaMemcpy(&h_loss, loss_data, sizeof(T), cudaMemcpyDeviceToHost));
    h_loss /= static_cast<T>(batch_size);
    CUDA_CHECK(cudaMemcpy(loss_data, &h_loss, sizeof(T), cudaMemcpyHostToDevice));
  }

  /*! \brief backward computation */
  void backward() override {
    auto logits = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[0]);
    auto targets = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[1]);
    auto output = std::dynamic_pointer_cast<cudaDataNode<T>>(this->outputs()[0]);
    
    T* logits_data = logits->data();
    T* target_data = targets->data();
    T* grad_output = output->grad();
    
    const auto& shape = logits->shape();
    size_t batch_size = 1;
    size_t num_classes = shape[shape.size() - 1];
    
    for (size_t i = 0; i < shape.size() - 1; ++i) {
      batch_size *= shape[i];
    }
    
    if (logits->requires_grad()) {
      T* grad_logits = logits->grad();
      T factor = static_cast<T>(1) / static_cast<T>(batch_size);
      
      // Launch kernel (one block per batch, one thread per class)
      // Note: This assumes num_classes <= 1024 (max threads per block)
      cross_entropy_softmax_backward_kernel<<<batch_size, num_classes>>>(
          logits_data, target_data, grad_output, grad_logits, 
          batch_size, num_classes, factor);
      CUDA_CHECK(cudaGetLastError());
    }
  }

 private:
  T epsilon_;  // Small value to prevent log(0)
};

} // namespace cuda

} // namespace modeldy

#endif // MODELDY_INCLUDE_CUDA_OPERATOR_LOSS_OP_H_
