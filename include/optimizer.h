/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef MODELDY_INCLUDE_OPTIMIZER_H_
#define MODELDY_INCLUDE_OPTIMIZER_H_

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <cmath>
#include <stdexcept>

#include <include/node.h>

#ifdef USE_CUDA
#include <include/cuda/cuda_check.h>
#include <include/cuda/node_cuda.h>
#endif

namespace modeldy {

template <typename T>
class DataNode;

namespace cpu {
  template <typename T>
  class cpuDataNode;
}

#ifdef USE_CUDA
template <typename T>
class cudaDataNode;
#endif

/*!
 * \brief Parameter structure to hold trainable parameters
 */
template <typename T>
struct Parameter {
  std::shared_ptr<DataNode<T>> node;
  std::string name;
  bool is_cuda;  // Flag to indicate if parameter is on CUDA
  
  Parameter(std::shared_ptr<DataNode<T>> n, const std::string& nm, bool cuda = false)
      : node(n), name(nm), is_cuda(cuda) {}
};

/*!
 * \brief Base class for optimizers
 */
template <typename T>
class Optimizer {
 public:
  explicit Optimizer(T learning_rate = static_cast<T>(0.01))
      : learning_rate_(learning_rate) {}
  
  virtual ~Optimizer() = default;
  
  /*! \brief Add a parameter to optimize */
  void add_parameter(const std::shared_ptr<DataNode<T>>& param, const std::string& name) {
    // Check if it's a CUDA node
    bool is_cuda = false;
#ifdef USE_CUDA
    is_cuda = (std::dynamic_pointer_cast<cudaDataNode<T>>(param) != nullptr);
#endif
    parameters_.emplace_back(param, name, is_cuda);
  }
  
  /*! \brief Update all parameters based on gradients */
  virtual void step() = 0;
  
  /*! \brief Zero all gradients */
  virtual void zero_grad() {
    for (auto& param : parameters_) {
      T* grad_ptr = param.node->grad();
      size_t total_size = 1;
      for (const auto& dim : param.node->shape()) {
        total_size *= dim;
      }
      for (size_t i = 0; i < total_size; ++i) {
        grad_ptr[i] = static_cast<T>(0);
      }
    }
  }
  
  /*! \brief Get learning rate */
  T learning_rate() const { return learning_rate_; }
  
  /*! \brief Set learning rate */
  void set_learning_rate(T lr) { learning_rate_ = lr; }
  
  /*! \brief Get parameters */
  const std::vector<Parameter<T>>& parameters() const { return parameters_; }
  
 protected:
  T learning_rate_;
  std::vector<Parameter<T>> parameters_;
};

/*!
 * \brief Stochastic Gradient Descent (SGD) optimizer
 */
template <typename T>
class SGD : public Optimizer<T> {
 public:
  explicit SGD(T learning_rate = static_cast<T>(0.01),
               T momentum = static_cast<T>(0),
               T weight_decay = static_cast<T>(0))
      : Optimizer<T>(learning_rate),
        momentum_(momentum),
        weight_decay_(weight_decay) {}
  
  void step() override {
    for (auto& param : this->parameters_) {
      if (param.is_cuda) {
        step_cuda(param);
      } else {
        step_cpu(param);
      }
    }
  }
  
 private:
  T momentum_;
  T weight_decay_;
  std::unordered_map<std::string, std::vector<T>> velocity_cpu_;
  std::unordered_map<std::string, T*> velocity_cuda_;  // CUDA device pointers
  
  /*! \brief CPU implementation */
  void step_cpu(Parameter<T>& param) {
    T* data_ptr = param.node->data();
    T* grad_ptr = param.node->grad();
    
    size_t total_size = 1;
    for (const auto& dim : param.node->shape()) {
      total_size *= dim;
    }
    
    // Initialize velocity if using momentum
    if (momentum_ > static_cast<T>(0) && velocity_cpu_.find(param.name) == velocity_cpu_.end()) {
      velocity_cpu_[param.name] = std::vector<T>(total_size, static_cast<T>(0));
    }
    
    for (size_t i = 0; i < total_size; ++i) {
      T grad = grad_ptr[i];
      
      // Add weight decay (L2 regularization)
      if (weight_decay_ > static_cast<T>(0)) {
        grad += weight_decay_ * data_ptr[i];
      }
      
      // Apply momentum
      if (momentum_ > static_cast<T>(0)) {
        velocity_cpu_[param.name][i] = momentum_ * velocity_cpu_[param.name][i] + grad;
        grad = velocity_cpu_[param.name][i];
      }
      
      // Update parameter
      data_ptr[i] -= this->learning_rate_ * grad;
    }
  }
  
  /*! \brief CUDA implementation (to be implemented with CUDA kernels) */
  void step_cuda(Parameter<T>& param) {
#ifdef USE_CUDA
    T* data_ptr = param.node->data();  // Device pointer
    T* grad_ptr = param.node->grad();  // Device pointer
    
    size_t total_size = 1;
    for (const auto& dim : param.node->shape()) {
      total_size *= dim;
    }
    
    // Allocate velocity buffer if using momentum
    if (momentum_ > static_cast<T>(0) && velocity_cuda_.find(param.name) == velocity_cuda_.end()) {
      T* velocity_ptr;
      CUDA_CHECK(cudaMalloc(&velocity_ptr, total_size * sizeof(T)));
      CUDA_CHECK(cudaMemset(velocity_ptr, 0, total_size * sizeof(T)));
      velocity_cuda_[param.name] = velocity_ptr;
    }
    
    // Call CUDA kernel (to be implemented)
    if (momentum_ > static_cast<T>(0)) {
      cuda::sgd_momentum_kernel_launch(
          data_ptr,
          grad_ptr,
          velocity_cuda_[param.name],
          total_size,
          this->learning_rate_,
          momentum_,
          weight_decay_
      );
    } else {
      cuda::sgd_kernel_launch(
          data_ptr,
          grad_ptr,
          total_size,
          this->learning_rate_,
          weight_decay_
      );
    }
#else
    throw std::runtime_error("CUDA support not enabled. Compile with USE_CUDA flag.");
#endif
  }
};

/*!
 * \brief Adam optimizer
 */
template <typename T>
class Adam : public Optimizer<T> {
 public:
  explicit Adam(T learning_rate = static_cast<T>(0.001),
                T beta1 = static_cast<T>(0.9),
                T beta2 = static_cast<T>(0.999),
                T epsilon = static_cast<T>(1e-8),
                T weight_decay = static_cast<T>(0))
      : Optimizer<T>(learning_rate),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        weight_decay_(weight_decay),
        t_(0) {}
  
  void step() override {
    t_++;
    
    for (auto& param : this->parameters_) {
      if (param.is_cuda) {
        step_cuda(param);
      } else {
        step_cpu(param);
      }
    }
  }
  
 private:
  T beta1_;
  T beta2_;
  T epsilon_;
  T weight_decay_;
  size_t t_;
  std::unordered_map<std::string, std::vector<T>> m_cpu_;  // First moment (CPU)
  std::unordered_map<std::string, std::vector<T>> v_cpu_;  // Second moment (CPU)
  std::unordered_map<std::string, T*> m_cuda_;  // First moment (CUDA)
  std::unordered_map<std::string, T*> v_cuda_;  // Second moment (CUDA)
  
  /*! \brief CPU implementation */
  void step_cpu(Parameter<T>& param) {
    T* data_ptr = param.node->data();
    T* grad_ptr = param.node->grad();
    
    size_t total_size = 1;
    for (const auto& dim : param.node->shape()) {
      total_size *= dim;
    }
    
    // Initialize first and second moments
    if (m_cpu_.find(param.name) == m_cpu_.end()) {
      m_cpu_[param.name] = std::vector<T>(total_size, static_cast<T>(0));
      v_cpu_[param.name] = std::vector<T>(total_size, static_cast<T>(0));
    }
    
    for (size_t i = 0; i < total_size; ++i) {
      T grad = grad_ptr[i];
      
      // Add weight decay (L2 regularization)
      if (weight_decay_ > static_cast<T>(0)) {
        grad += weight_decay_ * data_ptr[i];
      }
      
      // Update biased first moment estimate
      m_cpu_[param.name][i] = beta1_ * m_cpu_[param.name][i] + (static_cast<T>(1) - beta1_) * grad;
      
      // Update biased second raw moment estimate
      v_cpu_[param.name][i] = beta2_ * v_cpu_[param.name][i] + (static_cast<T>(1) - beta2_) * grad * grad;
      
      // Compute bias-corrected first moment estimate
      T m_hat = m_cpu_[param.name][i] / (static_cast<T>(1) - std::pow(beta1_, t_));
      
      // Compute bias-corrected second raw moment estimate
      T v_hat = v_cpu_[param.name][i] / (static_cast<T>(1) - std::pow(beta2_, t_));
      
      // Update parameter
      data_ptr[i] -= this->learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
    }
  }
  
  /*! \brief CUDA implementation (to be implemented with CUDA kernels) */
  void step_cuda(Parameter<T>& param) {
#ifdef USE_CUDA
    T* data_ptr = param.node->data();  // Device pointer
    T* grad_ptr = param.node->grad();  // Device pointer
    
    size_t total_size = 1;
    for (const auto& dim : param.node->shape()) {
      total_size *= dim;
    }
    
    // Allocate moment buffers if not exist
    if (m_cuda_.find(param.name) == m_cuda_.end()) {
      T* m_ptr;
      T* v_ptr;
      CUDA_CHECK(cudaMalloc(&m_ptr, total_size * sizeof(T)));
      CUDA_CHECK(cudaMalloc(&v_ptr, total_size * sizeof(T)));
      CUDA_CHECK(cudaMemset(m_ptr, 0, total_size * sizeof(T)));
      CUDA_CHECK(cudaMemset(v_ptr, 0, total_size * sizeof(T)));
      m_cuda_[param.name] = m_ptr;
      v_cuda_[param.name] = v_ptr;
    }
    
    // Call CUDA kernel (to be implemented)
    cuda::adam_kernel_launch(
        data_ptr,
        grad_ptr,
        m_cuda_[param.name],
        v_cuda_[param.name],
        total_size,
        this->learning_rate_,
        beta1_,
        beta2_,
        epsilon_,
        weight_decay_,
        t_
    );
#else
    throw std::runtime_error("CUDA support not enabled. Compile with USE_CUDA flag.");
#endif
  }
};

/*!
 * \brief RMSprop optimizer
 */
template <typename T>
class RMSprop : public Optimizer<T> {
 public:
  explicit RMSprop(T learning_rate = static_cast<T>(0.01),
                   T alpha = static_cast<T>(0.99),
                   T epsilon = static_cast<T>(1e-8),
                   T weight_decay = static_cast<T>(0))
      : Optimizer<T>(learning_rate),
        alpha_(alpha),
        epsilon_(epsilon),
        weight_decay_(weight_decay) {}
  
  void step() override {
    for (auto& param : this->parameters_) {
      if (param.is_cuda) {
        step_cuda(param);
      } else {
        step_cpu(param);
      }
    }
  }
  
 private:
  T alpha_;
  T epsilon_;
  T weight_decay_;
  std::unordered_map<std::string, std::vector<T>> square_avg_cpu_;
  std::unordered_map<std::string, T*> square_avg_cuda_;
  
  /*! \brief CPU implementation */
  void step_cpu(Parameter<T>& param) {
    T* data_ptr = param.node->data();
    T* grad_ptr = param.node->grad();
    
    size_t total_size = 1;
    for (const auto& dim : param.node->shape()) {
      total_size *= dim;
    }
    
    // Initialize squared gradient average
    if (square_avg_cpu_.find(param.name) == square_avg_cpu_.end()) {
      square_avg_cpu_[param.name] = std::vector<T>(total_size, static_cast<T>(0));
    }
    
    for (size_t i = 0; i < total_size; ++i) {
      T grad = grad_ptr[i];
      
      // Add weight decay
      if (weight_decay_ > static_cast<T>(0)) {
        grad += weight_decay_ * data_ptr[i];
      }
      
      // Update squared gradient average
      square_avg_cpu_[param.name][i] = alpha_ * square_avg_cpu_[param.name][i] + 
                                        (static_cast<T>(1) - alpha_) * grad * grad;
      
      // Update parameter
      data_ptr[i] -= this->learning_rate_ * grad / 
                     (std::sqrt(square_avg_cpu_[param.name][i]) + epsilon_);
    }
  }
  
  /*! \brief CUDA implementation (to be implemented with CUDA kernels) */
  void step_cuda(Parameter<T>& param) {
#ifdef USE_CUDA
    T* data_ptr = param.node->data();  // Device pointer
    T* grad_ptr = param.node->grad();  // Device pointer
    
    size_t total_size = 1;
    for (const auto& dim : param.node->shape()) {
      total_size *= dim;
    }
    
    // Allocate squared average buffer if not exist
    if (square_avg_cuda_.find(param.name) == square_avg_cuda_.end()) {
      T* sq_avg_ptr;
      CUDA_CHECK(cudaMalloc(&sq_avg_ptr, total_size * sizeof(T)));
      CUDA_CHECK(cudaMemset(sq_avg_ptr, 0, total_size * sizeof(T)));
      square_avg_cuda_[param.name] = sq_avg_ptr;
    }
    
    // Call CUDA kernel (to be implemented)
    cuda::rmsprop_kernel_launch(
        data_ptr,
        grad_ptr,
        square_avg_cuda_[param.name],
        total_size,
        this->learning_rate_,
        alpha_,
        epsilon_,
        weight_decay_
    );
#else
    throw std::runtime_error("CUDA support not enabled. Compile with USE_CUDA flag.");
#endif
  }
};

} // namespace modeldy

#endif // MODELDY_INCLUDE_OPTIMIZER_H_
