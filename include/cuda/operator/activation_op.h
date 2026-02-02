/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef MODELDY_INCLUDE_CUDA_OPERATOR_ACTIVATION_OP_H_
#define MODELDY_INCLUDE_CUDA_OPERATOR_ACTIVATION_OP_H_

#include <include/cuda/node_cuda.h>
#include <include/cuda/cuda_math.h>
#include <include/cuda/cuda_check.h>
#include <cassert>

namespace modeldy {

namespace cuda {

template <typename T>
class cudaReLU : public cudaComputeNode<T> {
 public:
  explicit cudaReLU(const std::vector<NodePtr<T>>& input,
                    const std::vector<NodePtr<T>>& output,
                    const std::string& name = "")
      : cudaComputeNode<T>(input, output, name) {
    this->validate_shape();
  }

  ~cudaReLU() override = default;

  /*! \brief validate the shape of the input and output*/
  void validate_shape() const override {
    assert(this->inputs().size() == 1 && "ReLUNode requires exactly one input");
    const auto& input_shape = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[0])->shape();
    const auto& output_shape = std::dynamic_pointer_cast<cudaDataNode<T>>(this->outputs()[0])->shape();
    assert(input_shape == output_shape && "Input and output shapes must match for ReLU");
  }

  /*! \brief forward computation */
  void forward() override {
    auto input = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[0]);
    auto output = std::dynamic_pointer_cast<cudaDataNode<T>>(this->outputs()[0]);
    
    T* input_data = input->data();
    T* output_data = output->data();
    
    size_t total_size = 1;
    for (const auto& dim : output->shape()) {
      total_size *= dim;
    }
    
    // Launch kernel
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    relu_forward_kernel<<<num_blocks, block_size>>>(input_data, output_data, total_size);
    CUDA_CHECK(cudaGetLastError());
  }

  /*! \brief backward computation */
  void backward() override {
    if (this->inputs()[0]->requires_grad()) {
      auto input = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[0]);
      auto output = std::dynamic_pointer_cast<cudaDataNode<T>>(this->outputs()[0]);
      
      T* input_data = input->data();
      T* grad_output = output->grad();
      T* grad_input = input->grad();
      
      size_t total_size = 1;
      for (const auto& dim : output->shape()) {
        total_size *= dim;
      }
      
      const int block_size = 256;
      const int num_blocks = (total_size + block_size - 1) / block_size;
      relu_backward_kernel<<<num_blocks, block_size>>>(input_data, grad_output, 
                                                        grad_input, total_size);
      CUDA_CHECK(cudaGetLastError());
    }
  }
};

template <typename T>
class cudaSigmoid : public cudaComputeNode<T> {
 public:
  explicit cudaSigmoid(const std::vector<NodePtr<T>>& input,
                       const std::vector<NodePtr<T>>& output,
                       const std::string& name = "")
      : cudaComputeNode<T>(input, output, name) {
    this->validate_shape();
  }

  ~cudaSigmoid() override = default;

  /*! \brief validate the shape of the input and output */
  void validate_shape() const override {
    assert(this->inputs().size() == 1 && "SigmoidNode requires exactly one input");
    const auto& input_shape = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[0])->shape();
    const auto& output_shape = std::dynamic_pointer_cast<cudaDataNode<T>>(this->outputs()[0])->shape();
    assert(input_shape == output_shape && "Input and output shapes must match for Sigmoid");
  }

  /*! \brief forward computation */
  void forward() override {
    auto input = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[0]);
    auto output = std::dynamic_pointer_cast<cudaDataNode<T>>(this->outputs()[0]);
    
    T* input_data = input->data();
    T* output_data = output->data();
    
    size_t total_size = 1;
    for (const auto& dim : output->shape()) {
      total_size *= dim;
    }
    
    // Launch kernel
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    sigmoid_forward_kernel<<<num_blocks, block_size>>>(input_data, output_data, total_size);
    CUDA_CHECK(cudaGetLastError());
  }

  /*! \brief backward computation */
  void backward() override {
    if (this->inputs()[0]->requires_grad()) {
      auto input = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[0]);
      auto output = std::dynamic_pointer_cast<cudaDataNode<T>>(this->outputs()[0]);
      
      T* output_data = output->data();
      T* grad_output = output->grad();
      T* grad_input = input->grad();
      
      size_t total_size = 1;
      for (const auto& dim : output->shape()) {
        total_size *= dim;
      }
      
      const int block_size = 256;
      const int num_blocks = (total_size + block_size - 1) / block_size;
      sigmoid_backward_kernel<<<num_blocks, block_size>>>(output_data, grad_output,
                                                           grad_input, total_size);
      CUDA_CHECK(cudaGetLastError());
    }
  }
};

template <typename T>
class cudaTanh : public cudaComputeNode<T> {
 public:
  explicit cudaTanh(const std::vector<NodePtr<T>>& input,
                    const std::vector<NodePtr<T>>& output,
                    const std::string& name = "")
      : cudaComputeNode<T>(input, output, name) {
    this->validate_shape();
  }

  ~cudaTanh() override = default;

  /*! \brief validate the shape of the input and output */
  void validate_shape() const override {
    assert(this->inputs().size() == 1 && "TanhNode requires exactly one input");
    const auto& input_shape = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[0])->shape();
    const auto& output_shape = std::dynamic_pointer_cast<cudaDataNode<T>>(this->outputs()[0])->shape();
    assert(input_shape == output_shape && "Input and output shapes must match for Tanh");
  }

  /*! \brief forward computation */
  void forward() override {
    auto input = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[0]);
    auto output = std::dynamic_pointer_cast<cudaDataNode<T>>(this->outputs()[0]);
    
    T* input_data = input->data();
    T* output_data = output->data();
    
    size_t total_size = 1;
    for (const auto& dim : output->shape()) {
      total_size *= dim;
    }
    
    // Launch kernel
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    tanh_forward_kernel<<<num_blocks, block_size>>>(input_data, output_data, total_size);
    CUDA_CHECK(cudaGetLastError());
  }

  /*! \brief backward computation */
  void backward() override {
    if (this->inputs()[0]->requires_grad()) {
      auto input = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[0]);
      auto output = std::dynamic_pointer_cast<cudaDataNode<T>>(this->outputs()[0]);
      
      T* output_data = output->data();
      T* grad_output = output->grad();
      T* grad_input = input->grad();
      
      size_t total_size = 1;
      for (const auto& dim : output->shape()) {
        total_size *= dim;
      }
      
      const int block_size = 256;
      const int num_blocks = (total_size + block_size - 1) / block_size;
      tanh_backward_kernel<<<num_blocks, block_size>>>(output_data, grad_output,
                                                        grad_input, total_size);
      CUDA_CHECK(cudaGetLastError());
    }
  }
};

} // namespace cuda

} // namespace modeldy

#endif // MODELDY_INCLUDE_CUDA_OPERATOR_ACTIVATION_OP_H_
