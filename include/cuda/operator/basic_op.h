/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef MODELDY_INCLUDE_CUDA_OPERATOR_BASIC_OP_H_
#define MODELDY_INCLUDE_CUDA_OPERATOR_BASIC_OP_H_

#include <include/cuda/node_cuda.h>
#include <include/cuda/cuda_math.h>
#include <include/cuda/cuda_check.h>
#include <cassert>

namespace modeldy {

namespace cuda {

template <typename T>
class cudaAdd : public cudaComputeNode<T> {
 public:
  explicit cudaAdd(const std::vector<NodePtr<T>>& input,
                   const std::vector<NodePtr<T>>& output,
                   const std::string& name = "")
      : cudaComputeNode<T>(input, output, name) {
    this->validate_shape();
  }

  ~cudaAdd() override = default;

  /*! \brief validate the shape of the input and output*/
  void validate_shape() const override {
    assert(this->inputs().size() == 2 && "AddNode requires exactly two inputs");
    const auto& shape1 = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[0])->shape();
    const auto& shape2 = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[1])->shape();
    assert(shape1 == shape2 && "Input shapes must match for addition");
    const auto& output_shape = std::dynamic_pointer_cast<cudaDataNode<T>>(this->outputs()[0])->shape();
    assert(shape1 == output_shape && "Output shape must match input shapes");
  }

  /*! \brief forward computation */
  void forward() override {
    auto input1 = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[0]);
    auto input2 = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[1]);
    auto output = std::dynamic_pointer_cast<cudaDataNode<T>>(this->outputs()[0]);
    
    T* input1_data = input1->data();
    T* input2_data = input2->data();
    T* output_data = output->data();
    
    size_t total_size = 1;
    for (const auto& dim : output->shape()) {
      total_size *= dim;
    }
    
    // Launch kernel
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    add_forward_kernel<<<num_blocks, block_size>>>(input1_data, input2_data, 
                                                    output_data, total_size);
    CUDA_CHECK(cudaGetLastError());
  }

  /*! \brief backward computation */
  void backward() override {
    auto output = std::dynamic_pointer_cast<cudaDataNode<T>>(this->outputs()[0]);
    T* grad_output = output->grad();
    
    size_t total_size = 1;
    for (const auto& dim : output->shape()) {
      total_size *= dim;
    }
    
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    if (this->inputs()[0]->requires_grad()) {
      auto input1 = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[0]);
      T* grad_input1 = input1->grad();
      add_backward_kernel<<<num_blocks, block_size>>>(grad_output, grad_input1, total_size);
      CUDA_CHECK(cudaGetLastError());
    }
    
    if (this->inputs()[1]->requires_grad()) {
      auto input2 = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[1]);
      T* grad_input2 = input2->grad();
      add_backward_kernel<<<num_blocks, block_size>>>(grad_output, grad_input2, total_size);
      CUDA_CHECK(cudaGetLastError());
    }
  }
};

template <typename T>
class cudaMul : public cudaComputeNode<T> {
 public:
  explicit cudaMul(const std::vector<NodePtr<T>>& input,
                   const std::vector<NodePtr<T>>& output,
                   const std::string& name = "")
      : cudaComputeNode<T>(input, output, name) {
    this->validate_shape();
  }

  ~cudaMul() override = default;

  /*! \brief validate the shape of the input and output*/
  void validate_shape() const override {
    assert(this->inputs().size() == 2 && "MulNode requires exactly two inputs");
    const auto& shape1 = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[0])->shape();
    const auto& shape2 = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[1])->shape();
    assert(shape1 == shape2 && "Input shapes must match for multiplication");
    const auto& output_shape = std::dynamic_pointer_cast<cudaDataNode<T>>(this->outputs()[0])->shape();
    assert(shape1 == output_shape && "Output shape must match input shapes");
  }

  /*! \brief forward computation */
  void forward() override {
    auto input1 = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[0]);
    auto input2 = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[1]);
    auto output = std::dynamic_pointer_cast<cudaDataNode<T>>(this->outputs()[0]);
    
    T* input1_data = input1->data();
    T* input2_data = input2->data();
    T* output_data = output->data();
    
    size_t total_size = 1;
    for (const auto& dim : output->shape()) {
      total_size *= dim;
    }
    
    // Launch kernel
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    mul_forward_kernel<<<num_blocks, block_size>>>(input1_data, input2_data, 
                                                    output_data, total_size);
    CUDA_CHECK(cudaGetLastError());
  }

  /*! \brief backward computation */
  void backward() override {
    auto output = std::dynamic_pointer_cast<cudaDataNode<T>>(this->outputs()[0]);
    T* grad_output = output->grad();
    
    size_t total_size = 1;
    for (const auto& dim : output->shape()) {
      total_size *= dim;
    }
    
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    if (this->inputs()[0]->requires_grad()) {
      auto input1 = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[0]);
      auto input2 = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[1]);
      T* input2_data = input2->data();
      T* grad_input1 = input1->grad();
      mul_backward_kernel<<<num_blocks, block_size>>>(input2_data, grad_output, 
                                                       grad_input1, total_size);
      CUDA_CHECK(cudaGetLastError());
    }
    
    if (this->inputs()[1]->requires_grad()) {
      auto input1 = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[0]);
      auto input2 = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs()[1]);
      T* input1_data = input1->data();
      T* grad_input2 = input2->grad();
      mul_backward_kernel<<<num_blocks, block_size>>>(input1_data, grad_output, 
                                                       grad_input2, total_size);
      CUDA_CHECK(cudaGetLastError());
    }
  }
};

} // namespace cuda

} // namespace modeldy

#endif // MODELDY_INCLUDE_CUDA_OPERATOR_BASIC_OP_H_
