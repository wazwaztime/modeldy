/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef MODELDY_INCLUDE_CPU_OPERATOR_BASIC_OP_H_
#define MODELDY_INCLUDE_CPU_OPERATOR_BASIC_OP_H_

#include <modeldy/include/cpu/node_cpu.h>
#include <cassert>

namespace modeldy {

namespace cpu {

template <typename T>
class cpuAdd : public cpuComputeNode<T> {
 public:
  explicit cpuAdd(const std::vector<NodePtr<T>>& input,
                   const std::vector<NodePtr<T>>& output,
                   const std::string& name = "")
      : cpuComputeNode<T>(input, output, name) {
    this->validate_shape();
  }

  ~cpuAdd() override = default;

  /*! \brief validate the shape of the input and output*/
  void validate_shape() const override {
    assert(this->inputs().size() == 2 && "AddNode requires exactly two inputs");
    const auto& shape1 = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[0])->shape();
    const auto& shape2 = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[1])->shape();
    assert(shape1 == shape2 && "Input shapes must match for addition");
    const auto& output_shape = std::dynamic_pointer_cast<cpuDataNode<T>>(this->outputs()[0])->shape();
    assert(shape1 == output_shape && "Output shape must match input shapes");
  }

  /*! \brief forward computation */
  void forward() override {
    auto input1 = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[0]);
    auto input2 = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[1]);
    auto output = std::dynamic_pointer_cast<cpuDataNode<T>>(this->outputs()[0]);
    T* input1_data = input1->data();
    T* input2_data = input2->data();
    T* output_data = output->data();
    size_t total_size = 1;
    for (const auto& dim : output->shape()) {
      total_size *= dim;
    }
    for (size_t i = 0; i < total_size; ++i) {
      output_data[i] = input1_data[i] + input2_data[i];
    }
  }

  /*! \brief backward computation */
  void backward() override {
    auto output = std::dynamic_pointer_cast<cpuDataNode<T>>(this->outputs()[0]);
    if (this->inputs()[0]->requires_grad()) {
      auto input1 = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[0]);
      T* grad_output = output->grad();
      T* grad_input1 = input1->grad();
      size_t total_size = 1;
      for (const auto& dim : output->shape()) {
        total_size *= dim;
      }
      for (size_t i = 0; i < total_size; ++i) {
        grad_input1[i] += grad_output[i];
      }
    }
    if (this->inputs()[1]->requires_grad()) {
      auto input2 = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[1]);
      T* grad_output = output->grad();
      T* grad_input2 = input2->grad();
      size_t total_size = 1;
      for (const auto& dim : output->shape()) {
        total_size *= dim;
      }
      for (size_t i = 0; i < total_size; ++i) {
        grad_input2[i] += grad_output[i];
      }
    }
  }
};

template <typename T>
class cpuMul : public cpuComputeNode<T> {
 public:
  explicit cpuMul(const std::vector<NodePtr<T>>& input,
                   const std::vector<NodePtr<T>>& output,
                   const std::string& name = "")
      : cpuComputeNode<T>(input, output, name) {
    this->validate_shape();
  }

  ~cpuMul() override = default;

  /*! \brief validate the shape of the input and output*/
  void validate_shape() const override {
    assert(this->inputs().size() == 2 && "MulNode requires exactly two inputs");
    const auto& shape1 = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[0])->shape();
    const auto& shape2 = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[1])->shape();
    assert(shape1 == shape2 && "Input shapes must match for multiplication");
    const auto& output_shape = std::dynamic_pointer_cast<cpuDataNode<T>>(this->outputs()[0])->shape();
    assert(shape1 == output_shape && "Output shape must match input shapes");
  }

  /*! \brief forward computation */
  void forward() override {
    auto input1 = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[0]);
    auto input2 = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[1]);
    auto output = std::dynamic_pointer_cast<cpuDataNode<T>>(this->outputs()[0]);
    T* input1_data = input1->data();
    T* input2_data = input2->data();
    T* output_data = output->data();
    size_t total_size = 1;
    for (const auto& dim : output->shape()) {
      total_size *= dim;
    }
    for (size_t i = 0; i < total_size; ++i) {
      output_data[i] = input1_data[i] * input2_data[i];
    }
  }

  /*! \brief backward computation */
  void backward() override {
    auto output = std::dynamic_pointer_cast<cpuDataNode<T>>(this->outputs()[0]);
    if (this->inputs()[0]->requires_grad()) {
      auto input1 = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[0]);
      auto input2 = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[1]);
      T* input2_data = input2->data();
      T* grad_output = output->grad();
      T* grad_input1 = input1->grad();
      size_t total_size = 1;
      for (const auto& dim : output->shape()) {
        total_size *= dim;
      }
      for (size_t i = 0; i < total_size; ++i) {
        grad_input1[i] += grad_output[i] * input2_data[i];
      }
    }
    if (this->inputs()[1]->requires_grad()) {
      auto input1 = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[0]);
      auto input2 = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[1]);
      T* input1_data = input1->data();
      T* grad_output = output->grad();
      T* grad_input2 = input2->grad();
      size_t total_size = 1;
      for (const auto& dim : output->shape()) {
        total_size *= dim;
      }
      for (size_t i = 0; i < total_size; ++i) {
        grad_input2[i] += grad_output[i] * input1_data[i];
      }
    }
  }
};

} // namespace cpu

} // namespace modeldy

#include <modeldy/include/operator_registry.h>

REGISTER_OPERATOR(float, Add, cpu);
REGISTER_OPERATOR(double, Add, cpu);
REGISTER_OPERATOR(float, Mul, cpu);
REGISTER_OPERATOR(double, Mul, cpu);

#endif // MODELDY_INCLUDE_CPU_OPERATOR_BASIC_OP_H_