/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef MODELDY_INCLUDE_CPU_OPERATOR_ACTIVATION_OP_H_
#define MODELDY_INCLUDE_CPU_OPERATOR_ACTIVATION_OP_H_

#include <include/cpu/node_cpu.h>
#include <include/cpu/cpu_math.h>

#include <cassert>

namespace modeldy {

namespace cpu {

template <typename T>
class cpuReLU : public cpuComputeNode<T> {
 public : 
  explicit cpuReLU(const std::vector<NodePtr<T>>& input,
                   const std::vector<NodePtr<T>>& output,
                   const std::string& name = "")
      : cpuComputeNode<T>(input, output, name) {
    this->validate_shape();
  }

  ~cpuReLU() override = default;

  /*! \brief validate the shape of the input and output*/
  void validate_shape() const override {
    assert(this->inputs().size() == 1 && "ReLUNode requires exactly one input");
    const auto& input_shape = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[0])->shape();
    const auto& output_shape = std::dynamic_pointer_cast<cpuDataNode<T>>(this->outputs()[0])->shape();
    assert(input_shape == output_shape && "Input and output shapes must match for ReLU");
  }

  /*! \brief forward computation */
  void forward() override {
    auto input = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[0]);
    auto output = std::dynamic_pointer_cast<cpuDataNode<T>>(this->outputs()[0]);
    T* input_data = input->data();
    T* output_data = output->data();
    size_t total_size = 1;
    for (const auto& dim : output->shape()) {
      total_size *= dim;
    }
    for (size_t i = 0; i < total_size; ++i) {
      output_data[i] = cpu_relu(input_data[i]);
    }
  }

  /*! \brief backward computation */
  void backward() override {
    if (this->inputs()[0]->requires_grad()) {
      auto input = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[0]);
      auto output = std::dynamic_pointer_cast<cpuDataNode<T>>(this->outputs()[0]);
      T* input_data = input->data();
      T* grad_output = output->grad();
      T* grad_input = input->grad();
      size_t total_size = 1;
      for (const auto& dim : output->shape()) {
        total_size *= dim;
      }
      for (size_t i = 0; i < total_size; ++i) {
        grad_input[i] += (input_data[i] > static_cast<T>(0) ? grad_output[i] : static_cast<T>(0));
      }
    }
  }
};

template <typename T>
class cpuSigmoid : public cpuComputeNode<T> {
 public :
  explicit cpuSigmoid(const std::vector<NodePtr<T>>& input,
                   const std::vector<NodePtr<T>>& output,
                   const std::string& name = "")
      : cpuComputeNode<T>(input, output, name) {
    this->validate_shape();
  }

  ~cpuSigmoid() override = default;

  /*! \brief validate the shape of the input and output */
  void validate_shape() const override {
    assert(this->inputs().size() == 1 && "SigmoidNode requires exactly one input");
    const auto& input_shape = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[0])->shape();
    const auto& output_shape = std::dynamic_pointer_cast<cpuDataNode<T>>(this->outputs()[0])->shape();
    assert(input_shape == output_shape && "Input and output shapes must match for Sigmoid");
  }

  /*! \brief forward computation */
  void forward() override {
    auto input = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[0]);
    auto output = std::dynamic_pointer_cast<cpuDataNode<T>>(this->outputs()[0]);
    T* input_data = input->data();
    T* output_data = output->data();
    size_t total_size = 1;
    for (const auto& dim : output->shape()) {
      total_size *= dim;
    }
    for (size_t i = 0; i < total_size; ++i) {
      output_data[i] = cpu_sigmoid(input_data[i]);
    }
  }

  /*! \brief backward computation */
  void backward() override {
    if (this->inputs()[0]->requires_grad()) {
      auto input = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[0]);
      auto output = std::dynamic_pointer_cast<cpuDataNode<T>>(this->outputs()[0]);
      T* output_data = output->data();
      T* grad_output = output->grad();
      T* grad_input = input->grad();
      size_t total_size = 1;
      for (const auto& dim : output->shape()) {
        total_size *= dim;
      }
      for (size_t i = 0; i < total_size; ++i) {
        T sigmoid_val = output_data[i];
        grad_input[i] += sigmoid_val * (static_cast<T>(1) - sigmoid_val) * grad_output[i];
      }
    }
  }
};

template <typename T>
class cpuTanh : public cpuComputeNode<T> {
 public :
  explicit cpuTanh(const std::vector<NodePtr<T>>& input,
                   const std::vector<NodePtr<T>>& output,
                   const std::string& name = "")
      : cpuComputeNode<T>(input, output, name) {
    this->validate_shape();
  }

  ~cpuTanh() override = default;

  /*! \brief validate the shape of the input and output */
  void validate_shape() const override {
    assert(this->inputs().size() == 1 && "TanhNode requires exactly one input");
    const auto& input_shape = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[0])->shape();
    const auto& output_shape = std::dynamic_pointer_cast<cpuDataNode<T>>(this->outputs()[0])->shape();
    assert(input_shape == output_shape && "Input and output shapes must match for Tanh");
  }

  /*! \brief forward computation */
  void forward() override {
    auto input = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[0]);
    auto output = std::dynamic_pointer_cast<cpuDataNode<T>>(this->outputs()[0]);
    T* input_data = input->data();
    T* output_data = output->data();
    size_t total_size = 1;
    for (const auto& dim : output->shape()) {
      total_size *= dim;
    }
    for (size_t i = 0; i < total_size; ++i) {
      output_data[i] = cpu_tanh(input_data[i]);
    }
  }

  /*! \brief backward computation */
  void backward() override {
    if (this->inputs()[0]->requires_grad()) {
      auto input = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[0]);
      auto output = std::dynamic_pointer_cast<cpuDataNode<T>>(this->outputs()[0]);
      T* output_data = output->data();
      T* grad_output = output->grad();
      T* grad_input = input->grad();
      size_t total_size = 1;
      for (const auto& dim : output->shape()) {
        total_size *= dim;
      }
      for (size_t i = 0; i < total_size; ++i) {
        T tanh_val = output_data[i];
        grad_input[i] += (static_cast<T>(1) - tanh_val * tanh_val) * grad_output[i];
      }
    }
  }
};

} // namespace cpu

} // namespace modeldy

#endif // MODELDY_INCLUDE_CPU_OPERATOR_ACTIVATION_OP_H_
