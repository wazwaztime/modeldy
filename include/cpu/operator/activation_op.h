/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef MODELDY_INCLUDE_CPU_OPERATOR_ACTIVATION_OP_H_
#define MODELDY_INCLUDE_CPU_OPERATOR_ACTIVATION_OP_H_

#include <modeldy/include/cpu/node_cpu.h>
#include <modeldy/include/cpu/cpu_math.h>

namespace modeldy {

namespace cpu {

template <typename T>
class CpuReLU : public CpuComputeNode<T> {
 public : 
  explicit CpuReLU(const std::vector<NodePtr<T>>& inputs,
                   const std::vector<NodePtr<T>>& outputs,
                   const std::string& name = "")
      : CpuComputeNode<T>(inputs, outputs, name) {
    this->validate_shape();
  }

  ~CpuReLU() override = default;

  /*! \brief validate the shape of the input and output*/
  void validate_shape() const override {
    assert(this->inputs().size() == 1 && "ReLUNode requires exactly one input");
    const auto& input_shape = this->inputs()[0]->shape();
    const auto& output_shape = this->outputs()[0]->shape();
    assert(input_shape == output_shape && "Input and output shapes must match for ReLU");
  }

  /*! \brief forward computation */
  void forward() override {
    T* input_data = this->inputs()[0]->data();
    T* output_data = this->outputs()[0]->data();
    size_t total_size = 1;
    for (const auto& dim : this->outputs()[0]->shape()) {
      total_size *= dim;
    }
    for (size_t i = 0; i < total_size; ++i) {
      output_data[i] = cpu_relu(input_data[i]);
    }
  }

  /*! \brief backward computation */
  void backward() override {
    if (this->inputs()[0]->requires_grad()) {
      T* input_data = this->inputs()[0]->data();
      T* grad_output = this->outputs()[0]->grad();
      T* grad_input = this->inputs()[0]->grad();
      size_t total_size = 1;
      for (const auto& dim : this->outputs()[0]->shape()) {
        total_size *= dim;
      }
      for (size_t i = 0; i < total_size; ++i) {
        grad_input[i] += (input_data[i] > static_cast<T>(0) ? grad_output[i] : static_cast<T>(0));
      }
    }
  }
};

} // namespace cpu

namespace cpu {

template <typename T>
class CpuSigmoid : public CpuComputeNode<T> {
 public :
  explicit CpuSigmoid(const std::vector<NodePtr<T>>& inputs,
                      const std::vector<NodePtr<T>>& outputs,
                      const std::string& name = "")
      : CpuComputeNode<T>(inputs, outputs, name) {
    this->validate_shape();
  }

  ~CpuSigmoid() override = default;

  /*! \brief validate the shape of the input and output */
  void validate_shape() const override {
    assert(this->inputs().size() == 1 && "SigmoidNode requires exactly one input");
    const auto& input_shape = this->inputs()[0]->shape();
    const auto& output_shape = this->outputs()[0]->shape();
    assert(input_shape == output_shape && "Input and output shapes must match for Sigmoid");
  }

  /*! \brief forward computation */
  void forward() override {
    T* input_data = this->inputs()[0]->data();
    T* output_data = this->outputs()[0]->data();
    size_t total_size = 1;
    for (const auto& dim : this->outputs()[0]->shape()) {
      total_size *= dim;
    }
    for (size_t i = 0; i < total_size; ++i) {
      output_data[i] = cpu_sigmoid(input_data[i]);
    }
  }

  /*! \brief backward computation */
  void backward() override {
    if (this->inputs()[0]->requires_grad()) {
      T* output_data = this->outputs()[0]->data();
      T* grad_output = this->outputs()[0]->grad();
      T* grad_input = this->inputs()[0]->grad();
      size_t total_size = 1;
      for (const auto& dim : this->outputs()[0]->shape()) {
        total_size *= dim;
      }
      for (size_t i = 0; i < total_size; ++i) {
        T sigmoid_val = output_data[i];
        grad_input[i] += sigmoid_val * (static_cast<T>(1) - sigmoid_val) * grad_output[i];
      }
    }
  }
};

} // namespace cpu

namespace cpu {

template <typename T>
class CpuTanh : public CpuComputeNode<T> {
 public :
  explicit CpuTanh(const std::vector<NodePtr<T>>& inputs,
                   const std::vector<NodePtr<T>>& outputs,
                   const std::string& name = "")
      : CpuComputeNode<T>(inputs, outputs, name) {
    this->validate_shape();
  }

  ~CpuTanh() override = default;

  /*! \brief validate the shape of the input and output */
  void validate_shape() const override {
    assert(this->inputs().size() == 1 && "TanhNode requires exactly one input");
    const auto& input_shape = this->inputs()[0]->shape();
    const auto& output_shape = this->outputs()[0]->shape();
    assert(input_shape == output_shape && "Input and output shapes must match for Tanh");
  }

  /*! \brief forward computation */
  void forward() override {
    T* input_data = this->inputs()[0]->data();
    T* output_data = this->outputs()[0]->data();
    size_t total_size = 1;
    for (const auto& dim : this->outputs()[0]->shape()) {
      total_size *= dim;
    }
    for (size_t i = 0; i < total_size; ++i) {
      output_data[i] = cpu_tanh(input_data[i]);
    }
  }

  /*! \brief backward computation */
  void backward() override {
    if (this->inputs()[0]->requires_grad()) {
      T* output_data = this->outputs()[0]->data();
      T* grad_output = this->outputs()[0]->grad();
      T* grad_input = this->inputs()[0]->grad();
      size_t total_size = 1;
      for (const auto& dim : this->outputs()[0]->shape()) {
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