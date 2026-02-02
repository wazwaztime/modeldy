/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef MODELDY_INCLUDE_CUDA_DATA_TRANSFER_NODE_H_
#define MODELDY_INCLUDE_CUDA_DATA_TRANSFER_NODE_H_

#include <include/cpu/node_cpu.h>
#include <include/cuda/node_cuda.h>
#include <include/cuda/cuda_check.h>

namespace modeldy {
/*! \brief Host to Device data transfer node */
template <typename T>
class HostToDeviceNode : public ComputeNode<T> {
 public:
  explicit HostToDeviceNode(const std::vector<NodePtr<T>>& inputs,
                            const std::vector<NodePtr<T>>& outputs,
                            const std::string& name = "")
    : ComputeNode<T>(inputs, outputs, name) {
    // Assert that all inputs are cpu::cpuDataNode
    for (const auto& input : inputs_) {
      assert(std::dynamic_pointer_cast<cpu::cpuDataNode<T>>(input) != nullptr && 
             "HostToDeviceNode inputs must be cpu::cpuDataNode instances");
    }
    // Assert that all outputs are cudaDataNode
    for (const auto& output : outputs_) {
      assert(std::dynamic_pointer_cast<cudaDataNode<T>>(output) != nullptr && 
             "HostToDeviceNode outputs must be cudaDataNode instances");
    }
    validate_shape();
  }

  ~HostToDeviceNode() override = default;

  /*! \brief validate the shape of the input and output*/
  void validate_shape() const override {
    assert(this->inputs_.size() == 1 && "HostToDeviceNode must have exactly one input");
    assert(this->outputs_.size() == 1 && "HostToDeviceNode must have exactly one output");
    auto input_data = std::dynamic_pointer_cast<DataNode<T>>(this->inputs_[0]);
    auto output_data = std::dynamic_pointer_cast<DataNode<T>>(this->outputs_[0]);
    assert(input_data && output_data && "Inputs and outputs must be DataNode instances");
    assert(input_data->shape() == output_data->shape() &&
           "Input and output shapes must match in HostToDeviceNode");
  }

  /*! \brief forward computation */
  void forward() override {
    auto input_node = std::dynamic_pointer_cast<cpu::cpuDataNode<T>>(this->inputs_[0]);
    auto output_node = std::dynamic_pointer_cast<cudaDataNode<T>>(this->outputs_[0]);
    size_t total_size = 1;
    for (const auto& dim : input_node->shape()) {
      total_size *= dim;
    }
    CUDA_CHECK(cudaMemcpy(output_node->data(), input_node->data(),
                          total_size * sizeof(T), cudaMemcpyHostToDevice));
  }

  /*! \brief backward computation */
  void backward() override {
    if (this -> requires_grad()) {
      auto input_node = std::dynamic_pointer_cast<cpu::cpuDataNode<T>>(this->inputs_[0]);
      auto output_node = std::dynamic_pointer_cast<cudaDataNode<T>>(this->outputs_[0]);
      size_t total_size = 1;
      for (const auto& dim : input_node->shape()) {
        total_size *= dim;
      }
      CUDA_CHECK(cudaMemcpy(input_node->grad(), output_node->grad(),
                            total_size * sizeof(T), cudaMemcpyDeviceToHost));
    }
  }

};

/*! \brief Device to Host data transfer node */
template <typename T>
class DeviceToHostNode : public ComputeNode<T> {
 public:
  explicit DeviceToHostNode(const std::vector<NodePtr<T>>& inputs,
                            const std::vector<NodePtr<T>>& outputs,
                            const std::string& name = "")
    : ComputeNode<T>(inputs, outputs, name) {
    // Assert that all inputs are cudaDataNode
    for (const auto& input : inputs_) {
      assert(std::dynamic_pointer_cast<cudaDataNode<T>>(input) != nullptr && 
             "DeviceToHostNode inputs must be cudaDataNode instances");
    }
    // Assert that all outputs are cpu::cpuDataNode
    for (const auto& output : outputs_) {
      assert(std::dynamic_pointer_cast<cpu::cpuDataNode<T>>(output) != nullptr && 
             "DeviceToHostNode outputs must be cpu::cpuDataNode instances");
    }
    validate_shape();
  }

  ~DeviceToHostNode() override = default;

  /*! \brief validate the shape of the input and output*/
  void validate_shape() const override {
    assert(this->inputs_.size() == 1 && "DeviceToHostNode must have exactly one input");
    assert(this->outputs_.size() == 1 && "DeviceToHostNode must have exactly one output");
    auto input_data = std::dynamic_pointer_cast<DataNode<T>>(this->inputs_[0]);
    auto output_data = std::dynamic_pointer_cast<DataNode<T>>(this->outputs_[0]);
    assert(input_data && output_data && "Inputs and outputs must be DataNode instances");
    assert(input_data->shape() == output_data->shape() &&
           "Input and output shapes must match in DeviceToHostNode");
  }

  /*! \brief forward computation */
  void forward() override {
    auto input_node = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs_[0]);
    auto output_node = std::dynamic_pointer_cast<cpu::cpuDataNode<T>>(this->outputs_[0]);
    size_t total_size = 1;
    for (const auto& dim : input_node->shape()) {
      total_size *= dim;
    }
    CUDA_CHECK(cudaMemcpy(output_node->data(), input_node->data(),
                          total_size * sizeof(T), cudaMemcpyDeviceToHost));
  }

  /*! \brief backward computation */
  void backward() override {
    if (this -> requires_grad()) {
      auto input_node = std::dynamic_pointer_cast<cudaDataNode<T>>(this->inputs_[0]);
      auto output_node = std::dynamic_pointer_cast<cpu::cpuDataNode<T>>(this->outputs_[0]);
      size_t total_size = 1;
      for (const auto& dim : input_node->shape()) {
        total_size *= dim;
      }
      CUDA_CHECK(cudaMemcpy(input_node->grad(), output_node->grad(),
                            total_size * sizeof(T), cudaMemcpyHostToDevice));
    }
  }
};

} // namespace modeldy


#endif // MODELDY_INCLUDE_CUDA_DATA_TRANSFER_NODE_H_
