/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef MODELDY_INCLUDE_CUDA_DATA_TRANSFER_NODE_H_
#define MODELDY_INCLUDE_CUDA_DATA_TRANSFER_NODE_H_

#include <modeldy/include/cpu/node_cpu.h>
#include <modeldy/include/cuda/node_cuda.h>
#include <modeldy/include/cuda/cuda_check.h>

namespace modeldy {
/*! \brief Host to Device data transfer node */
template <typename T>
class HostToDeviceNode : public ComputeNode<T> {
 public:
  explicit HostToDeviceNode(const std::vector<NodePtr<T>>& inputs,
                            const std::vector<NodePtr<T>>& outputs,
                            const std::string& name = "")
    : ComputeNode<T>(inputs, outputs, name) {
    // Assert that all inputs are CpuDataNode
    for (const auto& input : inputs_) {
      assert(std::dynamic_pointer_cast<CpuDataNode<T>>(input) != nullptr && 
             "HostToDeviceNode inputs must be CpuDataNode instances");
    }
    // Assert that all outputs are CudaDataNode
    for (const auto& output : outputs_) {
      assert(std::dynamic_pointer_cast<CudaDataNode<T>>(output) != nullptr && 
             "HostToDeviceNode outputs must be CudaDataNode instances");
    }
    validate_shape();
  }

  ~HostToDeviceNode() override = default;

  /*! \brief validate the shape of the input and output*/
  void validate_shape() const override {
    assert(this->inputs_.size() == 1 && "HostToDeviceNode must have exactly one input");
    assert(this->outputs_.size() == 1 && "HostToDeviceNode must have exactly one output");
    assert(this->inputs_[0]->shape() == this->outputs_[0]->shape() &&
           "Input and output shapes must match in HostToDeviceNode");
  }

  /*! \brief forward computation */
  void forward() override {
    auto input_node = std::dynamic_pointer_cast<CpuDataNode<T>>(this->inputs_[0]);
    auto output_node = std::dynamic_pointer_cast<CudaDataNode<T>>(this->outputs_[0]);
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
      auto input_node = std::dynamic_pointer_cast<CpuDataNode<T>>(this->inputs_[0]);
      auto output_node = std::dynamic_pointer_cast<CudaDataNode<T>>(this->outputs_[0]);
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
    // Assert that all inputs are CudaDataNode
    for (const auto& input : inputs_) {
      assert(std::dynamic_pointer_cast<CudaDataNode<T>>(input) != nullptr && 
             "DeviceToHostNode inputs must be CudaDataNode instances");
    }
    // Assert that all outputs are CpuDataNode
    for (const auto& output : outputs_) {
      assert(std::dynamic_pointer_cast<CpuDataNode<T>>(output) != nullptr && 
             "DeviceToHostNode outputs must be CpuDataNode instances");
    }
    validate_shape();
  }

  ~DeviceToHostNode() override = default;

  /*! \brief validate the shape of the input and output*/
  void validate_shape() const override {
    assert(this->inputs_.size() == 1 && "DeviceToHostNode must have exactly one input");
    assert(this->outputs_.size() == 1 && "DeviceToHostNode must have exactly one output");
    assert(this->inputs_[0]->shape() == this->outputs_[0]->shape() &&
           "Input and output shapes must match in DeviceToHostNode");
  }

  /*! \brief forward computation */
  void forward() override {
    auto input_node = std::dynamic_pointer_cast<CudaDataNode<T>>(this->inputs_[0]);
    auto output_node = std::dynamic_pointer_cast<CpuDataNode<T>>(this->outputs_[0]);
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
      auto input_node = std::dynamic_pointer_cast<CudaDataNode<T>>(this->inputs_[0]);
      auto output_node = std::dynamic_pointer_cast<CpuDataNode<T>>(this->outputs_[0]);
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