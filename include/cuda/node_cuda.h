/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef MODELDY_INCLUDE_CUDA_NODE_CUDA_H_
#define MODELDY_INCLUDE_CUDA_NODE_CUDA_H_

#include <modeldy/include/node.h>

#include <modeldy/include/cuda/cuda_check.h>

namespace modeldy {
/*! \brief CUDA-specific node implementations */
template <typename T>
class CudaDataNode : public DataNode<T> {
 public:
  explicit CudaDataNode(const std::vector<size_t>& shape,
                        bool requires_grad = false,
                        const std::string& name = "")
      : DataNode<T>(shape, requires_grad, name) {
    // Allocate CUDA memory for data
    size_t total_size = 1;
    for (const auto& dim : shape) {
      total_size *= dim;
    }
    T* data_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&data_ptr, total_size * sizeof(T)));
    data_ = std::unique_ptr<T[], Deleter>(data_ptr);

    if (requires_grad) {
      T* grad_ptr = nullptr;
      CUDA_CHECK(cudaMalloc(&grad_ptr, total_size * sizeof(T)));
      grad_ = std::unique_ptr<T[], Deleter>(grad_ptr);
    }
  }

  ~CudaDataNode() override = default;

  /*! \brief get the underlying data pointer */
  T* data() {
    return data_.get();
  }

  /*! \brief get the underlying gradient pointer */
  T* grad() {
    assert(this->requires_grad_ && "Gradient not allocated for this node");
    return grad_.get();
  }

 private:
  /*! \brief Custom deleter for CUDA memory */
  struct Deleter {
    void operator()(T* ptr) {
      CUDA_CHECK(cudaFree(ptr));
    }
  };

  std::unique_ptr<T[], Deleter> data_;
  std::unique_ptr<T[], Deleter> grad_;

};

namespace cuda {

template <typename T>
class CudaComputeNode : public ComputeNode<T> {
 public:
  explicit CudaComputeNode(const std::vector<NodePtr<T>>& inputs,
                           const std::vector<NodePtr<T>>& outputs,
                           const std::string& name = "")
      : ComputeNode<T>(inputs, outputs, name) {
    // Assert that all inputs are CudaDataNode
    for (const auto& input : inputs_) {
      assert(std::dynamic_pointer_cast<CudaDataNode<T>>(input) != nullptr &&
             "CudaComputeNode inputs must be CudaDataNode instances");
    }
    // Assert that all outputs are CudaDataNode
    for (const auto& output : outputs_) {
      assert(std::dynamic_pointer_cast<CudaDataNode<T>>(output) != nullptr &&
             "CudaComputeNode outputs must be CudaDataNode instances");
    }
  }
  
  ~CudaComputeNode() override = default;
};

} // namespace cuda

} // namespace modeldy

#endif // MODELDY_INCLUDE_CUDA_NODE_CUDA_H_