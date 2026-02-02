/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef MODELDY_INCLUDE_CUDA_NODE_CUDA_H_
#define MODELDY_INCLUDE_CUDA_NODE_CUDA_H_

#include <modeldy/include/node.h>
#include <modeldy/include/memory_pool.h>
#include <modeldy/include/cuda/cuda_check.h>

namespace modeldy {
/*! \brief CUDA-specific node implementations */
template <typename T>
class cudaDataNode : public DataNode<T> {
 public:
  explicit cudaDataNode(const std::vector<size_t>& shape,
                        bool requires_grad = false,
                        const std::string& name = "")
      : DataNode<T>(shape, requires_grad, name) {
    // Calculate total size
    size_t total_size = 1;
    for (const auto& dim : shape) {
      total_size *= dim;
    }
    total_size_ = total_size;
    
    // Allocate CUDA memory from pool
    auto& pool = cuda::cudaMemoryPool<T>::getInstance();
    T* data_ptr = pool.allocate(total_size);
    data_ = std::unique_ptr<T[], PoolDeleter>(data_ptr, PoolDeleter(total_size));

    if (requires_grad) {
      T* grad_ptr = pool.allocate(total_size);
      grad_ = std::unique_ptr<T[], PoolDeleter>(grad_ptr, PoolDeleter(total_size));
    }
  }

  ~cudaDataNode() override = default;

  /*! \brief get the underlying data pointer */
  T* data() {
    return data_.get();
  }

  /*! \brief get the underlying gradient pointer */
  T* grad() {
    assert(this->requires_grad_ && "Gradient not allocated for this node");
    return grad_.get();
  }

  /*! \brief copy data from host to device */
  void copy_from(const std::vector<T>& host_data) {
    assert(host_data.size() == total_size_ && "Data size does not match node shape");
    CUDA_CHECK(cudaMemcpy(data_.get(), host_data.data(), 
                          total_size_ * sizeof(T), 
                          cudaMemcpyHostToDevice));
  }

 private:
  /*! \brief Custom deleter that returns memory to pool */
  struct PoolDeleter {
    size_t size;
    explicit PoolDeleter(size_t s = 0) : size(s) {}
    
    void operator()(T* ptr) {
      if (ptr != nullptr && size > 0) {
        cuda::cudaMemoryPool<T>::getInstance().deallocate(ptr, size);
      }
    }
  };

  std::unique_ptr<T[], PoolDeleter> data_;
  std::unique_ptr<T[], PoolDeleter> grad_;
  size_t total_size_;

};

namespace cuda {

template <typename T>
class cudaComputeNode : public ComputeNode<T> {
 public:
  explicit cudaComputeNode(const std::vector<NodePtr<T>>& inputs,
                           const std::vector<NodePtr<T>>& outputs,
                           const std::string& name = "")
      : ComputeNode<T>(inputs, outputs, name) {
    // Assert that all inputs are cudaDataNode
    for (const auto& input : inputs_) {
      assert(std::dynamic_pointer_cast<cudaDataNode<T>>(input) != nullptr &&
             "cudaComputeNode inputs must be cudaDataNode instances");
    }
    // Assert that all outputs are cudaDataNode
    for (const auto& output : outputs_) {
      assert(std::dynamic_pointer_cast<cudaDataNode<T>>(output) != nullptr &&
             "cudaComputeNode outputs must be cudaDataNode instances");
    }
  }
  
  ~cudaComputeNode() override = default;
};

} // namespace cuda

} // namespace modeldy

#endif // MODELDY_INCLUDE_CUDA_NODE_CUDA_H_