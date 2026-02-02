/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef MODELDY_INCLUDE_CPU_NODE_CPU_H_
#define MODELDY_INCLUDE_CPU_NODE_CPU_H_

#include <include/node.h>
#include <include/memory_pool.h>

#include <vector>
#include <cassert>

namespace modeldy {
namespace cpu {

/*! \brief CPU-specific node implementations */
template <typename T>
class cpuDataNode : public DataNode<T> {
 public:
  explicit cpuDataNode(const std::vector<size_t>& shape,
                       bool requires_grad = false,
                       const std::string& name = "")
      : DataNode<T>(shape, requires_grad, name) {
    // Calculate total size
    size_t total_size = 1;
    for (const auto& dim : shape) {
      total_size *= dim;
    }
    total_size_ = total_size;
    
    // Allocate CPU memory from pool
    auto& pool = cpuMemoryPool<T>::getInstance();
    T* data_ptr = pool.allocate(total_size);
    data_ = std::unique_ptr<T[], PoolDeleter>(data_ptr, PoolDeleter(total_size));
    
    if (requires_grad) {
      T* grad_ptr = pool.allocate(total_size);
      grad_ = std::unique_ptr<T[], PoolDeleter>(grad_ptr, PoolDeleter(total_size));
    }
  }

  ~cpuDataNode() override = default;

  /*! \brief get the underlying data pointer */
  T* data() {
    return data_.get();
  }

  /*! \brief get the underlying gradient pointer */
  T* grad() {
    assert(this->requires_grad_ && "Gradient not allocated for this node");
    return grad_.get();
  }

  void copy_from(const std::vector<T>& src) {
    assert(src.size() == total_size_ && "Source size must match data size");
    std::copy(src.begin(), src.end(), data_.get());
  }

 private:
  /*! \brief Custom deleter that returns memory to pool */
  struct PoolDeleter {
    size_t size;
    explicit PoolDeleter(size_t s = 0) : size(s) {}
    
    void operator()(T* ptr) {
      if (ptr != nullptr && size > 0) {
        cpuMemoryPool<T>::getInstance().deallocate(ptr, size);
      }
    }
  };

  std::unique_ptr<T[], PoolDeleter> data_;
  std::unique_ptr<T[], PoolDeleter> grad_;
  size_t total_size_;
};

template <typename T>
class cpuComputeNode : public ComputeNode<T> {
 public:
  explicit cpuComputeNode(const std::vector<NodePtr<T>>& inputs,
                          const std::vector<NodePtr<T>>& outputs,
                          const std::string& name = "")
      : ComputeNode<T>(inputs, outputs, name) {
    // Assert that all inputs are cpuDataNode
    for (const auto& input : this->inputs()) {
      assert(std::dynamic_pointer_cast<cpuDataNode<T>>(input) != nullptr && 
             "cpuComputeNode inputs must be cpuDataNode instances");
    }
    // Assert that all outputs are cpuDataNode
    for (const auto& output : this->outputs()) {
      assert(std::dynamic_pointer_cast<cpuDataNode<T>>(output) != nullptr && 
             "cpuComputeNode outputs must be cpuDataNode instances");
    }
  }

  ~cpuComputeNode() override = default;
};

} // namespace cpu
} // namespace modeldy

#endif // MODELDY_INCLUDE_CPU_NODE_CPU_H_
