/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef MODELDY_INCLUDE_CPU_NODE_CPU_H_
#define MODELDY_INCLUDE_CPU_NODE_CPU_H_

#include <modeldy/include/node.h>

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
    // Allocate CPU memory for data
    size_t total_size = 1;
    for (const auto& dim : shape) {
      total_size *= dim;
    }
    data_.resize(total_size);
    if (requires_grad) {
        grad_.resize(total_size);
    }
  }

  ~cpuDataNode() override = default;

  /*! \brief get the underlying data pointer */
  T* data() {
    return data_.data();
  }

  /*! \brief get the underlying gradient pointer */
  T* grad() {
    assert(this->requires_grad_ && "Gradient not allocated for this node");
    return grad_.data();
  }

  void copy_from(const std::vector<T>& src) {
    assert(src.size() == data_.size() && "Source size must match data size");
    std::copy(src.begin(), src.end(), data_.begin());
  }

 private:
  std::vector<T> data_;
  std::vector<T> grad_;
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