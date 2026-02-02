/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef MODELDY_INCLUDE_CPU_NODE_CPU_H_
#define MODELDY_INCLUDE_CPU_NODE_CPU_H_

#include <modeldy/include/node.h>

#include <vector>

namespace modeldy {
/*! \brief CPU-specific node implementations */
template <typename T>
class CpuDataNode : public DataNode<T> {
 public:
  explicit CpuDataNode(const std::vector<size_t>& shape,
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

  ~CpuDataNode() override = default;

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

} // namespace modeldy

#endif // MODELDY_INCLUDE_CPU_NODE_CPU_H_