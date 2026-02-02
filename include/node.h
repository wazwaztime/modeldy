/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef MODELDY_INCLUDE_NODE_H_
#define MODELDY_INCLUDE_NODE_H_

#include <vector>
#include <memory>
#include <cstddef>
#include <algorithm>
#include <string>

namespace modeldy {

template <typename T>
class Node;

template <typename T>
class DataNode;

template <typename T>
class ComputeNode;

template <typename T>
using NodePtr = std::shared_ptr<Node<T>>;

/*! \brief node class */
template <typename T>
class Node : public std::enable_shared_from_this<Node<T>> {
 protected:
  std::vector<NodePtr<T>> inputs_;
  std::vector<NodePtr<T>> outputs_;
  bool requires_grad_ = false;
  std::string name_;

 public:
  explicit Node(const std::vector<NodePtr<T>>& inputs,
                const std::vector<NodePtr<T>>& outputs,
                const std::string& name)
    : inputs_(inputs), outputs_(outputs), name_(name) {}
  Node() = default;
  virtual ~Node() = default;

  /*! \brief get the input nodes */
  const std::vector<NodePtr<T>>& inputs() const {
    return inputs_;
  }

  /*! \brief get the output nodes */
  const std::vector<NodePtr<T>>& outputs() const {
    return outputs_;
  }

  /*! \brief get the name of the node */
  const std::string& name() const {
    return name_;
  }

  /*! \brief check whether the node requires gradient */
  bool requires_grad() const {
    return requires_grad_;
  }

 protected:
  /*! \brief add an input node */
  void add_input(const NodePtr<T>& input) {
    inputs_.push_back(input);
  }

  /*! \brief add an output node */
  void add_output(const NodePtr<T>& output) {
    outputs_.push_back(output);
  }
  
  friend class ComputeNode<T>;
};

/*! \brief data node class */
template <typename T>
class DataNode : public Node<T> {
 public:
  explicit DataNode(const std::vector<size_t>& shape,
                    bool requires_grad = false,
                    const std::string& name = "")
    : Node<T>(std::vector<NodePtr<T>>(), std::vector<NodePtr<T>>(), name), 
      shape_(shape) {
        this -> requires_grad_ = requires_grad;
      }
  
  ~DataNode() override = default;

  /*! \brief get the pointer to the data */
  virtual T* data() = 0;

  /*! \brief get the pointer to the gradient */
  virtual T* grad() = 0;

  /*! \brief get the shape of the data */
  const std::vector<size_t>& shape() const {
    return shape_;
  }

 private:
  std::vector<size_t> shape_; 
};

/*! \brief compute node class */
template <typename T>
class ComputeNode : public Node<T> {
 public:
  explicit ComputeNode(const std::vector<NodePtr<T>>& inputs,
                       const std::vector<NodePtr<T>>& outputs,
                       const std::string& name = "") 
   : Node<T>(inputs, outputs, name) {
    // Assert that all inputs are DataNode
    for (const auto& input : inputs_) {
      assert(std::dynamic_pointer_cast<DataNode<T>>(input) != nullptr && 
             "ComputeNode inputs must be DataNode instances");
    }
    // Assert that all outputs are DataNode
    for (const auto& output : outputs_) {
      assert(std::dynamic_pointer_cast<DataNode<T>>(output) != nullptr && 
             "ComputeNode outputs must be DataNode instances");
    }
    for (const auto& input : inputs_) {
      input->add_output(this->shared_from_this());
    }
    for (const auto& output : outputs_) {
      output->add_input(this->shared_from_this());
    }
    // Determine requires_grad based on inputs
    requires_grad_ = std::any_of(
      inputs_.begin(), inputs_.end(),
      [](const NodePtr<T>& node) { return node->requires_grad(); });
  }

  ~ComputeNode() override = default;

  /*! \brief validate the shape of the input and output*/
  virtual void validate_shape() const = 0;

  /*! \brief forward computation */
  virtual void forward() = 0;

  /*! \brief backward computation */
  virtual void backward() = 0;
  
};

} // namespace modeldy

#endif  // MODELDY_INCLUDE_NODE_H_