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
using NodePtr = std::shared_ptr<Node<T>>;
template <typename T>

/*! \brief node class */
template <typename T>
class Node : public std::enable_shared_from_this<Node<T>> {
 protected:
  std::vector<NodePtr<T>> inputs_;
  std::vector<NodePtr<T>> outputs_;
  bool requires_grad_ = false;
  std::string name_;

 public:
  virtual ~Node() = default;

  /*! \brief add an input node */
  void add_input(const NodePtr<T>& input) {
    inputs_.push_back(input);
  }

  /*! \brief add an output node */
  void add_output(const NodePtr<T>& output) {
    outputs_.push_back(output);
  }

  /*! \brief get the input nodes */
  const std::vector<NodePtr<T>>& inputs() const {
    return inputs_;
  }

  /*! \brief get the output nodes */
  const std::vector<NodePtr<T>>& outputs() const {
    return outputs_;
  }

  /*! \brief set the name of the node */
  void set_name(const std::string& name) {
    name_ = name;
  }

  /*! \brief get the name of the node */
  const std::string& name() const {
    return name_;
  }
};

/*! \brief data node class */
template <typename T, typename Deleter = std::default_delete<T[]>>
class DataNode : public Node<T> {
 public:
  /*! \brief get the shape of the data */
  const std::vector<size_t>& shape() const {
    return shape_;
  }

  /*! \brief get the pointer to the data */
  T* data() {
    return data_.get();
  }

  /*! \brief get the pointer to the gradient */
  T* grad() {
    return grad_.get();
  }
  
  /*! \brief set whether the node requires gradient */
  void set_requires_grad(bool requires_grad) {
    requires_grad_ = requires_grad;
  }

  /*! \brief check whether the node requires gradient */
  bool requires_grad() const {
    return requires_grad_;
  }

 private:
  std::vector<size_t> shape_; 
  std::unique_ptr<T[], Deleter> data_;
  std::unique_ptr<T[], Deleter> grad_;
};

/*! \brief compute node class */
template <typename T>
class ComputeNode : public Node<T> {
 public:
  ComputeNode(const std::vector<NodePtr<T>>& inputs,
             const std::vector<NodePtr<T>>& outputs,
             const std::string& name = "") 
   : inputs_(inputs), outputs_(outputs), name_(name) {
    for (const auto& input : inputs_) {
      input->add_output(this->shared_from_this());
    }
    for (const auto& output : outputs_) {
      output->add_input(this->shared_from_this());
    }
    requires_grad_ = std::any_of(
      inputs_.begin(), inputs_.end(),
      [](const NodePtr<T>& node) { return node->requires_grad(); });
  }

  ~ComputeNode() override = default;

  /*! \brief assert the shape of the input and output*/
  virtual bool validate_shape() = 0;

  /*! \brief forward computation */
  virtual void forward() = 0;

  /*! \brief backward computation */
  virtual void backward() = 0;
  
};

/*! \brief addition node class */
template <typename T>
class AddNode : public ComputeNode<T> {
 public:
  bool validate_shape() override {
    if (this->inputs_.size() != 2 || this->outputs_.size() != 1) {
      return false;
    }
    if (dynamic_cast<DataNode<T>*>(this->inputs_[0].get()) == nullptr ||
        dynamic_cast<DataNode<T>*>(this->inputs_[1].get()) == nullptr ||
        dynamic_cast<DataNode<T>*>(this->outputs_[0].get()) == nullptr) {
      return false;
    }
    auto shape1 = this->inputs_[0]->shape();
    auto shape2 = this->inputs_[1]->shape();
    auto out_shape = this->outputs_[0]->shape();
    return shape1 == shape2 && shape1 == out_shape;
  }

  void forward() override {
    if (!assertShape()) {
      throw std::runtime_error("Shape mismatch in AddNode forward");
    }
    for (size_t i = 0; i < this->inputs_[0]->shape()[0]; ++i) {
      this->outputs_[0]->data()[i] = this->inputs_[0]->data()[i] + this->inputs_[1]->data()[i];
    }
  }

  void backward() override {
    if (!assertShape()) {
      throw std::runtime_error("Shape mismatch in AddNode backward");
    }
    for (size_t i = 0; i < this->inputs_[0]->shape()[0]; ++i) {
      if (this->inputs_[0]->requires_grad()) {
        this->inputs_[0]->grad()[i] += this->outputs_[0]->grad()[i];
      }
      if (this->inputs_[1]->requires_grad()) {
        this->inputs_[1]->grad()[i] += this->outputs_[0]->grad()[i];
      }
    }
  }
};


} // namespace modeldy

#endif  // MODELDY_INCLUDE_NODE_H_