/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef MODELDY_INCLUDE_MODEL_H_
#define MODELDY_INCLUDE_MODEL_H_

#include <vector>
#include <unordered_map>
#include <memory>
#include <queue>

#include <modeldy/include/operator_registry.h>
#include <modeldy/include/cpu/operator.h>
#ifdef USE_CUDA
#include <modeldy/include/cuda/operator.h>
#endif // USE_CUDA

namespace modeldy {

/*! \brief model class */
template <typename T>
class Model {
 public:
  ~Model() = default;
  
  /*! \brief Create a new data node */
  void newDataNode(const std::string& name,
                   const std::vector<size_t>& shape,
                   bool requires_grad = false,
                   const std::string& device = "cpu") {
    compiled_ = false;
    if (node_map_.find(name) != node_map_.end()) {
      throw std::runtime_error("Node with name " + name + " already exists.");
    }
    std::shared_ptr<Node<T>> node;
    if (device == "cpu") {
      node = std::make_shared<cpu::cpuDataNode<T>>(shape, requires_grad, name);
    }
#ifdef USE_CUDA
    else if (device == "cuda") {
      node = std::make_shared<cuda::cudaDataNode<T>>(shape, requires_grad, name);
    }
#endif // USE_CUDA
    else {
      throw std::runtime_error("Unsupported device: " + device);
    }
    nodes_.push_back(node);
    node_map_[name] = node;
  }

  /*! \brief Create a new compute node */
  void newComputeNode(const std::string& class_name, 
                      const std::string& name, 
                      const std::vector<std::string>& input_names,
                      const std::vector<std::string>& output_names,
                      const std::string& device = "cpu"
                    ) {
    compiled_ = false;
    if (node_map_.find(name) != node_map_.end()) {
      throw std::runtime_error("Node with name " + name + " already exists.");
    }
    
    // Get input nodes
    std::vector<std::shared_ptr<Node<T>>> inputs;
    for (const auto& input_name : input_names) {
      auto it = node_map_.find(input_name);
      if (it == node_map_.end()) {
        throw std::runtime_error("Input node " + input_name + " not found.");
      }
      inputs.push_back(it->second);
    }
    
    // Get output nodes
    std::vector<std::shared_ptr<Node<T>>> outputs;
    for (const auto& output_name : output_names) {
      auto it = node_map_.find(output_name);
      if (it == node_map_.end()) {
        throw std::runtime_error("Output node " + output_name + " not found.");
      }
      outputs.push_back(it->second);
    }
    
    // Create compute node using registry
    auto node = OperatorRegistry<T>::getInstance().createOperator(
        class_name, device, inputs, outputs, name);
    
    // Initialize connections after the node is managed by shared_ptr
    if (auto compute_node = std::dynamic_pointer_cast<ComputeNode<T>>(node)) {
      compute_node->initialize_connections();
    }
    
    nodes_.push_back(node);
    node_map_[name] = node;
  }

  /*! \brief get data pointer of a data node by name */
  const T *data(const std::string& name) const {
    auto it = node_map_.find(name);
    if (it == node_map_.end()) {
      throw std::runtime_error("Node with name " + name + " not found.");
    }
    if (std::dynamic_pointer_cast<DataNode<T>>(it->second) == nullptr) {
      throw std::runtime_error("Node with name " + name + " is not a DataNode.");
    }
    return std::dynamic_pointer_cast<DataNode<T>>(it->second)->data();
  }

  /*! \brief get gradient pointer of a data node by name */
  const T *grad(const std::string& name) const {
    auto it = node_map_.find(name);
    if (it == node_map_.end()) {
      throw std::runtime_error("Node with name " + name + " not found.");
    }
    if (std::dynamic_pointer_cast<DataNode<T>>(it->second) == nullptr) {
      throw std::runtime_error("Node with name " + name + " is not a DataNode.");
    }
    if (!it->second->requires_grad()) {
      throw std::runtime_error("Node with name " + name + " does not require gradient.");
    }
    return std::dynamic_pointer_cast<DataNode<T>>(it->second)->grad();
  }

  /*! \brief set data of a data node by name */
  void setData(const std::string& name, const std::vector<T>& data) {
    auto it = node_map_.find(name);
    if (it == node_map_.end()) {
      throw std::runtime_error("Node with name " + name + " not found.");
    }
    if (std::dynamic_pointer_cast<DataNode<T>>(it->second) == nullptr) {
      throw std::runtime_error("Node with name " + name + " is not a DataNode.");
    }
    auto data_node = std::dynamic_pointer_cast<DataNode<T>>(it->second);
    size_t total_size = 1;
    for (const auto& dim : data_node->shape()) {
      total_size *= dim;
    }
    if (data.size() != total_size) {
      throw std::runtime_error("Data size does not match node shape.");
    }
    if (auto cpu_node = std::dynamic_pointer_cast<cpu::cpuDataNode<T>>(data_node)) {
      cpu_node->copy_from(data);
    }
#ifdef USE_CUDA
    else if (auto cuda_node = std::dynamic_pointer_cast<cuda::cudaDataNode<T>>(data_node)) {
      cuda_node->copy_from(data);
    }
#endif // USE_CUDA
    else {
      throw std::runtime_error("Unsupported device type for node " + name);
    }
  }

  /*! \brief set gradient of a data node by name */
  void setGrad(const std::string& name, const std::vector<T>& grad) {
    auto it = node_map_.find(name);
    if (it == node_map_.end()) {
      throw std::runtime_error("Node with name " + name + " not found.");
    }
    if (std::dynamic_pointer_cast<DataNode<T>>(it->second) == nullptr) {
      throw std::runtime_error("Node with name " + name + " is not a DataNode.");
    }
    auto data_node = std::dynamic_pointer_cast<DataNode<T>>(it->second);
    if (!data_node->requires_grad()) {
      throw std::runtime_error("Node with name " + name + " does not require gradient.");
    }
    size_t total_size = 1;
    for (const auto& dim : data_node->shape()) {
      total_size *= dim;
    }
    if (grad.size() != total_size) {
      throw std::runtime_error("Gradient size does not match node shape.");
    }
    T* grad_ptr = data_node->grad();
    for (size_t i = 0; i < total_size; ++i) {
      grad_ptr[i] = grad[i];
    }
  }

  /*! \brief zero all gradients */
  void zeroGrad() {
    for (const auto& node : nodes_) {
      if (node->requires_grad()) {
        if (auto data_node = std::dynamic_pointer_cast<DataNode<T>>(node)) {
          T* grad_ptr = data_node->grad();
          size_t total_size = 1;
          for (const auto& dim : data_node->shape()) {
            total_size *= dim;
          }
          for (size_t i = 0; i < total_size; ++i) {
            grad_ptr[i] = static_cast<T>(0);
          }
        }
      }
    }
  }

  /*! \brief compile the model (topological sort) */
  void compile() {
    if (compiled_) {
      return;
    }
    checkDAG();
    compute_nodes_ = std::move(topologicalSort());
    compiled_ = true;
  }

  /*! \brief perform prediction (forward pass) */
  void predict() {
    compile();
    for (const auto& node : compute_nodes_) {
      node->forward();
    }
  }

  /*! \brief perform training (backward pass) 
   * \param loss_node_name Optional name of the loss node to start backprop from.
   *                       If empty, will try to find a scalar output node automatically.
   * \param init_grad Initial gradient value for the loss node (default 1.0)
   */
  void backward(const std::string& loss_node_name = "", T init_grad = static_cast<T>(1)) {
    compile();
    
    // Initialize gradient for loss node
    if (!loss_node_name.empty()) {
      // User specified loss node
      auto it = node_map_.find(loss_node_name);
      if (it == node_map_.end()) {
        throw std::runtime_error("Loss node with name " + loss_node_name + " not found.");
      }
      if (auto data_node = std::dynamic_pointer_cast<DataNode<T>>(it->second)) {
        if (!data_node->requires_grad()) {
          throw std::runtime_error("Loss node " + loss_node_name + " does not require gradient.");
        }
        // Set initial gradient to init_grad
        T* grad_ptr = data_node->grad();
        size_t total_size = 1;
        for (const auto& dim : data_node->shape()) {
          total_size *= dim;
        }
        for (size_t i = 0; i < total_size; ++i) {
          grad_ptr[i] = (i == 0) ? init_grad : static_cast<T>(0);
        }
      }
    } else {
      // Auto-detect: find the last data node with requires_grad and shape [1]
      std::shared_ptr<DataNode<T>> loss_node = nullptr;
      for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it) {
        if (auto data_node = std::dynamic_pointer_cast<DataNode<T>>(*it)) {
          if (data_node->requires_grad()) {
            const auto& shape = data_node->shape();
            if (shape.size() == 1 && shape[0] == 1) {
              loss_node = data_node;
              break;
            }
          }
        }
      }
      if (loss_node) {
        T* grad_ptr = loss_node->grad();
        grad_ptr[0] = init_grad;
      }
    }
    
    // Perform backward pass
    for (auto it = compute_nodes_.rbegin(); it != compute_nodes_.rend(); ++it) {
      (*it)->backward();
    }
  }

 private:
  std::vector<std::shared_ptr<Node<T>>> nodes_;
  std::unordered_map<std::string, std::shared_ptr<Node<T>>> node_map_;
  std::vector<std::shared_ptr<ComputeNode<T>>> compute_nodes_;
  bool compiled_ = false;

  /*! \brief topological sort of compute nodes */
  std::vector<std::shared_ptr<ComputeNode<T>>> topologicalSort() {
    std::vector<std::shared_ptr<ComputeNode<T>>> sorted_nodes;
    std::unordered_map<std::shared_ptr<Node<T>>, int> in_degree;
    std::queue<std::shared_ptr<Node<T>>> zero_in_degree_queue;

    // Initialize in-degrees
    for (const auto& node : nodes_) {
      in_degree[node] = 0;
    }
    for (const auto& node : nodes_) {
      for (const auto& output : node->outputs()) {
        in_degree[output]++;
      }
    }

    // Enqueue nodes with zero in-degree
    for (const auto& pair : in_degree) {
      if (pair.second == 0) {
        zero_in_degree_queue.push(pair.first);
      }
    }

    // Perform topological sort
    while (!zero_in_degree_queue.empty()) {
      auto current_node = zero_in_degree_queue.front();
      zero_in_degree_queue.pop();

      if (auto compute_node = std::dynamic_pointer_cast<ComputeNode<T>>(current_node)) {
        sorted_nodes.push_back(compute_node);
      }

      for (const auto& output : current_node->outputs()) {
        in_degree[output]--;
        if (in_degree[output] == 0) {
          zero_in_degree_queue.push(output);
        }
      }
    }

    return sorted_nodes;
  }

  /*! \brief check if the graph is a Directed Acyclic Graph (DAG) */
  void checkDAG() {
    std::unordered_map<std::shared_ptr<Node<T>>, int> in_degree;
    for (const auto& node : nodes_) {
      in_degree[node] = 0;
    }
    for (const auto& node : nodes_) {
      for (const auto& output : node->outputs()) {
        in_degree[output]++;
      }
    }
    size_t visited_count = 0;
    std::queue<std::shared_ptr<Node<T>>> zero_in_degree_queue;
    for (const auto& pair : in_degree) {
      if (pair.second == 0) {
        zero_in_degree_queue.push(pair.first);
      }
    }
    while (!zero_in_degree_queue.empty()) {
      auto current_node = zero_in_degree_queue.front();
      zero_in_degree_queue.pop();
      visited_count++;
      for (const auto& output : current_node->outputs()) {
        in_degree[output]--;
        if (in_degree[output] == 0) {
          zero_in_degree_queue.push(output);
        }
      }
    }
    if (visited_count != nodes_.size()) {
      throw std::runtime_error("The computation graph is not a Directed Acyclic Graph (DAG).");
    }
  }

};

} // namespace modeldy

#endif  // MODELDY_INCLUDE_MODEL_H_