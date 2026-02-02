/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef MODELDY_INCLUDE_MODEL_H_
#define MODELDY_INCLUDE_MODEL_H_

#include <vector>
#include <unordered_map>
#include <memory>

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
  
  void newDataNode(const std::string& name,
                   const std::vector<size_t>& shape,
                   bool requires_grad = false,
                   const std::string& device = "cpu") {
    if (node_map_.find(name) != node_map_.end()) {
      throw std::runtime_error("Node with name " + name + " already exists.");
    }
    std::shared_ptr<Node<T>> node;
    if (device == "cpu") {
      node = std::make_shared<modeldy::cpu::cpuDataNode<T>>(shape, requires_grad, name);
    }
#ifdef USE_CUDA
    else if (device == "cuda") {
      node = std::make_shared<modeldy::cuda::cudaDataNode<T>>(shape, requires_grad, name);
    }
#endif // USE_CUDA
    else {
      throw std::runtime_error("Unsupported device: " + device);
    }
    nodes_.push_back(node);
    node_map_[name] = node;
  }

  void newComputeNode(const std::string& class_name, 
                      const std::string& name, 
                      const std::vector<std::string>& input_names,
                      const std::vector<std::string>& output_names,
                      const std::string& device = "cpu"
                    ) {
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
    
    nodes_.push_back(node);
    node_map_[name] = node;
  }

  void Predict();

 private:
  std::vector<std::shared_ptr<Node<T>>> nodes_;
  std::unordered_map<std::string, std::shared_ptr<Node<T>>> node_map_;
};

} // namespace modeldy

#endif  // MODELDY_INCLUDE_MODEL_H_