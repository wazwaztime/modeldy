/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef MODELDY_INCLUDE_OPERATOR_REGISTRY_H_
#define MODELDY_INCLUDE_OPERATOR_REGISTRY_H_

#include <vector>
#include <unordered_map>
#include <functional>
#include <memory>
#include <string>
#include <stdexcept>

namespace modeldy {

template <typename T>
class Node;

template <typename T>
class ComputeNode;

/*! \brief Factory function type for creating compute nodes */
template <typename T>
using ComputeNodeFactory = std::function<std::shared_ptr<ComputeNode<T>>(
    const std::vector<std::shared_ptr<Node<T>>>&,
    const std::vector<std::shared_ptr<Node<T>>>&,
    const std::string&)>;

/*! \brief Operator registry for registering and creating compute nodes */
template <typename T>
class OperatorRegistry {
 public:
  static OperatorRegistry<T>& getInstance() {
    static OperatorRegistry<T> instance;
    return instance;
  }

  void registerOperator(const std::string& class_name,
                        const std::string& device,
                        ComputeNodeFactory<T> factory) {
    std::string key = device + "::" + class_name;
    registry_[key] = factory;
  }

  std::shared_ptr<ComputeNode<T>> createOperator(
      const std::string& class_name,
      const std::string& device,
      const std::vector<std::shared_ptr<Node<T>>>& inputs,
      const std::vector<std::shared_ptr<Node<T>>>& outputs,
      const std::string& name) {
    std::string key = device + "::" + class_name;
    auto it = registry_.find(key);
    if (it == registry_.end()) {
      throw std::runtime_error("Operator " + class_name + " not registered for device " + device);
    }
    return it->second(inputs, outputs, name);
  }

  bool hasOperator(const std::string& class_name, const std::string& device) const {
    std::string key = device + "::" + class_name;
    return registry_.find(key) != registry_.end();
  }

 private:
  OperatorRegistry() = default;
  std::unordered_map<std::string, ComputeNodeFactory<T>> registry_;
};

/*! \brief Macro for registering operators */
#define REGISTER_OPERATOR(T, ClassName, DeviceType) \
  namespace { \
    struct ClassName##DeviceType##Registrar { \
      ClassName##DeviceType##Registrar() { \
        modeldy::OperatorRegistry<T>::getInstance().registerOperator( \
            #ClassName, \
            #DeviceType, \
            [](const std::vector<std::shared_ptr<modeldy::Node<T>>>& inputs, \
               const std::vector<std::shared_ptr<modeldy::Node<T>>>& outputs, \
               const std::string& name) -> std::shared_ptr<modeldy::ComputeNode<T>> { \
              return std::make_shared<modeldy::DeviceType::DeviceType##ClassName<T>>( \
                  inputs, outputs, name); \
            }); \
      } \
    }; \
    static ClassName##DeviceType##Registrar g_##ClassName##DeviceType##Registrar; \
  }

} // namespace modeldy

#endif  // MODELDY_INCLUDE_OPERATOR_REGISTRY_H_
