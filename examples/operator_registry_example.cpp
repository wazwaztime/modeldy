/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

/*
 * Example usage of the operator registration mechanism
 * 
 * This file demonstrates how to use the Model class with registered operators
 */

#include <modeldy/include/model.h>
#include <modeldy/include/operator_registry.h>
#include <iostream>

int main() {
  // Initialize operator registry (must be called before using Model)
  modeldy::InitializeOperatorRegistry<float>();
  
  // Create a model
  modeldy::Model<float> model;
  
  // Create data nodes
  model.newDataNode("x", {2, 3}, false, "cpu");
  model.newDataNode("y", {2, 3}, false, "cpu");
  model.newDataNode("sum", {2, 3}, false, "cpu");
  model.newDataNode("relu_out", {2, 3}, false, "cpu");
  
  // Create compute nodes using registered operators
  // Add operation: sum = x + y
  model.newComputeNode("Add", "add_node", {"x", "y"}, {"sum"}, "cpu");
  
  // ReLU operation: relu_out = ReLU(sum)
  model.newComputeNode("ReLU", "relu_node", {"sum"}, {"relu_out"}, "cpu");
  
  std::cout << "Model created successfully with registered operators!" << std::endl;
  
  return 0;
}
