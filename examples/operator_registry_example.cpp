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
  
  // Create a model
  modeldy::Model<float> model;
  
  // Create data nodes
  model.newDataNode("x", {2, 3}, false, "cpu");
  model.setData("x", {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f});
  model.newDataNode("y", {2, 3}, false, "cpu");
  model.setData("y", {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f});
  model.newDataNode("sum", {2, 3}, false, "cpu");
  model.newDataNode("relu_out", {2, 3}, false, "cpu");
  
  // Create compute nodes using registered operators
  // Add operation: sum = x + y
  model.newComputeNode("Add", "add_node", {"x", "y"}, {"sum"}, "cpu");
  
  // ReLU operation: relu_out = ReLU(sum)
  model.newComputeNode("ReLU", "relu_node", {"sum"}, {"relu_out"}, "cpu");

  model.predict();

  // Retrieve and print the output data
  const float* relu_data = model.data("relu_out");
  std::cout << "ReLU Output:" << std::endl;
  for (size_t i = 0; i < 2 * 3; ++i) {
    std::cout << relu_data[i] << " ";
  }
  std::cout << std::endl;
  return 0;
}
