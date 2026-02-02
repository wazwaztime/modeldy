/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

/*
 * Example usage of matrix multiplication (GEMM) operation
 * 
 * This file demonstrates how to perform matrix multiplication using the GemmOO operator
 * Test case: C = A * B
 * A: 2x3 matrix
 * B: 3x2 matrix
 * C: 2x2 matrix (result)
 */

#include <modeldy/include/model.h>
#include <modeldy/include/operator_registry.h>
#include <iostream>
#include <iomanip>

void print_matrix(const float* data, size_t rows, size_t cols, const std::string& name) {
  std::cout << name << " (" << rows << "x" << cols << "):" << std::endl;
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      std::cout << std::setw(8) << std::fixed << std::setprecision(2) << data[i * cols + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int main() {
  // Create a model
  modeldy::Model<float> model;
  
  // Create matrix A (2x3)
  // [1, 2, 3]
  // [4, 5, 6]
  model.newDataNode("A", {2, 3}, false, "cpu");
  model.setData("A", {1.0f, 2.0f, 3.0f, 
                      4.0f, 5.0f, 6.0f});
  
  // Create matrix B (3x2)
  // [7,  8]
  // [9, 10]
  // [11, 12]
  model.newDataNode("B", {3, 2}, false, "cpu");
  model.setData("B", {7.0f, 8.0f,
                      9.0f, 10.0f,
                      11.0f, 12.0f});
  
  // Create output matrix C (2x2)
  model.newDataNode("C", {2, 2}, false, "cpu");
  
  // Create GEMM compute node: C = A * B
  model.newComputeNode("GemmOO", "gemm_node", {"A", "B"}, {"C"}, "cpu");
  
  // Print input matrices
  std::cout << "Matrix Multiplication Test: C = A * B" << std::endl;
  std::cout << "======================================" << std::endl << std::endl;
  
  const float* A_data = model.data("A");
  const float* B_data = model.data("B");
  
  print_matrix(A_data, 2, 3, "Matrix A");
  print_matrix(B_data, 3, 2, "Matrix B");
  
  // Perform forward computation
  model.predict();
  
  // Retrieve and print the result
  const float* C_data = model.data("C");
  print_matrix(C_data, 2, 2, "Result C");
  
  // Expected result:
  // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
  // C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
  // C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
  // C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
  
  std::cout << "Expected result:" << std::endl;
  std::cout << "  58.00    64.00" << std::endl;
  std::cout << " 139.00   154.00" << std::endl << std::endl;
  
  // Verify results
  float expected[] = {58.0f, 64.0f, 139.0f, 154.0f};
  bool all_correct = true;
  for (size_t i = 0; i < 4; ++i) {
    if (std::abs(C_data[i] - expected[i]) > 1e-5) {
      all_correct = false;
      break;
    }
  }
  
  if (all_correct) {
    std::cout << "✓ Test PASSED: All results match expected values!" << std::endl;
  } else {
    std::cout << "✗ Test FAILED: Results do not match expected values!" << std::endl;
  }
  
  return 0;
}
