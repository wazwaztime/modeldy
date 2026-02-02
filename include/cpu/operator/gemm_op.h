/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef MODELDY_INCLUDE_CPU_OPERATOR_GEMM_OP_H_
#define MODELDY_INCLUDE_CPU_OPERATOR_GEMM_OP_H_

#include <include/cpu/node_cpu.h>
#include <cassert>

namespace modeldy {

namespace cpu {

template <typename T>
class cpuGemmOO : public cpuComputeNode<T> {
 public:
  explicit cpuGemmOO(const std::vector<NodePtr<T>>& input,
                     const std::vector<NodePtr<T>>& output,
                     const std::string& name = "",
                     T alpha = static_cast<T>(1),
                     T beta = static_cast<T>(0))
      : cpuComputeNode<T>(input, output, name),
        alpha_(alpha),
        beta_(beta) {
    this->validate_shape();
  }

  ~cpuGemmOO() override = default;

  /*! 
  * \brief validate the shape of the input and output
  * \details features must be the last dimension
  */
  void validate_shape() const override {
    assert(this->inputs().size() == 2 && "GemmOONode requires exactly two inputs (A and B)");
    assert(this->outputs().size() == 1 && "GemmOONode requires exactly one output (C)");
    const auto& shapeA = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[0])->shape();
    const auto& shapeB = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[1])->shape();
    const auto& shapeC = std::dynamic_pointer_cast<cpuDataNode<T>>(this->outputs()[0])->shape();
    assert(shapeA.size() >= 2 && shapeB.size() >= 2 && "Inputs A and B must be at least 2D tensors");
    assert(shapeC.size() >= 2 && "Output C must be at least a 2D tensor");
    size_t M = 1;
    for (size_t i = 0; i < shapeA.size() - 1; ++i) {
      M *= shapeA[i];
    }
    size_t K = shapeA[shapeA.size() - 1];
    size_t N = shapeB[shapeB.size() - 1];
    size_t B_K = 1;
    for (size_t i = 0; i < shapeB.size() - 1; ++i) {
      B_K *= shapeB[i];
    }
    assert(K == B_K && "Inner dimensions of A and B must match");
    size_t output_M = 1;
    for (size_t i = 0; i < shapeC.size() - 1; ++i) {
      output_M *= shapeC[i];
    }
    size_t output_N = shapeC[shapeC.size() - 1];
    assert(M == output_M && N == output_N && "Output shape must match the result of A*B");
  }

  /*! \brief forward computation */
  void forward() override {
    auto inputA = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[0]);
    auto inputB = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[1]);
    auto output = std::dynamic_pointer_cast<cpuDataNode<T>>(this->outputs()[0]);
    size_t M = 1;
    const auto& shapeA = inputA->shape();
    const auto& shapeB = inputB->shape();
    for (size_t i = 0; i < shapeA.size() - 1; ++i) {
      M *= shapeA[i];
    }
    size_t K = shapeA[shapeA.size() - 1];
    size_t N = shapeB[shapeB.size() - 1];
    T* A_data = inputA->data();
    T* B_data = inputB->data();
    T* C_data = output->data();
    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        T sum = static_cast<T>(0);
        for (size_t k = 0; k < K; ++k) {
          sum += A_data[m * K + k] * B_data[k * N + n];
        }
        C_data[m * N + n] = alpha_ * sum + beta_ * C_data[m * N + n];
      }
    }
  }

  /*! \brief backward computation */
  void backward() override {
    auto inputA = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[0]);
    auto inputB = std::dynamic_pointer_cast<cpuDataNode<T>>(this->inputs()[1]);
    auto output = std::dynamic_pointer_cast<cpuDataNode<T>>(this->outputs()[0]);
    size_t M = 1;
    const auto& shapeA = inputA->shape();
    const auto& shapeB = inputB->shape();
    for (size_t i = 0; i < shapeA.size() - 1; ++i) {
      M *= shapeA[i];
    }
    size_t K = shapeA[shapeA.size() - 1];
    size_t N = shapeB[shapeB.size() - 1];
    T* A_data = inputA->data();
    T* B_data = inputB->data();
    T* C_grad = output->grad();
    
    // Gradient for A: dL/dA[m,k] = sum_n(dL/dC[m,n] * B[k,n])
    if (this->inputs()[0]->requires_grad()) {
      T* A_grad = inputA->grad();
      for (size_t m = 0; m < M; ++m) {
        for (size_t k = 0; k < K; ++k) {
          T grad_sum_A = static_cast<T>(0);
          for (size_t n = 0; n < N; ++n) {
            grad_sum_A += C_grad[m * N + n] * B_data[k * N + n];
          }
          A_grad[m * K + k] += alpha_ * grad_sum_A;
        }
      }
    }
    
    // Gradient for B: dL/dB[k,n] = sum_m(dL/dC[m,n] * A[m,k])
    if (this->inputs()[1]->requires_grad()) {
      T* B_grad = inputB->grad();
      for (size_t k = 0; k < K; ++k) {
        for (size_t n = 0; n < N; ++n) {
          T grad_sum_B = static_cast<T>(0);
          for (size_t m = 0; m < M; ++m) {
            grad_sum_B += C_grad[m * N + n] * A_data[m * K + k];
          }
          B_grad[k * N + n] += alpha_ * grad_sum_B;
        }
      }
    }
  }

 private:
  T alpha_;
  T beta_;
};


} // namespace cpu

} // namespace modeldy

#endif // MODELDY_INCLUDE_CPU_OPERATOR_GEMM_OP_H_
