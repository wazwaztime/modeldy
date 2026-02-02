/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef MODELDY_INCLUDE_CUDA_OPTIMIZER_KERNELS_H_
#define MODELDY_INCLUDE_CUDA_OPTIMIZER_KERNELS_H_

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <modeldy/include/cuda/cuda_check.h>

namespace modeldy {
namespace cuda {

// ============================================================================
// SGD Kernels
// ============================================================================

/*!
 * \brief CUDA kernel for SGD update without momentum
 * \param data Device pointer to parameter data
 * \param grad Device pointer to gradients
 * \param size Number of elements
 * \param lr Learning rate
 * \param weight_decay Weight decay coefficient
 * 
 * Formula: data = data - lr * (grad + weight_decay * data)
 */
template <typename T>
__global__ void sgd_kernel(T* data, const T* grad, size_t size, T lr, T weight_decay) {
  // TODO: Implement SGD kernel
  // int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // if (idx < size) {
  //   T g = grad[idx];
  //   if (weight_decay > 0) {
  //     g += weight_decay * data[idx];
  //   }
  //   data[idx] -= lr * g;
  // }
}

/*!
 * \brief CUDA kernel for SGD update with momentum
 * \param data Device pointer to parameter data
 * \param grad Device pointer to gradients
 * \param velocity Device pointer to velocity buffer
 * \param size Number of elements
 * \param lr Learning rate
 * \param momentum Momentum coefficient
 * \param weight_decay Weight decay coefficient
 * 
 * Formula: 
 *   velocity = momentum * velocity + (grad + weight_decay * data)
 *   data = data - lr * velocity
 */
template <typename T>
__global__ void sgd_momentum_kernel(T* data, const T* grad, T* velocity,
                                   size_t size, T lr, T momentum, T weight_decay) {
  // TODO: Implement SGD with momentum kernel
  // int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // if (idx < size) {
  //   T g = grad[idx];
  //   if (weight_decay > 0) {
  //     g += weight_decay * data[idx];
  //   }
  //   velocity[idx] = momentum * velocity[idx] + g;
  //   data[idx] -= lr * velocity[idx];
  // }
}

// ============================================================================
// Adam Kernels
// ============================================================================

/*!
 * \brief CUDA kernel for Adam optimizer update
 * \param data Device pointer to parameter data
 * \param grad Device pointer to gradients
 * \param m Device pointer to first moment
 * \param v Device pointer to second moment
 * \param size Number of elements
 * \param lr Learning rate
 * \param beta1 Exponential decay rate for first moment
 * \param beta2 Exponential decay rate for second moment
 * \param epsilon Small constant for numerical stability
 * \param weight_decay Weight decay coefficient
 * \param t Current timestep (for bias correction)
 * 
 * Formula:
 *   g = grad + weight_decay * data
 *   m = beta1 * m + (1 - beta1) * g
 *   v = beta2 * v + (1 - beta2) * g^2
 *   m_hat = m / (1 - beta1^t)
 *   v_hat = v / (1 - beta2^t)
 *   data = data - lr * m_hat / (sqrt(v_hat) + epsilon)
 */
template <typename T>
__global__ void adam_kernel(T* data, const T* grad, T* m, T* v,
                           size_t size, T lr, T beta1, T beta2,
                           T epsilon, T weight_decay, size_t t) {
  // TODO: Implement Adam kernel
  // int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // if (idx < size) {
  //   T g = grad[idx];
  //   if (weight_decay > 0) {
  //     g += weight_decay * data[idx];
  //   }
  //   
  //   m[idx] = beta1 * m[idx] + (1 - beta1) * g;
  //   v[idx] = beta2 * v[idx] + (1 - beta2) * g * g;
  //   
  //   T m_hat = m[idx] / (1 - pow(beta1, t));
  //   T v_hat = v[idx] / (1 - pow(beta2, t));
  //   
  //   data[idx] -= lr * m_hat / (sqrt(v_hat) + epsilon);
  // }
}

// ============================================================================
// RMSprop Kernels
// ============================================================================

/*!
 * \brief CUDA kernel for RMSprop optimizer update
 * \param data Device pointer to parameter data
 * \param grad Device pointer to gradients
 * \param square_avg Device pointer to squared gradient average
 * \param size Number of elements
 * \param lr Learning rate
 * \param alpha Decay rate
 * \param epsilon Small constant for numerical stability
 * \param weight_decay Weight decay coefficient
 * 
 * Formula:
 *   g = grad + weight_decay * data
 *   square_avg = alpha * square_avg + (1 - alpha) * g^2
 *   data = data - lr * g / (sqrt(square_avg) + epsilon)
 */
template <typename T>
__global__ void rmsprop_kernel(T* data, const T* grad, T* square_avg,
                              size_t size, T lr, T alpha,
                              T epsilon, T weight_decay) {
  // TODO: Implement RMSprop kernel
  // int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // if (idx < size) {
  //   T g = grad[idx];
  //   if (weight_decay > 0) {
  //     g += weight_decay * data[idx];
  //   }
  //   
  //   square_avg[idx] = alpha * square_avg[idx] + (1 - alpha) * g * g;
  //   data[idx] -= lr * g / (sqrt(square_avg[idx]) + epsilon);
  // }
}

// ============================================================================
// Kernel Launch Functions (Template Specializations)
// ============================================================================

// SGD kernel launchers
template <typename T>
void sgd_kernel_launch(T* data, const T* grad, size_t size, T lr, T weight_decay) {
  // TODO: Implement kernel launch
  // int block_size = 256;
  // int grid_size = (size + block_size - 1) / block_size;
  // sgd_kernel<<<grid_size, block_size>>>(data, grad, size, lr, weight_decay);
  // CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void sgd_momentum_kernel_launch(T* data, const T* grad, T* velocity,
                               size_t size, T lr, T momentum, T weight_decay) {
  // TODO: Implement kernel launch
  // int block_size = 256;
  // int grid_size = (size + block_size - 1) / block_size;
  // sgd_momentum_kernel<<<grid_size, block_size>>>(
  //     data, grad, velocity, size, lr, momentum, weight_decay);
  // CUDA_CHECK(cudaGetLastError());
}

// Adam kernel launcher
template <typename T>
void adam_kernel_launch(T* data, const T* grad, T* m, T* v, size_t size,
                       T lr, T beta1, T beta2, T epsilon, T weight_decay, size_t t) {
  // TODO: Implement kernel launch
  // int block_size = 256;
  // int grid_size = (size + block_size - 1) / block_size;
  // adam_kernel<<<grid_size, block_size>>>(
  //     data, grad, m, v, size, lr, beta1, beta2, epsilon, weight_decay, t);
  // CUDA_CHECK(cudaGetLastError());
}

// RMSprop kernel launcher
template <typename T>
void rmsprop_kernel_launch(T* data, const T* grad, T* square_avg, size_t size,
                          T lr, T alpha, T epsilon, T weight_decay) {
  // TODO: Implement kernel launch
  // int block_size = 256;
  // int grid_size = (size + block_size - 1) / block_size;
  // rmsprop_kernel<<<grid_size, block_size>>>(
  //     data, grad, square_avg, size, lr, alpha, epsilon, weight_decay);
  // CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace modeldy

#endif // USE_CUDA

#endif // MODELDY_INCLUDE_CUDA_OPTIMIZER_KERNELS_H_
