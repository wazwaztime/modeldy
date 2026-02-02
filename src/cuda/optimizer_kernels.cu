/*!  
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <include/cuda/optimizer_kernels.h>

namespace modeldy {
namespace cuda {

// SGD kernel launcher
template <typename T>
void sgd_kernel_launch(T* data, const T* grad, size_t size, T lr, T weight_decay) {
  int block_size = 256;
  int grid_size = (size + block_size - 1) / block_size;
  sgd_kernel<<<grid_size, block_size>>>(data, grad, size, lr, weight_decay);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void sgd_momentum_kernel_launch(T* data, const T* grad, T* velocity,
                               size_t size, T lr, T momentum, T weight_decay) {
  int block_size = 256;
  int grid_size = (size + block_size - 1) / block_size;
  sgd_momentum_kernel<<<grid_size, block_size>>>(
      data, grad, velocity, size, lr, momentum, weight_decay);
  CUDA_CHECK(cudaGetLastError());
}

// Adam kernel launcher
template <typename T>
void adam_kernel_launch(T* data, const T* grad, T* m, T* v, size_t size,
                       T lr, T beta1, T beta2, T epsilon, T weight_decay, size_t t) {
  int block_size = 256;
  int grid_size = (size + block_size - 1) / block_size;
  adam_kernel<<<grid_size, block_size>>>(
      data, grad, m, v, size, lr, beta1, beta2, epsilon, weight_decay, t);
  CUDA_CHECK(cudaGetLastError());
}

// RMSprop kernel launcher
template <typename T>
void rmsprop_kernel_launch(T* data, const T* grad, T* square_avg, size_t size,
                          T lr, T alpha, T epsilon, T weight_decay) {
  int block_size = 256;
  int grid_size = (size + block_size - 1) / block_size;
  rmsprop_kernel<<<grid_size, block_size>>>(
      data, grad, square_avg, size, lr, alpha, epsilon, weight_decay);
  CUDA_CHECK(cudaGetLastError());
}

// Explicit instantiation for float
template void sgd_kernel_launch<float>(float*, const float*, size_t, float, float);
template void sgd_momentum_kernel_launch<float>(float*, const float*, float*, size_t, float, float, float);
template void adam_kernel_launch<float>(float*, const float*, float*, float*, size_t, float, float, float, float, float, size_t);
template void rmsprop_kernel_launch<float>(float*, const float*, float*, size_t, float, float, float, float);

}  // namespace cuda
}  // namespace modeldy
