/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef MODELDY_INCLUDE_CUDA_CUDA_MATH_H_
#define MODELDY_INCLUDE_CUDA_CUDA_MATH_H_

#include <cuda_runtime.h>
#include <cmath>

namespace modeldy {

namespace cuda {

#ifdef __CUDACC__
// atomicAdd for double (for older compute capabilities that don't support native double atomicAdd)
__device__ __forceinline__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// Unified atomic add function
template <typename T>
__device__ __forceinline__ void atomic_add_wrapper(T* address, T val) {
    atomicAdd(address, val);
}

template <>
__device__ __forceinline__ void atomic_add_wrapper<double>(double* address, double val) {
    atomicAddDouble(address, val);
}
#endif // __CUDACC__

// Device math functions
template <typename T>
__device__ __forceinline__ T cuda_relu(T x) {
  return x > static_cast<T>(0) ? x : static_cast<T>(0);
}

template <typename T>
__device__ __forceinline__ T cuda_sigmoid(T x) {
  return static_cast<T>(1) / (static_cast<T>(1) + exp(-x));
}

template <typename T>
__device__ __forceinline__ T cuda_tanh(T x) {
  return tanh(x);
}

// Element-wise kernels
template <typename T>
__global__ void relu_forward_kernel(const T* input, T* output, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = cuda_relu(input[idx]);
  }
}

template <typename T>
__global__ void relu_backward_kernel(const T* input, const T* grad_output, 
                                     T* grad_input, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    grad_input[idx] += (input[idx] > static_cast<T>(0)) ? grad_output[idx] : static_cast<T>(0);
  }
}

template <typename T>
__global__ void sigmoid_forward_kernel(const T* input, T* output, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = cuda_sigmoid(input[idx]);
  }
}

template <typename T>
__global__ void sigmoid_backward_kernel(const T* output, const T* grad_output,
                                        T* grad_input, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    T sigmoid_val = output[idx];
    grad_input[idx] += sigmoid_val * (static_cast<T>(1) - sigmoid_val) * grad_output[idx];
  }
}

template <typename T>
__global__ void tanh_forward_kernel(const T* input, T* output, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = cuda_tanh(input[idx]);
  }
}

template <typename T>
__global__ void tanh_backward_kernel(const T* output, const T* grad_output,
                                     T* grad_input, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    T tanh_val = output[idx];
    grad_input[idx] += (static_cast<T>(1) - tanh_val * tanh_val) * grad_output[idx];
  }
}

// Element-wise binary operations
template <typename T>
__global__ void add_forward_kernel(const T* input1, const T* input2, 
                                   T* output, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input1[idx] + input2[idx];
  }
}

template <typename T>
__global__ void add_backward_kernel(const T* grad_output, T* grad_input, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    grad_input[idx] += grad_output[idx];
  }
}

template <typename T>
__global__ void mul_forward_kernel(const T* input1, const T* input2, 
                                   T* output, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input1[idx] * input2[idx];
  }
}

template <typename T>
__global__ void mul_backward_kernel(const T* input, const T* grad_output,
                                    T* grad_input, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    grad_input[idx] += grad_output[idx] * input[idx];
  }
}

// Loss function kernels
template <typename T>
__global__ void mse_forward_kernel(const T* predictions, const T* targets,
                                   T* partial_sums, size_t size) {
  extern __shared__ T sdata[];
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  
  T sum = 0;
  if (idx < size) {
    T diff = predictions[idx] - targets[idx];
    sum = diff * diff;
  }
  sdata[tid] = sum;
  __syncthreads();
  
  // Reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  
  if (tid == 0) {
    partial_sums[blockIdx.x] = sdata[0];
  }
}

template <typename T>
__global__ void mse_backward_kernel(const T* predictions, const T* targets,
                                    const T* grad_output, T* grad_predictions,
                                    size_t size, T factor) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    grad_predictions[idx] += factor * grad_output[0] * (predictions[idx] - targets[idx]);
  }
}

template <typename T>
__global__ void bce_forward_kernel(const T* predictions, const T* targets,
                                   T* partial_sums, size_t size, T epsilon) {
  extern __shared__ T sdata[];
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  
  T sum = 0;
  if (idx < size) {
    T pred = max(epsilon, min(static_cast<T>(1) - epsilon, predictions[idx]));
    sum = targets[idx] * log(pred) + 
          (static_cast<T>(1) - targets[idx]) * log(static_cast<T>(1) - pred);
  }
  sdata[tid] = sum;
  __syncthreads();
  
  // Reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  
  if (tid == 0) {
    partial_sums[blockIdx.x] = sdata[0];
  }
}

template <typename T>
__global__ void bce_backward_kernel(const T* predictions, const T* targets,
                                    const T* grad_output, T* grad_predictions,
                                    size_t size, T epsilon, T factor) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    T pred = max(epsilon, min(static_cast<T>(1) - epsilon, predictions[idx]));
    grad_predictions[idx] += factor * grad_output[0] * 
                            (-(targets[idx] / pred) + 
                             (static_cast<T>(1) - targets[idx]) / (static_cast<T>(1) - pred));
  }
}

template <typename T>
__global__ void cross_entropy_softmax_forward_kernel(const T* logits, const T* targets,
                                                     T* loss, size_t batch_size, 
                                                     size_t num_classes) {
  int batch_idx = blockIdx.x;
  if (batch_idx >= batch_size) return;
  
  const T* batch_logits = logits + batch_idx * num_classes;
  const T* batch_targets = targets + batch_idx * num_classes;
  
  // Find max logit for numerical stability
  T max_logit = batch_logits[0];
  for (size_t c = 1; c < num_classes; ++c) {
    max_logit = max(max_logit, batch_logits[c]);
  }
  
  // Compute softmax and loss
  T sum_exp = 0;
  for (size_t c = 0; c < num_classes; ++c) {
    sum_exp += exp(batch_logits[c] - max_logit);
  }
  
  T batch_loss = 0;
  for (size_t c = 0; c < num_classes; ++c) {
    T prob = exp(batch_logits[c] - max_logit) / sum_exp;
    batch_loss += -batch_targets[c] * log(max(prob, static_cast<T>(1e-7)));
  }
  
  atomic_add_wrapper(loss, batch_loss);
}

template <typename T>
__global__ void cross_entropy_softmax_backward_kernel(const T* logits, const T* targets,
                                                      const T* grad_output, T* grad_logits,
                                                      size_t batch_size, size_t num_classes,
                                                      T factor) {
  int batch_idx = blockIdx.x;
  int class_idx = threadIdx.x;
  
  if (batch_idx >= batch_size || class_idx >= num_classes) return;
  
  const T* batch_logits = logits + batch_idx * num_classes;
  const T* batch_targets = targets + batch_idx * num_classes;
  T* batch_grad = grad_logits + batch_idx * num_classes;
  
  // Compute softmax
  __shared__ T max_logit;
  __shared__ T sum_exp;
  
  if (class_idx == 0) {
    T max_val = batch_logits[0];
    for (size_t c = 1; c < num_classes; ++c) {
      max_val = max(max_val, batch_logits[c]);
    }
    max_logit = max_val;
  }
  __syncthreads();
  
  // Compute sum of exponentials
  T local_exp = exp(batch_logits[class_idx] - max_logit);
  
  // Parallel reduction to compute sum_exp
  __shared__ T exp_values[1024]; // Assuming max 1024 classes
  exp_values[class_idx] = local_exp;
  __syncthreads();
  
  if (class_idx == 0) {
    T total = 0;
    for (size_t c = 0; c < num_classes; ++c) {
      total += exp_values[c];
    }
    sum_exp = total;
  }
  __syncthreads();
  
  // Compute gradient: (softmax - target) * factor
  T softmax_val = local_exp / sum_exp;
  batch_grad[class_idx] += factor * grad_output[0] * (softmax_val - batch_targets[class_idx]);
}

#endif // __CUDACC__

} // namespace cuda

} // namespace modeldy

#endif // MODELDY_INCLUDE_CUDA_CUDA_MATH_H_
