/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <include/cuda/operator/activation_op.h>
#include <include/operator_registry.h>

// Explicit instantiation for float
template class modeldy::cuda::cudaReLU<float>;
template class modeldy::cuda::cudaSigmoid<float>;
template class modeldy::cuda::cudaTanh<float>;

// Register operators for float
REGISTER_OPERATOR(float, ReLU, cuda);
REGISTER_OPERATOR(float, Sigmoid, cuda);
REGISTER_OPERATOR(float, Tanh, cuda);
