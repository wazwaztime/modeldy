/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <include/cuda/operator/basic_op.h>
#include <include/operator_registry.h>

// Explicit instantiation for float
template class modeldy::cuda::cudaAdd<float>;
template class modeldy::cuda::cudaMul<float>;

// Register operators for float
REGISTER_OPERATOR(float, Add, cuda);
REGISTER_OPERATOR(float, Mul, cuda);
