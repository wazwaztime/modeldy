/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <include/cuda/operator/loss_op.h>
#include <include/operator_registry.h>

// Explicit instantiation for float
template class modeldy::cuda::cudaMSELoss<float>;
template class modeldy::cuda::cudaBCELoss<float>;
template class modeldy::cuda::cudaCrossEntropyLoss<float>;

// Register operators for float
REGISTER_OPERATOR(float, MSELoss, cuda);
REGISTER_OPERATOR(float, BCELoss, cuda);
REGISTER_OPERATOR(float, CrossEntropyLoss, cuda);
