/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <include/cpu/operator/gemm_op.h>
#include <include/operator_registry.h>

// Explicit instantiation for float
template class modeldy::cpu::cpuGemmOO<float>;

// Explicit instantiation for double
template class modeldy::cpu::cpuGemmOO<double>;

// Register operators for float
REGISTER_OPERATOR(float, GemmOO, cpu);

// Register operators for double
REGISTER_OPERATOR(double, GemmOO, cpu);
