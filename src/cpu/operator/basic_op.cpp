/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <include/cpu/operator/basic_op.h>
#include <include/operator_registry.h>

// Explicit instantiation for float
template class modeldy::cpu::cpuAdd<float>;
template class modeldy::cpu::cpuMul<float>;

// Explicit instantiation for double
template class modeldy::cpu::cpuAdd<double>;
template class modeldy::cpu::cpuMul<double>;

// Register operators for float
REGISTER_OPERATOR(float, Add, cpu);
REGISTER_OPERATOR(float, Mul, cpu);

// Register operators for double
REGISTER_OPERATOR(double, Add, cpu);
REGISTER_OPERATOR(double, Mul, cpu);
