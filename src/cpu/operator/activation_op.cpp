/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <include/cpu/operator/activation_op.h>
#include <include/operator_registry.h>

// Explicit instantiation for float
template class modeldy::cpu::cpuReLU<float>;
template class modeldy::cpu::cpuSigmoid<float>;
template class modeldy::cpu::cpuTanh<float>;

// Explicit instantiation for double
template class modeldy::cpu::cpuReLU<double>;
template class modeldy::cpu::cpuSigmoid<double>;
template class modeldy::cpu::cpuTanh<double>;

// Register operators for float
REGISTER_OPERATOR(float, ReLU, cpu);
REGISTER_OPERATOR(float, Sigmoid, cpu);
REGISTER_OPERATOR(float, Tanh, cpu);

// Register operators for double
REGISTER_OPERATOR(double, ReLU, cpu);
REGISTER_OPERATOR(double, Sigmoid, cpu);
REGISTER_OPERATOR(double, Tanh, cpu);
