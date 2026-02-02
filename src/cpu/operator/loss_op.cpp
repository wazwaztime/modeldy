/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <include/cpu/operator/loss_op.h>
#include <include/operator_registry.h>

// Explicit instantiation for float
template class modeldy::cpu::cpuMSELoss<float>;
template class modeldy::cpu::cpuBCELoss<float>;
template class modeldy::cpu::cpuCrossEntropyLoss<float>;

// Explicit instantiation for double
template class modeldy::cpu::cpuMSELoss<double>;
template class modeldy::cpu::cpuBCELoss<double>;
template class modeldy::cpu::cpuCrossEntropyLoss<double>;

// Register operators for float
REGISTER_OPERATOR(float, MSELoss, cpu);
REGISTER_OPERATOR(float, BCELoss, cpu);
REGISTER_OPERATOR(float, CrossEntropyLoss, cpu);

// Register operators for double
REGISTER_OPERATOR(double, MSELoss, cpu);
REGISTER_OPERATOR(double, BCELoss, cpu);
REGISTER_OPERATOR(double, CrossEntropyLoss, cpu);
