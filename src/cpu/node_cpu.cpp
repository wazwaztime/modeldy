/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <include/cpu/node_cpu.h>

namespace modeldy {
namespace cpu {

// Explicit instantiation for float
template class cpuDataNode<float>;
template class cpuComputeNode<float>;

// Explicit instantiation for double
template class cpuDataNode<double>;
template class cpuComputeNode<double>;

} // namespace cpu
} // namespace modeldy
