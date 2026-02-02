/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <include/memory_pool.h>

namespace modeldy {
namespace cpu {

// Explicit instantiation for float
template class cpuMemoryPool<float>;

// Explicit instantiation for double
template class cpuMemoryPool<double>;

} // namespace cpu

#ifdef USE_CUDA
namespace cuda {

// Explicit instantiation for float
template class cudaMemoryPool<float>;

// Explicit instantiation for double
template class cudaMemoryPool<double>;

} // namespace cuda
#endif

} // namespace modeldy
