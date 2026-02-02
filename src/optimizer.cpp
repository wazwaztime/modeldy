/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <include/optimizer.h>

#ifdef USE_CUDA
#include <include/cuda/optimizer_kernels.h>
#endif

namespace modeldy {

// Explicit instantiation for float
template class Optimizer<float>;
template class SGD<float>;
template class Adam<float>;
template class RMSprop<float>;

// Double precision not supported for CUDA due to atomicAdd limitations
// Only float version is instantiated

} // namespace modeldy
