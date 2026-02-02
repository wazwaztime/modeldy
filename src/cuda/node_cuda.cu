/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <include/cuda/node_cuda.h>

namespace modeldy {

// Explicit instantiation for float
template class cudaDataNode<float>;

// Explicit instantiation for double
template class cudaDataNode<double>;

namespace cuda {

// Explicit instantiation for float
template class cudaComputeNode<float>;

// Explicit instantiation for double
template class cudaComputeNode<double>;

} // namespace cuda
} // namespace modeldy
