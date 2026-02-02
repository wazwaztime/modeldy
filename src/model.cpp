/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <include/model.h>

namespace modeldy {

// Explicit instantiation for float
template class Model<float>;

// Double precision not supported for CUDA due to atomicAdd limitations
// Only float version is instantiated

} // namespace modeldy
