/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <include/cuda/data_transfer_node.h>

namespace modeldy {

// Explicit instantiation for float
template class HostToDeviceNode<float>;
template class DeviceToHostNode<float>;

// Explicit instantiation for double
template class HostToDeviceNode<double>;
template class DeviceToHostNode<double>;

} // namespace modeldy
