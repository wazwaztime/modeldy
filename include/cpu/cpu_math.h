/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef MODELDY_INCLUDE_CPU_MATH_H_
#define MODELDY_INCLUDE_CPU_MATH_H_

#include <cmath>

namespace modeldy {

namespace cpu {

template <typename T>
inline T cpu_sigmoid(T x) {
  return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
}

template <typename T>
inline T cpu_tanh(T x) {
  return std::tanh(x);
}

template <typename T>
inline T cpu_relu(T x) {
  return x > static_cast<T>(0) ? x : static_cast<T>(0);
}

} // namespace cpu

} // namespace modeldy

#endif // MODELDY_INCLUDE_CPU_MATH_H_
