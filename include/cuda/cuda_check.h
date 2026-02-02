/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef MODELDY_INCLUDE_CUDA_CHECK_H_
#define MODELDY_INCLUDE_CUDA_CHECK_H_

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <iostream>

/*!
 * \brief Macro to check CUDA function calls for errors
 * \param call The CUDA function call to check
 * \throws std::runtime_error if the CUDA call returns an error
 */
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = (call);                                                \
    if (error != cudaSuccess) {                                                \
      std::string error_msg = "CUDA error at " + std::string(__FILE__) + ":" + \
                              std::to_string(__LINE__) + " - " +               \
                              std::string(cudaGetErrorString(error));          \
      std::cerr << "[ERROR] " << error_msg << std::endl;                       \
      throw std::runtime_error(error_msg);                                     \
    }                                                                          \
  } while (0)

/*!
 * \brief Macro to check CUDA function calls for errors with a custom message
 * \param call The CUDA function call to check
 * \param custom_msg A custom message to prepend to the error description
 * \throws std::runtime_error if the CUDA call returns an error
 */
#define CUDA_CHECK_MSG(call, custom_msg)                                       \
  do {                                                                         \
    cudaError_t error = (call);                                                \
    if (error != cudaSuccess) {                                                \
      std::string error_msg = "CUDA error at " + std::string(__FILE__) + ":" + \
                              std::to_string(__LINE__) + " - " +               \
                              std::string(custom_msg) + " - " +                \
                              std::string(cudaGetErrorString(error));          \
      std::cerr << "[ERROR] " << error_msg << std::endl;                       \
      throw std::runtime_error(error_msg);                                     \
    }                                                                          \
  } while (0)

#endif // MODELDY_INCLUDE_CUDA_CHECK_H_