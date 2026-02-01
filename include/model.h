/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef MODELDY_INCLUDE_MODEL_H_
#define MODELDY_INCLUDE_MODEL_H_

namespace modeldy {

/*! \brief model class */
template <typename T>
class Model {
 public:
  Model();
  ~Model();

  void Train();
  void Predict();
};

} // namespace modeldy

#endif  // MODELDY_INCLUDE_MODEL_H_