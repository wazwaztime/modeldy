/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef MODELDY_INCLUDE_MODEL_H_
#define MODELDY_INCLUDE_MODEL_H_

#include <modeldy/include/node.h>

#include <vector>

namespace modeldy {

/*! \brief model class */
template <typename T>
class Model {
 public:
  Model();
  ~Model();

  void Train();
  void Predict();

 private:
  std::vector<std::shared_ptr<Node<T>>> nodes_;
  

};

} // namespace modeldy

#endif  // MODELDY_INCLUDE_MODEL_H_