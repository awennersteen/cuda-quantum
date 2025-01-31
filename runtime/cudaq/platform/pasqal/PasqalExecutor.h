/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/Executor.h"

namespace cudaq {

class PasqalExecutor : public Executor {
public:
  PasqalExecutor() = default;
  ~PasqalExecutor() = default;

  /// @brief Execute the provided Pasqal task
  details::future execute(std::vector<KernelExecution> &codesToExecute,
                          bool isObserve) override;
};

} // namespace cudaq