/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/SampleResult.h"
#include <cstddef>
#include <string>

namespace cudaq {

cudaq::sample_result emulateRydbergDynamics(const std::string &programString,
                                            std::size_t shots, double rydbergC6,
                                            std::size_t seed);

} // namespace cudaq
