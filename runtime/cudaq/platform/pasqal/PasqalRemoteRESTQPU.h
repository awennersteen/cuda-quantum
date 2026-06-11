/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/AnalogRemoteRESTQPU.h"

namespace cudaq {

/// @brief The PasqalRemoteRESTQPU is a subtype of QPU that enables the
/// execution of Analog Hamiltonian Programs via a REST Client.
class PasqalRemoteRESTQPU : public AnalogRemoteRESTQPU {
#ifdef CUDAQ_PASQAL_ENABLE_DYNAMICS_EMULATION
  static constexpr double rydbergC6 = 5.42e-24;

protected:
  cudaq::sample_result emulateProgram(const std::string &programString,
                                      std::size_t shots,
                                      std::size_t seed) override;
#endif

public:
  PasqalRemoteRESTQPU() : AnalogRemoteRESTQPU() {}
  PasqalRemoteRESTQPU(PasqalRemoteRESTQPU &&) = delete;
  ~PasqalRemoteRESTQPU() override;
};

} // namespace cudaq
