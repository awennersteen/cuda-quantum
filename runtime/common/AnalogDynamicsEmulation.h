/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/AnalogHamiltonian.h"
#include "common/SampleResult.h"
#include "cudaq/operators.h"
#include <cstddef>
#include <string>

namespace cudaq {

namespace ahs {
/// Hamiltonian in rad/s, with time parameter `t` in seconds and site 0 at the
/// least significant tensor index, following CUDA-Q's operator convention.
/// Use |g> = |0>, |r> = |1>, and Omega/2 (cos(phi) X + sin(phi) Y).
/// Amplitude and detuning are linear; phase is constant between time points.
struct RydbergModel {
  sum_op<matrix_handler> hamiltonian;
  dimension_map dimensions;
  std::vector<double> times;
  double maxStep;
};

RydbergModel makeRydbergModel(const Program &program,
                              const DeviceSpecification &device);

/// Sample a qubit state vector with site 0 in the least significant index.
/// Output strings list trap sites in register order: ground/vacant=0,
/// Rydberg=1. The state vector contains only occupied sites when a filling is
/// supplied.
sample_result sampleStateVector(const std::vector<std::complex<double>> &state,
                                std::size_t shots, std::size_t seed,
                                const std::vector<int> &filling = {});
} // namespace ahs

cudaq::sample_result
emulateRydbergDynamics(const ahs::Program &program, std::size_t shots,
                       const ahs::DeviceSpecification &device,
                       std::size_t seed);

} // namespace cudaq
