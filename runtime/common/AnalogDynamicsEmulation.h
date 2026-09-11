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
enum class ResultFormat { Binary, AtomState };
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
/// Output strings list sites in register order. Binary uses ground=0,
/// Rydberg=1; AtomState uses ground=2, Rydberg=1 and includes pre/post readout
/// registers.
sample_result sampleStateVector(const std::vector<std::complex<double>> &state,
                                std::size_t shots, std::size_t seed,
                                ResultFormat format = ResultFormat::Binary);
} // namespace ahs

cudaq::sample_result
emulateRydbergDynamics(const std::string &programString, std::size_t shots,
                       const ahs::DeviceSpecification &device, std::size_t seed,
                       ahs::ResultFormat format = ahs::ResultFormat::Binary);

} // namespace cudaq
