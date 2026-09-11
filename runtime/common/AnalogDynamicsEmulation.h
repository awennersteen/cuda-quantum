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
/// Physics parameters for ideal AHS emulation, independent of a transport.
/// C6 includes 1/hbar and is measured in rad m^6 / s. Hardware constraints
/// and calibration data are not part of this specification.
struct DeviceSpecification {
  double rydbergC6;
  /// Sign of the Y quadrature: +1 for Pulser, -1 for Braket/Aquila.
  double phaseSign = 1.0;
};

/// FRESNEL_CAN1 cloud specification (2026-09-11): Rb-87, 60S.
/// Pulser's level-60 coefficient is 865723.02 rad us^-1 um^6.
inline constexpr DeviceSpecification fresnelCan{865723.02e-30};
/// QuEra Aquila: the Braket c6Coefficient in rad m^6 / s.
inline constexpr DeviceSpecification aquila{5.42e-24, -1.0};

DeviceSpecification deviceSpecification(const std::string &name);

/// Hamiltonian in rad/s, with time parameter `t` in seconds and site 0 at the
/// least significant tensor index, following CUDA-Q's operator convention.
/// Use |g> = |0>, |r> = |1>, and
/// Omega/2 (cos(phi) X + device.phaseSign sin(phi) Y).
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
