/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/AnalogDynamicsEmulation.h"
#include "common/EvolveResult.h"
#include "common/SimulationState.h"
#include "nlohmann/json.hpp"
#include "cudaq/algorithms/evolve_internal.h"
#include "cudaq/algorithms/integrator.h"
#include "cudaq/schedule.h"
#include <cuda_runtime_api.h>

namespace cudaq {
cudaq::sample_result
emulateRydbergDynamics(const std::string &programString, std::size_t shots,
                       const ahs::DeviceSpecification &device, std::size_t seed,
                       ahs::ResultFormat format) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
    throw std::runtime_error(
        "Analog emulation requires a CUDA-capable GPU for the CUDA-Q dynamics "
        "backend.");

  const auto program = nlohmann::json::parse(programString).get<ahs::Program>();
  const auto model = ahs::makeRydbergModel(program, device);
  cudaq::schedule schedule(model.times, {"t"});
  cudaq::integrators::runge_kutta integrator(4, model.maxStep);
  auto result = cudaq::detail::evolveSingle(
      model.hamiltonian, model.dimensions, schedule, cudaq::InitialState::ZERO,
      integrator, {}, {}, cudaq::IntermediateResultSave::None, std::nullopt);
  const auto &state = result.states.value().back();
  std::vector<std::complex<double>> amplitudes(1UL << state.get_num_qubits());
  state.to_host(amplitudes.data(), amplitudes.size());
  return ahs::sampleStateVector(amplitudes, shots, seed, format);
}

} // namespace cudaq
