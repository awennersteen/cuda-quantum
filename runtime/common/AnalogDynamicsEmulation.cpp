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
#include "cudaq/algorithms/evolve_internal.h"
#include "cudaq/algorithms/integrator.h"
#include "cudaq/schedule.h"
#include <algorithm>
#include <cmath>
#include <cuda_runtime_api.h>
#include <limits>

namespace cudaq {
namespace {
// RK4's final stage must use the left limit of a waveform interval. The next
// integrate() call starts at the same time with the new (right-continuous)
// phase.
class AhsRungeKutta : public integrators::runge_kutta {
  std::shared_ptr<double> intervalEnd;

public:
  AhsRungeKutta(double step, std::shared_ptr<double> end)
      : integrators::runge_kutta(4, step), intervalEnd(std::move(end)) {}

  void integrate(double targetTime) override {
    *intervalEnd = targetTime;
    integrators::runge_kutta::integrate(targetTime);
  }
};
} // namespace

cudaq::sample_result
emulateRydbergDynamics(const ahs::Program &program, std::size_t shots,
                       const ahs::DeviceSpecification &device,
                       std::size_t seed) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
    throw std::runtime_error(
        "Analog emulation requires a CUDA-capable GPU for the CUDA-Q dynamics "
        "backend.");

  const auto model = ahs::makeRydbergModel(program, device);
  if (model.dimensions.empty())
    return ahs::sampleStateVector({1.0}, shots, seed,
                                  program.setup.ahs_register.filling);
  auto intervalEnd = std::make_shared<double>(0.0);
  cudaq::schedule schedule(
      std::vector<std::complex<double>>(model.times.begin(), model.times.end()),
      {"t"},
      [intervalEnd](const std::string &, const std::complex<double> &time) {
        return std::complex<double>(
            std::min(time.real(),
                     std::nextafter(*intervalEnd,
                                    -std::numeric_limits<double>::infinity())));
      });
  AhsRungeKutta integrator(model.maxStep, intervalEnd);
  auto result = cudaq::detail::evolveSingle(
      model.hamiltonian, model.dimensions, schedule, cudaq::InitialState::ZERO,
      integrator, {}, {}, cudaq::IntermediateResultSave::None, std::nullopt);
  const auto &state = result.states.value().back();
  std::vector<std::complex<double>> amplitudes(1UL << state.get_num_qubits());
  state.to_host(amplitudes.data(), amplitudes.size());
  return ahs::sampleStateVector(amplitudes, shots, seed,
                                program.setup.ahs_register.filling);
}

} // namespace cudaq
