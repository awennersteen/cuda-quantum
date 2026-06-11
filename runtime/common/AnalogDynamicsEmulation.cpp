/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/AnalogDynamicsEmulation.h"
#include "common/AnalogHamiltonian.h"
#include "common/EvolveResult.h"
#include "common/SimulationState.h"
#include "cudaq/algorithms/evolve_internal.h"
#include "cudaq/algorithms/integrator.h"
#include "cudaq/operators.h"
#include "cudaq/schedule.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cuda_runtime_api.h>
#include <functional>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace cudaq {
namespace {

double interpolate(const cudaq::ahs::TimeSeries &series, double time) {
  if (series.times.empty())
    return 0.0;
  if (time <= series.times.front())
    return series.values.front();
  for (std::size_t i = 1; i < series.times.size(); ++i) {
    if (time <= series.times[i]) {
      const auto t0 = series.times[i - 1];
      const auto t1 = series.times[i];
      const auto v0 = series.values[i - 1];
      const auto v1 = series.values[i];
      return v0 + (v1 - v0) * ((time - t0) / (t1 - t0));
    }
  }
  return series.values.back();
}

std::vector<double> uniqueTimes(const cudaq::ahs::Program &program) {
  std::vector<double> times;
  for (const auto &drive : program.hamiltonian.drivingFields) {
    times.insert(times.end(), drive.amplitude.time_series.times.begin(),
                 drive.amplitude.time_series.times.end());
    times.insert(times.end(), drive.phase.time_series.times.begin(),
                 drive.phase.time_series.times.end());
    times.insert(times.end(), drive.detuning.time_series.times.begin(),
                 drive.detuning.time_series.times.end());
  }
  std::sort(times.begin(), times.end());
  times.erase(std::unique(times.begin(), times.end()), times.end());
  return times;
}

double distance(const std::vector<double> &a, const std::vector<double> &b) {
  const auto dx = a[0] - b[0];
  const auto dy = a[1] - b[1];
  return std::sqrt(dx * dx + dy * dy);
}

void validateUniform(const cudaq::ahs::PhysicalField &field) {
  if (field.pattern.patternStr != "uniform" ||
      !field.pattern.patternVals.empty())
    throw std::runtime_error(
        "Analog emulation supports only uniform driving fields.");
}

cudaq::scalar_operator
fieldCoefficient(const cudaq::ahs::TimeSeries &series,
                 const std::function<double(double)> &scale) {
  return cudaq::scalar_operator(
      [series,
       scale](const std::unordered_map<std::string, std::complex<double>>
                  &parameters) {
        const auto time = parameters.at("t").real();
        return std::complex<double>(scale(interpolate(series, time)), 0.0);
      },
      {{"t", "time"}});
}

cudaq::sum_op<cudaq::matrix_handler>
rydbergHamiltonian(const cudaq::ahs::Program &program, double rydbergC6) {
  const auto &drive = program.hamiltonian.drivingFields.front();
  const auto xCoefficient = cudaq::scalar_operator(
      [drive](const std::unordered_map<std::string, std::complex<double>>
                  &parameters) {
        const auto time = parameters.at("t").real();
        const auto omega = interpolate(drive.amplitude.time_series, time);
        const auto phase = interpolate(drive.phase.time_series, time);
        return std::complex<double>(0.5 * omega * std::cos(phase), 0.0);
      },
      {{"t", "time"}});
  const auto yCoefficient = cudaq::scalar_operator(
      [drive](const std::unordered_map<std::string, std::complex<double>>
                  &parameters) {
        const auto time = parameters.at("t").real();
        const auto omega = interpolate(drive.amplitude.time_series, time);
        const auto phase = interpolate(drive.phase.time_series, time);
        return std::complex<double>(-0.5 * omega * std::sin(phase), 0.0);
      },
      {{"t", "time"}});
  const auto detuningCoefficient = fieldCoefficient(
      drive.detuning.time_series, [](double delta) { return -0.5 * delta; });

  auto hamiltonian = cudaq::spin_op::empty();
  const auto &sites = program.setup.ahs_register.sites;
  for (std::size_t i = 0; i < sites.size(); ++i) {
    hamiltonian += xCoefficient * cudaq::spin_op::x(i);
    hamiltonian += yCoefficient * cudaq::spin_op::y(i);
    hamiltonian += detuningCoefficient *
                   (cudaq::spin_op::identity(i) - cudaq::spin_op::z(i));
  }

  for (std::size_t i = 0; i < sites.size(); ++i) {
    for (std::size_t j = i + 1; j < sites.size(); ++j) {
      const auto interaction =
          0.25 * rydbergC6 / std::pow(distance(sites[i], sites[j]), 6);
      hamiltonian +=
          interaction *
          (cudaq::spin_op::identity(i) * cudaq::spin_op::identity(j) -
           cudaq::spin_op::identity(i) * cudaq::spin_op::z(j) -
           cudaq::spin_op::z(i) * cudaq::spin_op::identity(j) +
           cudaq::spin_op::z(i) * cudaq::spin_op::z(j));
    }
  }
  return hamiltonian;
}

cudaq::sample_result sampleFinalState(cudaq::state state, std::size_t shots,
                                      std::size_t seed) {
  const auto numQubits = state.get_num_qubits();
  std::vector<double> probabilities;
  probabilities.reserve(1UL << numQubits);
  std::vector<std::string> bitstrings;
  bitstrings.reserve(1UL << numQubits);
  for (std::size_t basis = 0; basis < (1UL << numQubits); ++basis) {
    std::string bitstring(numQubits, '0');
    for (std::size_t qubit = 0; qubit < numQubits; ++qubit)
      bitstring[qubit] = ((basis >> (numQubits - qubit - 1)) & 1) ? '1' : '0';
    probabilities.emplace_back(std::norm(state({basis}, 0)));
    bitstrings.emplace_back(std::move(bitstring));
  }

  std::random_device randomDevice;
  std::mt19937 generator(seed ? seed : randomDevice());
  std::discrete_distribution<std::size_t> distribution(probabilities.begin(),
                                                       probabilities.end());
  cudaq::CountsDictionary counts;
  for (std::size_t i = 0; i < shots; ++i)
    counts[bitstrings[distribution(generator)]]++;
  return cudaq::sample_result(cudaq::ExecutionResult(counts));
}

cudaq::ahs::Program parseAndValidateProgram(const std::string &programString) {
  auto program =
      nlohmann::json::parse(programString).get<cudaq::ahs::Program>();
  if (program.hamiltonian.drivingFields.size() != 1)
    throw std::runtime_error(
        "Analog emulation supports one uniform driving field.");
  if (!program.hamiltonian.localDetuning.empty())
    throw std::runtime_error(
        "Analog emulation does not support local detuning.");
  for (auto filled : program.setup.ahs_register.filling)
    if (filled != 1)
      throw std::runtime_error(
          "Analog emulation does not support empty trap sites.");
  const auto &drive = program.hamiltonian.drivingFields.front();
  validateUniform(drive.amplitude);
  validateUniform(drive.phase);
  validateUniform(drive.detuning);
  return program;
}

} // namespace

cudaq::sample_result emulateRydbergDynamics(const std::string &programString,
                                            std::size_t shots, double rydbergC6,
                                            std::size_t seed) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
    throw std::runtime_error(
        "Analog emulation requires a CUDA-capable GPU for the CUDA-Q dynamics "
        "backend.");

  auto program = parseAndValidateProgram(programString);
  cudaq::dimension_map dimensions;
  for (std::size_t i = 0; i < program.setup.ahs_register.sites.size(); ++i)
    dimensions[i] = 2;

  const auto times = uniqueTimes(program);
  cudaq::schedule schedule(times, {"t"});
  /* Hardcoded 4th order Runge-Kutta with a 1ns time step, almost guaranteed to
   * be stable */
  cudaq::integrators::runge_kutta integrator(4, 1e-9);
  auto result = cudaq::detail::evolveSingle(
      rydbergHamiltonian(program, rydbergC6), dimensions, schedule,
      cudaq::InitialState::ZERO, integrator, {}, {},
      cudaq::IntermediateResultSave::None, std::nullopt);
  return sampleFinalState(result.states.value().back(), shots, seed);
}

} // namespace cudaq
