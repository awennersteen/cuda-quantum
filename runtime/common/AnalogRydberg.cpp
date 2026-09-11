/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/AnalogDynamicsEmulation.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <functional>
#include <random>

namespace cudaq {
namespace {

double interpolate(const cudaq::ahs::TimeSeries &series, double time,
                   bool constant = false) {
  if (series.times.empty())
    return 0.0;
  if (constant) {
    auto next =
        std::upper_bound(series.times.begin(), series.times.end(), time);
    return series
        .values[next == series.times.begin() ? 0
                                             : next - series.times.begin() - 1];
  }
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
        const auto phase = interpolate(drive.phase.time_series, time, true);
        return std::complex<double>(0.5 * omega * std::cos(phase), 0.0);
      },
      {{"t", "time"}});
  const auto yCoefficient = cudaq::scalar_operator(
      [drive](const std::unordered_map<std::string, std::complex<double>>
                  &parameters) {
        const auto time = parameters.at("t").real();
        const auto omega = interpolate(drive.amplitude.time_series, time);
        const auto phase = interpolate(drive.phase.time_series, time, true);
        return std::complex<double>(0.5 * omega * std::sin(phase), 0.0);
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

void validateProgram(const cudaq::ahs::Program &program) {
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
}

} // namespace

sample_result
ahs::sampleStateVector(const std::vector<std::complex<double>> &state,
                       std::size_t shots, std::size_t seed,
                       ResultFormat format) {
  const std::size_t numSites = std::bit_width(state.size()) - 1;
  std::vector<double> probabilities;
  probabilities.reserve(state.size());
  for (auto amplitude : state)
    probabilities.push_back(std::norm(amplitude));
  std::mt19937 generator(seed ? seed : std::random_device{}());
  std::discrete_distribution<std::size_t> distribution(probabilities.begin(),
                                                       probabilities.end());
  CountsDictionary counts;
  for (std::size_t shot = 0; shot < shots; ++shot) {
    const auto basis = distribution(generator);
    std::string bits(numSites, '0');
    for (std::size_t site = 0; site < numSites; ++site)
      bits[site] = ((basis >> site) & 1) ? '1' : '0';
    counts[bits]++;
  }
  if (format == ResultFormat::Binary)
    return sample_result(ExecutionResult(counts));

  // Native AHS readout: 0=empty, 1=Rydberg, 2=ground, with pre/post registers.
  CountsDictionary atomStates, preSequence, postSequence;
  for (const auto &[bits, count] : counts) {
    auto atoms = bits;
    auto post = bits;
    for (std::size_t site = 0; site < numSites; ++site) {
      atoms[site] = bits[site] == '1' ? '1' : '2';
      post[site] = bits[site] == '1' ? '0' : '1';
    }
    atomStates[atoms] += count;
    postSequence[post] += count;
    preSequence[std::string(numSites, '1')] += count;
  }
  return sample_result(std::vector<ExecutionResult>{
      ExecutionResult(atomStates), ExecutionResult(preSequence, "pre_sequence"),
      ExecutionResult(postSequence, "post_sequence")});
}

ahs::DeviceSpecification ahs::deviceSpecification(const std::string &name) {
  if (name == "FRESNEL_CAN1" || name == "FRESNEL_CAN")
    return fresnelCan;
  if (name == "AnalogDevice")
    return analogDevice;
  if (name == "Aquila")
    return aquila;
  throw std::invalid_argument("Unknown AHS device specification: " + name);
}

ahs::RydbergModel ahs::makeRydbergModel(const Program &program,
                                        const DeviceSpecification &device) {
  validateProgram(program);
  dimension_map dimensions;
  const auto &sites = program.setup.ahs_register.sites;
  for (std::size_t i = 0; i < sites.size(); ++i)
    dimensions[i] = 2;

  const auto maxAbs = [](const TimeSeries &series) {
    double value = 0.0;
    for (auto v : series.values)
      value = std::max(value, std::abs(v));
    return value;
  };
  const auto &drive = program.hamiltonian.drivingFields.front();
  double bound = sites.size() * (0.5 * maxAbs(drive.amplitude.time_series) +
                                 maxAbs(drive.detuning.time_series));
  for (std::size_t i = 0; i < sites.size(); ++i)
    for (std::size_t j = i + 1; j < sites.size(); ++j)
      bound += std::abs(device.rydbergC6) /
               std::pow(distance(sites[i], sites[j]), 6);

  // Limit the RK4 step by the Hamiltonian norm as well as the waveform scale.
  // A fixed 1 ns step is unstable for tightly spaced atoms.
  const auto maxStep = bound > 0.0 ? std::min(1e-9, 0.1 / bound) : 1e-9;
  return {rydbergHamiltonian(program, device.rydbergC6), std::move(dimensions),
          uniqueTimes(program), maxStep};
}

} // namespace cudaq
