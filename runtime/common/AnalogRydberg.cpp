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
  for (const auto &local : program.hamiltonian.localDetuning)
    times.insert(times.end(), local.magnitude.time_series.times.begin(),
                 local.magnitude.time_series.times.end());
  std::sort(times.begin(), times.end());
  times.erase(std::unique(times.begin(), times.end()), times.end());
  return times;
}

double distance(const std::vector<double> &a, const std::vector<double> &b) {
  const auto dx = a[0] - b[0];
  const auto dy = a[1] - b[1];
  return std::sqrt(dx * dx + dy * dy);
}

double siteScale(const cudaq::ahs::PhysicalField &field, std::size_t site) {
  return field.pattern.patternStr == "uniform"
             ? 1.0
             : field.pattern.patternVals[site];
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
rydbergHamiltonian(const cudaq::ahs::Program &program, double rydbergC6,
                   const std::vector<std::size_t> &occupied) {
  auto hamiltonian = cudaq::spin_op::empty();
  for (const auto &drive : program.hamiltonian.drivingFields) {
    for (std::size_t i = 0; i < occupied.size(); ++i) {
      const auto site = occupied[i];
      const auto xCoefficient = cudaq::scalar_operator(
          [drive,
           site](const std::unordered_map<std::string, std::complex<double>>
                     &parameters) {
            const auto time = parameters.at("t").real();
            const auto omega = siteScale(drive.amplitude, site) *
                               interpolate(drive.amplitude.time_series, time);
            const auto phase = siteScale(drive.phase, site) *
                               interpolate(drive.phase.time_series, time, true);
            return std::complex<double>(0.5 * omega * std::cos(phase), 0.0);
          },
          {{"t", "time"}});
      const auto yCoefficient = cudaq::scalar_operator(
          [drive,
           site](const std::unordered_map<std::string, std::complex<double>>
                     &parameters) {
            const auto time = parameters.at("t").real();
            const auto omega = siteScale(drive.amplitude, site) *
                               interpolate(drive.amplitude.time_series, time);
            const auto phase = siteScale(drive.phase, site) *
                               interpolate(drive.phase.time_series, time, true);
            return std::complex<double>(0.5 * omega * std::sin(phase), 0.0);
          },
          {{"t", "time"}});
      const auto detuningCoefficient =
          fieldCoefficient(drive.detuning.time_series,
                           [scale = siteScale(drive.detuning, site)](
                               double delta) { return -0.5 * scale * delta; });
      hamiltonian += xCoefficient * cudaq::spin_op::x(i);
      hamiltonian += yCoefficient * cudaq::spin_op::y(i);
      hamiltonian += detuningCoefficient *
                     (cudaq::spin_op::identity(i) - cudaq::spin_op::z(i));
    }
  }
  for (const auto &local : program.hamiltonian.localDetuning)
    for (std::size_t i = 0; i < occupied.size(); ++i)
      hamiltonian +=
          fieldCoefficient(local.magnitude.time_series,
                           [scale = siteScale(local.magnitude, occupied[i])](
                               double delta) { return -0.5 * scale * delta; }) *
          (cudaq::spin_op::identity(i) - cudaq::spin_op::z(i));

  const auto &sites = program.setup.ahs_register.sites;
  for (std::size_t i = 0; i < occupied.size(); ++i) {
    for (std::size_t j = i + 1; j < occupied.size(); ++j) {
      const auto interaction =
          0.25 * rydbergC6 /
          std::pow(distance(sites[occupied[i]], sites[occupied[j]]), 6);
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
  const auto &atoms = program.setup.ahs_register;
  if (atoms.sites.size() != atoms.filling.size())
    throw std::invalid_argument(
        "AHS sites and filling must have equal lengths.");
  for (std::size_t i = 0; i < atoms.sites.size(); ++i) {
    if (atoms.sites[i].size() != 2 || !std::isfinite(atoms.sites[i][0]) ||
        !std::isfinite(atoms.sites[i][1]))
      throw std::invalid_argument(
          "AHS sites must contain two finite coordinates.");
    if (atoms.filling[i] != 0 && atoms.filling[i] != 1)
      throw std::invalid_argument("AHS filling must contain only 0 or 1.");
    for (std::size_t j = 0; j < i; ++j)
      if (atoms.filling[i] && atoms.filling[j] &&
          atoms.sites[i] == atoms.sites[j])
        throw std::invalid_argument(
            "Occupied AHS sites must have distinct coordinates.");
  }
  auto validateField = [&atoms](const cudaq::ahs::PhysicalField &field) {
    const auto &series = field.time_series;
    if (series.times.size() != series.values.size())
      throw std::invalid_argument(
          "AHS times and values must have equal lengths.");
    for (std::size_t i = 0; i < series.times.size(); ++i)
      if (!std::isfinite(series.times[i]) || !std::isfinite(series.values[i]) ||
          series.times[i] < 0.0 ||
          (i && series.times[i] <= series.times[i - 1]))
        throw std::invalid_argument("AHS waveforms require finite values and "
                                    "strictly increasing nonnegative times.");
    const auto &pattern = field.pattern;
    if (pattern.patternStr != "uniform" &&
        (!pattern.patternStr.empty() ||
         pattern.patternVals.size() != atoms.sites.size()))
      throw std::invalid_argument(
          "AHS patterns must be uniform or have one scale per trap site.");
    for (auto scale : pattern.patternVals)
      if (!std::isfinite(scale) || scale < 0.0 || scale > 1.0)
        throw std::invalid_argument(
            "AHS spatial scales must be between 0 and 1.");
  };
  for (const auto &drive : program.hamiltonian.drivingFields) {
    validateField(drive.amplitude);
    validateField(drive.phase);
    validateField(drive.detuning);
  }
  for (const auto &local : program.hamiltonian.localDetuning)
    validateField(local.magnitude);
}

} // namespace

sample_result
ahs::sampleStateVector(const std::vector<std::complex<double>> &state,
                       std::size_t shots, std::size_t seed,
                       const std::vector<int> &filling) {
  const std::size_t numSites =
      filling.empty() ? std::bit_width(state.size()) - 1 : filling.size();
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
    std::size_t atom = 0;
    for (std::size_t site = 0; site < numSites; ++site)
      if (filling.empty() || filling[site])
        bits[site] = ((basis >> atom++) & 1) ? '1' : '0';
    counts[bits]++;
  }
  return sample_result(ExecutionResult(counts));
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
  std::vector<std::size_t> occupied;
  for (std::size_t i = 0; i < sites.size(); ++i)
    if (program.setup.ahs_register.filling[i]) {
      dimensions[occupied.size()] = 2;
      occupied.push_back(i);
    }

  const auto maxAbs = [](const TimeSeries &series) {
    double value = 0.0;
    for (auto v : series.values)
      value = std::max(value, std::abs(v));
    return value;
  };
  double bound = 0.0;
  for (const auto &drive : program.hamiltonian.drivingFields)
    for (auto site : occupied)
      bound +=
          0.5 * siteScale(drive.amplitude, site) *
              maxAbs(drive.amplitude.time_series) +
          siteScale(drive.detuning, site) * maxAbs(drive.detuning.time_series);
  for (const auto &local : program.hamiltonian.localDetuning)
    for (auto site : occupied)
      bound += siteScale(local.magnitude, site) *
               maxAbs(local.magnitude.time_series);
  for (std::size_t i = 0; i < occupied.size(); ++i)
    for (std::size_t j = i + 1; j < occupied.size(); ++j)
      bound += std::abs(device.rydbergC6) /
               std::pow(distance(sites[occupied[i]], sites[occupied[j]]), 6);

  // Limit the RK4 step by the Hamiltonian norm as well as the waveform scale.
  // A fixed 1 ns step is unstable for tightly spaced atoms.
  const auto maxStep = bound > 0.0 ? std::min(1e-9, 0.1 / bound) : 1e-9;
  return {rydbergHamiltonian(program, device.rydbergC6, occupied),
          std::move(dimensions), uniqueTimes(program), maxStep};
}

} // namespace cudaq
