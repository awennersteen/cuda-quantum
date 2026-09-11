/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/AnalogDynamicsEmulation.h"
#include "common/BaseRemoteRESTQPU.h"
#include "cudaq/platform/qpu_utils.h"
#include <future>

namespace cudaq {

/// @brief Base QPU class for analog platforms like `quera` and `pasqal`.
/// Provides common functionality and implementation.
class AnalogRemoteRESTQPU : public BaseRemoteRESTQPU {
protected:
  ahs::DeviceSpecification device;
  using Emulator = decltype(&emulateRydbergDynamics);
  Emulator emulator = nullptr;
  using ResultAdapter = sample_result (*)(const sample_result &,
                                          const ahs::AtomArrangement &);
  ResultAdapter resultAdapter;

  bool requiresRemoteBackend() const override { return !emulate; }

public:
  explicit AnalogRemoteRESTQPU(ahs::DeviceSpecification device,
                               ResultAdapter adapter = nullptr)
      : device(device), resultAdapter(adapter) {}

  /// @brief Check if this is a remote target
  virtual bool isRemote() override { return !emulate; }

  /// @brief Check if this is an emulated target
  virtual bool isEmulated() override { return emulate; }

  using BaseRemoteRESTQPU::getCompileTarget;
  using BaseRemoteRESTQPU::launchKernel;

  CompileTarget
  getCompileTarget(bool skipPipelineSubstitutions = false) override {
    return {.overrideAOTCompilation = false};
  }

  /// @brief Launch a kernel with the given arguments
  /// Only analog Hamiltonian kernels are supported
  detail::future launchKernelCommon(const sample_policy &policy,
                                    const CompiledModule &module,
                                    KernelArgs args) {
    const auto &kernelName = module.getName();
    if (!cudaq::detail::isAnalogHamiltonianKernel(kernelName))
      throw std::runtime_error(
          "Arbitrary kernel execution is not supported on this target.");

    CUDAQ_INFO("Launching analog kernel ({})", kernelName);
    std::vector<cudaq::KernelExecution> codes;
    std::string name = kernelName;
    const auto packed = args.getPacked();
    if (!packed)
      throw std::runtime_error(
          "Analog Hamiltonian launch requires a packed JSON payload.");
    std::string strArgs(reinterpret_cast<const char *>(packed->data.data()),
                        packed->data.size());
    if (emulate) {
      if (!emulator)
        throw std::runtime_error(
            "AHS emulation requires a build with the CUDA-Q dynamics backend.");
      auto specification = device;
      if (auto it = backendConfig.find("rydberg_c6"); it != backendConfig.end())
        specification.rydbergC6 = std::stod(it->second);
      auto seed = cudaq::get_random_seed();
      auto program = ahs::fromJsonString(strArgs);
      return detail::future(std::async(
          std::launch::async,
          [specification, simulate = emulator, adapter = resultAdapter,
           program = std::move(program), shots = policy.options.shots, seed]() {
            auto result = simulate(program, shots, specification, seed);
            return adapter ? adapter(result, program.setup.ahs_register)
                           : result;
          }));
    }
    codes.push_back(KernelExecution{.name = name, .code = strArgs});

    executor->setShots(policy.options.shots);
    return executor->execute(codes);
  }

  async_sample_result launchKernel(const async_sample_policy &policy,
                                   const CompiledModule &module,
                                   KernelArgs args) override {
    // Keep this asynchronous if requested
    return async_sample_result(
        launchKernelCommon(policy.inner, module, std::move(args)));
  }

  sample_result launchKernel(const sample_policy &policy,
                             const CompiledModule &module,
                             KernelArgs args) override {
    // Otherwise make this synchronous
    return launchKernelCommon(policy, module, std::move(args)).get();
  }
};

} // namespace cudaq
