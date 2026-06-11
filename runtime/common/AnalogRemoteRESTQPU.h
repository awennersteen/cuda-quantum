/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/BaseRemoteRESTQPU.h"
#include "cudaq/platform/qpu_utils.h"
#include <future>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace cudaq {

/// @brief Base QPU class for analog platforms like `quera` and `pasqal`.
/// Provides common functionality and implementation.
class AnalogRemoteRESTQPU : public BaseRemoteRESTQPU {
protected:
  virtual cudaq::sample_result emulateProgram(const std::string &programString,
                                              std::size_t shots,
                                              std::size_t seed) {
    throw std::runtime_error(
        "Analog emulation is not supported on this target.");
  }

public:
  /// @brief Check if this is a remote target
  virtual bool isRemote() override { return !emulate; }

  /// @brief Check if this is an emulated target
  virtual bool isEmulated() override { return emulate; }

  /// @brief Launch a kernel with the given arguments
  /// Only analog Hamiltonian kernels are supported
  KernelThunkResultType unifiedLaunchModule(const AnyModule &module,
                                            KernelArgs args) override {
    if (!std::holds_alternative<SourceModule>(module))
      throw std::runtime_error(
          "AnalogRemoteRESTQPU does not support pre-compiled module launch.");

    const auto &src = std::get<SourceModule>(module);
    const auto &kernelName = src.getName();
    auto executionContext = cudaq::getExecutionContext();

    if (!cudaq::detail::isAnalogHamiltonianKernel(kernelName))
      throw std::runtime_error(
          "Arbitrary kernel execution is not supported on this target.");

    const auto packed = args.getPacked();
    std::string strArgs = packed ? (char *)packed->data.data() : "";

    if (executionContext && emulate) {
      std::size_t localShots = 1000;
      if (executionContext->shots != std::numeric_limits<std::size_t>::max() &&
          executionContext->shots != 0)
        localShots = executionContext->shots;

      auto seed = cudaq::get_random_seed();
      if (executionContext->asyncExec) {
        auto worker =
            std::async(std::launch::async, [this, strArgs, localShots, seed]() {
              return emulateProgram(strArgs, localShots, seed);
            });
        cudaq::detail::future future(std::move(worker));
        executionContext->asyncResult = async_sample_result(std::move(future));
        return {};
      }
      executionContext->result = emulateProgram(strArgs, localShots, seed);
      return {};
    }

    CUDAQ_INFO("Launching remote kernel ({})", kernelName);
    std::vector<cudaq::KernelExecution> codes;
    std::vector<std::size_t> mapping_reorder_idx;
    codes.emplace_back(kernelName, strArgs, std::nullopt, std::nullopt,
                       mapping_reorder_idx);
    if (executionContext) {
      std::size_t localShots = 1000;
      if (executionContext->shots != std::numeric_limits<std::size_t>::max() &&
          executionContext->shots != 0)
        localShots = executionContext->shots;
      executor->setShots(localShots);
      if (getEnvBool("DISABLE_REMOTE_SEND", false))
        return {};
      cudaq::detail::future future;
      future = executor->execute(codes);
      // Keep this asynchronous if requested
      if (executionContext->asyncExec) {
        executionContext->asyncResult = async_sample_result(std::move(future));
        return {};
      }
      // Otherwise make this synchronous
      executionContext->result = future.get();
    }
    return {};
  }
};

} // namespace cudaq
