/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QuEraRemoteRESTQPU.h"

cudaq::sample_result cudaq::QuEraRemoteRESTQPU::formatEmulatedResult(
    const sample_result &result, const ahs::AtomArrangement &atoms) {
  CountsDictionary states, preSequence, postSequence;
  std::string pre(atoms.filling.size(), '0');
  for (std::size_t site = 0; site < pre.size(); ++site)
    pre[site] += atoms.filling[site];
  for (const auto &[bits, count] : result) {
    std::string state(bits.size(), '0'), post(bits.size(), '0');
    for (std::size_t site = 0; site < bits.size(); ++site) {
      if (atoms.filling[site]) {
        state[site] = bits[site] == '1' ? '1' : '2';
        post[site] = bits[site] == '1' ? '0' : '1';
      }
    }
    states[state] += count;
    preSequence[pre] += count;
    postSequence[post] += count;
  }
  return sample_result(std::vector<ExecutionResult>{
      ExecutionResult(states), ExecutionResult(preSequence, "pre_sequence"),
      ExecutionResult(postSequence, "post_sequence")});
}

cudaq::QuEraRemoteRESTQPU::~QuEraRemoteRESTQPU() = default;

CUDAQ_REGISTER_TYPE(cudaq::QPU, cudaq::QuEraRemoteRESTQPU, quera)
