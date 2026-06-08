/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PasqalRemoteRESTQPU.h"
#ifdef CUDAQ_PASQAL_ENABLE_DYNAMICS_EMULATION
#include "common/AnalogDynamicsEmulation.h"
#endif

cudaq::PasqalRemoteRESTQPU::~PasqalRemoteRESTQPU() = default;

#ifdef CUDAQ_PASQAL_ENABLE_DYNAMICS_EMULATION
cudaq::sample_result cudaq::PasqalRemoteRESTQPU::emulateProgram(
    const std::string &programString, std::size_t shots, std::size_t seed) {
  return cudaq::emulateRydbergDynamics(programString, shots, rydbergC6, seed);
}
#endif

CUDAQ_REGISTER_TYPE(cudaq::QPU, cudaq::PasqalRemoteRESTQPU, pasqal)
