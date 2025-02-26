/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "common/AnalogHamiltonian.h"

const std::string sampleSequence = R"(
{
  "setup": {
    "ahs_register": {
      "sites": [["0.0", "0.0"], ["0.0", "0.000006"], ["0.000006", "0.0"]],
      "filling": [1, 1, 1]
    }
  },
  "hamiltonian": {
    "drivingFields": [
      {
        "amplitude": {
          "time_series": {
            "values": ["0.0", "10700000.0", "10700000.0", "0.0"],
            "times": ["0.0", "0.000001", "0.000002", "0.000003"]
          },
            "pattern": "uniform"
        },
        "phase": {
          "time_series": {"values": ["0.0"], "times": ["0.0"]},
          "pattern": "uniform"
        },
        "detuning": {
          "time_series": {
            "values": ["-5400000.0", "5400000.0"],
            "times": ["0.0", "0.000003"]
          },
          "pattern": "uniform"
        }
      }
    ],
    "localDetuning": []
  }
})";

const std::string responseSample = R"(
{"status":"success","message":"OK.","code":"200","data":{"id":"6ebb665b-9445-450e-8a86-741677a5c869","status":"PENDING","result":{}}}
)";

const std::string responseSampleResult = R"(
{'id': '6ebb665b-9445-450e-8a86-741677a5c869', 'status': 'DONE', 'result': {'daab9abe-0ab9-41ac-8c6b-23be1f8d342d': {'010': 25, '000': 39, '001': 18, '100': 18}}}
)";

CUDAQ_TEST(PasqalTester, checkHamiltonianJson) {
  cudaq::ahs::AtomArrangement layout;
  layout.sites = {{0.0, 0.0}, {0.0, 6.0e-6}, {6.0e-6, 0.0}};
  layout.filling = {1, 1, 1};

  cudaq::ahs::PhysicalField amplitude;
  amplitude.time_series = std::vector<std::pair<double, double>>{
      {0.0, 0.0}, {1.07e7, 1.0e-6}, {1.07e7, 2.0e-6}, {0.0, 3.0e-6}};

  cudaq::ahs::PhysicalField detuning;
  detuning.time_series =
      std::vector<std::pair<double, double>>{{-5.4e6, 0.0}, {5.4e6, 3.0e-6}};

  cudaq::ahs::PhysicalField phase;
  phase.time_series = std::vector<std::pair<double, double>>{{0.0, 0.0}};

  cudaq::ahs::DrivingField drive;
  drive.amplitude = amplitude;
  drive.detuning = detuning;
  drive.phase = phase;

  cudaq::ahs::Program sequence;
  sequence.setup.ahs_register = layout;
  sequence.hamiltonian.drivingFields = {drive};

  nlohmann::json serializedSequence = sequence;
  cudaq::ahs::Program refSequence = nlohmann::json::parse(sampleSequence);
  EXPECT_EQ(serializedSequence, refSequence);
}

CUDAQ_TEST(PasqalTester, checkGetJobResponse) {
  auto resp1 = nlohmann::json::parse(responseSample);
  cudaq::ahs::PasqalJobResponse resp = nlohmann::json::parse(responseSample);
  EXPECT_EQ(resp1, resp);
}

CUDAQ_TEST(PasqalTester, checkGetJobResponseResult) {
  auto resp1 = nlohmann::json::parse(responseSampleResult);
  cudaq::ahs::PasqalJobResponse resp =
      nlohmann::json::parse(responseSampleResult);
  EXPECT_EQ(resp1, resp);
}