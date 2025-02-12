/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/AnalogHamiltonian.h"
#include "common/Logger.h"
#include "PasqalServerHelper.h"

#include <unordered_set>

namespace cudaq {

void PasqalServerHelper::initialize(BackendConfig config) {
  cudaq::info("Initialize Pasqal Cloud.");

  // Hard-coded for now.
  const std::string FRESNEL = "Fresnel";
  auto machine = FRESNEL;
  const int MAX_QUBITS = 100;

  cudaq::info("Running on device {}", machine);

  config["machine"] = machine;
  config["qubits"] = MAX_QUBITS;

  if(!config["shots"].empty())
    setShots(std::stoul(config["shots"]));

  parseConfigForCommonParams(config);

  backendConfig = std::move(config);
}

RestHeaders PasqalServerHelper::getHeaders() {
  std::string token;

  if (auto auth_token = std::getenv("PASQAL_AUTH_TOKEN"))
    token = "Bearer " + std::string(auth_token);
  else
    token = "Bearer ";

  std::map<std::string, std::string> headers{
    {"Authorization", token},
    {"Content-Type", "application/json"},
    {"User-Agent", "cudaq/Pasqal"},
    {"Connection", "keep-alive"},
    {"Accept", "*/*"}};

  return headers;
}

ServerJobPayload
PasqalServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {
  std::vector<ServerMessage> tasks;

  for (auto &circuitCode : circuitCodes) {
    ServerMessage message;
    message["name"] = circuitCode.name;
    message["device"] = backendConfig.at("machine");
    message["shots"] = shots;

    auto action = nlohmann::json::parse(circuitCode.code);
    message["action"] = action.dump();

    tasks.push_back(message);
  }

  cudaq::info("Created job payload for Pasqal, targeting device {}",
              backendConfig.at("machine"));
  
  // Return a tuple containing the job path, headers, and the job message
  return std::make_tuple(baseUrl + apiPath + "/cudaq/job", getHeaders(), tasks);
}

std::string PasqalServerHelper::extractJobId(ServerMessage &postResponse) {
    return postResponse["id"].get<std::string>();
}

std::string PasqalServerHelper::constructGetJobPath(std::string &jobId) {
  return baseUrl + apiPath + "/jobs/" + jobId + "/results_link";
}

std::string
PasqalServerHelper::constructGetJobPath(ServerMessage &postResponse) {
    return baseUrl + apiPath + "/jobs/" +
      postResponse["id"].get<std::string>() + "/results_link";
}

bool PasqalServerHelper::jobIsDone(ServerMessage &getJobResponse) {
  std::unordered_set<std::string>
  terminals = {"PENDING", "RUNNING", "DONE", "ERROR", "CANCEL"};
  
  auto jobStatus = getJobResponse["status"].get<std::string>();
  return terminals.find(jobStatus) != terminals.end();
}

// TODO: Implementing `processResults`.
sample_result
PasqalServerHelper::processResults(ServerMessage &postJobResponse,
                                   std::string &jobId) {

  auto jobStatus = postJobResponse["status"].get<std::string>();
  if (jobStatus != "DONE") 
    throw std::runtime_error("Job status: " + jobStatus);

  sample_result res;
  return res;
}

} // namespace cudaq

// Register the Pasqal server helper in the CUDA-Q server helper factory
CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::PasqalServerHelper, pasqal)