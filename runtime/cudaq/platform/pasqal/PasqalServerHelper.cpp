/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PasqalServerHelper.h"
#include "PasqalUtils.h"
#include "common/AnalogHamiltonian.h"
#include "cudaq/runtime/logger/logger.h"

#include <cstdlib>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace cudaq {
namespace {

struct PasqalConfig {
  std::optional<std::string> token;
  std::optional<std::string> projectId;
};

std::string trim(std::string_view value) {
  const auto begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string_view::npos)
    return "";
  const auto end = value.find_last_not_of(" \t\r\n");
  return std::string(value.substr(begin, end - begin + 1));
}

std::string stripQuotes(std::string value) {
  if (value.size() < 2)
    return value;
  const auto front = value.front();
  const auto back = value.back();
  if ((front == '"' && back == '"') || (front == '\'' && back == '\''))
    return value.substr(1, value.size() - 2);
  return value;
}

std::string toLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::optional<std::string> readNonEmptyEnv(const char *name) {
  if (auto *value = std::getenv(name)) {
    auto normalized = trim(value);
    if (!normalized.empty())
      return normalized;
  }
  return std::nullopt;
}

std::optional<std::string> findReadableConfigPath() {
  std::vector<std::string> candidates;

  if (auto configRoot = readNonEmptyEnv("PASQAL_CONFIG_ROOT"))
    candidates.push_back(*configRoot + "/.pasqal/config");
  if (auto home = readNonEmptyEnv("HOME"))
    candidates.push_back(*home + "/.pasqal/config");

  for (const auto &candidate : candidates) {
    std::ifstream stream(candidate);
    if (stream.good())
      return candidate;
  }

  return std::nullopt;
}

PasqalConfig readPasqalConfig() {
  auto configPath = findReadableConfigPath();
  if (!configPath)
    return {};

  std::ifstream stream(*configPath);
  if (!stream.is_open())
    return {};

  PasqalConfig config;
  std::string line;
  while (std::getline(stream, line)) {
    auto normalized = trim(line);
    if (normalized.empty() || normalized.front() == '#' ||
        normalized.front() == ';')
      continue;

    auto equalsPos = normalized.find('=');
    if (equalsPos == std::string::npos)
      continue;

    auto key = toLower(trim(normalized.substr(0, equalsPos)));
    auto value = trim(stripQuotes(trim(normalized.substr(equalsPos + 1))));
    if (key.empty() || value.empty())
      continue;

    if (key == "token")
      config.token = value;
    else if (key == "project_id")
      config.projectId = value;
  }

  return config;
}

} // namespace

void PasqalServerHelper::initialize(BackendConfig config) {
  CUDAQ_INFO("Initialize Pasqal Cloud.");

  // Defaults
  const std::string MACHINE = "EMU_MPS";
  const int MAX_QUBITS = 100;

  if (!config.contains("machine"))
    config["machine"] = MACHINE;

  CUDAQ_INFO("Running on Pasqal machine {}", config["machine"]);

  config["qubits"] = MAX_QUBITS;

  if (!config["shots"].empty())
    setShots(std::stoul(config["shots"]));

  auto pasqalConfig = readPasqalConfig();

  if (auto authToken = readNonEmptyEnv("PASQAL_AUTH_TOKEN")) {
    authToken_ = *authToken;
    CUDAQ_INFO("Using Pasqal auth token from PASQAL_AUTH_TOKEN.");
  } else if (pasqalConfig.token) {
    authToken_ = *pasqalConfig.token;
    CUDAQ_INFO("Using Pasqal auth token from ~/.pasqal/config.");
  } else {
    authToken_.clear();
  }

  if (auto projectId = readNonEmptyEnv("PASQAL_PROJECT_ID")) {
    config["project_id"] = *projectId;
    CUDAQ_INFO("Using Pasqal project id from PASQAL_PROJECT_ID.");
  } else if (pasqalConfig.projectId) {
    config["project_id"] = *pasqalConfig.projectId;
    CUDAQ_INFO("Using Pasqal project id from ~/.pasqal/config.");
  } else {
    config["project_id"] = "";
  }

  parseConfigForCommonParams(config);

  backendConfig = std::move(config);
}

RestHeaders PasqalServerHelper::getHeaders() {
  if (authToken_.empty())
    CUDAQ_WARN("No PASQAL_AUTH_TOKEN found.");
  else
    CUDAQ_INFO("Using PASQAL_AUTH_TOKEN for authentication.");
  
  auto token = std::string("Bearer ") + authToken_;

  std::map<std::string, std::string> headers{
      {"Authorization", token},
      {"Content-Type", "application/json"},
      {"User-Agent", "Cudaq/Pasqal"},
      {"Connection", "keep-alive"},
      {"Accept", "*/*"}};

  return headers;
}

ServerJobPayload
PasqalServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {
  std::vector<ServerMessage> tasks;

  for (auto &circuitCode : circuitCodes) {
    ServerMessage message;
    message["machine"] = backendConfig.at("machine");
    message["shots"] = shots;
    message["project_id"] = backendConfig.at("project_id");
    message["sequence"] = nlohmann::json::parse(circuitCode.code);
    tasks.push_back(message);
  }

  CUDAQ_INFO("Created Pasqal payload, targeting machine {}",
             backendConfig.at("machine"));

  // Return a tuple containing the job path, headers, and the job message
  return std::make_tuple(baseUrl + apiPath + "/v1/cudaq/job", getHeaders(),
                         tasks);
}

std::string PasqalServerHelper::extractJobId(ServerMessage &postResponse) {
  return postResponse["data"]["id"].get<std::string>();
}

std::string PasqalServerHelper::constructGetJobPath(std::string &jobId) {
  return baseUrl + apiPath + "/v1/cudaq/job/" + jobId;
}

std::string
PasqalServerHelper::constructGetJobPath(ServerMessage &postResponse) {
  return baseUrl + apiPath + "/v1/cudaq/job/" +
         postResponse["data"]["id"].get<std::string>();
}

bool PasqalServerHelper::jobIsDone(ServerMessage &getJobResponse) {
  std::unordered_set<std::string> terminals = {"DONE", "ERROR", "CANCELED",
                                               "TIMED_OUT", "PAUSED"};
  auto jobStatus = getJobResponse["data"]["status"].get<std::string>();
  return terminals.find(jobStatus) != terminals.end();
}

sample_result PasqalServerHelper::processResults(ServerMessage &postJobResponse,
                                                 std::string &jobId) {
  auto status = postJobResponse["data"]["status"].get<std::string>();
  if (status != "DONE")
    throw std::runtime_error("Job status: " + status);

  std::vector<ExecutionResult> results;
  auto jobs = postJobResponse["data"]["result"];

  // loop over jobs in batch to get results
  // Current implementation only has 1 job
  for (auto &job : jobs) {
    results.push_back(pasqal::parseExecutionResult(job));
  }

  return sample_result(results);
}
} // namespace cudaq

// Avoid duplicate "pasqal" registration from the Python dialect extension
// build; the runtime plugin provides the canonical registration.
#ifndef CUDAQuantumPythonModules_extension__quakeDialects_dso_EXPORTS
CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::PasqalServerHelper, pasqal)
#endif
