// Compile and run with:
// ```
// nvq++ --target pasqal pasqal.cpp -o out.x
// ./out.x
// ```
// For direct execution, credentials are read from `~/.pasqal/config`
// (`token`, `project_id`) or from `PASQAL_AUTH_TOKEN` + `PASQAL_PROJECT_ID`.
//
// QRMI flow (HPC scheduler)):
// ```
// nvq++ --target pasqal --pasqal-machine qrmi pasqal.cpp -o out.x
// sbatch --qpu=EMU_FREE my_pasqal_job.sh
// ```
// In QRMI mode the backend comes from the scheduler via
// `SLURM_JOB_QPU_RESOURCES`.

#include "cudaq/algorithms/evolve.h"
#include "cudaq/algorithms/integrator.h"
#include "cudaq/operators.h"
#include "cudaq/schedule.h"
#include <cmath>
#include <map>
#include <vector>

// This example illustrates how to use `Pasqal's` EMU_MPS emulator over
// `Pasqal's` cloud via CUDA-Q. Contact Pasqal at help@pasqal.com or through
// https://community.pasqal.com for assistance.

int main() {
  // Topology initialization
  const double a = 5e-6; // Inter-atomic spacing in meters
  std::vector<std::pair<double, double>> register_sites;
  // We create a 2D register. Here 3 atoms on a line (1D)
  register_sites.push_back(std::make_pair(a, 0.0));
  register_sites.push_back(std::make_pair(2 * a, 0.0));
  register_sites.push_back(std::make_pair(3 * a, 0.0));

  // Simulation Timing
  const double time_ramp = 0.000001; // seconds
  const double time_max = 0.000003;  // seconds
  // Hamiltonian Parameters
  const double omega_max = 1000000; // rad/sec
  const double delta_end = 1000000;
  const double delta_start = 0.0;
  const double phi = 0.0;

  std::vector<std::complex<double>> steps = {0.0, time_ramp,
                                             time_max - time_ramp, time_max};
  cudaq::schedule schedule(steps, {"t"}, {});

  // Rabi frequency (amplitude) omega is ramped up from 0 to omega_max during
  // the, first time_ramp seconds, then kept constant at omega_max until
  // time_max, and finally ramped down to 0 at the end of the schedule.
  auto omega = cudaq::scalar_operator(
      [time_ramp, time_max,
       omega_max](const std::unordered_map<std::string, std::complex<double>>
                      &parameters) {
        double t = std::real(parameters.at("t"));
        return std::complex<double>(
            (t > time_ramp && t < time_max) ? omega_max : 0.0, 0.0);
      });

  // We keep the phase phi constant at 0 for this example
  auto phi = cudaq::scalar_operator(phi);

  // Detuning delta is ramped up from 0 to delta_end during the
  // irst time_ramp seconds, then kept constant at delta_end until time_max,
  // and finally ramped down to 0 at the end of the schedule.
  auto delta = cudaq::scalar_operator(
      [time_ramp, time_max, delta_start,
       delta_end](const std::unordered_map<std::string, std::complex<double>>
                      &parameters) {
        double t = std::real(parameters.at("t"));
        return std::complex<double>(
            (t > time_ramp && t < time_max) ? delta_end : delta_start, 0.0);
      });

  auto hamiltonian =
      cudaq::rydberg_hamiltonian(register_sites, omega, phi, delta);

  // Evolve the system
  auto result = cudaq::evolve(hamiltonian, schedule, 100);
  result.sampling_result->dump();

  return 0;
}
