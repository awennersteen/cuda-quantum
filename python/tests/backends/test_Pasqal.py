# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
from cudaq.dynamics import Schedule
from cudaq.operators import RydbergHamiltonian, ScalarOperator
import json
import numpy as np
import os
import pytest
import subprocess
import sys

pytestmark = pytest.mark.skipif(
    not (cudaq.has_target("pasqal")),
    reason='Could not find `pasqal` in installation')
skipIfPasqalEmulationUnavailable = pytest.mark.skipif(
    not (cudaq.has_target("pasqal") and cudaq.has_target("dynamics") and
         cudaq.num_available_gpus() > 0),
    reason='Pasqal emulation requires the dynamics backend and a CUDA GPU')


@pytest.fixture(scope="session", autouse=True)
def set_up_target():
    # NOTE: Credentials can be set with environment variables.
    # This test covers the direct `pasqal` backend only.
    # QRMI-routed execution is validated separately because it requires a
    # supported QRMI build and a compatible cluster resource manager.
    cudaq.set_target("pasqal")
    yield "Running the tests."
    cudaq.reset_target()


def test_JSON_payload(monkeypatch):
    """Exercise the launch payload without submitting a cloud job."""
    monkeypatch.setenv("DISABLE_REMOTE_SEND", "1")
    input = {
        "setup": {
            "ahs_register": {
                "sites": [[0, 0], [0, 0.000004], [0.000004, 0]],
                "filling": [1, 1, 1]
            }
        },
        "hamiltonian": {
            "drivingFields": [{
                "amplitude": {
                    "time_series": {
                        "values": [0, 15700000, 15700000, 0],
                        "times": [0, 0.000001, 0.000002, 0.000003]
                    },
                    "pattern": "uniform"
                },
                "phase": {
                    "time_series": {
                        "values": [0, 0],
                        "times": [0, 0.000003]
                    },
                    "pattern": "uniform"
                },
                "detuning": {
                    "time_series": {
                        "values": [-54000000, 54000000],
                        "times": [0, 0.000003]
                    },
                    "pattern": "uniform"
                }
            }],
            "localDetuning": []
        }
    }
    # NOTE: For internal testing only, not user-level API; this does not return results
    cudaq.cudaq_runtime.launch_analog_kernel("__analog_hamiltonian_kernel__",
                                             json.dumps(input), 100)


@skipIfPasqalEmulationUnavailable
def test_evolve_emulate():
    cudaq.reset_target()
    cudaq.set_target("pasqal", emulate=True)
    a = 5.7e-6
    register = [
        tuple(np.array([0.0, 0.0]) * a),
        tuple(np.array([1.0, 0.0]) * a),
        tuple(np.array([0.0, 1.0]) * a)
    ]
    omega = ScalarOperator(lambda t: 6.3e6 if t.real > 1e-7 else 0.0)
    phi = ScalarOperator.const(0.0)
    delta = ScalarOperator(lambda t: 3.15e7 if t.real > 1e-7 else -3.15e7)
    result = cudaq.evolve(RydbergHamiltonian(atom_sites=register,
                                             amplitude=omega,
                                             phase=phi,
                                             delta_global=delta),
                          schedule=Schedule([0.0, 1e-7, 4e-7], ["t"]),
                          shots_count=10)
    assert result.get_total_shots() == 10
    assert all(len(bits) == len(register) for bits in result)


@skipIfPasqalEmulationUnavailable
def test_evolve_emulate_no_drive():
    cudaq.reset_target()
    cudaq.set_target("pasqal", emulate=True)
    register = [(0.0, 0.0), (5.7e-6, 0.0)]
    result = cudaq.evolve(RydbergHamiltonian(
        atom_sites=register,
        amplitude=ScalarOperator.const(0.0),
        phase=ScalarOperator.const(0.0),
        delta_global=ScalarOperator.const(0.0)),
                          schedule=Schedule([0.0, 1e-7], ["t"]),
                          shots_count=10)
    assert result.get_total_shots() == 10
    assert result["00"] == 10


@skipIfPasqalEmulationUnavailable
def test_evolve_emulate_async_seed():
    cudaq.reset_target()
    cudaq.set_target("pasqal", emulate=True)
    a = 5.7e-6

    def make_problem():
        return RydbergHamiltonian(
            atom_sites=[(0.0, 0.0), (a, 0.0), (0.0, a)],
            amplitude=ScalarOperator(lambda t: 6.3e6 if t.real > 1e-7 else 0.0),
            phase=ScalarOperator.const(0.0),
            delta_global=ScalarOperator(lambda t: 3.15e7
                                        if t.real > 1e-7 else -3.15e7))

    cudaq.set_random_seed(13)
    first = cudaq.evolve_async(make_problem(),
                               schedule=Schedule([0.0, 1e-7, 4e-7], ["t"]),
                               shots_count=20).get()
    cudaq.set_random_seed(13)
    second = cudaq.evolve_async(make_problem(),
                                schedule=Schedule([0.0, 1e-7, 4e-7], ["t"]),
                                shots_count=20).get()
    assert {
        bits: first[bits] for bits in first
    } == {
        bits: second[bits] for bits in second
    }


@skipIfPasqalEmulationUnavailable
@pytest.mark.parametrize("device", ["FRESNEL_CAN1", "Aquila"])
def test_emulate_pi_pulse(device):
    """Apply a pi pulse in SI units using either device specification."""
    cudaq.set_target("pasqal", emulate=True, device=device)
    result = cudaq.evolve(RydbergHamiltonian(
        atom_sites=[(0., 0.)],
        amplitude=ScalarOperator.const(np.pi / 2e-7),
        phase=ScalarOperator.const(0.7),
        delta_global=ScalarOperator.const(0.)),
                          schedule=Schedule([0., 2e-7], ["t"]),
                          shots_count=100)
    assert result["1"] == 100


@skipIfPasqalEmulationUnavailable
def test_emulate_phase_sequence():
    """Resolve the phase sign with a detuned two-pulse experiment."""
    from scipy.linalg import expm

    cudaq.set_target("pasqal", emulate=True)
    cudaq.set_random_seed(21)
    x = np.array([[0., 1.], [1., 0.]])
    y = np.array([[0., -1j], [1j, 0.]])
    number = np.diag([0., 1.])
    first = 4. * x - 4. * number
    second = 4. * y - 4. * number
    state = expm(-1j * second * 0.2) @ expm(-1j * first * 0.2) @ [1., 0.]
    result = cudaq.evolve(RydbergHamiltonian(
        atom_sites=[(0., 0.)],
        amplitude=ScalarOperator.const(8e6),
        phase=ScalarOperator(lambda t: 0. if t.real < 2e-7 else np.pi / 2),
        delta_global=ScalarOperator.const(4e6)),
                          schedule=Schedule([0., 2e-7, 4e-7], ["t"]),
                          shots_count=10000)
    assert result.probability("1") == pytest.approx(abs(state[1])**2, abs=0.025)


@skipIfPasqalEmulationUnavailable
@pytest.mark.parametrize("target", ["pasqal", "quera"])
def test_emulate_command_line(target):
    """Launch an analog evolution using the command-line emulation flag."""
    if not cudaq.has_target(target):
        pytest.skip(f"Could not find {target} in installation")
    code = """
import cudaq
from cudaq.dynamics import Schedule
from cudaq.operators import RydbergHamiltonian, ScalarOperator
h = RydbergHamiltonian([(0., 0.)], ScalarOperator.const(0.),
                      ScalarOperator.const(0.), ScalarOperator.const(0.))
result = cudaq.evolve(h, schedule=Schedule([0., 1e-8], ["t"]), shots_count=19)
assert result["2" if cudaq.get_target().name == "quera" else "0"] == 19
"""
    subprocess.run([sys.executable, "-c", code, "-target", target, "--emulate"],
                   env={
                       **os.environ, "DISABLE_REMOTE_SEND": "1"
                   },
                   check=True,
                   capture_output=True,
                   text=True,
                   timeout=60)


@skipIfPasqalEmulationUnavailable
def test_terminal_phase_does_not_evolve():
    """Ignore a phase change occurring only at the final waveform point."""
    cudaq.set_target("pasqal", emulate=True)
    results = []
    for final_phase in [0., np.pi / 2]:
        cudaq.set_random_seed(41)
        h = RydbergHamiltonian(
            [(0., 0.)], ScalarOperator.const(4e6),
            ScalarOperator(lambda t: final_phase if t.real >= 4e-7 else 0.),
            ScalarOperator.const(2e6))
        results.append(
            dict(
                cudaq.evolve(h,
                             schedule=Schedule([0., 4e-7], ["t"]),
                             shots_count=20000).items()))
    assert results[0] == results[1]


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
