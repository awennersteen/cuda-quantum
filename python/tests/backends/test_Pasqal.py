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

skipIfPasqalNotInstalled = pytest.mark.skipif(
    not (cudaq.has_target("pasqal")),
    reason='Could not find `pasqal` in installation')
skipIfPasqalEmulationUnavailable = pytest.mark.skipif(
    not (cudaq.has_target("pasqal") and cudaq.num_available_gpus() > 0),
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


@skipIfPasqalNotInstalled
def test_JSON_payload():
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
    cudaq.cudaq_runtime.pyAltLaunchAnalogKernel("__analog_hamiltonian_kernel__",
                                                json.dumps(input))


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


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
