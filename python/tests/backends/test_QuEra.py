# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
from cudaq.operators import RydbergHamiltonian, ScalarOperator
from cudaq.dynamics import Schedule
import numpy as np
import os
import pytest

# The `quera` target only exists when CUDA-Q was configured with the Braket
# backend enabled (AWSSDK_ROOT and CUDAQ_ENABLE_BRAKET_BACKEND). That is a
# separate condition from having credentials, which the per-test skips below
# cover: credentials cannot help if the target was never built.
pytestmark = pytest.mark.skipif(not cudaq.has_target("quera"),
                                reason="Could not find `quera` in installation")


@pytest.fixture(autouse=True)
def set_up_target():
    yield "Running the tests."
    cudaq.reset_target()


@pytest.mark.skip(reason="Amazon Braket credentials required")
def test_ahs_hello():
    '''
    Test based on
    https://docs.aws.amazon.com/braket/latest/developerguide/braket-get-started-hello-ahs.html
    '''
    cudaq.set_target("quera")
    a = 5.7e-6
    register = []
    register.append(tuple(np.array([0.5, 0.5 + 1 / np.sqrt(2)]) * a))
    register.append(tuple(np.array([0.5 + 1 / np.sqrt(2), 0.5]) * a))
    register.append(tuple(np.array([0.5 + 1 / np.sqrt(2), -0.5]) * a))
    register.append(tuple(np.array([0.5, -0.5 - 1 / np.sqrt(2)]) * a))
    register.append(tuple(np.array([-0.5, -0.5 - 1 / np.sqrt(2)]) * a))
    register.append(tuple(np.array([-0.5 - 1 / np.sqrt(2), -0.5]) * a))
    register.append(tuple(np.array([-0.5 - 1 / np.sqrt(2), 0.5]) * a))
    register.append(tuple(np.array([-0.5, 0.5 + 1 / np.sqrt(2)]) * a))

    time_max = 4e-6  # seconds
    time_ramp = 1e-7  # seconds
    omega_max = 6300000.0  # rad / sec
    delta_start = -5 * omega_max
    delta_end = 5 * omega_max

    omega = ScalarOperator(lambda t: omega_max
                           if time_ramp < t.real < time_max else 0.0)
    phi = ScalarOperator.const(0.0)
    delta = ScalarOperator(lambda t: delta_end
                           if time_ramp < t.real < time_max else delta_start)

    # Schedule of time steps.
    steps = [0.0, time_ramp, time_max - time_ramp, time_max]
    schedule = Schedule(steps, ["t"])

    evolution_result = cudaq.evolve(RydbergHamiltonian(atom_sites=register,
                                                       amplitude=omega,
                                                       phase=phi,
                                                       delta_global=delta),
                                    schedule=schedule,
                                    shots_count=2)
    evolution_result.dump()


@pytest.mark.skipif(
    not (cudaq.has_target("dynamics") and cudaq.num_available_gpus() > 0),
    reason="AHS emulation requires dynamics and a CUDA GPU")
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("filling, expected, pre", [([1, 1], "22", "11"),
                                                    ([0, 1], "02", "01"),
                                                    ([0, 0], "00", "00")])
def test_aquila_emulation_readout(is_async, filling, expected, pre,
                                  monkeypatch):
    """Preserve Aquila's native atom-state and pre/post readout registers."""
    monkeypatch.setenv("DISABLE_REMOTE_SEND", "1")
    cudaq.set_target("quera", emulate=True)
    evolve = cudaq.evolve_async if is_async else cudaq.evolve
    result = evolve(RydbergHamiltonian(atom_sites=[(0., 0.), (5.7e-6, 0.)],
                                       amplitude=ScalarOperator.const(0.),
                                       phase=ScalarOperator.const(0.),
                                       delta_global=ScalarOperator.const(0.),
                                       atom_filling=filling),
                    schedule=Schedule([0., 1e-8], ["t"]),
                    shots_count=23)
    result = result.get() if is_async else result
    assert result[expected] == 23
    assert result.get_register_counts("pre_sequence")[pre] == 23
    assert result.get_register_counts("post_sequence")[pre] == 23


@pytest.mark.skipif(
    not (cudaq.has_target("dynamics") and cudaq.num_available_gpus() > 0),
    reason="AHS emulation requires dynamics and a CUDA GPU")
@pytest.mark.parametrize("is_async", [False, True])
def test_aquila_emulation_phase_sequence(is_async, monkeypatch):
    """Resolve Aquila's AWS phase convention with a detuned two-pulse sequence."""
    from scipy.linalg import expm

    # AWS: Omega/2 (exp(i phi)|g><r| + exp(-i phi)|r><g|) - Delta n.
    # A single constant-phase pulse from |g> cannot distinguish the signs.
    first = np.array([[0., 4.], [4., -4.]])
    second = np.array([[0., 4j], [-4j, -4.]])
    state = expm(-0.2j * second) @ expm(-0.2j * first) @ [1., 0.]
    monkeypatch.setenv("DISABLE_REMOTE_SEND", "1")
    cudaq.set_target("quera", emulate=True)
    cudaq.set_random_seed(21)
    evolve = cudaq.evolve_async if is_async else cudaq.evolve
    result = evolve(RydbergHamiltonian(
        atom_sites=[(0., 0.)],
        amplitude=ScalarOperator.const(8e6),
        phase=ScalarOperator(lambda t: 0. if t.real < 2e-7 else np.pi / 2),
        delta_global=ScalarOperator.const(4e6)),
                    schedule=Schedule([0., 2e-7, 4e-7], ["t"]),
                    shots_count=10000)
    result = result.get() if is_async else result
    # QuEra's native readout encodes a Rydberg atom as 1 and a ground atom as 2.
    assert result.probability("1") == pytest.approx(abs(state[1])**2, abs=0.025)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
