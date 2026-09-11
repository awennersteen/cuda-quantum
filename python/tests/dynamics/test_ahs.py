# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import dataclasses

import numpy as np
import pytest

from cudaq import cudaq_runtime


def _program(sites, omega, phase, detuning, times):
    ahs = cudaq_runtime.ahs
    program = ahs.Program()
    program.setup.ahs_register.sites = sites
    program.setup.ahs_register.filling = [1] * len(sites)
    drive = ahs.DrivingField()
    for name, values in (("amplitude", omega), ("phase", phase), ("detuning",
                                                                  detuning)):
        field = ahs.PhysicalField()
        field.time_series = ahs.TimeSeries(list(zip(values, times)))
        setattr(drive, name, field)
    program.hamiltonian.drivingFields = [drive]
    return program


@pytest.mark.parametrize("num_atoms", [1, 2, 3])
@pytest.mark.parametrize("phase", [0.0, 0.7])
def test_pulser_hamiltonian_and_state_parity(num_atoms, phase):
    """Compare the AHS Hamiltonian and coherent evolution with Pulser."""
    pulser = pytest.importorskip("pulser")
    simulation = pytest.importorskip("pulser_simulation")
    from scipy.integrate import solve_ivp

    sites_um = [(0., 0.), (6., 0.), (0., 8.)][:num_atoms]
    device = dataclasses.replace(pulser.devices.MockDevice, rydberg_level=60)
    sequence = pulser.Sequence(
        pulser.Register.from_coordinates(sites_um, center=False, prefix="q"),
        device)
    sequence.declare_channel("drive", "rydberg_global")
    sequence.add(pulser.Pulse.ConstantPulse(800, 4.0, 1.5, phase), "drive")
    reference = simulation.QutipEmulator.from_sequence(sequence,
                                                       sampling_rate=1.0,
                                                       with_modulation=False)
    program = _program([(x * 1e-6, y * 1e-6) for x, y in sites_um], [4e6, 4e6],
                       [phase, phase], [1.5e6, 1.5e6], [0., 4e-7])
    ahs = cudaq_runtime.ahs
    spec = ahs.device_specification("FRESNEL_CAN1")
    assert spec.rydberg_c6 == pytest.approx(device.interaction_coeff * 1e-30)
    operator = ahs.rydberg_hamiltonian(program, spec)
    matrix = operator.to_matrix(
        {i: 2 for i in range(num_atoms)}, t=2e-7, invert_order=True) * 1e-6
    # Pulser puts site 0 first in the tensor product and uses |r>, |g>.
    # CUDA-Q uses site 0 as the least significant index and |g>, |r>.
    expected = reference.get_hamiltonian(200).full()[::-1, ::-1]
    np.testing.assert_allclose(matrix, expected, atol=1e-10)
    initial = np.zeros(2**num_atoms, dtype=complex)
    initial[0] = 1.
    evolved = solve_ivp(lambda t, state: -1j * matrix @ state, (0., 0.4),
                        initial,
                        rtol=1e-10,
                        atol=1e-12).y[:, -1]
    # Compare before Pulser's implicit zero sample at the end of a pulse.
    expected_state = reference.run(rtol=1e-10, atol=1e-12).get_state(
        0.4, ignore_global_phase=False).full().ravel()[::-1]
    np.testing.assert_allclose(evolved, expected_state, atol=1e-7, rtol=1e-7)


def test_frozen_pulser_hamiltonian_and_state_parity():
    """Check a fixed Pulser reference without requiring Pulser to be installed."""
    # Pulser/pulser-simulation 1.9.1, MockDevice level 60, ConstantPulse
    # (800 ns, 4 rad/us, 1.5 rad/us, phase 0.7), no modulation, sampling_rate=1.
    # get_hamiltonian(200), reversed into |g>, |r> with site 0 first.
    expected = np.diag([
        0., -1.5, -1.5, -2.13427698, -1.5, 0.30247123718261726,
        15.555448816872428, 18.223643074055044
    ]).astype(complex)
    for i, j in [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7),
                 (4, 5), (4, 6), (5, 7), (6, 7)]:
        expected[i, j] = 1.5296843745689768 - 1.2884353744753825j
        expected[j, i] = expected[i, j].conjugate()
    program = _program([(0., 0.), (6e-6, 0.), (0., 8e-6)], [4e6] * 2, [0.7] * 2,
                       [1.5e6] * 2, [0., 4e-7])
    ahs = cudaq_runtime.ahs
    operator = ahs.rydberg_hamiltonian(program,
                                       ahs.device_specification("FRESNEL_CAN1"))
    matrix = operator.to_matrix({
        0: 2,
        1: 2,
        2: 2
    }, t=2e-7, invert_order=True) * 1e-6
    np.testing.assert_allclose(matrix, expected, atol=1e-10, rtol=1e-12)
    # Pulser get_state(0.4, ignore_global_phase=False), rtol=1e-10, atol=1e-12.
    expected_state = np.array([
        0.3308467330777477 - 0.13892272361725103j,
        0.23960009327203305 - 0.20646028774714997j,
        0.3650697083546202 - 0.21793064675246762j,
        0.17670328828213308 - 0.40872876500980404j,
        0.3797685246725184 - 0.18669475720672554j,
        0.021736946006670907 - 0.44326848064178115j,
        -0.10159740193158978 - 0.005599642652013225j,
        -0.05736830935751294 + 0.049708453262244066j
    ])
    energies, vectors = np.linalg.eigh(matrix)
    evolved = vectors @ (np.exp(-0.4j * energies) * vectors[0].conj())
    np.testing.assert_allclose(evolved, expected_state, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("device_name", ["Aquila", "FRESNEL_CAN1"])
def test_device_phase_convention(device_name):
    """Compare nonzero drive phases with the device's raising/lowering terms."""
    # AWS uses exp(+i phi)|g><r| + exp(-i phi)|r><g|; Pulser reverses phi.
    # https://docs.aws.amazon.com/braket/latest/developerguide/braket-quera-submitting-analog-program-aquila.html
    phase = 0.7
    sign = -1. if device_name == "Aquila" else 1.
    ahs = cudaq_runtime.ahs
    spec = ahs.device_specification(device_name)
    assert spec.phase_sign == sign
    program = _program([(0., 0.)], [4e6] * 2, [phase] * 2, [1.5e6] * 2,
                       [0., 4e-7])
    coupling = 2e6 * np.exp(-1j * sign * phase)
    expected = np.array([[0., coupling], [coupling.conjugate(), -1.5e6]])
    for device in (spec, ahs.DeviceSpecification(spec.rydberg_c6, sign)):
        matrix = ahs.rydberg_hamiltonian(program, device).to_matrix({0: 2},
                                                                    t=0.)
        np.testing.assert_allclose(matrix, expected, atol=1e-9, rtol=1e-12)


def test_phase_jump_and_nanosecond_payload():
    """Keep phase jumps and nanosecond times in the AHS representation."""
    import json

    program = _program([(0., 0.)], [2e6] * 3, [0., np.pi / 2, 0.], [0.] * 3,
                       [0., 1e-9, 2e-9])
    encoded = json.loads(program.to_json())
    assert [
        float(t) for t in encoded["hamiltonian"]["drivingFields"][0]["phase"]
        ["time_series"]["times"]
    ] == [0., 1e-9, 2e-9]
    ahs = cudaq_runtime.ahs
    operator = ahs.rydberg_hamiltonian(program,
                                       ahs.device_specification("Aquila"))
    before = operator.to_matrix({0: 2}, t=0.5e-9)
    after = operator.to_matrix({0: 2}, t=1.5e-9)
    assert before[0, 1] == pytest.approx(1e6)
    assert after[0, 1] == pytest.approx(1j * 1e6)


def test_vendor_neutral_device_coefficient():
    """Build the same AHS program with a caller-supplied device coefficient."""
    program = _program([(0., 0.), (5e-6, 0.)], [0., 0.], [0., 0.], [0., 0.],
                       [0., 1e-6])
    ahs = cudaq_runtime.ahs
    operator = ahs.rydberg_hamiltonian(program, ahs.DeviceSpecification(1e-24))
    matrix = operator.to_matrix({0: 2, 1: 2}, t=0.)
    assert matrix[3, 3] == pytest.approx(1e-24 / (5e-6)**6)


@pytest.mark.parametrize("num_atoms", [2, 3])
def test_emulation_pulser_sampling_parity(num_atoms, monkeypatch):
    """Compare GPU emulation samples with Pulser on an asymmetric register."""
    import cudaq

    if not (cudaq.has_target("pasqal") and cudaq.has_target("dynamics") and
            cudaq.num_available_gpus() > 0):
        pytest.skip(
            "AHS emulation requires the PASQAL target, dynamics and a GPU")
    pulser = pytest.importorskip("pulser")
    simulation = pytest.importorskip("pulser_simulation")
    sites_um = [(0., 0.), (6., 0.), (0., 8.)][:num_atoms]
    device = dataclasses.replace(pulser.devices.MockDevice, rydberg_level=60)
    sequence = pulser.Sequence(
        pulser.Register.from_coordinates(sites_um, center=False, prefix="q"),
        device)
    sequence.declare_channel("drive", "rydberg_global")
    sequence.add(pulser.Pulse.ConstantPulse(800, 4., 1.5, 0.7), "drive")
    reference = simulation.QutipEmulator.from_sequence(sequence,
                                                       sampling_rate=1.,
                                                       with_modulation=False)
    state = reference.run(rtol=1e-10,
                          atol=1e-12).get_state(0.4).full().ravel()[::-1]
    program = _program([(x * 1e-6, y * 1e-6) for x, y in sites_um], [4e6] * 2,
                       [0.7] * 2, [1.5e6] * 2, [0., 4e-7])
    monkeypatch.setenv("DISABLE_REMOTE_SEND", "1")
    cudaq.set_target("pasqal", emulate=True)
    cudaq.set_random_seed(31)
    result = cudaq_runtime.launch_analog_kernel(
        "__analog_hamiltonian_kernel__pulser_parity", program.to_json(), 20000)
    assert result.get_total_shots() == 20000
    for basis, probability in enumerate(np.abs(state)**2):
        assert result.probability(format(
            basis, f"0{num_atoms}b")) == pytest.approx(probability, abs=0.02)


@pytest.mark.parametrize("is_async", [False, True])
def test_ahs_routing_uses_hamiltonian_type(monkeypatch, is_async):
    """Route AHS evolution without requiring a vendor name in a target list."""
    from types import SimpleNamespace
    from cudaq.dynamics import evolution, Schedule
    from cudaq.operators import RydbergHamiltonian, ScalarOperator

    launches = []
    monkeypatch.setattr(cudaq_runtime, "get_target",
                        lambda: SimpleNamespace(name="qilimanjaro"))
    launcher = "launch_analog_kernel_async" if is_async else "launch_analog_kernel"
    monkeypatch.setattr(cudaq_runtime, launcher,
                        lambda *args: launches.append(args))
    evolve = evolution.evolve_async if is_async else evolution.evolve
    evolve(RydbergHamiltonian([(0., 0.)], ScalarOperator.const(0.),
                              ScalarOperator.const(0.),
                              ScalarOperator.const(0.)),
           schedule=Schedule([0., 1e-9], ["t"]),
           shots_count=17)
    assert len(launches) == 1
    assert launches[0][0].startswith(
        "__analog_hamiltonian_kernel__qilimanjaro_")
    assert launches[0][2] == 17


@pytest.mark.parametrize("target, c6_override", [("pasqal", None),
                                                 ("quera", None),
                                                 ("pasqal", 5.42e-24)])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("ramp", [False, True])
def test_local_detuning_with_vacancy_e2e(target, c6_override, is_async, ramp,
                                         monkeypatch):
    """Evolve occupied atoms with local detuning through the real QPU launch path."""
    import cudaq
    from cudaq.dynamics import Schedule
    from cudaq.operators import RydbergHamiltonian, ScalarOperator
    from scipy.integrate import solve_ivp

    if not (cudaq.has_target(target) and cudaq.has_target("dynamics") and
            cudaq.num_available_gpus()):
        pytest.skip("AHS emulation requires the target, dynamics and a GPU")
    monkeypatch.setenv("DISABLE_REMOTE_SEND", "1")
    options = {} if c6_override is None else {"rydberg_c6": str(c6_override)}
    cudaq.set_target(target, emulate=True, **options)
    cudaq.set_random_seed(47)
    h = RydbergHamiltonian(
        [(0., 0.), (1e-6, 0.), (6e-6, 0.)],
        ScalarOperator.const(4e6),
        ScalarOperator.const(0.7),
        ScalarOperator.const(1e6),
        atom_filling=[1, 0, 1],
        delta_local=(
            ScalarOperator(lambda t: 4e6 * t.real / 4e-7 if ramp else 4e6),
            [0.2, 0.9, 0.8]))
    evolve = cudaq.evolve_async if is_async else cudaq.evolve
    result = evolve(h, schedule=Schedule([0., 4e-7], ["t"]), shots_count=20000)
    result = result.get() if is_async else result
    x = np.array([[0., 1.], [1., 0.]])
    y = np.array([[0., -1j], [1j, 0.]])
    n, identity = np.diag([0., 1.]), np.eye(2)
    drive = 2e6 * (np.cos(0.7) * x + np.sin(0.7) * y)
    c6 = c6_override if c6_override is not None else (
        865723.02e-30 if target == "pasqal" else 5.42e-24)
    matrix = (np.kron(drive - 1e6 * n, identity) +
              np.kron(identity, drive - 1e6 * n) + c6 /
              (6e-6)**6 * np.kron(n, n))
    local = -4e6 * (0.2 * np.kron(n, identity) + 0.8 * np.kron(identity, n))
    state = solve_ivp(lambda t, state: -1j *
                      (matrix + (t / 4e-7
                                 if ramp else 1.) * local) @ state, (0., 4e-7),
                      np.array([1., 0., 0., 0.], dtype=complex),
                      rtol=1e-10,
                      atol=1e-12).y[:, -1]
    assert result.get_total_shots() == 20000
    for basis, probability in enumerate(np.abs(state)**2):
        left, right = format(basis, "02b")
        bits = left + "0" + right if target == "pasqal" else (
            "1" if left == "1" else "2") + "0" + ("1" if right == "1" else "2")
        assert result.probability(bits) == pytest.approx(probability, abs=0.02)
    if target == "quera":
        assert result.get_register_counts("pre_sequence")["101"] == 20000


@pytest.mark.parametrize("is_async", [False, True])
def test_malformed_waveform_emulation(is_async):
    """Report mismatched waveform arrays through sync and async emulation."""
    import json
    import cudaq

    if not (cudaq.has_target("pasqal") and cudaq.has_target("dynamics") and
            cudaq.num_available_gpus()):
        pytest.skip("AHS emulation requires PASQAL, dynamics and a GPU")
    cudaq.set_target("pasqal", emulate=True)
    payload = json.loads(
        _program([(0., 0.)], [1e6, 1e6], [0., 0.], [0., 0.],
                 [0., 1e-7]).to_json())
    payload["hamiltonian"]["drivingFields"][0]["amplitude"]["time_series"][
        "values"] = ["1e6"]
    launch = (cudaq_runtime.launch_analog_kernel_async
              if is_async else cudaq_runtime.launch_analog_kernel)
    with pytest.raises(ValueError,
                       match="times and values must have equal lengths"):
        result = launch("__analog_hamiltonian_kernel__malformed",
                        json.dumps(payload), 10)
        if is_async:
            result.get()


@pytest.mark.parametrize("c6, message", [
    ("nan", "finite C6 coefficient"),
    ("inf", "finite C6 coefficient"),
    ("-inf", "finite C6 coefficient"),
    ("1e300", "Hamiltonian norm bound must be finite"),
    ("1.0", "step is too small to advance waveform times"),
])
@pytest.mark.parametrize("is_async", [False, True])
def test_numerically_invalid_emulation(c6, message, is_async):
    """Reject invalid numerical scales through sync and async emulation."""
    import cudaq

    if not (cudaq.has_target("pasqal") and cudaq.has_target("dynamics") and
            cudaq.num_available_gpus()):
        pytest.skip("AHS emulation requires PASQAL, dynamics and a GPU")
    cudaq.set_target("pasqal", emulate=True, rydberg_c6=c6)
    program = _program([(0., 0.), (6e-6, 0.)], [0., 0.], [0., 0.], [0., 0.],
                       [0., 1e-7])
    launch = (cudaq_runtime.launch_analog_kernel_async
              if is_async else cudaq_runtime.launch_analog_kernel)
    with pytest.raises(ValueError, match=message):
        result = launch("__analog_hamiltonian_kernel__invalid_scale",
                        program.to_json(), 10)
        if is_async:
            result.get()


@pytest.mark.parametrize("target", ["pasqal", "quera"])
def test_cpp_emulation_e2e(target, tmp_path, monkeypatch):
    """Compile and run C++ AHS evolution with the actual nvq++ emulation flag."""
    import cudaq
    import shutil
    import subprocess

    compiler = shutil.which("nvq++")
    if not (compiler and cudaq.has_target(target) and
            cudaq.has_target("dynamics") and cudaq.num_available_gpus()):
        pytest.skip(
            "C++ AHS emulation requires nvq++, the target, dynamics and a GPU")
    source = tmp_path / "ahs.cpp"
    executable = tmp_path / "ahs"
    source.write_text(r'''
#include <cudaq.h>
#include <cudaq/algorithms/integrator.h>
#include <cudaq/algorithms/evolve.h>
#include <cassert>
#include <numbers>

int main() {
  cudaq::rydberg_hamiltonian pulse(
      {{0., 0.}, {6e-6, 0.}}, cudaq::scalar_operator(std::numbers::pi / 2e-7),
      cudaq::scalar_operator(0.7), cudaq::scalar_operator(-1e6), {0, 1},
      std::make_pair(cudaq::scalar_operator(2e6), std::vector<double>{1., 0.5}));
  cudaq::schedule times(std::vector<double>{0., 2e-7}, {"t"});
  auto sync = cudaq::evolve(pulse, times, 31);
  assert(sync.sampling_result->count("01") == 31);
  auto async = cudaq::evolve_async(pulse, times, 37).get();
  assert(async.sampling_result->count("01") == 37);
  cudaq::rydberg_hamiltonian idle({{0., 0.}, {6e-6, 0.}},
      cudaq::scalar_operator(0.), cudaq::scalar_operator(0.),
      cudaq::scalar_operator(0.), {0, 1});
  auto ground = cudaq::evolve(idle, times, 19);
  assert(ground.sampling_result->count("EXPECTED_GROUND") == 19);
}
'''.replace("EXPECTED_GROUND", "02" if target == "quera" else "00"))
    monkeypatch.setenv("DISABLE_REMOTE_SEND", "1")
    compiled = subprocess.run([
        compiler, "--target", target, "--emulate",
        str(source), "-o",
        str(executable)
    ],
                              capture_output=True,
                              text=True,
                              timeout=120)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    executed = subprocess.run([str(executable)],
                              capture_output=True,
                              text=True,
                              timeout=60)
    assert executed.returncode == 0, executed.stdout + executed.stderr
