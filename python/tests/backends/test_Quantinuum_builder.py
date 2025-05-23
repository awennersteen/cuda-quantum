# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq, pytest, os
import numpy as np
from typing import List
from cudaq import spin
from multiprocessing import Process
from network_utils import check_server_connection
try:
    from utils.mock_qpu.quantinuum import startServer
except:
    print("Mock qpu not available, skipping Quantinuum tests.")
    pytest.skip("Mock qpu not available.", allow_module_level=True)

# Define the port for the mock server
port = 62440


def assert_close(got) -> bool:
    return got < -1.5 and got > -1.9


@pytest.fixture(scope="session", autouse=True)
def startUpMockServer():
    # We need a Fake Credentials Config file
    credsName = '{}/QuantinuumFakeConfig.config'.format(os.environ["HOME"])
    f = open(credsName, 'w')
    f.write('key: {}\nrefresh: {}\ntime: 0'.format("hello", "rtoken"))
    f.close()

    cudaq.set_random_seed(13)

    # Set the targeted QPU
    cudaq.set_target('quantinuum', url='http://localhost:{}'.format(port))

    # Launch the Mock Server
    p = Process(target=startServer, args=(port,))
    p.start()

    if not check_server_connection(port):
        p.terminate()
        pytest.exit("Mock server did not start in time, skipping tests.",
                    returncode=1)

    yield credsName

    # Kill the server, remove the file
    p.terminate()
    os.remove(credsName)


@pytest.fixture(scope="function", autouse=True)
def configureTarget(startUpMockServer):

    # Set the targeted QPU with credentials
    cudaq.set_target('quantinuum',
                     url='http://localhost:{}'.format(port),
                     credentials=startUpMockServer)

    yield "Running the test."
    cudaq.reset_target()


def test_quantinuum_sample():
    # Create the kernel we'd like to execute on Quantinuum
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    kernel.h(qubits[0])
    kernel.cx(qubits[0], qubits[1])
    kernel.mz(qubits)
    print(kernel)

    # Run sample synchronously, this is fine
    # here in testing since we are targeting a mock
    # server. In reality you'd probably not want to
    # do this with the remote job queue.
    counts = cudaq.sample(kernel)
    counts.dump()
    assert (len(counts) == 2)
    assert ('00' in counts)
    assert ('11' in counts)

    # Run sample, but do so asynchronously. This enters
    # the execution job into the remote Quantinuum job queue.
    future = cudaq.sample_async(kernel)
    # We could go do other work, but since this
    # is a mock server, get the result
    counts = future.get()
    assert (len(counts) == 2)
    assert ('00' in counts)
    assert ('11' in counts)

    # Ok now this is the most likely scenario, launch the
    # job asynchronously, this puts it in the queue, now
    # you can take the future and persist it to file for later.
    future = cudaq.sample_async(kernel)
    print(future)

    # Persist the future to a file (or here a string,
    # could write this string to file for later)
    futureAsString = str(future)

    # Later you can come back and read it in and get
    # the results, which are now present because the job
    # made it through the queue
    futureReadIn = cudaq.AsyncSampleResult(futureAsString)
    counts = futureReadIn.get()
    assert (len(counts) == 2)
    assert ('00' in counts)
    assert ('11' in counts)


def test_quantinuum_observe():
    # Create the parameterized ansatz
    kernel, theta = cudaq.make_kernel(float)
    qreg = kernel.qalloc(2)
    kernel.x(qreg[0])
    kernel.ry(theta, qreg[1])
    kernel.cx(qreg[1], qreg[0])

    # Define its spin Hamiltonian.
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    # Run the observe task on quantinuum synchronously
    res = cudaq.observe(kernel, hamiltonian, .59)
    assert assert_close(res.expectation())

    # Launch it asynchronously, enters the job into the queue
    future = cudaq.observe_async(kernel, hamiltonian, .59)
    # Retrieve the results (since we're on a mock server)
    res = future.get()
    assert assert_close(res.expectation())

    # Launch the job async, job goes in the queue, and
    # we're free to dump the future to file
    future = cudaq.observe_async(kernel, hamiltonian, .59)
    print(future)
    futureAsString = str(future)

    # Later you can come back and read it in
    # You must provide the spin_op so we can reconstruct
    # the results from the term job ids.
    futureReadIn = cudaq.AsyncObserveResult(futureAsString, hamiltonian)
    res = futureReadIn.get()
    assert assert_close(res.expectation())


def test_quantinuum_state_preparation():
    kernel, state = cudaq.make_kernel(List[complex])
    qubits = kernel.qalloc(state)

    state = [1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.]
    counts = cudaq.sample(kernel, state)
    assert '00' in counts
    assert '10' in counts
    assert not '01' in counts
    assert not '11' in counts


def test_quantinuum_state_synthesis_from_simulator():
    kernel, state = cudaq.make_kernel(cudaq.State)
    qubits = kernel.qalloc(state)

    state = cudaq.State.from_data(
        np.array([1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.], dtype=complex))

    counts = cudaq.sample(kernel, state)
    assert "00" in counts
    assert "10" in counts
    assert len(counts) == 2


def test_quantinuum_state_synthesis():

    init, n = cudaq.make_kernel(int)
    qubits = init.qalloc(n)
    init.x(qubits[0])

    s = cudaq.get_state(init, 2)

    kernel, state = cudaq.make_kernel(cudaq.State)
    qubits = kernel.qalloc(state)
    kernel.x(qubits[1])

    s = cudaq.get_state(kernel, s)
    counts = cudaq.sample(kernel, s)
    assert '10' in counts
    assert len(counts) == 1


def test_exp_pauli():
    test = cudaq.make_kernel()
    q = test.qalloc(2)
    test.exp_pauli(1.0, q, "XX")

    counts = cudaq.sample(test)
    assert '00' in counts
    assert '11' in counts
    assert not '01' in counts
    assert not '10' in counts


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
