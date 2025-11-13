import pytest
from unittest.mock import MagicMock
from qiskit import QuantumCircuit
import run_cli

@pytest.fixture
def cli():
    """Fixture to create a QiskitCLI instance with a mocked logger."""
    logger = MagicMock()
    return run_cli.QiskitCLI(logger)

def test_add_measurement_incrementally(cli):
    """
    Tests that adding measurements incrementally maps qubits to the correct classical bits.
    This test replicates the bug where subsequent measurements overwrite the classical bit.
    """
    # 1. Create a circuit
    cli.create_circuit('2')
    assert cli.current_circuit is not None
    assert cli.current_circuit.num_qubits == 2

    # 2. Add the first measurement for qubit 0
    cli.add_measurement(['0'])

    # 3. Add the second measurement for qubit 1
    cli.add_measurement(['1'])

    # 4. Verify the circuit's measurements
    # Expected: Qubit 0 is mapped to classical bit 0, Qubit 1 to classical bit 1.
    # Bug behavior: Qubit 0 is mapped to classical bit 0, then Qubit 1 is ALSO mapped to classical bit 0.

    instructions = cli.current_circuit.data
    measure_instructions = [inst for inst in instructions if inst.operation.name == 'measure']

    assert len(measure_instructions) == 2, "There should be two measurement instructions."

    # Check the first measurement: qubit 0 -> clbit 0
    assert measure_instructions[0].qubits[0] == cli.current_circuit.qubits[0]
    assert measure_instructions[0].clbits[0] == cli.current_circuit.clbits[0]

    # Check the second measurement: qubit 1 -> clbit 1
    # This is the part that will fail due to the bug.
    assert measure_instructions[1].qubits[0] == cli.current_circuit.qubits[1]
    assert measure_instructions[1].clbits[0] == cli.current_circuit.clbits[1]
