from qiskit import QuantumCircuit
import numpy as np

PHI = (1 + np.sqrt(5)) / 2  # número áureo
delta = 0.1
n = 3

theta = np.pi * (n + delta) * (1 + PHI)

qc = QuantumCircuit(1)
qc.rz(theta, 0)
qc.rx(np.pi / 4, 0)
qc.draw()
