from qiskit import Aer
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.opflow import PauliSumOp
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils import QuantumInstance
from qiskit.opflow.primitive_ops import PauliOp
import numpy as np

def generate_annni_hamiltonian(n, kappa=0.5, h=1.0):
    terms = []
    for i in range(n - 1):
        zz = ["I"] * n
        zz[i] = "Z"
        zz[i + 1] = "Z"
        terms.append(PauliOp.from_label("".join(zz)))
    for i in range(n - 2):
        zzz = ["I"] * n
        zzz[i] = "Z"
        zzz[i + 2] = "Z"
        terms.append(kappa * PauliOp.from_label("".join(zzz)))
    for i in range(n):
        x = ["I"] * n
        x[i] = "X"
        terms.append(h * PauliOp.from_label("".join(x)))
    return sum(terms).reduce()

def hardware_ansatz(n, depth):
    circuit = QuantumCircuit(n)
    params = ParameterVector("θ", length=n * depth * 2)
    idx = 0
    for d in range(depth):
        for i in range(n):
            circuit.ry(params[idx], i)
            idx += 1
        for i in range(n - 1):
            circuit.cx(i, i + 1)
        for i in range(n):
            circuit.ry(params[idx], i)
            idx += 1
    return circuit, params

n_qubits = 6
hamiltonian = generate_annni_hamiltonian(n_qubits)

depth = 3
ansatz, parameters = hardware_ansatz(n_qubits, depth)

backend = Aer.get_backend("aer_simulator_statevector")
qi = QuantumInstance(backend=backend, shots=1024, seed_simulator=42, seed_transpiler=42)
vqe = VQE(ansatz=ansatz, optimizer=COBYLA(maxiter=200), quantum_instance=qi)
result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)
print("Estimated Ground State Energy:", result.eigenvalue.real)

