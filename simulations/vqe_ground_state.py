from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.opflow import I, X, Z
from qiskit.circuit.library import EfficientSU2
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import COBYLA

def construct_annni_hamiltonian(n_qubits, kappa=0.5, h=1.0):
    terms = []

    for i in range(n_qubits - 1):
        op = [I] * n_qubits
        op[i] = Z
        op[i + 1] = Z
        terms.append(op[0])
        for j in range(1, n_qubits):
            terms[-1] = terms[-1] ^ op[j]

    for i in range(n_qubits - 2):
        op = [I] * n_qubits
        op[i] = Z
        op[i + 2] = Z
        term = op[0]
        for j in range(1, n_qubits):
            term = term ^ op[j]
        terms.append(kappa * term)

    for i in range(n_qubits):
        op = [I] * n_qubits
        op[i] = X
        term = op[0]
        for j in range(1, n_qubits):
            term = term ^ op[j]
        terms.append(h * term)

    return sum(terms)

if __name__ == "__main__":
    n = 6  
    depth = 3
    kappa = 0.5
    h = 1.0

    hamiltonian = construct_annni_hamiltonian(n, kappa, h)

    ansatz = EfficientSU2(n, entanglement='linear', reps=depth)

    backend = Aer.get_backend("aer_simulator_statevector")
    qinst = QuantumInstance(backend, seed_simulator=123, seed_transpiler=123, shots=1024)

    optimizer = COBYLA(maxiter=250)
    vqe = VQE(ansatz=ansatz, optimizer=optimizer, quantum_instance=qinst)

    result = vqe.calculate_minimum_eigenvalue(operator=hamiltonian)
    energy = result.eigenvalue.real
    print("estimated ground state energy:", round(energy, 6))
