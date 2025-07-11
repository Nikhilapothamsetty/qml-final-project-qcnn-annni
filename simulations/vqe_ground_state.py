from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.opflow import PauliOp
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils import QuantumInstance
import numpy as np

def annni_ham(n):
    terms = []
    for i in range(n - 1):
        z = ["I"] * n
        z[i] = "Z"
        z[i + 1] = "Z"
        terms.append(PauliOp.from_label("".join(z)))
    for i in range(n - 2):
        z = ["I"] * n
        z[i] = "Z"
        z[i + 2] = "Z"
        terms.append(0.5 * PauliOp.from_label("".join(z)))
    for i in range(n):
        x = ["I"] * n
        x[i] = "X"
        terms.append(1.0 * PauliOp.from_label("".join(x)))
    return sum(terms).reduce()

def ansatz(n):
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(np.pi / 4, i)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc

nq = 4
h = annni_ham(nq)
c = ansatz(nq)

b = Aer.get_backend("aer_simulator_statevector")
qi = QuantumInstance(backend=b)
v = VQE(ansatz=c, optimizer=COBYLA(), quantum_instance=qi)
r = v.compute_minimum_eigenvalue(operator=h)

print("Energy:")
print(r.eigenvalue.real)


  



