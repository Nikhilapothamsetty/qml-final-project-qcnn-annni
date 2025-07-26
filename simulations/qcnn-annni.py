import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector
from qiskit.opflow import I, X, Z
from scipy.linalg import eigh
from sklearn.metrics import accuracy_score

def annni_hamiltonian(N=4, J=1.0, g=0.5, kappa=0.5):
    H = 0 * I ^ N
    for i in range(N - 1):
        term = 1
        for j in range(N):
            if j == i or j == i + 1:
                term = term ^ Z if isinstance(term, int) else term ^ Z
            else:
                term = term ^ I if isinstance(term, int) else term ^ I
        H += -J * term

    for i in range(N):
        term = 1
        for j in range(N):
            if j == i:
                term = term ^ X if isinstance(term, int) else term ^ X
            else:
                term = term ^ I if isinstance(term, int) else term ^ I
        H += -g * term

    for i in range(N - 2):
        term = 1
        for j in range(N):
            if j == i or j == i + 2:
                term = term ^ Z if isinstance(term, int) else term ^ Z
            else:
                term = term ^ I if isinstance(term, int) else term ^ I
        H += -kappa * term

    return H.to_matrix().real

def get_ground_state(H):
    evals, evecs = eigh(H)
    return evals[0], evecs[:, 0]

def original_qcnn(params, num_qubits=4):
    qc = QuantumCircuit(num_qubits)
    for i in range(0, num_qubits - 1, 2):
        qc.cx(i, i + 1)
    for i in range(num_qubits):
        qc.ry(params[i], i)
        qc.rz(params[i + num_qubits], i)
    for i in range(num_qubits):
        qc.cz(i, (i + 1) % num_qubits)
    return qc

def classify_state_parity(statevec, circuit):
    evolved = statevec.evolve(circuit)
    counts = evolved.probabilities_dict()
    even_parity = 0
    odd_parity = 0
    for bitstring, prob in counts.items():
        parity = sum(int(b) for b in bitstring) % 2
        if parity == 0:
            even_parity += prob
        else:
            odd_parity += prob
    return 0 if even_parity >= odd_parity else 1

kappas = [0.1, 0.5, 1.2, 0.8, 0.3]
labels = [0 if k < 0.6 else 1 for k in kappas]
states = []

for k in kappas:
    H = annni_hamiltonian(kappa=k)
    _, gs = get_ground_state(H)
    states.append(Statevector(gs))

np.random.seed(42)
params = 2 * np.pi * np.random.rand(8)

preds = []
for state in states:
    qc = original_qcnn(params)
    pred = classify_state_parity(state, qc)
    preds.append(pred)

acc = accuracy_score(labels, preds)
print(f"Classification Accuracy: {acc * 100:.2f}%")




