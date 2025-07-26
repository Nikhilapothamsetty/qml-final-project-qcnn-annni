import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.opflow import I, X, Z
from scipy.linalg import eigh
from sklearn.metrics import accuracy_score

def annni_hamiltonian(n=4, J=1.0, g=0.5, kappa=0.5):
    H = 0 * (I ^ n)
    for i in range(n - 1):
        term = 1
        for j in range(n):
            term = Z if j == i or j == i + 1 else I if term == 1 else term ^ (Z if j == i or j == i + 1 else I)
        H += -J * term
    for i in range(n):
        term = 1
        for j in range(n):
            term = X if j == i else I if term == 1 else term ^ (X if j == i else I)
        H += -g * term
    for i in range(n - 2):
        term = 1
        for j in range(n):
            term = Z if j == i or j == i + 2 else I if term == 1 else term ^ (Z if j == i or j == i + 2 else I)
        H += -kappa * term
    return H.to_matrix().real

def get_ground_state(H):
    eigvals, eigvecs = eigh(H)
    return eigvals[0], eigvecs[:, 0]

def build_qcnn(params, n_qubits=4):
    qc = QuantumCircuit(n_qubits)
    for i in range(0, n_qubits - 1, 2):
        qc.cx(i, i + 1)
    for i in range(n_qubits):
        qc.ry(params[i], i)
        qc.rz(params[i + n_qubits], i)
    for i in range(n_qubits):
        qc.cz(i, (i + 1) % n_qubits)
    return qc

def classify_parity(statevec, circuit):
    evolved = statevec.evolve(circuit)
    probs = evolved.probabilities_dict()
    even = sum(p for bitstring, p in probs.items() if bitstring.count('1') % 2 == 0)
    return 0 if even >= 0.5 else 1

kappas = [0.1, 0.5, 1.2, 0.8, 0.3]
true_labels = [0 if k < 0.6 else 1 for k in kappas]
ground_states = []

print("Generating ground states for different kappa values...")
for k in kappas:
    H = annni_hamiltonian(kappa=k)
    _, psi0 = get_ground_state(H)
    ground_states.append(Statevector(psi0))
print("Ground states generated!\n")

np.random.seed(42)
params = 2 * np.pi * np.random.rand(8)

preds = []
for idx, sv in enumerate(ground_states):
    circuit = build_qcnn(params)
    pred = classify_parity(sv, circuit)
    preds.append(pred)
    print(f"Kappa={kappas[idx]:.2f}, Predicted={pred}, True={true_labels[idx]}")

acc = accuracy_score(true_labels, preds)
print(f"\nFinal QCNN Classification Accuracy: {acc * 100:.2f}%")






