# 🧠 QCNN Phase Classification on ANNNI Model (Qiskit)

This project implements a simplified **Quantum Convolutional Neural Network (QCNN)** using [Qiskit](https://qiskit.org/) to classify the quantum ground states of the **Axial Next-Nearest-Neighbor Ising (ANNNI)** model.
The goal is to predict the phase of the system based on the value of a physical parameter `κ (kappa)`.

---

## 📜 Overview

We simulate a 4-qubit quantum system using the ANNNI Hamiltonian. For different κ values, we:

* Compute the ground state using exact diagonalization,
* Use a shallow QCNN-like circuit to classify the parity of the state,
* Predict whether κ lies below or above a threshold (here, 0.6) — a **binary phase classification task**.

The QCNN is untrained (i.e., parameters are randomly initialized) and acts as a proof-of-concept.

---

## 🧩 Code Breakdown

### 1. **Imports**

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.opflow import I, X, Z
from scipy.linalg import eigh
from sklearn.metrics import accuracy_score
```

These libraries are used to:

* Construct the quantum Hamiltonian (`I, X, Z` from `qiskit.opflow`),
* Simulate quantum circuits and states (`Statevector`, `QuantumCircuit`),
* Compute ground states via matrix diagonalization (`eigh` from `scipy`),
* Evaluate performance (`accuracy_score` from `sklearn`).

---

### 2. **Hamiltonian Construction**

```python
def annni_hamiltonian(n=4, J=1.0, g=0.5, kappa=0.5):
```

Defines the Hamiltonian for the ANNNI model with:

* `n`: number of qubits
* `J`: nearest-neighbor interaction strength
* `g`: transverse field
* `kappa`: next-nearest-neighbor interaction strength

It includes 3 terms:

1. Nearest-neighbor Z-Z coupling
2. Local X field
3. Next-nearest-neighbor Z-Z coupling

Returns a **real-valued matrix** representation of the Hamiltonian.

---

### 3. **Ground State Extraction**

```python
def get_ground_state(H):
    eigvals, eigvecs = eigh(H)
    return eigvals[0], eigvecs[:, 0]
```

Uses **exact diagonalization** to compute the **lowest energy eigenstate** (ground state) of the Hamiltonian.

---

### 4. **QCNN Circuit Construction**

```python
def build_qcnn(params, n_qubits=4):
```

Builds a simple variational quantum circuit with:

* **Entanglement layer**: Controlled-X gates (CX)
* **Rotation layer**: Parameterized `Ry` and `Rz` on each qubit
* **Pooling layer**: Controlled-Z (CZ) gates

This mimics the convolution → activation → pooling structure of a classical CNN.

---

### 5. **Parity Classification Logic**

```python
def classify_parity(statevec, circuit):
```

* Evolves a quantum state using the QCNN circuit.
* Measures probabilities of all basis states.
* If the number of 1s in the bitstring is **even**, predicts class `0`; otherwise, class `1`.

Used here as a binary classifier (e.g., whether κ < 0.6 or not).

---

### 6. **Main Loop: Generate Ground States**

```python
kappas = [0.1, 0.5, 1.2, 0.8, 0.3]
true_labels = [0 if k < 0.6 else 1 for k in kappas]
```

Defines the input κ values and corresponding **true phase labels** (0 or 1).

```python
for k in kappas:
    H = annni_hamiltonian(kappa=k)
    _, psi0 = get_ground_state(H)
    ground_states.append(Statevector(psi0))
```

For each κ, compute the ground state and convert it into a `Statevector`.

---

### 7. **QCNN Inference**

```python
params = 2 * np.pi * np.random.rand(8)
```

Randomly initialize 8 circuit parameters.

```python
for idx, sv in enumerate(ground_states):
    circuit = build_qcnn(params)
    pred = classify_parity(sv, circuit)
```

For each state:

* Build the QCNN,
* Classify it using the parity logic,
* Compare predicted vs. true label.

---

### 8. **Accuracy Evaluation**

```python
acc = accuracy_score(true_labels, preds)
print(f"Final QCNN Classification Accuracy: {acc * 100:.2f}%")
```

Computes and prints the model accuracy.

---

