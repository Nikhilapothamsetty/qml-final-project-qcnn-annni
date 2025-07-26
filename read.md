# 🧠 QCNN Phase Classification on ANNNI Model (Qiskit)

This project implements a simplified **Quantum Convolutional Neural Network (QCNN)** using [Qiskit](https://qiskit.org/) to classify the quantum ground states of the **Axial Next-Nearest-Neighbor Ising (ANNNI)** model.  
The goal is to predict  **phase of the system** based on the value of a physical parameter `κ (kappa)`.

---

## 📜 Overview

We simulate a 4-qubit quantum system described by the ANNNI Hamiltonian. For different values of `κ`, we:

- Compute the ground state using exact diagonalization,
- Use a shallow QCNN-like circuit to classify the parity of the state,
- Predict whether `κ` lies **below or above a threshold (0.6)** — a **binary classification** representing two different quantum phases.

This QCNN is **untrained** (random parameters) and is meant to serve as a **proof-of-concept**.

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
````

These imports allow us to:

* `numpy`: Handle numerical operations and generate random numbers.
* `qiskit.QuantumCircuit`: Create quantum circuits.
* `qiskit.quantum_info.Statevector`: Simulate quantum states.
* `qiskit.opflow.I, X, Z`: Define operators for building Hamiltonians.
* `scipy.linalg.eigh`: Diagonalize matrices to get eigenvalues/eigenvectors.
* `sklearn.metrics.accuracy_score`: Evaluate how well our classifier performs.

---

### 2. **Hamiltonian Construction**

```python
def annni_hamiltonian(n=4, J=1.0, g=0.5, kappa=0.5):
```

Defines the ANNNI model Hamiltonian with:

* `n`: Number of qubits
* `J`: Nearest-neighbor interaction strength
* `g`: Transverse magnetic field strength
* `kappa`: Next-nearest-neighbor interaction strength

The Hamiltonian has **three types of terms**:

1. Z-Z coupling between neighboring qubits
2. A transverse magnetic field on each qubit (X terms)
3. Z-Z coupling between next-nearest neighbors

```python
    H = 0 * (I ^ n)
```

Start with a zero Hamiltonian as a placeholder. `I ^ n` creates an identity operator on all qubits.

```python
    for i in range(n - 1):
        H += -J * (Z ^ i) @ (Z ^ (i + 1)) ^ (I ^ (n - i - 2))
```

Adds **nearest-neighbor Z-Z interactions** between each pair of adjacent qubits.

```python
    for i in range(n):
        H += -g * (I ^ i) @ X ^ (I ^ (n - i - 1))
```

Adds a **transverse field (X term)** on each individual qubit.

```python
    for i in range(n - 2):
        H += -kappa * (Z ^ i) @ (Z ^ (i + 2)) ^ (I ^ (n - i - 3))
```

Adds **next-nearest-neighbor Z-Z interactions**, controlled by `kappa`.

```python
    return H.to_matrix()
```

Converts the Hamiltonian to a matrix so we can diagonalize it later.

---

### 3. **Ground State Extraction**

```python
def get_ground_state(H):
    eigvals, eigvecs = eigh(H)
    return eigvals[0], eigvecs[:, 0]
```

* Uses **exact diagonalization** (`eigh`) to find all eigenvalues and eigenvectors.
* Returns the **ground state energy** and corresponding eigenvector (the lowest energy state).

---

### 4. **QCNN Circuit Construction**

```python
def build_qcnn(params, n_qubits=4):
    qc = QuantumCircuit(n_qubits)
```

Creates a new quantum circuit with 4 qubits.

```python
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
```

Adds **entanglement** between adjacent qubits using **CNOT (CX)** gates.

```python
    for i in range(n_qubits):
        qc.ry(params[i], i)
        qc.rz(params[i + 4], i)
```

Applies learnable **Ry and Rz rotations**. These act like weights in classical neural networks.
We use 8 parameters total: 4 for Ry, 4 for Rz.

```python
    for i in range(0, n_qubits - 1, 2):
        qc.cz(i, i + 1)
```

Adds **pooling** using Controlled-Z (CZ) gates between alternate qubit pairs.

```python
    return qc
```

Returns the final quantum circuit.

---

### 5. **Parity Classification Logic**

```python
def classify_parity(statevec, circuit):
    evolved = statevec.evolve(circuit)
    probs = evolved.probabilities_dict()
```

* Applies the QCNN circuit to the ground state.
* Gets a dictionary of measurement outcomes and their probabilities.

```python
    total = 0
    for bitstring, prob in probs.items():
        ones = bitstring.count('1')
        if ones % 2 == 0:
            total += prob
        else:
            total -= prob
```

* For each bitstring:

  * Count how many `1`s it contains.
  * If the number is **even**, add its probability to total.
  * If **odd**, subtract it.

```python
    return 0 if total >= 0 else 1
```

* Final decision:

  * If even-parity dominates → return **0**
  * If odd-parity dominates → return **1**

---

### 6. **Main Loop: Generate Ground States**

```python
kappas = [0.1, 0.5, 1.2, 0.8, 0.3]
true_labels = [0 if k < 0.6 else 1 for k in kappas]
```

We want to classify the system as:

* **Class 0**: κ < 0.6
* **Class 1**: κ ≥ 0.6

```python
ground_states = []
for k in kappas:
    H = annni_hamiltonian(kappa=k)
    _, psi0 = get_ground_state(H)
    ground_states.append(Statevector(psi0))
```

* For each κ:

  * Generate the Hamiltonian
  * Find the ground state
  * Convert to `Statevector` and store

---

### 7. **QCNN Inference**

```python
params = 2 * np.pi * np.random.rand(8)
```

Initialize 8 random rotation angles between 0 and 2π.

```python
preds = []
for sv in ground_states:
    circuit = build_qcnn(params)
    pred = classify_parity(sv, circuit)
    preds.append(pred)
```

* Build the QCNN with these parameters
* Apply it to each ground state
* Store the predicted class (0 or 1)

---

### 8. **Accuracy Evaluation**

```python
acc = accuracy_score(true_labels, preds)
print(f"Final QCNN Classification Accuracy: {acc * 100:.2f}%")
```

* Compare predictions with true labels
* Print the final **accuracy**

---



