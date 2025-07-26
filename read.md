#  QCNN Phase Classification on ANNNI Model (Qiskit)

This project uses a simple quantum circuit to study how a quantum system behaves.
It tries to tell which phase the system is in by checking the value of a setting called kappa (κ).

---

##  Overview 

I created a small 4-qubit quantum system using a physics model called ANNNI. For different values of a setting called kappa (κ) we:

Find the system’s most stable state (called the ground state),

Use a basic quantum neural network (QCNN) to look at that state,

Predict if κ is less than or greater than 0.6 which tells us which phase the system is in.

Our QCNN isn’t trained — it just uses random values. This project is like a demo to show that even this simple setup can still give useful result

---

##  Code explanation 

### 1. **Imports**

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.opflow import I, X, Z
from scipy.linalg import eigh
from sklearn.metrics import accuracy_score
````

These tools help us build and run our quantum model:

numpy – For math operations and making random numbers.

QuantumCircuit – Lets us design quantum circuits (like building blocks for quantum computers).

Statevector – Helps us simulate the behavior of quantum states.

I, X, Z – These are basic quantum operators used to build the system’s physics (called the Hamiltonian).

eigh – Finds the energy levels and states of our quantum system.

accuracy_score – Checks how accurate our predictions are by comparing them to the correct answers.

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

Adds **nearest-neighbor Z-Z interactions** btw each pair of adjacent qubits.

```python
    for i in range(n):
        H += -g * (I ^ i) @ X ^ (I ^ (n - i - 1))
```

Adds  **transverse field (X term)** on each individual qubit.

```python
    for i in range(n - 2):
        H += -kappa * (Z ^ i) @ (Z ^ (i + 2)) ^ (I ^ (n - i - 3))
```

Adds **next-nearest-neighbor Z-Z interactions** controlled by `kappa`.

```python
    return H.to_matrix()
```

Converts the Hamiltonian to  matrix so we can diagonalize it later.

---

### 3. **Ground State Extraction**

```python
def get_ground_state(H):
    eigvals, eigvecs = eigh(H)
    return eigvals[0], eigvecs[:, 0]
```
we use eigh to break down the Hamiltonian and find its energy levels and quantum states.
Then we return the lowest energy value and its corresponding state which is called the ground state — the most stable configuration of the system.

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
We use 8 parameters total: 4 for Ry, 4 for Rz

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
  * If the number is **even** add its probability to total.
  * If **odd** then  subtract it.

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



