from qiskit import QuantumCircuit, Aer, execute
from qiskit.algorithms.optimizers import SPSA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def generate_data(n_samples=60):
    data = []
    labels = []
    for i in range(n_samples):
        label = i % 3
        qc = QuantumCircuit(3)
        if label == 0:
            qc.ry(np.pi/4, 0)
        elif label == 1:
            qc.ry(np.pi/2, 1)
            qc.cx(1, 2)
        else:
            qc.ry(3*np.pi/4, 0)
            qc.cx(0, 1)
            qc.cx(1, 2)
        data.append(qc)
        labels.append(label)
    return data, np.array(labels)

def create_qcnn(params):
    qc = QuantumCircuit(3)
    for i in range(3):
        qc.ry(params[i], i)
    qc.cx(0, 1)
    qc.cx(1, 2)
    for i in range(3):
        qc.ry(params[3 + i], i)
    qc.cx(0, 1)
    qc.cx(1, 2)
    for i in range(3):
        qc.ry(params[6 + i], i)
    qc.cx(0, 1)
    qc.ry(params[9], 0)
    qc.measure_all()
    return qc

def predict(qc_template, params):
    backend = Aer.get_backend('qasm_simulator')
    result = execute(qc_template, backend=backend, shots=1024).result()
    counts = result.get_counts()
    prediction = max(counts, key=counts.get)
    return prediction

def objective(params, X, y_true):
    y_pred = []
    for i, qc_data in enumerate(X):
        full_circuit = qc_data.compose(create_qcnn(params))
        full_circuit.measure_all()
        backend = Aer.get_backend('qasm_simulator')
        result = execute(full_circuit, backend=backend, shots=1024).result()
        counts = result.get_counts()
        guess = max(counts, key=counts.get)
        pred_class = int(guess[-1])
        y_pred.append(pred_class)
    return 1 - accuracy_score(y_true, y_pred)

X, y = generate_data(60)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

params = 2 * np.pi * np.random.rand(10)
optimizer = SPSA(maxiter=20)
opt_params, _, _ = optimizer.optimize(
    num_vars=len(params),
    objective_function=lambda p: objective(p, X_train, y_train),
    initial_point=params
)

def evaluate(X, y, params):
    y_pred = []
    for qc in X:
        full_circuit = qc.compose(create_qcnn(params))
        full_circuit.measure_all()
        result = execute(full_circuit, Aer.get_backend('qasm_simulator'), shots=1024).result()
        counts = result.get_counts()
        guess = max(counts, key=counts.get)
        pred_class = int(guess[-1])
        y_pred.append(pred_class)
    return accuracy_score(y, y_pred)

train_acc = evaluate(X_train, y_train, opt_params)
test_acc = evaluate(X_test, y_test, opt_params)

print(f"{train_acc * 100:.2f}")
print(f"{test_acc * 100:.2f}")

accuracies = [evaluate(X_train, y_train, params)]
for i in range(20):
    new_params = params + 0.1 * np.random.randn(10)
    acc = evaluate(X_train, y_train, new_params)
    accuracies.append(acc)

plt.plot(accuracies)
plt.title("QCNN Accuracy Over Random Trials")
plt.xlabel("Trial")
plt.ylabel("Accuracy")
plt.grid()
plt.show()


