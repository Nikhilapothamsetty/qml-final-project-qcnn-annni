from qiskit import QuantumCircuit, Aer, execute
from qiskit.algorithms.optimizers import SPSA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def generate_data(n_samples=60):
    circuits, labels = [], []
    for i in range(n_samples):
        label = i % 3
        qc = QuantumCircuit(3)
        if label == 0:
            qc.ry(np.pi / 4, 0)
        elif label == 1:
            qc.ry(np.pi / 2, 1)
            qc.cx(1, 2)
        else:
            qc.ry(3 * np.pi / 4, 0)
            qc.cx(0, 1)
            qc.cx(1, 2)
        circuits.append(qc)
        labels.append(label)
    return circuits, np.array(labels)

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

def evaluate_model(X, y, weights):
    predictions = []
    backend = Aer.get_backend('qasm_simulator')
    for x in X:
        qc = x.compose(create_qcnn(weights))
        qc.measure_all()
        counts = execute(qc, backend=backend, shots=1024).result().get_counts()
        output_bit = max(counts, key=counts.get)
        pred = int(output_bit[-1])
        predictions.append(pred)
    return accuracy_score(y, predictions)

def cost_function(params, X, y):
    return 1 - evaluate_model(X, y, params)

X, y = generate_data(60)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

initial_params = 2 * np.pi * np.random.rand(10)
optimizer = SPSA(maxiter=20)
opt_params, _, _ = optimizer.optimize(
    num_vars=len(initial_params),
    objective_function=lambda p: cost_function(p, X_train, y_train),
    initial_point=initial_params
)

train_accuracy = evaluate_model(X_train, y_train, opt_params)
test_accuracy = evaluate_model(X_test, y_test, opt_params)

print(f"{train_accuracy * 100:.2f}")
print(f"{test_accuracy * 100:.2f}")

# 🔵 MODIFIED VISUALIZATION BLOCK 🔵
random_scores = []
for shift in np.linspace(-0.2, 0.2, 20):
    shifted = opt_params + shift * np.random.randn(10)
    score = evaluate_model(X_test, y_test, shifted)
    random_scores.append(score)

plt.plot(range(len(random_scores)), random_scores, marker='o', color='navy')
plt.title("Test Accuracy with Parameter Variations")
plt.xlabel("Trial")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.show()



