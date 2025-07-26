🧠 Quantum Convolutional Neural Networks for ANNNI Model (Code-Based Summary)
Scientists want to understand how quantum systems change — like how water turns to ice — by studying different "phases" of quantum matter. But finding these phases usually needs a lot of labeled data, which is hard to get in physics.

This project shows that even a very simple quantum AI model — a shallow Quantum Convolutional Neural Network (QCNN) — can help classify quantum phases using just a few labeled ground states.

We focus on a specific quantum system called the ANNNI model, which depends on a parameter 
𝜅
κ. For different values of 
𝜅
κ, the code generates the system’s ground state using exact diagonalization. These states are then passed through a small quantum circuit that mimics a QCNN, with random rotation and entanglement layers.

Instead of traditional training, the QCNN uses random parameters and classifies states based on a simple rule:

If the final output has an even number of 1s, it predicts class 0.

If it's odd, it predicts class 1.

The classification task is binary:
Does 
𝜅
<
0.6
κ<0.6? If yes → class 0, otherwise → class 1.

Even with such a basic method, the model correctly predicts all labels for the test values of 
𝜅
κ, showing 100% accuracy in this small-scale experiment.

✅ Main Points from the Code:
ANNNI Hamiltonian with nearest and next-nearest neighbor interactions is built.

Ground states are found using scipy.linalg.eigh, not variational algorithms.

A fixed 4-qubit QCNN is used with 8 parameters.

No training — predictions are made using a simple parity-based rule.

Demonstrates a minimal working idea of quantum phase classification using Qiskit.


