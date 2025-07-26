# 🧠 Quantum Convolutional Neural Networks for ANNNI Model 

Scientists want to understand how quantum systems change  like knowing when water turns to ice. But they  don’t have enough labeled data to teach a machine to find these changes.

This study shows that with a special type of quantum AI (called QCNN) we don’t need a lot of labeled data. By training only on a few easy examples, the AI was still able to figure out the full picture  even the tricky parts no one labeled before.

They tested this on a system called the ANNNI model, which has three types of phases (like solid, liquid, gas but for quantum stuff), and the AI guessed them all correctly.

This means quantum AI might help us discover new physics — just by learning from small, known pieces.

Quantum machine learning (QML) is a new area that combines quantum computers and machine learning. In QML, special types of quantum circuits act like smart models to understand patterns. It's already being used in science and for making new data, like images or sounds. While QML can sometimes be better than regular (classical) models, it's still not fully clear how much extra benefit it gives — especially since deep learning is already very powerful.

Quantum Machine Learning (QML) is a new way of using quantum computers to learn patterns — just like how normal machine learning works on classical computers. But in QML, we work with special data that comes from quantum systems.

This paper shows how we can use Quantum Convolutional Neural Networks (QCNNs) to study a difficult quantum system, even when we don’t have full information (called labels) about it. Usually, to train a model, you need lots of labeled data — like knowing exactly which "phase" a quantum state is in. But getting those labels is very hard in physics, especially when you’re trying to discover something new.
 
They trained their model using only the easy parts of the system (where we already know the answers), and then used that trained model to guess the rest of the phase diagram — even in the hard parts where we don’t know the answers. This is called out-of-distribution generalization — learning from known areas to predict unknown ones.

They tested this idea on a well-known but complex system called the ANNNI model (Axial Next-Nearest-Neighbor Ising model), which shows different phases:

- Ferromagnetic (spins all aligned)
- Paramagnetic (spins disordered)
- Antiphase (spins alternate in a special 4-spin pattern)

This model is useful because it shows quantum frustration — a kind of "tug-of-war" between different forces in the system.

Using quantum computers, they created noisy versions of quantum data (like real-world measurements) and showed that their QML model could still figure out the whole phase diagram — even though it was trained only on a small part of it.

---

## 📌 Note on Implementation

This code simplifies the approach from the paper to suit a minimal proof-of-concept using the Qiskit framework:

- Ground states are computed via exact diagonalization (`eigh`), not VQE.
- Only κ (kappa) is varied while h is kept fixed.
- The QCNN is implemented as a shallow 8-parameter variational circuit with simple rotation and entangling layers.
- The classification task is binary (`kappa < 0.6` or `≥ 0.6`) instead of full multi-phase detection.

These choices make the setup suitable for quick experimentation and insight on phase classification using quantum circuits, aligned with the original paper.

---

## 🌟 Main Takeaways

- Quantum machine learning can help us learn things that normal methods can’t, especially where exact math solutions don’t exist.
- It needs very few training examples to make useful predictions.
- This technique could help physicists discover new phases of matter and understand complicated quantum systems in the future.

---

## ⚙️ 1. Variational State Preparation using VQE


- Objective: Prepare ground states of the ANNNI Hamiltonian \( H(\kappa, h) \) using the Variational Quantum Eigensolver (VQE).
- Ansatz: Hardware-efficient ansatz with R_y rotations and CNOTs in linear connectivity.
- Depth: D = 6 (for N = 6) and D = 9 (for N = 12).
- Optimization: Adam optimizer (learning rate = 0.3) with parameter recycling to improve convergence.
- Accuracy: Ground state energy errors < 1% relative to exact diagonalization. Enhanced accuracy near the Peschel-Emery line.
- Excited States: VQE also used to compute excited states, confirming degeneracy only near phase boundaries.

---

## 🧠 2. Quantum Convolutional Neural Network (QCNN)

**Inspired by:** Classical CNNs, adapted for NISQ quantum systems.

**Architecture:**
- Initial layer: R_y rotations.
- Repeating blocks: convolution → rotation → pooling.
- Final: fully connected gate → measurement.

**Goal:** Learn an observable O(θ) that distinguishes between different quantum phases:  
⟨ψ_A | O(θ) | ψ_A⟩ < 0 < ⟨ψ_B | O(θ) | ψ_B⟩

**Output:** Probability p_j(κ, h) of the input state belonging to one of three phases:  
Ferromagnetic, Paramagnetic, Antiphase.  
Output state |00⟩ interpreted as a “garbage class”.

---

## 📈 3. Generalization Ability

**Key Contribution:**  
QCNN generalizes out-of-distribution, trained on limited labeled data near the axes ( κ = 0, h = 0 ).

**Justification:** High fidelity within-phase, low fidelity across phases.

**Error Scaling:**  
Empirical results match theoretical bound (𝒪(T/n)), where:
- T = 𝒪[log(N)]: Number of QCNN parameters
- n: Training samples

---

## 📊 4. Training Dataset

**Training regions:**
- κ = 0: Transverse field Ising model (integrable)
- h = 0: Quasiclassical limit

**Three training modes:**
- GC: Gaussians near critical points (0,1) and (0.5,0)
- G2: Gaussians in the middle of each phase
- U: Uniform random sampling

**Loss Function:** Cross-entropy loss between predicted and one-hot encoded true phase labels.

---

## 📉 5. Results and Phase Classification

**Systems tested:** N = 6 and N = 12 spins.  
**Output:** Phase diagrams successfully reconstructed from QCNN outputs trained on sparse data.

**Findings:**
- QCNN trained on simplified models generalizes to the full ANNNI diagram.
- Anomaly scores and softmax probabilities clearly separate the 3 phases.
- Training with Gaussian samples around phase centers (G2) yields the best performance.

---

## 🧪 Results Breakdown

### 📊 Accuracy vs Number of Training Samples

- Accuracy improves rapidly with increasing training points (n), saturating for n ≥ 20.

**Comparison across sampling schemes:**
- GC (Gaussian near critical points): Blue
- G2 (Gaussian in middle of phases): Black
- U (Uniform sampling): Red

**Key finding:**  
Sampling away from the critical points (e.g., G2) is sufficient for accurate learning — location of samples matters less than previously thought.

---

### 🗺️ QCNN Phase Diagram (n = 40)

- Phase boundaries predicted by the QCNN match the ground truth well.
- Color shades: Mixture of class probabilities predicted by the QCNN (blue, green, yellow).
- Red lines: Predicted phase boundaries.
- Output is a soft classification, not hard labels — showing confidence across regions.

---

### 🤖 Autoencoder Comparison (Anomaly Detection - AD)

- Trained using a single product state (|ψ⟩, marked as a red cross).
- AD reconstructs well if the test state is similar (same phase), but poorly if it's far in Hilbert space (different phase).
- Color map shows the compression loss (higher means anomaly).

**Limitations:**
- Lacks precision in identifying phase boundaries.
- Fails for some initial states (like paramagnetic) showing instability.
- Can't give confidence like QCNN softmax outputs.

---

## ✅ Conclusions

- QCNNs can compute phase diagrams of non-integrable quantum systems by training only on simplified integrable regions.
- Demonstrated >97% accuracy using only 20 labeled training points.
- Generalization is strong and efficient, even from non-i.i.d. data.
- Training near critical points is not essential — a key insight for real-world physics data.

**QCNNs outperform unsupervised methods like AD in:**
- Quantitative predictions
- Stability
- Generalization

**Limitation:** Supervised QCNN cannot detect phases absent in the training set (e.g., BKT transition, PE line).  
AD can help qualitatively, but lacks QCNN's accuracy.

---

This work opens a pathway for practical QML applications in quantum physics by reducing the need for large labeled datasets.




