from qiskit import Aer  #Pretend to run the quantum circuit on a fake computer
from qiskit.circuit import QuantumCircuit, ParameterVector  # Make circuits and angle dials
from qiskit.opflow import PauliOp  # Write energy rules using Z and X
from qiskit.algorithms import VQE  # Tool to guess the lowest energy
from qiskit.algorithms.optimizers import COBYLA  # Helps VQE find the best guess
from qiskit.utils import QuantumInstance  # Runs the circuit on the simulator

def build_annni_hamiltonian(n_qubits, kappa=0.5, h=1.0): # Makes the Hamiltonian (energy rules) for ANNNI model
    terms = []  # Start with an empty list of rules

    for i in range(n_qubits - 1):  # Loop through neighbor qubits
        pauli_str = ["I"] * n_qubits  # Start with identity gates
        pauli_str[i] = "Z"  # Put Z on qubit i
        pauli_str[i + 1] = "Z"  # Put Z on qubit i+1
        terms.append(PauliOp.from_label("".join(pauli_str)))  # Add this ZZ rule

    for i in range(n_qubits - 2):  # Loop for next-nearest neighbors
        pauli_str = ["I"] * n_qubits
        pauli_str[i] = "Z"
        pauli_str[i + 2] = "Z"
        terms.append(kappa * PauliOp.from_label("".join(pauli_str)))  # Add this weaker ZZ rule

    for i in range(n_qubits):  # Loop through each qubit
        pauli_str = ["I"] * n_qubits
        pauli_str[i] = "X"
        terms.append(h * PauliOp.from_label("".join(pauli_str)))  # Add X (flip) rule for each qubit

    return sum(terms).reduce()  # Combine all rules into one and simplify

def layered_ansatz(n_qubits, depth):  # Make a layered quantum circuit with tunable angles
    qc = QuantumCircuit(n_qubits)  # Start with a blank circuit with n_qubits
    params = ParameterVector("θ", length=n_qubits * depth * 2)  # Make a list of angle knobs for all rotations
    idx = 0  # Start at first angle

    for d in range(depth):  # Repeat layers
        for q in range(n_qubits):  # Rotate each qubit
            qc.ry(params[idx], q)
            idx += 1
        for q in range(n_qubits - 1):  # Add CX gates between neighbors
            qc.cx(q, q + 1)
        for q in range(n_qubits):  # Rotate again
            qc.ry(params[idx], q)
            idx += 1

    return qc, params  # Return circuit and angles

if __name__ == "__main__":  
    n = 6  # Number of qubits
    depth = 3  # Number of ansatz layers
    kappa = 0.5  # Next-nearest neighbor interaction strength
    h = 1.0  # Magnetic field strength

    H = build_annni_hamiltonian(n, kappa=kappa, h=h)  # Build the Hamiltonian
    ansatz, theta = layered_ansatz(n, depth)  # Build the ansatz circuit

    backend = Aer.get_backend("aer_simulator_statevector")  # Use the simulator
    qinst = QuantumInstance(backend, seed_simulator=123, seed_transpiler=123, shots=1024)  # Set up simulation settings

    optimizer = COBYLA(maxiter=250)  # Set the optimizer with max 250 tries
    vqe = VQE(ansatz=ansatz, optimizer=optimizer, quantum_instance=qinst)  # Prepare the VQE solver
    result = vqe.compute_minimum_eigenvalue(operator=H)  # Run VQE to get energy

    print(f"{result.eigenvalue.real:.6f}")  # Print the estimated ground state energy


