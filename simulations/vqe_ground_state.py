from qiskit import Aer # Simulates a quantum computer on your regular laptop
from qiskit.circuit import QuantumCircuit, ParameterVector  # Make circuits and angle dials
from qiskit.opflow import PauliOp  # Write energy rules using Z and X
from qiskit.algorithms import VQE  # Tool to guess the lowest energy
from qiskit.algorithms.optimizers import COBYLA  # Helps VQE find the best guess
from qiskit.utils import QuantumInstance  # Runs the circuit on the simulator

def build_annni_hamiltonian(n_qubits, kappa=0.5, h=1.0): # Creates the ANNNI Hamiltonian using number of qubits, interaction strength kappa, and magnetic field h
    terms = []  # List to store each term of the Hamiltonian

    for i in range(n_qubits - 1):  # Loop through neighbour qubits
        pauli_str = ["I"] * n_qubits  # Creates a list like ["I", "I", "I", ...]
        pauli_str[i] = "Z"  # Put Z on qubit i
        pauli_str[i + 1] = "Z"  # Put Z on qubit i+1
        terms.append(PauliOp.from_label("".join(pauli_str))) #Converts the string into a Pauli operator 

    for i in range(n_qubits - 2):  # Look at qubits that are 2 positions gap
        pauli_str = ["I"] * n_qubits  # Start with all identities (do nothing)
        pauli_str[i] = "Z"  # Add a Z on one qubit
        pauli_str[i + 2] = "Z"  # Add another Z two steps ahead
        terms.append(kappa * PauliOp.from_label("".join(pauli_str))) 

    for i in range(n_qubits):  # Loop through each qubit
        pauli_str = ["I"] * n_qubits
        pauli_str[i] = "X"
        terms.append(h * PauliOp.from_label("".join(pauli_str)))  # Add X (flip) rule for each qubit

    return sum(terms).reduce()  # Adds all the terms together and simplifies the result(full  Hamiltonian)

def layered_ansatz(n_qubits, depth):  # Builds a parameterized quantum circuit of specified depth
    qc = QuantumCircuit(n_qubits)  # Start with a blank circuit with n_qubits
    params = ParameterVector("θ", length=n_qubits * depth * 2)  # Make a list of angle knobs for all rotations
    idx = 0  # Start at first angle

    for d in range(depth):  # Adds multiple layers to the circuit
        for q in range(n_qubits):  # Rotate each qubit
            qc.ry(params[idx], q)
            idx += 1                #First round of RY rotations for each qubit
        for q in range(n_qubits - 1):  # Add CX gates between neighbors
            qc.cx(q, q + 1)
        for q in range(n_qubits):  # Rotate again
            qc.ry(params[idx], q)
            idx += 1               #Second round of RY rotations

    return qc, params  # Return circuit and angles

if __name__ == "__main__":  
    n = 6  # No of qubits
    depth = 3  # No of layers in the quantum circuit 
    kappa = 0.5  # Next-nearest neighbor interaction strength
    h = 1.0  # Magnetic field strength

    H = build_annni_hamiltonian(n, kappa=kappa, h=h)  # Builds the Hamiltonian for the ANNNI model
    ansatz, theta = layered_ansatz(n, depth)  # Build the ansatz circuit

    backend = Aer.get_backend("aer_simulator_statevector")  # Use the simulator
    qinst = QuantumInstance(backend, seed_simulator=123, seed_transpiler=123, shots=1024)  # Set up simulation settings

    optimizer = COBYLA(maxiter=250)  # Set the optimizer with max 250 tries
    vqe = VQE(ansatz=ansatz, optimizer=optimizer, quantum_instance=qinst)  # Prepare the VQE solver
    result = vqe.compute_minimum_eigenvalue(operator=H)  # Run VQE to get minimum energy eigenvalue of the Hamiltonian
    energy = result.eigenvalue.real  #  real part of the calculated energy
    print("Estimated ground state energy:", round(energy, 6)) # Print the estimated ground state energy
