import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_bloch_multivector
from qiskit.quantum_info import Statevector, partial_trace

def create_teleportation_circuit():
    """Quantum teleportation circuit with arbitrary state input"""
    qc = QuantumCircuit(3, 2)

    # Prepare arbitrary quantum state on qubit 0
    theta, phi = np.pi/3, np.pi/4
    qc.ry(theta, 0)
    qc.rz(phi, 0)

    # Create entangled Bell pair between qubit 1 and 2
    qc.h(1)
    qc.cx(1, 2)

    # Bell-state measurement on qubit 0 and 1
    qc.cx(0, 1)
    qc.h(0)
    qc.measure([0, 1], [0, 1])

    # Apply conditional corrections
    qc.cx(1, 2)
    qc.cz(0, 2)

    return qc

def visualize_state_evolution():
    """Visualize initial and final quantum states in teleportation"""
    qc_statevec = QuantumCircuit(3)
    theta, phi = np.pi/3, np.pi/4
    qc_statevec.ry(theta, 0)
    qc_statevec.rz(phi, 0)

    initial_state = Statevector.from_instruction(qc_statevec)
    initial_qubit0 = partial_trace(initial_state, [1, 2])

    qc_statevec.h(1)
    qc_statevec.cx(1, 2)
    qc_statevec.cx(0, 1)
    qc_statevec.h(0)

    final_state = Statevector.from_instruction(qc_statevec)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_bloch_multivector(initial_qubit0, ax=ax1, title="Initial Qubit 0")
    plot_bloch_multivector(final_state, ax=ax2, title="Final Statevector")
    plt.tight_layout()
    plt.show()

    return initial_state, final_state

def phase_space_localization_demo():
    """Classical FFT-based analogy of vibrational localization"""
    x = np.linspace(-10, 10, 256)
    y = np.linspace(-10, 10, 256)
    X, Y = np.meshgrid(x, y)

    sigma = 1.0
    k_x, k_y = 2.0, 1.0

    psi = np.exp(-(X**2 + Y**2)/(4*sigma**2)) * np.exp(1j*(k_x*X + k_y*Y))
    phase_shift = np.exp(1j * (3.0*X + 2.0*Y))
    psi_shifted = psi * phase_shift

    prob_original = np.abs(psi)**2
    prob_shifted = np.abs(psi_shifted)**2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(prob_original, extent=[-10,10,-10,10])
    ax1.set_title("Original Localization")

    ax2.imshow(prob_shifted, extent=[-10,10,-10,10])
    ax2.set_title("Shifted Localization (Phase Modulated)")

    plt.tight_layout()
    plt.show()

def run_all():
    circuit = create_teleportation_circuit()
    print("Quantum Teleportation Circuit:")
    print(circuit.draw())

    visualize_state_evolution()
    phase_space_localization_demo()

if __name__ == "__main__":
    run_all()
