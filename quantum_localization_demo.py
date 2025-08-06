"""
Quantum Localization Demo - Core Concepts
==========================================

This file provides a streamlined demonstration of the key quantum localization concepts
for quick testing and educational purposes. For a comprehensive analysis, 
use the enhanced system in src/quantum_localization_enhanced.py

Created by Vers3Dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_bloch_multivector
from qiskit.quantum_info import Statevector, partial_trace, state_fidelity
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_teleportation_circuit(theta=np.pi/3, phi=np.pi/4):
    """
    Create a basic quantum teleportation circuit with parameterized input state
    
    Args:
        theta: Polar angle for input qubit state
        phi: Azimuthal angle for input qubit state
        
    Returns:
        QuantumCircuit: Teleportation circuit
    """
    qc = QuantumCircuit(3, 2)

    # Prepare arbitrary quantum state on qubit 0 (Alice's unknown state)
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    qc.barrier()

    # Create entangled Bell pair between qubit 1 (Alice's ancilla) and 2 (Bob's qubit)
    qc.h(1)
    qc.cx(1, 2)
    qc.barrier()

    # Alice's Bell-state measurement
    qc.cx(0, 1)  # Entangle unknown state with Alice's ancilla
    qc.h(0)      # Complete Bell measurement
    qc.barrier()

    # Measure Alice's qubits
    qc.measure([0, 1], [0, 1])
    qc.barrier()

    # Bob's conditional corrections based on Alice's measurement results
    qc.cx(1, 2)  # Apply X correction if needed
    qc.cz(0, 2)  # Apply Z correction if needed

    return qc

def demonstrate_state_evolution():
    """
    Visualize quantum state evolution during teleportation process
    """
    logger.info("Demonstrating quantum state evolution in teleportation")
    
    # Parameters for the state to be teleported
    theta, phi = np.pi/3, np.pi/4
    
    # Create circuit for state evolution analysis (no measurements)
    qc_statevec = QuantumCircuit(3)
    
    # Step 1: Prepare initial state
    qc_statevec.ry(theta, 0)
    qc_statevec.rz(phi, 0)
    initial_state = Statevector.from_instruction(qc_statevec)
    initial_qubit0 = partial_trace(initial_state, [1, 2])  # Extract qubit 0 state
    
    # Step 2: Create Bell pair and perform teleportation operations
    qc_statevec.h(1)
    qc_statevec.cx(1, 2)
    qc_statevec.cx(0, 1)
    qc_statevec.h(0)
    
    # Final state after Alice's operations (before measurement)
    final_state = Statevector.from_instruction(qc_statevec)
    bob_state = partial_trace(final_state, [0, 1])  # Extract Bob's qubit state
    
    # Calculate fidelity between original and Bob's state
    fidelity = state_fidelity(initial_qubit0, bob_state)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot initial state
    plot_bloch_multivector(initial_qubit0, ax=ax1, title="Initial State (Alice's Qubit)")
    
    # Plot final state on Bob's qubit
    plot_bloch_multivector(bob_state, ax=ax2, title=f"Final State (Bob's Qubit)\nFidelity: {fidelity:.4f}")
    
    plt.suptitle("Quantum Teleportation: State Transfer Demonstration", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    logger.info(f"Teleportation fidelity: {fidelity:.6f}")
    return initial_state, final_state, fidelity

def phase_space_localization_demo(grid_size=128):
    """
    Demonstrate classical analogy of vibrational localization using phase modulation
    
    Args:
        grid_size: Resolution of spatial grid
    """
    logger.info("Running phase space localization demonstration")
    
    # Create spatial grid
    x = np.linspace(-8, 8, grid_size)
    y = np.linspace(-8, 8, grid_size)
    X, Y = np.meshgrid(x, y)

    # Parameters for Gaussian wavepacket
    sigma = 1.2  # Width parameter
    k_x, k_y = 2.0, 1.0  # Initial momentum components

    # Original wavepacket (Gaussian with momentum)
    psi_original = np.exp(-(X**2 + Y**2)/(4*sigma**2)) * np.exp(1j*(k_x*X + k_y*Y))
    
    # Apply phase shift to demonstrate coordinate transformation
    shift_x, shift_y = 3.0, -2.0  # Desired translation
    phase_shift = np.exp(1j * (shift_x*X/sigma**2 + shift_y*Y/sigma**2))
    psi_shifted = psi_original * phase_shift

    # Calculate probability densities
    prob_original = np.abs(psi_original)**2
    prob_shifted = np.abs(psi_shifted)**2

    # Calculate centroids
    total_prob_orig = np.sum(prob_original)
    total_prob_shift = np.sum(prob_shifted)
    
    x_center_orig = np.sum(prob_original * X) / total_prob_orig
    y_center_orig = np.sum(prob_original * Y) / total_prob_orig
    
    x_center_shift = np.sum(prob_shifted * X) / total_prob_shift
    y_center_shift = np.sum(prob_shifted * Y) / total_prob_shift

    # Visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    extent = [-8, 8, -8, 8]
    
    # Original probability density
    im1 = ax1.imshow(prob_original, extent=extent, cmap='Blues', origin='lower')
    ax1.plot(x_center_orig, y_center_orig, 'ro', markersize=10, label=f'Center: ({x_center_orig:.2f}, {y_center_orig:.2f})')
    ax1.set_title('Original Localization')
    ax1.set_xlabel('Position X')
    ax1.set_ylabel('Position Y')
    ax1.legend()
    plt.colorbar(im1, ax=ax1)
    
    # Shifted probability density
    im2 = ax2.imshow(prob_shifted, extent=extent, cmap='Reds', origin='lower')
    ax2.plot(x_center_shift, y_center_shift, 'go', markersize=10, label=f'Center: ({x_center_shift:.2f}, {y_center_shift:.2f})')
    ax2.set_title('Phase-Shifted Localization')
    ax2.set_xlabel('Position X')
    ax2.set_ylabel('Position Y')
    ax2.legend()
    plt.colorbar(im2, ax=ax2)
    
    # Original wavefunction phase
    phase_orig = np.angle(psi_original)
    im3 = ax3.imshow(phase_orig, extent=extent, cmap='hsv', origin='lower')
    ax3.set_title('Original Phase Distribution')
    ax3.set_xlabel('Position X')
    ax3.set_ylabel('Position Y')
    plt.colorbar(im3, ax=ax3, label='Phase (radians)')
    
    # Shifted wavefunction phase
    phase_shift_applied = np.angle(psi_shifted)
    im4 = ax4.imshow(phase_shift_applied, extent=extent, cmap='hsv', origin='lower')
    ax4.set_title('Phase-Shifted Distribution')
    ax4.set_xlabel('Position X')
    ax4.set_ylabel('Position Y')
    plt.colorbar(im4, ax=ax4, label='Phase (radians)')
    
    plt.suptitle('Vibrational Localization via Phase Modulation', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Calculate and report transformation accuracy
    actual_shift_x = x_center_shift - x_center_orig
    actual_shift_y = y_center_shift - y_center_orig
    error_x = abs(actual_shift_x - shift_x)
    error_y = abs(actual_shift_y - shift_y)
    
    logger.info(f"Intended shift: ({shift_x:.2f}, {shift_y:.2f})")
    logger.info(f"Actual shift: ({actual_shift_x:.2f}, {actual_shift_y:.2f})")
    logger.info(f"Transformation error: ({error_x:.4f}, {error_y:.4f})")
    
    return {
        'original_wavefunction': psi_original,
        'shifted_wavefunction': psi_shifted,
        'original_centroid': (x_center_orig, y_center_orig),
        'shifted_centroid': (x_center_shift, y_center_shift),
        'transformation_error': (error_x, error_y)
    }

def quick_fidelity_test(num_tests=50):
    """
    Quick fidelity test across multiple random states
    
    Args:
        num_tests: Number of random states to test
        
    Returns:
        dict: Statistics about teleportation fidelity
    """
    logger.info(f"Running quick fidelity test with {num_tests} random states")
    
    fidelities = []
    
    for i in range(num_tests):
        # Generate random state parameters
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2*np.pi)
        
        # Create original state
        qc_orig = QuantumCircuit(1)
        qc_orig.ry(theta, 0)
        qc_orig.rz(phi, 0)
        original_state = Statevector.from_instruction(qc_orig)
        
        # Simulate teleportation (statevector method for ideal case)
        qc_teleport = QuantumCircuit(3)
        qc_teleport.ry(theta, 0)  # Prepare state
        qc_teleport.h(1)          # Bell pair
        qc_teleport.cx(1, 2)
        qc_teleport.cx(0, 1)      # Teleportation
        qc_teleport.h(0)
        
        final_state = Statevector.from_instruction(qc_teleport)
        bob_state = partial_trace(final_state, [0, 1])
        
        # Calculate fidelity
        fidelity = state_fidelity(original_state, bob_state)
        fidelities.append(fidelity)
    
    # Statistics
    fidelities = np.array(fidelities)
    stats = {
        'mean': np.mean(fidelities),
        'std': np.std(fidelities),
        'min': np.min(fidelities),
        'max': np.max(fidelities),
        'median': np.median(fidelities),
        'all_fidelities': fidelities
    }
    
    logger.info(f"Fidelity statistics - Mean: {stats['mean']:.6f} ± {stats['std']:.6f}")
    logger.info(f"Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
    
    return stats

def run_basic_demo():
    """
    Run the basic demonstration of quantum localization concepts
    """
    print("="*60)
    print("QUANTUM LOCALIZATION DEMO - BASIC CONCEPTS")
    print("Created by Vers3Dynamics")
    print("="*60)
    
    # 1. Show basic teleportation circuit
    logger.info("Creating quantum teleportation circuit")
    circuit = create_teleportation_circuit()
    print("\nQuantum Teleportation Circuit:")
    print(circuit.draw())
    
    # 2. Demonstrate state evolution
    print("\n" + "-"*50)
    print("QUANTUM STATE EVOLUTION ANALYSIS")
    print("-"*50)
    initial_state, final_state, fidelity = demonstrate_state_evolution()
    
    # 3. Phase space localization demo  
    print("\n" + "-"*50)
    print("PHASE SPACE LOCALIZATION DEMO")
    print("-"*50)
    localization_results = phase_space_localization_demo()
    
    # 4. Quick fidelity test
    print("\n" + "-"*50)
    print("TELEPORTATION FIDELITY ANALYSIS")
    print("-"*50)
    fidelity_stats = quick_fidelity_test()
    
    # Summary
    print("\n" + "="*60)
    print("DEMO SUMMARY")
    print("="*60)
    print(f"Teleportation Fidelity: {fidelity:.6f}")
    print(f"Average Fidelity ({len(fidelity_stats['all_fidelities'])} tests): {fidelity_stats['mean']:.6f}")
    print(f"Localization Error: {localization_results['transformation_error']}")
    print(f"System Status: {'✓ OPERATIONAL' if fidelity_stats['mean'] > 0.99 else '⚠ NEEDS OPTIMIZATION'}")
    print("="*60)
    
    return {
        'circuit': circuit,
        'state_evolution': (initial_state, final_state, fidelity),
        'localization_results': localization_results,
        'fidelity_stats': fidelity_stats
    }

if __name__ == "__main__":
    # Run basic demonstration
    try:
        results = run_basic_demo()
        logger.info("Basic demo completed successfully")
        
        print("\nFor comprehensive DARPA analysis, run:")
        print("python src/quantum_localization_enhanced.py")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise
