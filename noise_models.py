"""
Noise Models for Quantum Hardware Platforms
===========================================

This module provides functions to create Qiskit `NoiseModel` objects from
hardware profiles. These models are used to simulate the performance of
different quantum hardware platforms.
"""

import json
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, phase_damping_error

def create_noise_model_from_profile(profile: dict) -> NoiseModel:
    """
    Creates a Qiskit NoiseModel from a hardware profile.

    Args:
        profile: A dictionary containing the hardware profile information.

    Returns:
        A Qiskit NoiseModel object representing the noise characteristics of the platform.
    """
    noise_model = NoiseModel()
    params = profile['parameters']

    # Gate errors from fidelity
    single_qubit_error_rate = 1.0 - params['gate_fidelity']['single_qubit']
    two_qubit_error_rate = 1.0 - params['gate_fidelity']['two_qubit']

    # Depolarizing error for gates
    single_qubit_error = depolarizing_error(single_qubit_error_rate, 1)
    two_qubit_error = depolarizing_error(two_qubit_error_rate, 2)

    noise_model.add_all_qubit_quantum_error(single_qubit_error, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx'])

    # Platform-specific noise
    if profile['name'] in ['Trapped Ions', 'Cavity QED']:
        # T1 and T2 thermal relaxation
        if 't1_time' in params and 't2_time' in params:
            t1 = params['t1_time']
            t2 = params['t2_time']
            # Qiskit's thermal_relaxation_error requires gate times, which are not in the profile.
            # As a simplification, we can use amplitude and phase damping errors.
            # This is an approximation of T1 and T2 effects.
            # The error per gate would be roughly gate_time/T_coherence.
            # Since we don't have gate times, we'll stick to the depolarizing error from fidelity,
            # which is a more direct measure of gate performance.
            # A more advanced implementation would include gate times in the profiles.
            pass
    elif profile['name'] == 'Photonics':
        # Photon loss modeled as amplitude damping
        if 'photon_loss' in params:
            photon_loss_error = amplitude_damping_error(params['photon_loss'])
            noise_model.add_all_qubit_quantum_error(photon_loss_error, ['u1', 'u2', 'u3'])
        # Dephasing error
        if 'dephasing_error' in params:
            dephasing = phase_damping_error(params['dephasing_error'])
            noise_model.add_all_qubit_quantum_error(dephasing, ['u1', 'u2', 'u3'])


    # Readout error
    if 'readout_error' in params:
        readout_error = params['readout_error']
        # This would be applied during simulation, not directly in the noise model for gates.
        # The simulator's `run` method has options for measurement error.
        # We will handle this in the simulation engine.
        pass

    return noise_model

def load_profile(filepath: str) -> dict:
    """
    Loads a hardware profile from a JSON file.

    Args:
        filepath: The path to the JSON file.

    Returns:
        A dictionary containing the hardware profile.
    """
    with open(filepath, 'r') as f:
        return json.load(f)
