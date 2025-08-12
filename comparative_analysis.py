"""
Comparative Analysis of Quantum Hardware Platforms
===================================================

This module provides the main orchestration for the quantum hardware
comparative analysis. It loads hardware profiles, runs simulations with
hardware-specific noise models, and generates a comparative analysis of
the results.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF

from noise_models import load_profile, create_noise_model_from_profile
from src.quantum_localization_enhanced import QuantumLocalizationSystem

def generate_comparison_report(results: list):
    """
    Generates a PDF report comparing the hardware platforms.

    Args:
        results: A list of dictionaries containing the simulation results.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Quantum Hardware Comparative Analysis", 0, 1, 'C')
    pdf.ln(10)

    # --- Comparison Table ---
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. Comparative Analysis Table", 0, 1)
    pdf.set_font("Arial", '', 8)

    col_widths = [35, 30, 30, 45, 45]
    headers = ["Platform", "Sim. Fidelity", "Exp. Fidelity", "Sim. Pos. Uncertainty", "Exp. Pos. Uncertainty"]
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 10, header, 1, 0, 'C')
    pdf.ln()

    for res in results:
        pdf.cell(col_widths[0], 10, res['platform'], 1)
        pdf.cell(col_widths[1], 10, f"{res['simulated_fidelity']:.4f}", 1)
        pdf.cell(col_widths[2], 10, f"{res['experimental_fidelity']:.4f}", 1)
        pdf.cell(col_widths[3], 10, f"{res['simulated_positional_uncertainty']:.4f}", 1)
        pdf.cell(col_widths[4], 10, f"{res['experimental_positional_uncertainty']:.4f}", 1)
        pdf.ln()
    pdf.ln(10)

    # --- Overlay Graphs ---
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. Performance Visualization", 0, 1)

    platforms = [res['platform'] for res in results]
    sim_fidelities = [res['simulated_fidelity'] for res in results]
    exp_fidelities = [res['experimental_fidelity'] for res in results]
    sim_uncertainties = [res['simulated_positional_uncertainty'] for res in results]
    exp_uncertainties = [res['experimental_positional_uncertainty'] for res in results]

    # Fidelity Comparison Chart
    plt.figure(figsize=(10, 6))
    x = np.arange(len(platforms))
    width = 0.35
    plt.bar(x - width/2, sim_fidelities, width, label='Simulated')
    plt.bar(x + width/2, exp_fidelities, width, label='Experimental')
    plt.ylabel('Fidelity')
    plt.title('Fidelity Comparison by Platform')
    plt.xticks(x, platforms)
    plt.legend()
    plt.tight_layout()
    fidelity_chart_path = "fidelity_comparison.png"
    plt.savefig(fidelity_chart_path)
    pdf.image(fidelity_chart_path, x=10, y=None, w=180)
    plt.close()

    # Positional Uncertainty Comparison Chart
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, sim_uncertainties, width, label='Simulated')
    plt.bar(x + width/2, exp_uncertainties, width, label='Experimental')
    plt.ylabel('Positional Uncertainty (λ)')
    plt.title('Positional Uncertainty Comparison by Platform')
    plt.xticks(x, platforms)
    plt.legend()
    plt.tight_layout()
    uncertainty_chart_path = "uncertainty_comparison.png"
    plt.savefig(uncertainty_chart_path)
    pdf.image(uncertainty_chart_path, x=10, y=None, w=180)
    plt.close()
    pdf.ln(10)

    # --- Citations ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "3. References", 0, 1)
    pdf.set_font("Arial", '', 10)
    for i, res in enumerate(results):
        source_text = res['source'].replace("–", "-")
        pdf.multi_cell(0, 5, f"[{i+1}] {res['platform']}: {source_text} (DOI: {res['doi']})")
        pdf.ln(5)

    pdf_output_path = "hardware_comparison_report.pdf"
    pdf.output(pdf_output_path)
    print(f"Comparison report saved to {pdf_output_path}")


def run_simulation_for_profile(profile: dict, qls: QuantumLocalizationSystem) -> dict:
    """
    Runs the simulation for a given hardware profile.
    """
    print(f"Running simulation for {profile['name']}...")
    noise_model = create_noise_model_from_profile(profile)
    teleportation_results = qls.analyze_teleportation_fidelity(
        num_trials=100,
        noise_model=noise_model,
        readout_error=profile['parameters'].get('readout_error', 0.0)
    )
    localization_results = qls.vibrational_localization_analysis(
        encoding_method=profile.get('encoding_method', 'vibrational_modes')
    )
    results = {
        "platform": profile['name'],
        "simulated_fidelity": teleportation_results['mean_fidelity'],
        "simulated_positional_uncertainty": np.mean(localization_results['position_uncertainty']),
        "experimental_fidelity": profile['literature_data']['fidelity_achieved'],
        "experimental_positional_uncertainty": profile['literature_data']['positional_uncertainty'],
        "source": profile['literature_data']['source'],
        "doi": profile['literature_data']['doi']
    }
    print(f"Finished simulation for {profile['name']}.")
    return results

def run_full_comparison():
    """
    Runs the full comparative analysis and generates a report.
    """
    print("Starting full hardware comparison...")
    hardware_profiles_dir = "hardware_profiles"
    all_results = []
    qls = QuantumLocalizationSystem(grid_size=64)

    profile_files = [f for f in os.listdir(hardware_profiles_dir) if f.endswith(".json")]
    if not profile_files:
        print("No hardware profiles found in 'hardware_profiles' directory.")
        return

    for filename in profile_files:
        filepath = os.path.join(hardware_profiles_dir, filename)
        profile = load_profile(filepath)
        simulation_result = run_simulation_for_profile(profile, qls)
        all_results.append(simulation_result)

    print("Full hardware comparison finished.")

    if all_results:
        generate_comparison_report(all_results)

    return all_results

if __name__ == '__main__':
    print("Running comparative analysis directly...")
    run_full_comparison()
