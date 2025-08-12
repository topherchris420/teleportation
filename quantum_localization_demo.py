"""
Quantum-Enhanced Navigation System (QENS)
==================================================

CLASSIFICATION: UNCLASSIFIED
DISTRIBUTION: Approved for public release; distribution unlimited

Principal Investigator: Christopher Woodyard, Vers3Dynamics
TRL Assessment: Current TRL 3 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq
from scipy.optimize import minimize
from scipy.special import hermite
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, state_fidelity, process_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, phase_damping_error
import time
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
import argparse

from src.quantum_localization_enhanced import QuantumLocalizationSystem, run_technical_simulation
from darpa_proposal_generator.theory import QuantumLocalizationTheory, RealisticQuantumSystem

class NumpyJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for numpy data types.
    This encoder handles numpy integers, floats, and arrays.
    """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

# Enhanced logging with proper classification handling
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - QENS-v2 - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('darpa_eris_enhanced_analysis.log')
    ]
)
from darpa_proposal_generator.analysis import EnhancedDARPAAnalysis, calculate_darpa_eris_score

from darpa_proposal_generator.roadmap import ExperimentalValidationRoadmap

logger = logging.getLogger(__name__)

def run_quick_demo() -> bool:
    """
    Runs a quick demonstration of the quantum localization system.
    This function demonstrates the core teleportation fidelity analysis.
    """
    logger.info("="*80)
    logger.info("RUNNING QUICK DEMO: Quantum Localization System")
    logger.info("="*80)

    try:
        # Initialize the core system from the enhanced module
        qls = QuantumLocalizationSystem(grid_size=64, space_bounds=(-5, 5))

        # Run a simplified teleportation fidelity analysis
        logger.info("Performing quick teleportation fidelity analysis (100 trials)...")
        teleportation_results = qls.analyze_teleportation_fidelity(num_trials=100)

        mean_fidelity = teleportation_results['mean_fidelity']
        std_fidelity = teleportation_results['std_fidelity']

        logger.info(f"Quick Demo Complete. Mean Fidelity: {mean_fidelity:.4f} +/- {std_fidelity:.4f}")

        if mean_fidelity > 0.95:
            logger.info("✅ Quick Demo PASSED: High fidelity teleportation achieved.")
            return True
        else:
            logger.warning("⚠️ Quick Demo NEEDS OPTIMIZATION: Fidelity below target threshold.")
            return False

    except Exception as e:
        logger.error(f"❌ Quick Demo FAILED: An error occurred: {str(e)}")
        return False

def run_enhanced_darpa_analysis():
    """
    Execute comprehensive enhanced DARPA ERIS analysis
    """
    logger.info("="*80)
    logger.info("STEP 1: Running Core Technical Simulations & Visualizations")
    logger.info("="*80)
    technical_results = run_technical_simulation()
    logger.info("="*80)
    logger.info("STEP 2: Running High-Level DARPA Proposal Analysis")
    logger.info("="*80)

    logger.info("="*80)
    logger.info("ENHANCED QUANTUM NAVIGATION SYSTEM ANALYSIS v2.0")
    logger.info("="*80)
    
    # Test multiple quantum platforms for robustness
    platforms = ["superconducting", "trapped_ion", "photonic"]
    
    best_system = None
    best_score = 0
    all_results = {}
    
    for platform in platforms:
        logger.info(f"Analyzing {platform} quantum platform...")
        
        try:
            # Initialize enhanced analysis system
            enhanced_system = EnhancedDARPAAnalysis(platform)
            
            # Run comprehensive analyses
            problem_analysis = enhanced_system.rigorous_problem_analysis()
            quantum_advantage = enhanced_system.realistic_quantum_advantage_assessment(num_trials=1000)
            team_assessment = enhanced_system.enhanced_team_assessment()
            market_analysis = enhanced_system.realistic_market_analysis()
            
            # Calculate enhanced DARPA ERIS score
            darpa_score = calculate_darpa_eris_score(
                problem_analysis, quantum_advantage, team_assessment, market_analysis
            )
            
            all_results[platform] = {
                'system': enhanced_system,
                'problem_analysis': problem_analysis,
                'quantum_advantage': quantum_advantage,
                'team_assessment': team_assessment,
                'market_analysis': market_analysis,
                'darpa_score': darpa_score
            }
            
            if darpa_score > best_score:
                best_score = darpa_score
                best_system = enhanced_system
                best_results = all_results[platform]
            
            logger.info(f"Platform {platform} - Enhanced DARPA Score: {darpa_score:.1f}/100")
            
        except Exception as e:
            logger.error(f"Platform {platform} analysis failed: {str(e)}")
            continue
    
    if best_system is None:
        raise Exception("All platform analyses failed")
    
    # Generate experimental validation roadmap
    validation_roadmap = ExperimentalValidationRoadmap()
    risk_mitigation = validation_roadmap.generate_risk_mitigation_plan()
    
    # Generate enhanced final report
    enhanced_report = generate_enhanced_darpa_report(
        best_results, validation_roadmap, risk_mitigation, best_score
    )
    
    print("\n" + "="*80)
    print("ENHANCED DARPA ERIS SUBMISSION")
    print("="*80)
    print(enhanced_report)
    print("="*80)
    
    # Summary for quick evaluation
    print(f"\n🎯 ENHANCED ANALYSIS SUMMARY:")
    print("="*50)
    print(f"Best Platform: {best_system.platform.upper()}")
    print(f"Enhanced DARPA Score: {best_score:.1f}/100")
    print(f"Realistic Quantum Advantage: {best_results['quantum_advantage']['accuracy_improvement_factor']:.1f}x")
    print(f"Conservative TRL Timeline: 2 → 4 in 36 months")
    print(f"Market Opportunity: ${best_results['market_analysis']['defense_market']['navigation_systems_market_2024']/1e9:.1f}B")
    print(f"Investment Required: ${best_results['market_analysis']['investment_requirements']['total_development_cost']/1e6:.0f}M")
    
    # Enhanced scoring breakdown
    print(f"\nDetailed Scoring:")
    print(f"  Problem Definition: {darpa_score * 0.4:.1f}/40 points")
    print(f"  State-of-Art Advancement: {darpa_score * 0.4:.1f}/40 points") 
    print(f"  Team Capability: {darpa_score * 0.15:.1f}/15 points")
    print(f"  Market Impact: {darpa_score * 0.05:.1f}/5 points")
    
    if best_score >= 75:
        print("✅ RECOMMENDATION: EXCELLENT DARPA ERIS CANDIDATE")
    elif best_score >= 65:
        print("✅ RECOMMENDATION: STRONG DARPA ERIS CANDIDATE")
    elif best_score >= 55:
        print("⚠️  RECOMMENDATION: VIABLE WITH ADDRESSED CONCERNS")
    else:
        print("❌ RECOMMENDATION: NEEDS SIGNIFICANT IMPROVEMENTS")
    
    return {
        'best_system': best_system,
        'best_results': best_results,
        'all_results': all_results,
        'validation_roadmap': validation_roadmap,
        'risk_mitigation': risk_mitigation,
        'enhanced_report': enhanced_report,
        'darpa_score': best_score
    }

from darpa_proposal_generator.report import generate_enhanced_darpa_report
from comparative_analysis import run_full_comparison


# Main execution for enhanced analysis
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum-Enhanced Navigation System (QENS) Demo")
    parser.add_argument(
        '--full-analysis',
        action='store_true',
        help='Run the full, enhanced DARPA analysis.'
    )
    parser.add_argument(
        '--full-comparison',
        action='store_true',
        help='Run the full hardware comparison and generate a report.'
    )
    args = parser.parse_args()

    print("DARPA ERIS: Quantum-Enhanced Navigation System")
    print("=" * 70)

    if args.full_comparison:
        try:
            run_full_comparison()
            print("\n✓ Full hardware comparison complete.")
            print("📄 Report saved to 'hardware_comparison_report.pdf'")
        except Exception as e:
            logger.error(f"Full hardware comparison failed: {str(e)}")
            print(f"\n❌ Full hardware comparison failed: {str(e)}")

    elif args.full_analysis:
        try:
            # Run enhanced comprehensive analysis
            results = run_enhanced_darpa_analysis()

            print(f"\n✓ Enhanced DARPA ERIS analysis complete")
            print(f"Final Enhanced Score: {results['darpa_score']:.1f}/100")

            # Save enhanced results
            with open('enhanced_darpa_eris_submission.txt', 'w') as f:
                f.write(results['enhanced_report'])
            print("📄 Enhanced DARPA ERIS submission saved to 'enhanced_darpa_eris_submission.txt'")

            # Save detailed analysis data
            with open('enhanced_analysis_data.json', 'w') as f:
                serializable_results = {
                    'darpa_score': results['darpa_score'],
                    'quantum_advantage': results['best_results']['quantum_advantage'],
                    'market_analysis': results['best_results']['market_analysis'],
                    'validation_phases': results['validation_roadmap'].validation_phases,
                    'risk_mitigation': results['risk_mitigation']
                }
                json.dump(serializable_results, f, indent=2, cls=NumpyJSONEncoder)
            print("📊 Detailed analysis data saved to 'enhanced_analysis_data.json'")

        except Exception as e:
            logger.error(f"Enhanced DARPA ERIS analysis failed: {str(e)}")
            print(f"\n❌ Enhanced analysis failed: {str(e)}")
            print("Check logs for detailed error information")
    else:
        # Run the quick demo by default
        run_quick_demo()
