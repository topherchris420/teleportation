"""
Quantum Localization System - Setup Script
Created by Vers3Dynamics
"""

from setuptools import setup, find_packages
import os

# Read long description from README
def read_readme():
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    return "Quantum Localization using Vibrational Variables as Location Coordinates"

# Read requirements
def read_requirements():
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh 
                   if line.strip() and not line.startswith("#")]
    return ["qiskit", "matplotlib", "numpy", "scipy"]

setup(
    name="quantum-localization",
    version="1.0.0",
    author="Christopher Woodyard",
    author_email="ciao_chris@proton.me",  
    description="Quantum Localization using Vibrational Variables as Location Coordinates",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/topherchris420/teleportation",
    
    # Find packages automatically
    packages=find_packages(),
    
    # Real PyPI classifiers (these are official)
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10", 
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    python_requires=">=3.8",
    install_requires=read_requirements(),
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
        ],
        "plotting": [
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
        ],
    },
    
    # Command line tools (optional)
    entry_points={
        "console_scripts": [
            "quantum-demo=quantum_localization_demo:run_basic_demo",
        ],
    },
    
    # Include data files
    include_package_data=True,
    zip_safe=False,
    
    keywords=[
        "quantum computing",
        "quantum localization", 
        "quantum teleportation",
        "physics simulation",
    ],
)
