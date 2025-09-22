#!/usr/bin/env python3
"""
Setup script for Qiskit Runtime CLI
by SmokAppSoftware jako with Claude AI, Gemini AI, COPILOT
"""

from setuptools import setup, find_packages
from pathlib import Path

# Leer el README para la descripción larga
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="qiskit-runtime-cli",
    version="2.2.0",
    author="SmokAppSoftware jako",
    author_email="jakocrazykings@gmail.com",
    description="Herramienta de línea de comandos interactiva para Qiskit Runtime y circuitos cuánticos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smokappstore/CLI-Qis-kit-",
    packages=find_packages(),
    py_modules=[
        "main",
        "qiskit_cli", 
        "run_cli"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "qiskit>=0.45.0",
        "numpy>=1.21.0",
        "qiskit-aer>=0.12.0",
        "matplotlib>=3.5.0",
        "Pillow>=9.0.0",
    ],
    extras_require={
        "ibm": ["qiskit-ibm-runtime>=0.15.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
        "full": [
            "qiskit-ibm-runtime>=0.15.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "qiskit-cli=main:main",
            "qcli=main:main",
        ],
    },
    keywords="qiskit quantum computing cli runtime ibm circuit quantum-circuits",
    project_urls={
        "Bug Reports": "https://github.com/smokappstore/CLI-Qis-kit-/tree/main/issues",
        "Source": "https://github.com/smokappstore/CLI-Qis-kit-/tree/main",
        "Documentation": "https://github.com/smokappstore/CLI-Qis-kit-/blob/main/README.md",
    },
    include_package_data=True,
    zip_safe=False,
)
