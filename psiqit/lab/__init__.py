# psiqit/lab/__init__.py 

"""
PSIQIT - Python Scientific Quantum Information Toolkit
Virtual Lab Module

This module provides a virtual quantum laboratory for running experiments.

Submodules:
-----------
virtual_lab.py - Virtual quantum lab with experiments and results
"""

# ============================================================================
# Version and metadata
# ============================================================================

from ..version import __version__

__all__ = [
    # Module information
    '__version__',
    
    # ========================================================================
    # VIRTUAL LAB (from virtual_lab.py)
    # ========================================================================
    'ExperimentStatus',
    'ExperimentResult',
    'Experiment',
    'QuantumLab',
    'PredefinedExperiments',
]

# ============================================================================
# Lazy imports to avoid loading everything at once
# ============================================================================

def _lazy_import(module_name: str, attr_name: str):
    """Helper for lazy imports"""
    import importlib
    module = importlib.import_module(f'.{module_name}', package='psiqit.lab')
    return getattr(module, attr_name)


# ============================================================================
# Import all functions (with fallback for missing modules)
# ============================================================================

# Try importing from each module
try:
    from .virtual_lab import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import virtual_lab module: {e}")


# ============================================================================
# Module information
# ============================================================================

__doc__ = """
PSIQIT Virtual Lab Module
=========================

This module provides a virtual quantum laboratory for running experiments:

Submodules:
-----------
virtual_lab.py - Virtual quantum lab with experiments and results

Quick Start:
------------
>>> from psiqit.lab import QuantumLab, PredefinedExperiments

# Create a lab
>>> lab = QuantumLab()

# Run a predefined experiment
>>> exp = PredefinedExperiments.bell_state()
>>> result = lab.run_experiment(exp)
>>> print(result.summary())

# Create a custom experiment
>>> exp = lab.create_experiment("My Experiment", n_qubits=2)
>>> exp.add_gate('h', 0)
>>> exp.add_gate('cx', 0, 1)
>>> result = lab.run_experiment(exp)
>>> print(result.summary())

# Get state vector
>>> state = lab.get_state_vector(exp)
>>> print(state)

# Get Bloch coordinates
>>> x, y, z = lab.get_bloch_coordinates(exp, 0)
>>> print(f"Bloch: ({x:.3f}, {y:.3f}, {z:.3f})")
"""


# ============================================================================
# Version check
# ============================================================================

def info() -> dict:
    """
    Get information about the virtual lab module
    
    Returns:
        dict: Module information
    """
    return {
        'module': 'psiqit.lab',
        'version': __version__,
        'submodules': ['virtual_lab'],
        'exports': len(__all__),
    }


def test() -> dict:
    """
    Run basic tests for the virtual lab module
    
    Returns:
        dict: Test results
    """
    results = {}
    
    # Test QuantumLab
    try:
        from .virtual_lab import QuantumLab
        lab = QuantumLab()
        results['QuantumLab'] = len(lab.experiments) == 0
    except Exception as e:
        results['QuantumLab'] = str(e)
    
    # Test Experiment creation
    try:
        from .virtual_lab import Experiment
        exp = Experiment("Test", n_qubits=2)
        results['Experiment'] = exp.n_qubits == 2
    except Exception as e:
        results['Experiment'] = str(e)
    
    # Test PredefinedExperiments
    try:
        from .virtual_lab import PredefinedExperiments
        exp = PredefinedExperiments.bell_state()
        results['PredefinedExperiments'] = len(exp.gates) == 2
    except Exception as e:
        results['PredefinedExperiments'] = str(e)
    
    # Test running experiment
    try:
        from .virtual_lab import QuantumLab, PredefinedExperiments
        lab = QuantumLab()
        exp = PredefinedExperiments.bell_state()
        result = lab.run_experiment(exp, shots=100)
        results['run_experiment'] = result.status.value == 'completed'
    except Exception as e:
        results['run_experiment'] = str(e)
    
    return results


# ============================================================================
# Clean up namespace (remove helper functions)
# ============================================================================

del _lazy_import