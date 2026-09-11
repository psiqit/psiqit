# psiqit/variational/__init__.py

"""
PSIQIT - Python Scientific Quantum Information Toolkit
Variational Module

This module provides variational quantum algorithms and methods.

Submodules:
-----------
vqe.py                - Variational Quantum Eigensolver (VQE)
qaoa.py               - Quantum Approximate Optimization Algorithm (QAOA)
variational_advanced.py - Advanced variational methods (SSVQE, ADAPT-VQE, VQD)
variational_methods.py - General variational methods (Rayleigh-Ritz, TDVP, VMC, Hartree-Fock)
"""

# ============================================================================
# Version and metadata
# ============================================================================

from ..version import __version__

__all__ = [
    # Module information
    '__version__',
    
    # ========================================================================
    # VQE (from vqe.py)
    # ========================================================================
    'VQEResult',
    'VQE',
    
    # ========================================================================
    # QAOA (from qaoa.py)
    # ========================================================================
    'QAOResult',
    'QAOA',
    'maxcut_hamiltonian',
    'qubo_to_hamiltonian',
    'maxcut_from_graph',
    
    # ========================================================================
    # VARIATIONAL ADVANCED (from variational_advanced.py)
    # ========================================================================
    'MultiStateResult',
    'SSVQE',
    'ADAPTVQE',
    'VariationalQuantumDeflation',
    
    # ========================================================================
    # VARIATIONAL METHODS (from variational_methods.py)
    # ========================================================================
    'VariationalResult',
    'RayleighRitz',
    'TDVP',
    'VariationalMonteCarlo',
    'HartreeFock',
]

# ============================================================================
# Lazy imports to avoid loading everything at once
# ============================================================================

def _lazy_import(module_name: str, attr_name: str):
    """Helper for lazy imports"""
    import importlib
    module = importlib.import_module(f'.{module_name}', package='psiqit.variational')
    return getattr(module, attr_name)


# ============================================================================
# Import all functions (with fallback for missing modules)
# ============================================================================

# Try importing from each module
try:
    from .vqe import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import vqe module: {e}")

try:
    from .qaoa import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import qaoa module: {e}")

try:
    from .variational_advanced import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import variational_advanced module: {e}")

try:
    from .variational_methods import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import variational_methods module: {e}")


# ============================================================================
# Module information
# ============================================================================

__doc__ = """
PSIQIT Variational Module
=========================

This module provides variational quantum algorithms and methods:

Submodules:
-----------
vqe.py                - Variational Quantum Eigensolver (VQE)
qaoa.py               - Quantum Approximate Optimization Algorithm (QAOA)
variational_advanced.py - Advanced variational methods (SSVQE, ADAPT-VQE, VQD)
variational_methods.py - General variational methods (Rayleigh-Ritz, TDVP, VMC, Hartree-Fock)

Quick Start:
------------
>>> from psiqit.variational import VQE, QAOA, SSVQE, VariationalMonteCarlo

# VQE: Find ground state energy
>>> hamiltonian = {'Z0Z1': 1.0, 'Z0': 0.5, 'Z1': -0.5}
>>> vqe = VQE(n_qubits=2, hamiltonian=hamiltonian, n_layers=2)
>>> result = vqe.run(n_iterations=100)
>>> print(result.optimal_energy)

# QAOA: Solve MaxCut problem
>>> edges = [(0, 1), (1, 2), (0, 2)]
>>> H = maxcut_hamiltonian(edges)
>>> qaoa = QAOA(n_qubits=3, hamiltonian=H, p=2)
>>> result = qaoa.run(n_iterations=100)
>>> print(result.optimal_energy)

# SSVQE: Find multiple excited states
>>> ssvqe = SSVQE(n_qubits=2, hamiltonian=hamiltonian, n_states=3)
>>> result = ssvqe.run(n_iterations=100)
>>> print(result.energies)

# Variational Monte Carlo
>>> def wavefunction(x, params):
...     return np.exp(-params[0] * x**2)
>>> def potential(x):
...     return 0.5 * x**2
>>> vmc = VariationalMonteCarlo(wavefunction, potential, n_params=1)
>>> result = vmc.optimize(n_samples=1000, n_iterations=50)
>>> print(result.optimal_params)
"""


# ============================================================================
# Version check
# ============================================================================

def info() -> dict:
    """
    Get information about the variational module
    
    Returns:
        dict: Module information
    """
    return {
        'module': 'psiqit.variational',
        'version': __version__,
        'submodules': ['vqe', 'qaoa', 'variational_advanced', 'variational_methods'],
        'exports': len(__all__),
    }


def test() -> dict:
    """
    Run basic tests for the variational module
    
    Returns:
        dict: Test results
    """
    results = {}
    
    # Test VQE
    try:
        from .vqe import VQE
        hamiltonian = {'Z0Z1': 1.0}
        vqe = VQE(n_qubits=2, hamiltonian=hamiltonian, n_layers=1)
        results['VQE'] = vqe.n_qubits == 2
    except Exception as e:
        results['VQE'] = str(e)
    
    # Test QAOA
    try:
        from .qaoa import QAOA, maxcut_hamiltonian
        edges = [(0, 1)]
        H = maxcut_hamiltonian(edges)
        qaoa = QAOA(n_qubits=2, hamiltonian=H, p=1)
        results['QAOA'] = qaoa.n_qubits == 2
    except Exception as e:
        results['QAOA'] = str(e)
    
    # Test SSVQE
    try:
        from .variational_advanced import SSVQE
        hamiltonian = {'Z0Z1': 1.0}
        ssvqe = SSVQE(n_qubits=2, hamiltonian=hamiltonian, n_states=2)
        results['SSVQE'] = ssvqe.n_qubits == 2
    except Exception as e:
        results['SSVQE'] = str(e)
    
    # Test VariationalMonteCarlo
    try:
        from .variational_methods import VariationalMonteCarlo
        def wf(x, p): return np.exp(-p[0] * x**2)
        def pot(x): return 0.5 * x**2
        vmc = VariationalMonteCarlo(wf, pot, n_params=1)
        results['VariationalMonteCarlo'] = vmc.n_params == 1
    except Exception as e:
        results['VariationalMonteCarlo'] = str(e)
    
    return results


# ============================================================================
# Clean up namespace (remove helper functions)
# ============================================================================

del _lazy_import