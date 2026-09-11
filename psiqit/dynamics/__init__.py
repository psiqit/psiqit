# psiqit/dynamics/__init__.py

"""
PSIQIT - Python Scientific Quantum Information Toolkit
Dynamics Module

This module provides quantum dynamics and time evolution capabilities.

Submodules:
-----------
adiabatic.py      - Adiabatic evolution and quantum annealing
heisenberg.py     - Heisenberg picture evolution
interaction.py    - Interaction picture dynamics
lindblad.py       - Lindblad master equation for open quantum systems
monte_carlo.py    - Quantum trajectories (Monte Carlo wavefunction method)
schrodinger.py    - Schrödinger equation (time-independent and time-dependent)
time_evolution.py - Numerical time evolution methods (Trotter, Chebyshev, Krylov)
"""

# ============================================================================
# Version and metadata
# ============================================================================

from ..version import __version__

__all__ = [
    # Module information
    '__version__',
    
    # ========================================================================
    # ADIABATIC (from adiabatic.py)
    # ========================================================================
    'AdiabaticResult',
    'AdiabaticEvolution',
    'QuantumAnnealing',
    'ising_hamiltonian',
    'maxcut_hamiltonian',
    'qubo_to_hamiltonian',
    
    # ========================================================================
    # HEISENBERG (from heisenberg.py)
    # ========================================================================
    'HeisenbergResult',
    'HeisenbergEvolution',
    'HeisenbergEquation',
    'harmonic_oscillator_hamiltonian',
    'spin_hamiltonian',
    'two_level_hamiltonian',
    
    # ========================================================================
    # INTERACTION (from interaction.py)
    # ========================================================================
    'InteractionResult',
    'InteractionPicture',
    'create_interaction_hamiltonian',
    'rotating_frame',
    
    # ========================================================================
    # LINDBLAD (from lindblad.py)
    # ========================================================================
    'LindbladResult',
    'LindbladSolver',
    'thermal_state',
    'dephasing_channel',
    'amplitude_damping_channel',
    
    # ========================================================================
    # MONTE CARLO (from monte_carlo.py)
    # ========================================================================
    'MonteCarloResult',
    'QuantumTrajectory',
    'jump_operator',
    'photon_loss_channel',
    'dephasing_operator',
    'amplitude_damping_operator',
    
    # ========================================================================
    # SCHRODINGER (from schrodinger.py)
    # ========================================================================
    'WaveFunction',
    'solve_time_independent',
    'solve_time_dependent',
    'time_evolution_operator',
    'propagate_state',
    'expectation_value',
    'uncertainty_relation',
    'gaussian_wavepacket',
    'plane_wave',
    'square_well_ground_state',
    'harmonic_oscillator_state',
    'infinite_well_state',
    
    # ========================================================================
    # TIME EVOLUTION (from time_evolution.py)
    # ========================================================================
    'EvolutionResult',
    'TrotterEvolution',
    'ChebyshevEvolution',
    'KrylovEvolution',
]

# ============================================================================
# Lazy imports to avoid loading everything at once
# ============================================================================

def _lazy_import(module_name: str, attr_name: str):
    """Helper for lazy imports"""
    import importlib
    module = importlib.import_module(f'.{module_name}', package='psiqit.dynamics')
    return getattr(module, attr_name)


# ============================================================================
# Import all functions (with fallback for missing modules)
# ============================================================================

# Try importing from each module
try:
    from .adiabatic import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import adiabatic module: {e}")

try:
    from .heisenberg import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import heisenberg module: {e}")

try:
    from .interaction import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import interaction module: {e}")

try:
    from .lindblad import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import lindblad module: {e}")

try:
    from .monte_carlo import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import monte_carlo module: {e}")

try:
    from .schrodinger import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import schrodinger module: {e}")

try:
    from .time_evolution import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import time_evolution module: {e}")


# ============================================================================
# Module information
# ============================================================================

__doc__ = """
PSIQIT Dynamics Module
======================

This module provides comprehensive quantum dynamics capabilities:

Submodules:
-----------
adiabatic.py      - Adiabatic evolution and quantum annealing
heisenberg.py     - Heisenberg picture evolution
interaction.py    - Interaction picture dynamics
lindblad.py       - Lindblad master equation for open quantum systems
monte_carlo.py    - Quantum trajectories (Monte Carlo wavefunction method)
schrodinger.py    - Schrödinger equation (time-independent and time-dependent)
time_evolution.py - Numerical time evolution methods (Trotter, Chebyshev, Krylov)

Quick Start:
------------
>>> from psiqit.dynamics import (
...     WaveFunction,
...     solve_time_independent,
...     solve_time_dependent,
...     LindbladSolver,
...     QuantumTrajectory,
...     AdiabaticEvolution,
...     TrotterEvolution
... )
>>> import numpy as np

# Wavefunction
>>> x = np.linspace(-5, 5, 100)
>>> psi = np.exp(-x**2/2) / np.sqrt(np.sqrt(np.pi))
>>> wf = WaveFunction(x, psi)
>>> print(wf.expectation_x())  # 0.0

# Time-independent Schrödinger
>>> def harmonic(x): return 0.5 * x**2
>>> energies, states = solve_time_independent(harmonic, (-5, 5), n_states=5)

# Lindblad master equation
>>> from psiqit.quantum import pauli_z, pauli_x
>>> H = pauli_z()
>>> L = pauli_x()
>>> solver = LindbladSolver(H, [L], gamma=[0.1])
>>> rho0 = np.array([[1, 0], [0, 0]])
>>> result = solver.evolve(rho0, t_max=10.0, dt=0.01)

# Adiabatic evolution
>>> H_i = pauli_x()
>>> H_f = pauli_z()
>>> evo = AdiabaticEvolution(H_i, H_f, T=10.0, n_steps=100)
>>> from psiqit.quantum import zero
>>> result = evo.evolve(zero())

# Trotter evolution
>>> H = pauli_x() + pauli_z()
>>> trotter = TrotterEvolution(H, n_steps=100)
>>> state = trotter.evolve(zero(), t=1.0)
"""


# ============================================================================
# Version check
# ============================================================================

def info() -> dict:
    """
    Get information about the dynamics module
    
    Returns:
        dict: Module information
    """
    return {
        'module': 'psiqit.dynamics',
        'version': __version__,
        'submodules': [
            'adiabatic',
            'heisenberg',
            'interaction',
            'lindblad',
            'monte_carlo',
            'schrodinger',
            'time_evolution'
        ],
        'exports': len(__all__),
    }


def test() -> dict:
    """
    Run basic tests for the dynamics module
    
    Returns:
        dict: Test results
    """
    results = {}
    
    # Test adiabatic
    try:
        from .adiabatic import AdiabaticEvolution
        from ..quantum.operator import pauli_x, pauli_z
        H_i = pauli_x()
        H_f = pauli_z()
        evo = AdiabaticEvolution(H_i, H_f, T=1.0, n_steps=10)
        results['adiabatic'] = evo.dim == 2
    except Exception as e:
        results['adiabatic'] = str(e)
    
    # Test heisenberg
    try:
        from .heisenberg import HeisenbergEvolution
        from ..quantum.operator import pauli_x, pauli_z
        H = pauli_z()
        evo = HeisenbergEvolution(H)
        results['heisenberg'] = evo.dim == 2
    except Exception as e:
        results['heisenberg'] = str(e)
    
    # Test interaction
    try:
        from .interaction import InteractionPicture
        from ..quantum.operator import pauli_x, pauli_z
        H0 = pauli_z()
        V = pauli_x()
        ip = InteractionPicture(H0, V)
        results['interaction'] = ip.dim == 2
    except Exception as e:
        results['interaction'] = str(e)
    
    # Test lindblad
    try:
        from .lindblad import LindbladSolver
        from ..quantum.operator import pauli_x, pauli_z
        H = pauli_z()
        L = pauli_x()
        solver = LindbladSolver(H, [L])
        results['lindblad'] = solver.dim == 2
    except Exception as e:
        results['lindblad'] = str(e)
    
    # Test monte_carlo
    try:
        from .monte_carlo import QuantumTrajectory
        from ..quantum.operator import pauli_x, pauli_z
        H = pauli_z()
        L = pauli_x()
        qt = QuantumTrajectory(H, [L])
        results['monte_carlo'] = qt.dim == 2
    except Exception as e:
        results['monte_carlo'] = str(e)
    
    # Test schrodinger
    try:
        from .schrodinger import WaveFunction
        import numpy as np
        x = np.linspace(0, 1, 10)
        psi = np.exp(-x**2/2)
        wf = WaveFunction(x, psi)
        results['schrodinger'] = len(wf.x) == 10
    except Exception as e:
        results['schrodinger'] = str(e)
    
    # Test time_evolution
    try:
        from .time_evolution import TrotterEvolution
        from ..quantum.operator import pauli_x, pauli_z
        H = pauli_x() + pauli_z()
        trotter = TrotterEvolution(H, n_steps=10)
        results['time_evolution'] = trotter.dim == 2
    except Exception as e:
        results['time_evolution'] = str(e)
    
    return results


# ============================================================================
# Clean up namespace (remove helper functions)
# ============================================================================

del _lazy_import