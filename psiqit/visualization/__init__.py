# psiqit/visualization/__init__.py

"""
PSIQIT - Python Scientific Quantum Information Toolkit
Visualization Module

This module provides visualization tools for quantum states and circuits.

Submodules:
-----------
bloch.py          - Bloch sphere visualization
circuit_drawer.py - Quantum circuit drawing
wigner.py         - Wigner function visualization
husimi.py         - Husimi Q-function visualization
animation.py      - Animation tools
symbolic_plot.py  - Symbolic equation rendering
"""

# ============================================================================
# Version and metadata
# ============================================================================

from ..version import __version__

__all__ = [
    # Module information
    '__version__',
    
    # ========================================================================
    # BLOCH (from bloch.py)
    # ========================================================================
    # Coordinate conversions
    'state_to_bloch',
    'bloch_to_state',
    'spherical_to_bloch',
    'bloch_to_spherical',
    
    # Plotting
    'bloch_sphere',
    'plot_multiple_states',
    'animate_bloch',
    
    # Additional utilities
    'get_bloch_coordinates',
    'bloch_distance',
    
    # ========================================================================
    # CIRCUIT DRAWER (from circuit_drawer.py)
    # ========================================================================
    'draw_circuit',
    '_draw_ascii',
    '_draw_unicode',
    'circuit_to_text',
    'circuit_statistics',
    'draw_circuit_latex',
    'draw_circuit_qiskit',
    
    # ========================================================================
    # WIGNER (from wigner.py)
    # ========================================================================
    # Wigner functions
    'wigner_function',
    'wigner_function_analytic',
    'wigner_function_gaussian',
    'wigner_function_coherent_state',
    'wigner_function_squeezed_state',
    
    # Plotting
    'plot_wigner',
    'plot_wigner_3d',
    
    # Utilities
    'wigner_negativity',
    'wigner_marginals',
    'wigner_variance',
    
    # ========================================================================
    # HUSIMI (from husimi.py)
    # ========================================================================
    # Husimi functions
    'husimi_function_gaussian',
    'husimi_function_coherent_state',
    'husimi_function_fock_state',
    'husimi_function_thermal_state',
    'husimi_function_wavefunction',
    'husimi_function_density_matrix',
    
    # Plotting
    'plot_husimi',
    
    # Utilities
    'husimi_negativity',
    'husimi_entropy',
    'husimi_variance',
    
    # ========================================================================
    # ANIMATION (from animation.py)
    # ========================================================================
    'animate_wavefunction',
    'animate_bloch_sphere',
    'animate_parametric_curve',
    'animate_quantum_circuit',
    
    # ========================================================================
    # SYMBOLIC PLOT (from symbolic_plot.py)
    # ========================================================================
    'render_equation',
    'schrodinger_equation',
    'dirac_notation',
    'pauli_matrices',
    'bloch_sphere_equation',
    'heisenberg_uncertainty',
    'commutation_relation',
    'maxwell_equations',
    'quantum_equations',
    'render_equation_as_image',
]

# ============================================================================
# Lazy imports to avoid loading everything at once
# ============================================================================

def _lazy_import(module_name: str, attr_name: str):
    """Helper for lazy imports"""
    import importlib
    module = importlib.import_module(f'.{module_name}', package='psiqit.visualization')
    return getattr(module, attr_name)


# ============================================================================
# Import all functions (with fallback for missing modules)
# ============================================================================

# Try importing from each module
try:
    from .bloch import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import bloch module: {e}")

try:
    from .circuit_drawer import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import circuit_drawer module: {e}")

try:
    from .wigner import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import wigner module: {e}")

try:
    from .husimi import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import husimi module: {e}")

try:
    from .animation import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import animation module: {e}")

try:
    from .symbolic_plot import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import symbolic_plot module: {e}")


# ============================================================================
# Module information
# ============================================================================

__doc__ = """
PSIQIT Visualization Module
===========================

This module provides visualization tools for quantum states and circuits:

Submodules:
-----------
bloch.py          - Bloch sphere visualization
circuit_drawer.py - Quantum circuit drawing
wigner.py         - Wigner function visualization
husimi.py         - Husimi Q-function visualization
animation.py      - Animation tools
symbolic_plot.py  - Symbolic equation rendering

Quick Start:
------------
>>> from psiqit.visualization import (
...     bloch_sphere,
...     draw_circuit,
...     plot_wigner,
...     animate_bloch_sphere,
...     render_equation
... )
>>> from psiqit.quantum import zero, plus, bell_phi_plus

# Bloch sphere
>>> bloch_sphere(plus(), title="|+⟩ State")

# Multiple states
>>> states = [zero(), plus(), bell_phi_plus()]
>>> labels = ['|0⟩', '|+⟩', '|Φ⁺⟩']
>>> plot_multiple_states(states, labels=labels)

# Circuit drawing
>>> from psiqit.circuits import QuantumCircuit
>>> circ = QuantumCircuit(2)
>>> circ.h(0).cx(0, 1)
>>> print(draw_circuit(circ))

# Wigner function
>>> from psiqit.quantum import coherent_state
>>> alpha = 1.0 + 1j*0.5
>>> state = coherent_state(alpha, n_levels=20)
>>> W, x, p = wigner_function(state, (-3, 3), (-3, 3))
>>> plot_wigner(W, x, p, title="Coherent State")

# Animation
>>> states = [zero(), plus(), minus(), zero()]
>>> animate_bloch_sphere(states, interval=500)

# Symbolic equations
>>> print(render_equation(r"E = mc^2"))
>>> print(schrodinger_equation())
>>> print(heisenberg_uncertainty())
"""


# ============================================================================
# Version check
# ============================================================================

def info() -> dict:
    """
    Get information about the visualization module
    
    Returns:
        dict: Module information
    """
    return {
        'module': 'psiqit.visualization',
        'version': __version__,
        'submodules': ['bloch', 'circuit_drawer', 'wigner', 'husimi', 'animation', 'symbolic_plot'],
        'exports': len(__all__),
    }


def test() -> dict:
    """
    Run basic tests for the visualization module
    
    Returns:
        dict: Test results
    """
    results = {}
    
    # Test bloch
    try:
        from .bloch import state_to_bloch, bloch_sphere
        from ..quantum.state import zero
        x, y, z = state_to_bloch(zero())
        results['bloch'] = abs(z - 1.0) < 1e-10
    except Exception as e:
        results['bloch'] = str(e)
    
    # Test circuit_drawer
    try:
        from .circuit_drawer import draw_circuit, circuit_statistics
        from ..circuits.circuit import QuantumCircuit
        circ = QuantumCircuit(2)
        circ.h(0).cx(0, 1)
        draw = draw_circuit(circ)
        stats = circuit_statistics(circ)
        results['circuit_drawer'] = stats['total_gates'] == 2
    except Exception as e:
        results['circuit_drawer'] = str(e)
    
    # Test wigner
    try:
        from .wigner import wigner_function_gaussian, plot_wigner
        W, x, p = wigner_function_gaussian()
        results['wigner'] = W.shape == (100, 100)
    except Exception as e:
        results['wigner'] = str(e)
    
    # Test husimi
    try:
        from .husimi import husimi_function_gaussian, plot_husimi
        Q, x, p = husimi_function_gaussian()
        results['husimi'] = Q.shape == (100, 100)
    except Exception as e:
        results['husimi'] = str(e)
    
    # Test animation
    try:
        from .animation import animate_bloch_sphere
        # Just check import
        results['animation'] = True
    except Exception as e:
        results['animation'] = str(e)
    
    # Test symbolic_plot
    try:
        from .symbolic_plot import render_equation, schrodinger_equation
        eq = schrodinger_equation()
        results['symbolic_plot'] = 'hbar' in eq
    except Exception as e:
        results['symbolic_plot'] = str(e)
    
    return results


# ============================================================================
# Clean up namespace (remove helper functions)
# ============================================================================

del _lazy_import
