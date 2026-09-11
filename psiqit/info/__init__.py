# psiqit/info/__init__.py
"""
PSIQIT - Python Scientific Quantum Information Toolkit
Information Theory Module

This module provides quantum information theory tools including
entropy measures and entanglement measures.

Submodules:
-----------
entropy.py      - Entropy measures (Shannon, von Neumann, Rényi, etc.)
entanglement.py - Entanglement measures (concurrence, negativity, etc.)
"""

# ============================================================================
# Version and metadata
# ============================================================================

from ..version import __version__

__all__ = [
    # Module information
    '__version__',
    
    # ========================================================================
    # ENTROPY (from entropy.py)
    # ========================================================================
    'shannon_entropy',
    'von_neumann_entropy',
    'renyi_entropy',
    'collision_entropy',
    'partial_trace',
    'mutual_information',
    'relative_entropy',
    'purity',
    'linear_entropy',
    'min_entropy',
    'max_entropy',
    
    # ========================================================================
    # ENTANGLEMENT (from entanglement.py)
    # ========================================================================
    'EntanglementResult',
    'concurrence_pure',
    'concurrence_mixed',
    'concurrence',
    'negativity',
    'logarithmic_negativity',
    'entanglement_entropy',
    'schmidt_decomposition',
    'schmidt_rank',
    'is_entangled',
    'entanglement_of_formation',
    'distillable_entanglement',
]

# ============================================================================
# Lazy imports to avoid loading everything at once
# ============================================================================

def _lazy_import(module_name: str, attr_name: str):
    """Helper for lazy imports"""
    import importlib
    module = importlib.import_module(f'.{module_name}', package='psiqit.info')
    return getattr(module, attr_name)


# ============================================================================
# Import all functions (with fallback for missing modules)
# ============================================================================

# Try importing from each module
try:
    from .entropy import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import entropy module: {e}")

try:
    from .entanglement import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import entanglement module: {e}")


# ============================================================================
# Module information
# ============================================================================

__doc__ = """
PSIQIT Information Theory Module
================================

This module provides quantum information theory tools:

Submodules:
-----------
entropy.py      - Entropy measures (Shannon, von Neumann, Rényi, etc.)
entanglement.py - Entanglement measures (concurrence, negativity, etc.)

Quick Start:
------------
>>> from psiqit.info import (
...     shannon_entropy,
...     von_neumann_entropy,
...     concurrence,
...     negativity,
...     is_entangled
... )
>>> import numpy as np

# Shannon entropy
>>> probs = [0.5, 0.5]
>>> H = shannon_entropy(probs, base='2')
>>> print(H)  # 1.0 bit

# von Neumann entropy
>>> rho = np.array([[0.5, 0], [0, 0.5]])
>>> S = von_neumann_entropy(rho, base='2')
>>> print(S)  # 1.0 bit

# Concurrence
>>> from psiqit.quantum import bell_phi_plus
>>> C = concurrence(bell_phi_plus())
>>> print(C)  # 1.0

# Negativity
>>> N = negativity(bell_phi_plus())
>>> print(N)  # 0.5

# Entanglement detection
>>> is_ent = is_entangled(bell_phi_plus())
>>> print(is_ent)  # True
"""


# ============================================================================
# Version check
# ============================================================================

def info() -> dict:
    """
    Get information about the information theory module
    
    Returns:
        dict: Module information
    """
    return {
        'module': 'psiqit.info',
        'version': __version__,
        'submodules': ['entropy', 'entanglement'],
        'exports': len(__all__),
    }


def test() -> dict:
    """
    Run basic tests for the information theory module
    
    Returns:
        dict: Test results
    """
    results = {}
    
    # Test entropy
    try:
        from .entropy import shannon_entropy, von_neumann_entropy, purity
        probs = [0.5, 0.5]
        H = shannon_entropy(probs)
        results['shannon_entropy'] = abs(H - 0.693147) < 1e-6
        
        rho = np.array([[0.5, 0], [0, 0.5]])
        S = von_neumann_entropy(rho)
        results['von_neumann_entropy'] = abs(S - 0.693147) < 1e-6
        
        p = purity(rho)
        results['purity'] = abs(p - 0.5) < 1e-6
    except Exception as e:
        results['entropy'] = str(e)
    
    # Test entanglement
    try:
        from .entanglement import concurrence, negativity, is_entangled
        from ..quantum.state import bell_phi_plus, zero
        
        C = concurrence(bell_phi_plus())
        results['concurrence'] = abs(C - 1.0) < 1e-6
        
        N = negativity(bell_phi_plus())
        results['negativity'] = abs(N - 0.5) < 1e-6
        
        is_ent = is_entangled(bell_phi_plus())
        results['is_entangled_bell'] = is_ent
        
        is_ent = is_entangled(zero())
        results['is_entangled_zero'] = not is_ent
    except Exception as e:
        results['entanglement'] = str(e)
    
    return results


# ============================================================================
# Clean up namespace (remove helper functions)
# ============================================================================

del _lazy_import