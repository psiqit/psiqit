# psiqit/noise_canceling/__init__.py

"""
PSIQIT - Python Scientific Quantum Information Toolkit
Noise Canceling Module

This module provides quantum noise models and error mitigation techniques.

Submodules:
-----------
noise_models.py - Noise channels and error mitigation
"""

# ============================================================================
# Version and metadata
# ============================================================================

from ..version import __version__

__all__ = [
    # Module information
    '__version__',
    
    # ========================================================================
    # NOISE MODELS (from noise_models.py)
    # ========================================================================
    'NoiseResult',
    'bit_flip_channel',
    'phase_flip_channel',
    'bit_phase_flip_channel',
    'depolarizing_channel',
    'amplitude_damping_channel',
    'phase_damping_channel',
    'readout_error',
    'zero_noise_extrapolation',
    'probabilistic_error_cancellation',
    'compose_noise_channels',
]

# ============================================================================
# Lazy imports to avoid loading everything at once
# ============================================================================

def _lazy_import(module_name: str, attr_name: str):
    """Helper for lazy imports"""
    import importlib
    module = importlib.import_module(f'.{module_name}', package='psiqit.noise_canceling')
    return getattr(module, attr_name)


# ============================================================================
# Import all functions (with fallback for missing modules)
# ============================================================================

try:
    from .noise_models import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import noise_models module: {e}")


# ============================================================================
# Module information
# ============================================================================

__doc__ = """
PSIQIT Noise Canceling Module
==============================

This module provides quantum noise models and error mitigation techniques:

Submodules:
-----------
noise_models.py - Noise channels and error mitigation

Quick Start:
------------
>>> from psiqit.noise_canceling import (
...     bit_flip_channel,
...     depolarizing_channel,
...     zero_noise_extrapolation,
...     NoiseResult
... )
>>> from psiqit.quantum import zero

# Apply noise to a state
>>> state = zero()
>>> result = depolarizing_channel(state, p=0.1)
>>> print(result.fidelity)

# Zero-noise extrapolation
>>> results = []
>>> for p in [0.1, 0.2, 0.3]:
...     result = depolarizing_channel(state, p)
...     results.append(result)
>>> value = zero_noise_extrapolation(results)
>>> print(f"Extrapolated fidelity: {value:.4f}")
"""


# ============================================================================
# Version check
# ============================================================================

def info() -> dict:
    """
    Get information about the noise canceling module
    
    Returns:
        dict: Module information
    """
    return {
        'module': 'psiqit.noise_canceling',
        'version': __version__,
        'submodules': ['noise_models'],
        'exports': len(__all__),
    }


def test() -> dict:
    """
    Run basic tests for the noise canceling module
    
    Returns:
        dict: Test results
    """
    results = {}
    
    try:
        from .noise_models import bit_flip_channel, depolarizing_channel, NoiseResult
        from ..quantum.state import zero
        
        state = zero()
        result = bit_flip_channel(state, p=0.1)
        results['bit_flip'] = result.fidelity > 0
    except Exception as e:
        results['bit_flip'] = str(e)
    
    try:
        from .noise_models import depolarizing_channel
        from ..quantum.state import zero
        
        state = zero()
        result = depolarizing_channel(state, p=0.1)
        results['depolarizing'] = result.fidelity > 0
    except Exception as e:
        results['depolarizing'] = str(e)
    
    try:
        from .noise_models import amplitude_damping_channel
        from ..quantum.state import one
        
        state = one()
        result = amplitude_damping_channel(state, gamma=0.5)
        results['amplitude_damping'] = result.fidelity > 0
    except Exception as e:
        results['amplitude_damping'] = str(e)
    
    return results


# ============================================================================
# Clean up namespace (remove helper functions)
# ============================================================================

del _lazy_import