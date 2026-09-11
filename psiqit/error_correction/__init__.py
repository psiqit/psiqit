"""
PSIQIT - Python Scientific Quantum Information Toolkit
Error Correction Module

This module provides quantum error correction codes and tools.

Submodules:
-----------
codes.py - Quantum error correction codes (Bit-flip, Phase-flip, Shor, Steane)
"""

# ============================================================================
# Version and metadata
# ============================================================================

from ..version import __version__

__all__ = [
    # Module information
    '__version__',
    
    # ========================================================================
    # CODES (from codes.py)
    # ========================================================================
    'CorrectionResult',
    'BitFlipCode',
    'PhaseFlipCode',
    'ShorCode',
    'SteaneCode',
    'detect_error',
]

# ============================================================================
# Lazy imports to avoid loading everything at once
# ============================================================================

def _lazy_import(module_name: str, attr_name: str):
    """Helper for lazy imports"""
    import importlib
    module = importlib.import_module(f'.{module_name}', package='psiqit.error_correction')
    return getattr(module, attr_name)


# ============================================================================
# Import all functions (with fallback for missing modules)
# ============================================================================

# Try importing from each module
try:
    from .codes import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import codes module: {e}")


# ============================================================================
# Module information
# ============================================================================

__doc__ = """
PSIQIT Error Correction Module
==============================

This module provides quantum error correction capabilities:

Submodules:
-----------
codes.py - Quantum error correction codes (Bit-flip, Phase-flip, Shor, Steane)

Quick Start:
------------
>>> from psiqit.error_correction import BitFlipCode, PhaseFlipCode, ShorCode, SteaneCode
>>> from psiqit.quantum import zero, one, plus

# Bit-flip code (3-qubit repetition code)
>>> code = BitFlipCode(n=3)
>>> state = zero()
>>> encoded = code.encode(state)
>>> result = code.decode(encoded)
>>> print(result.success)  # True

# Phase-flip code (3-qubit repetition code in Hadamard basis)
>>> code = PhaseFlipCode(n=3)
>>> state = plus()
>>> encoded = code.encode(state)
>>> result = code.decode(encoded)
>>> print(result.success)  # True

# Shor's 9-qubit code
>>> code = ShorCode()
>>> state = zero()
>>> encoded = code.encode(state)
>>> result = code.decode(encoded)
>>> print(result.success)  # True

# Steane's 7-qubit code
>>> code = SteaneCode()
>>> state = zero()
>>> encoded = code.encode(state)
>>> result = code.decode(encoded)
>>> print(result.success)  # True
"""


# ============================================================================
# Version check
# ============================================================================

def info() -> dict:
    """
    Get information about the error correction module
    
    Returns:
        dict: Module information
    """
    return {
        'module': 'psiqit.error_correction',
        'version': __version__,
        'submodules': ['codes'],
        'exports': len(__all__),
    }


def test() -> dict:
    """
    Run basic tests for the error correction module
    
    Returns:
        dict: Test results
    """
    results = {}
    
    # Test BitFlipCode
    try:
        from .codes import BitFlipCode
        from ..quantum.state import zero
        code = BitFlipCode(n=3)
        state = zero()
        encoded = code.encode(state)
        result = code.decode(encoded)
        results['BitFlipCode'] = result.success
    except Exception as e:
        results['BitFlipCode'] = str(e)
    
    # Test PhaseFlipCode
    try:
        from .codes import PhaseFlipCode
        from ..quantum.state import plus
        code = PhaseFlipCode(n=3)
        state = plus()
        encoded = code.encode(state)
        result = code.decode(encoded)
        results['PhaseFlipCode'] = result.success
    except Exception as e:
        results['PhaseFlipCode'] = str(e)
    
    # Test ShorCode
    try:
        from .codes import ShorCode
        from ..quantum.state import zero
        code = ShorCode()
        state = zero()
        encoded = code.encode(state)
        result = code.decode(encoded)
        results['ShorCode'] = result.success
    except Exception as e:
        results['ShorCode'] = str(e)
    
    # Test SteaneCode
    try:
        from .codes import SteaneCode
        from ..quantum.state import zero
        code = SteaneCode()
        state = zero()
        encoded = code.encode(state)
        result = code.decode(encoded)
        results['SteaneCode'] = result.success
    except Exception as e:
        results['SteaneCode'] = str(e)
    
    # Test detect_error
    try:
        from .codes import detect_error
        from ..circuits.circuit import QuantumCircuit
        circ = QuantumCircuit(5)
        syndrome = detect_error(circ, [3, 4])
        results['detect_error'] = len(syndrome) == 2
    except Exception as e:
        results['detect_error'] = str(e)
    
    return results


# ============================================================================
# Clean up namespace (remove helper functions)
# ============================================================================

del _lazy_import