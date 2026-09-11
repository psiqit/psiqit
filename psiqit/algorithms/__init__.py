# psiqit/algorithms/__init__.py

"""
PSIQIT - Python Scientific Quantum Information Toolkit
Algorithms Module

This module provides quantum algorithms for various computational tasks.

Submodules:
-----------
grover.py           - Grover's search algorithm (quadratic speedup for search)
deutsch_jozsa.py    - Deutsch-Jozsa algorithm (determine if function is constant or balanced)
bernstein_vazirani.py - Bernstein-Vazirani algorithm (find hidden string)
simon.py            - Simon's algorithm (find period of a function)
qft.py              - Quantum Fourier Transform (key component for many algorithms)
qpe.py              - Quantum Phase Estimation (estimate eigenvalues of unitary operators)
shor.py             - Shor's factoring algorithm (exponential speedup for factoring)
"""

# ============================================================================
# Version and metadata
# ============================================================================

from ..version import __version__

__all__ = [
    # Module information
    '__version__',
    
    # ========================================================================
    # GROVER (from grover.py)
    # ========================================================================
    'grover_search',
    'grover_search_simulated',
    'grover_with_custom_oracle',
    'optimal_grover_iterations',
    'grover_probability',
    'grover_amplitude',
    
    # ========================================================================
    # DEUTSCH-JOZSA (from deutsch_jozsa.py)
    # ========================================================================
    'deutsch_jozsa',
    'deutsch_algorithm',
    'create_balanced_oracle',
    'create_constant_oracle',
    'deutsch_jozsa_simulated',
    'is_constant_function',
    'is_balanced_function',
    'analyze_function_classically',
    
    # ========================================================================
    # BERNSTEIN-VAZIRANI (from bernstein_vazirani.py)
    # ========================================================================
    'bernstein_vazirani',
    'bernstein_vazirani_circuit',
    'bernstein_vazirani_simulated',
    'bernstein_vazirani_classical',
    'generate_random_hidden_string',
    'create_oracle_from_hidden_string',
    
    # ========================================================================
    # SIMON (from simon.py)
    # ========================================================================
    'simon_algorithm',
    'simon_classical',
    'create_simon_oracle',
    'is_periodic',
    
    # ========================================================================
    # QUANTUM FOURIER TRANSFORM (from qft.py)
    # ========================================================================
    'qft',
    'qft_circuit',
    'iqft',
    'qft_matrix',
    'iqft_matrix',
    'qft_controlled',
    'qft_basis_state',
    'qft_product_state',
    'compare_qft_to_fft',
    'qft_circuit_comparison',
    'qft_phase_estimation_prep',
    'qft_phase_estimation_measure',
    
    # ========================================================================
    # QUANTUM PHASE ESTIMATION (from qpe.py)
    # ========================================================================
    'quantum_phase_estimation',
    'qpe_circuit',
    'qpe_pauli_z',
    'qpe_rotation_gate',
    'phase_to_angle',
    'angle_to_phase',
    'phase_to_eigenvalue',
    'eigenvalue_to_phase',
    'qpe_accuracy',
    
    # ========================================================================
    # SHOR (from shor.py)
    # ========================================================================
    'shor_factor',
    'shor_classical',
    'shor_algorithm',
    'shor_steps',
]

# ============================================================================
# Lazy imports to avoid loading everything at once
# ============================================================================

def _lazy_import(module_name: str, attr_name: str):
    """Helper for lazy imports"""
    import importlib
    module = importlib.import_module(f'.{module_name}', package='psiqit.algorithms')
    return getattr(module, attr_name)


# ============================================================================
# Import all functions (with fallback for missing modules)
# ============================================================================

# Try importing from each module
try:
    from .grover import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import grover module: {e}")

try:
    from .deutsch_jozsa import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import deutsch_jozsa module: {e}")

try:
    from .bernstein_vazirani import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import bernstein_vazirani module: {e}")

try:
    from .simon import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import simon module: {e}")

try:
    from .qft import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import qft module: {e}")

try:
    from .qpe import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import qpe module: {e}")

try:
    from .shor import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import shor module: {e}")


# ============================================================================
# Module information
# ============================================================================

__doc__ = """
PSIQIT Algorithms Module
========================

This module provides quantum algorithms for various computational tasks:

Submodules:
-----------
grover.py           - Grover's search algorithm (quadratic speedup for search)
deutsch_jozsa.py    - Deutsch-Jozsa algorithm (determine if function is constant or balanced)
bernstein_vazirani.py - Bernstein-Vazirani algorithm (find hidden string)
simon.py            - Simon's algorithm (find period of a function)
qft.py              - Quantum Fourier Transform (key component for many algorithms)
qpe.py              - Quantum Phase Estimation (estimate eigenvalues of unitary operators)
shor.py             - Shor's factoring algorithm (exponential speedup for factoring)

Quick Start:
------------
>>> from psiqit.algorithms import (
...     grover_search,
...     deutsch_jozsa,
...     bernstein_vazirani,
...     simon_algorithm,
...     qft,
...     quantum_phase_estimation,
...     shor_factor
... )

# Grover's search
>>> result = grover_search(n_qubits=3, target=5, shots=1024)
>>> print(result['most_likely'])  # 5

# Deutsch-Jozsa
>>> result = deutsch_jozsa(function_type='balanced', n_qubits=3)
>>> print(result['result'])  # 'balanced'

# Bernstein-Vazirani
>>> result = bernstein_vazirani(hidden_string='101')
>>> print(result['found_string'])  # '101'

# Simon's algorithm
>>> from psiqit.algorithms import create_simon_oracle
>>> oracle = create_simon_oracle('101')
>>> result = simon_algorithm(oracle_function=oracle, n_qubits=3)
>>> print(result['found_string'])  # '101'

# Quantum Fourier Transform
>>> from psiqit.quantum import Ket
>>> state = Ket([1, 0, 0, 0])
>>> qft_state = qft(state)
>>> print(qft_state)

# Quantum Phase Estimation
>>> from psiqit.quantum import pauli_z, zero
>>> result = quantum_phase_estimation(pauli_z(), n_qubits=3, state=zero())
>>> print(result['phase'])  # 0.0

# Shor's factoring
>>> factors = shor_factor(15)
>>> print(factors)  # (3, 5)
"""


# ============================================================================
# Version check
# ============================================================================

def info() -> dict:
    """
    Get information about the algorithms module
    
    Returns:
        dict: Module information
    """
    return {
        'module': 'psiqit.algorithms',
        'version': __version__,
        'submodules': [
            'grover',
            'deutsch_jozsa',
            'bernstein_vazirani',
            'simon',
            'qft',
            'qpe',
            'shor'
        ],
        'exports': len(__all__),
    }


def test() -> dict:
    """
    Run basic tests for the algorithms module
    
    Returns:
        dict: Test results
    """
    results = {}
    
    # Test grover
    try:
        from .grover import grover_search
        result = grover_search(n_qubits=3, target=5, shots=100)
        results['grover'] = 'most_likely' in result
    except Exception as e:
        results['grover'] = str(e)
    
    # Test deutsch_jozsa
    try:
        from .deutsch_jozsa import deutsch_jozsa
        result = deutsch_jozsa(function_type='constant_zero', n_qubits=3, shots=100)
        results['deutsch_jozsa'] = result['result'] == 'constant'
    except Exception as e:
        results['deutsch_jozsa'] = str(e)
    
    # Test bernstein_vazirani
    try:
        from .bernstein_vazirani import bernstein_vazirani
        result = bernstein_vazirani(hidden_string='101', shots=100)
        results['bernstein_vazirani'] = result['found_string'] == '101'
    except Exception as e:
        results['bernstein_vazirani'] = str(e)
    
    # Test simon
    try:
        from .simon import simon_algorithm, create_simon_oracle
        oracle = create_simon_oracle('101')
        result = simon_algorithm(oracle_function=oracle, n_qubits=3, shots=100, max_attempts=5)
        results['simon'] = result['found_string'] == '101'
    except Exception as e:
        results['simon'] = str(e)
    
    # Test qft
    try:
        from .qft import qft
        from ..quantum.state import Ket
        state = Ket([1, 0, 0, 0])
        result = qft(state)
        results['qft'] = result.dim == 4
    except Exception as e:
        results['qft'] = str(e)
    
    # Test qpe
    try:
        from .qpe import quantum_phase_estimation
        from ..quantum.operator import pauli_z
        from ..quantum.state import zero
        result = quantum_phase_estimation(pauli_z(), n_qubits=3, state=zero(), shots=100)
        results['qpe'] = 'phase' in result
    except Exception as e:
        results['qpe'] = str(e)
    
    # Test shor
    try:
        from .shor import shor_factor
        p, q = shor_factor(15, use_quantum=False, max_attempts=5)
        results['shor'] = p * q == 15
    except Exception as e:
        results['shor'] = str(e)
    
    return results


# ============================================================================
# Clean up namespace (remove helper functions)
# ============================================================================

del _lazy_import