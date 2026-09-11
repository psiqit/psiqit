# psiqit/qml/__init__.py

"""
PSIQIT - Python Scientific Quantum Information Toolkit
Quantum Machine Learning Module

This module provides quantum machine learning algorithms and models.

Submodules:
-----------
qgan.py           - Quantum Generative Adversarial Networks
qnn.py            - Quantum Neural Networks
qsvm.py           - Quantum Support Vector Machines
vqc.py            - Variational Quantum Classifiers
quantum_kernel.py - Quantum Kernel Methods
"""

# ============================================================================
# Version and metadata
# ============================================================================

from ..version import __version__

__all__ = [
    # Module information
    '__version__',
    
    # ========================================================================
    # QGAN (from qgan.py)
    # ========================================================================
    'QGANResult',
    'QuantumGenerator',
    'QuantumDiscriminator',
    'ClassicalDiscriminator',
    'QGAN',
    
    # ========================================================================
    # QNN (from qnn.py)
    # ========================================================================
    'QNNResult',
    'QNNLayer',
    'QNN',
    'QuantumClassifier',
    'VariationalQuantumCircuit',
    
    # ========================================================================
    # QSVM (from qsvm.py)
    # ========================================================================
    'SVMResult',
    'QSVM',
    
    # ========================================================================
    # VQC (from vqc.py)
    # ========================================================================
    'VQCResult',
    'VariationalLayer',
    'VQC',
    'HamiltonianVQE',
    
    # ========================================================================
    # QUANTUM KERNEL (from quantum_kernel.py)
    # ========================================================================
    'KernelResult',
    'QuantumKernel',
    'QuantumKernelEstimator',
]

# ============================================================================
# Lazy imports to avoid loading everything at once
# ============================================================================

def _lazy_import(module_name: str, attr_name: str):
    """Helper for lazy imports"""
    import importlib
    module = importlib.import_module(f'.{module_name}', package='psiqit.qml')
    return getattr(module, attr_name)


# ============================================================================
# Import all functions (with fallback for missing modules)
# ============================================================================

# Try importing from each module
try:
    from .qgan import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import qgan module: {e}")

try:
    from .qnn import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import qnn module: {e}")

try:
    from .qsvm import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import qsvm module: {e}")

try:
    from .vqc import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import vqc module: {e}")

try:
    from .quantum_kernel import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import quantum_kernel module: {e}")


# ============================================================================
# Module information
# ============================================================================

__doc__ = """
PSIQIT Quantum Machine Learning Module
=======================================

This module provides quantum machine learning algorithms and models:

Submodules:
-----------
qgan.py           - Quantum Generative Adversarial Networks
qnn.py            - Quantum Neural Networks
qsvm.py           - Quantum Support Vector Machines
vqc.py            - Variational Quantum Classifiers
quantum_kernel.py - Quantum Kernel Methods

Quick Start:
------------
>>> from psiqit.qml import (
...     QGAN,
...     QNN,
...     QSVM,
...     VQC,
...     HamiltonianVQE,
...     QuantumKernel
... )

# Quantum Neural Network
>>> qnn = QNN(n_qubits=2, n_layers=3)
>>> result = qnn.train(X_train, y_train, epochs=50)
>>> print(result.accuracy)

# Quantum Support Vector Machine
>>> qsvm = QSVM(n_qubits=2, kernel_type='quantum')
>>> result = qsvm.fit(X_train, y_train)
>>> print(result.accuracy)

# Variational Quantum Circuit
>>> vqc = VQC(n_qubits=2, n_layers=3)
>>> result = vqc.optimize(n_iterations=50)

# Hamiltonian VQE
>>> hamiltonian = {'Z0Z1': 1.0, 'Z0': 0.5}
>>> vqe = HamiltonianVQE(n_qubits=2, hamiltonian=hamiltonian)
>>> result = vqe.run(n_iterations=50)
>>> print(result.optimal_cost)

# Quantum Kernel
>>> kernel = QuantumKernel(n_qubits=2, feature_map='zz')
>>> K = kernel.kernel_matrix(X)
>>> print(K.shape)

# QGAN
>>> from psiqit.quantum import bell_phi_plus
>>> target = [bell_phi_plus()]
>>> qgan = QGAN(n_qubits=2, n_latent=2, n_layers=2)
>>> result = qgan.train(target, epochs=50)
>>> print(result.final_fidelity)
"""


# ============================================================================
# Version check
# ============================================================================

def info() -> dict:
    """
    Get information about the quantum machine learning module
    
    Returns:
        dict: Module information
    """
    return {
        'module': 'psiqit.qml',
        'version': __version__,
        'submodules': ['qgan', 'qnn', 'qsvm', 'vqc', 'quantum_kernel'],
        'exports': len(__all__),
    }


def test() -> dict:
    """
    Run basic tests for the quantum machine learning module
    
    Returns:
        dict: Test results
    """
    results = {}
    
    # Test QGAN
    try:
        from .qgan import QGAN, QuantumGenerator, QuantumDiscriminator
        gen = QuantumGenerator(n_qubits=2, n_latent=2, n_layers=2)
        results['QuantumGenerator'] = gen.n_qubits == 2
    except Exception as e:
        results['QuantumGenerator'] = str(e)
    
    # Test QNN
    try:
        from .qnn import QNN
        qnn = QNN(n_qubits=2, n_layers=3)
        results['QNN'] = qnn.n_qubits == 2
    except Exception as e:
        results['QNN'] = str(e)
    
    # Test QSVM
    try:
        from .qsvm import QSVM
        qsvm = QSVM(n_qubits=2, kernel_type='quantum')
        results['QSVM'] = qsvm.n_qubits == 2
    except Exception as e:
        results['QSVM'] = str(e)
    
    # Test VQC
    try:
        from .vqc import VQC, HamiltonianVQE
        vqc = VQC(n_qubits=2, n_layers=3)
        results['VQC'] = vqc.n_qubits == 2
    except Exception as e:
        results['VQC'] = str(e)
    
    # Test QuantumKernel
    try:
        from .quantum_kernel import QuantumKernel
        kernel = QuantumKernel(n_qubits=2, feature_map='zz')
        results['QuantumKernel'] = kernel.n_qubits == 2
    except Exception as e:
        results['QuantumKernel'] = str(e)
    
    return results


# ============================================================================
# Clean up namespace (remove helper functions)
# ============================================================================

del _lazy_import