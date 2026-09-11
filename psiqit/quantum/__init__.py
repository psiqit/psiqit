#psiqit/quantum/__init__.py 
"""
PSIQIT - Python Scientific Quantum Information Toolkit
Quantum Module

This module provides quantum state representations, operators,
measurements, and quantum communication protocols.

Submodules:
-----------
state.py       - Ket, Bra, and quantum states (Bell, GHZ, W, etc.)
operator.py    - Quantum operators and gates (Pauli, Hadamard, CNOT, etc.)
measurement.py - Measurements, POVM, projective measurement, tomography
interference.py - Double-slit and Mach-Zehnder interferometers
parties.py     - Quantum communication protocols (BB84, Teleportation, Superdense Coding)
"""

# ============================================================================
# Version and metadata
# ============================================================================

from ..version import __version__

__all__ = [
    # Module information
    '__version__',
    
    # ========================================================================
    # STATE (from state.py)
    # ========================================================================
    'Ket', 'Bra',
    'ket', 'basis',
    'zero', 'one', 'plus', 'minus', 'ip', 'im',
    'bell_phi_plus', 'bell_phi_minus', 'bell_psi_plus', 'bell_psi_minus', 'bell_state',
    'ghz', 'w_state',
    'random_state',
    'is_orthogonal', 'is_same', 'fidelity',
    'fock_state', 'coherent_state', 'squeezed_state', 'thermal_state',
    'alice_bell_state', 'alice_bob_ghz', 'alice_bob_w', 'ep_pair',
    'phase_state', 'dual_rail_qubit',
    
    # ========================================================================
    # OPERATOR (from operator.py)
    # ========================================================================
    'Operator',
    'identity', 'pauli_x', 'pauli_y', 'pauli_z',
    'hadamard', 'phase', 's_gate', 't_gate',
    'rx', 'ry', 'rz',
    'cnot', 'cz', 'swap',
    'toffoli', 'fredkin',
    'tensor_product', 'expectation', 'projector', 'pauli_string',
    
    # ========================================================================
    # MEASUREMENT (from measurement.py)
    # ========================================================================
    'measure', 'measure_observable',
    'expectation', 'variance', 'standard_deviation',
    'born_rule',
    'POVM', 'povm_z_basis', 'povm_x_basis', 'povm_y_basis',
    'ProjectiveMeasurement',
    'measurement_statistics', 'state_tomography',
    
    # ========================================================================
    # INTERFERENCE (from interference.py)
    # ========================================================================
    'InterferenceResult',
    'DoubleSlit',
    'MachZehnderInterferometer',
    'fringe_visibility', 'optical_path_difference', 'phase_from_optical_path',
    
    # ========================================================================
    # PARTIES (from parties.py)
    # ========================================================================
    'Alice', 'Bob', 'Charlie',
    'BB84', 'SuperdenseCoding', 'QuantumTeleportation',
]

# ============================================================================
# Lazy loading with __getattr__
# ============================================================================

_module_cache = {}


def __getattr__(name):
    """
    Lazy import for modules to avoid circular imports
    
    Args:
        name: Name of the attribute to import
        
    Returns:
        The imported attribute
    """
    if name in _module_cache:
        return _module_cache[name]
    
    # Mapping of attribute names to (module_name, attribute_name)
    attr_map = {
        # State
        'Ket': ('state', 'Ket'),
        'Bra': ('state', 'Bra'),
        'ket': ('state', 'ket'),
        'basis': ('state', 'basis'),
        'zero': ('state', 'zero'),
        'one': ('state', 'one'),
        'plus': ('state', 'plus'),
        'minus': ('state', 'minus'),
        'ip': ('state', 'ip'),
        'im': ('state', 'im'),
        'bell_phi_plus': ('state', 'bell_phi_plus'),
        'bell_phi_minus': ('state', 'bell_phi_minus'),
        'bell_psi_plus': ('state', 'bell_psi_plus'),
        'bell_psi_minus': ('state', 'bell_psi_minus'),
        'bell_state': ('state', 'bell_state'),
        'ghz': ('state', 'ghz'),
        'w_state': ('state', 'w_state'),
        'random_state': ('state', 'random_state'),
        'is_orthogonal': ('state', 'is_orthogonal'),
        'is_same': ('state', 'is_same'),
        'fidelity': ('state', 'fidelity'),
        'fock_state': ('state', 'fock_state'),
        'coherent_state': ('state', 'coherent_state'),
        'squeezed_state': ('state', 'squeezed_state'),
        'thermal_state': ('state', 'thermal_state'),
        'alice_bell_state': ('state', 'alice_bell_state'),
        'alice_bob_ghz': ('state', 'alice_bob_ghz'),
        'alice_bob_w': ('state', 'alice_bob_w'),
        'ep_pair': ('state', 'ep_pair'),
        'phase_state': ('state', 'phase_state'),
        'dual_rail_qubit': ('state', 'dual_rail_qubit'),
        
        # Operator
        'Operator': ('operator', 'Operator'),
        'identity': ('operator', 'identity'),
        'pauli_x': ('operator', 'pauli_x'),
        'pauli_y': ('operator', 'pauli_y'),
        'pauli_z': ('operator', 'pauli_z'),
        'hadamard': ('operator', 'hadamard'),
        'phase': ('operator', 'phase'),
        's_gate': ('operator', 's_gate'),
        't_gate': ('operator', 't_gate'),
        'rx': ('operator', 'rx'),
        'ry': ('operator', 'ry'),
        'rz': ('operator', 'rz'),
        'cnot': ('operator', 'cnot'),
        'cz': ('operator', 'cz'),
        'swap': ('operator', 'swap'),
        'toffoli': ('operator', 'toffoli'),
        'fredkin': ('operator', 'fredkin'),
        'tensor_product': ('operator', 'tensor_product'),
        'expectation': ('operator', 'expectation'),
        'projector': ('operator', 'projector'),
        'pauli_string': ('operator', 'pauli_string'),
        
        # Measurement
        'measure': ('measurement', 'measure'),
        'measure_observable': ('measurement', 'measure_observable'),
        'expectation': ('measurement', 'expectation'),
        'variance': ('measurement', 'variance'),
        'standard_deviation': ('measurement', 'standard_deviation'),
        'born_rule': ('measurement', 'born_rule'),
        'POVM': ('measurement', 'POVM'),
        'povm_z_basis': ('measurement', 'povm_z_basis'),
        'povm_x_basis': ('measurement', 'povm_x_basis'),
        'povm_y_basis': ('measurement', 'povm_y_basis'),
        'ProjectiveMeasurement': ('measurement', 'ProjectiveMeasurement'),
        'measurement_statistics': ('measurement', 'measurement_statistics'),
        'state_tomography': ('measurement', 'state_tomography'),
        
        # Interference
        'InterferenceResult': ('interference', 'InterferenceResult'),
        'DoubleSlit': ('interference', 'DoubleSlit'),
        'MachZehnderInterferometer': ('interference', 'MachZehnderInterferometer'),
        'fringe_visibility': ('interference', 'fringe_visibility'),
        'optical_path_difference': ('interference', 'optical_path_difference'),
        'phase_from_optical_path': ('interference', 'phase_from_optical_path'),
        
        # Parties
        'Alice': ('parties', 'Alice'),
        'Bob': ('parties', 'Bob'),
        'Charlie': ('parties', 'Charlie'),
        'BB84': ('parties', 'BB84'),
        'SuperdenseCoding': ('parties', 'SuperdenseCoding'),
        'QuantumTeleportation': ('parties', 'QuantumTeleportation'),
    }
    
    if name not in attr_map:
        raise AttributeError(f"module 'psiqit.quantum' has no attribute '{name}'")
    
    module_name, attr_name = attr_map[name]
    
    # Import the module
    import importlib
    module = importlib.import_module(f'.{module_name}', package='psiqit.quantum')
    
    # Get the attribute
    value = getattr(module, attr_name)
    
    # Cache it
    _module_cache[name] = value
    
    return value


# ============================================================================
# Module information
# ============================================================================

__doc__ = """
PSIQIT Quantum Module
=====================

This module provides core quantum computing functionality:

Submodules:
-----------
state.py       - Ket, Bra, and quantum states (Bell, GHZ, W, etc.)
operator.py    - Quantum operators and gates (Pauli, Hadamard, CNOT, etc.)
measurement.py - Measurements, POVM, projective measurement, tomography
interference.py - Double-slit and Mach-Zehnder interferometers
parties.py     - Quantum communication protocols (BB84, Teleportation, Superdense Coding)
"""


# ============================================================================
# Version check
# ============================================================================

def info() -> dict:
    """
    Get information about the quantum module
    
    Returns:
        dict: Module information
    """
    return {
        'module': 'psiqit.quantum',
        'version': __version__,
        'submodules': ['state', 'operator', 'measurement', 'interference', 'parties'],
        'exports': len(__all__),
    }


def test() -> dict:
    """
    Run basic tests for the quantum module
    
    Returns:
        dict: Test results
    """
    results = {}
    
    # فقط تست‌های ساده بدون import مستقیم
    results['module'] = "Quantum module loaded"
    
    return results


def __dir__():
    """Return list of available attributes"""
    return __all__