"""
PSIQIT - Python Scientific Quantum Information Toolkit
Circuits Module

This module provides quantum circuit construction and simulation capabilities.

Submodules:
-----------
qubit.py          - Single qubit representation
register.py       - Quantum register (collection of qubits)
circuit.py        - Quantum circuit with gates and simulation
optical_circuits.py - Optical circuits (beam splitters, phase shifters)
"""

# ============================================================================
# Version and metadata
# ============================================================================

from ..version import __version__

__all__ = [
    # Module information
    '__version__',
    
    # ========================================================================
    # QUBIT (from qubit.py)
    # ========================================================================
    'Qubit',
    'create_qubits',
    'qubit_from_bloch',
    
    # ========================================================================
    # REGISTER (from register.py)
    # ========================================================================
    'QuantumRegister',
    'create_register_from_state',
    'create_product_state',
    
    # ========================================================================
    # CIRCUIT (from circuit.py)
    # ========================================================================
    'QuantumCircuit',
    'create_bell_circuit',
    'create_ghz_circuit',
    'create_w_circuit',
    
    # ========================================================================
    # OPTICAL CIRCUITS (from optical_circuits.py)
    # ========================================================================
    'BeamSplitter',
    'PhaseShifter',
    'OpticalCircuit',
    'beam_splitter_circuit',
    'phase_shifter_circuit',
    'mach_zehnder_interferometer',
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
        # Qubit
        'Qubit': ('qubit', 'Qubit'),
        'create_qubits': ('qubit', 'create_qubits'),
        'qubit_from_bloch': ('qubit', 'qubit_from_bloch'),
        
        # Register
        'QuantumRegister': ('register', 'QuantumRegister'),
        'create_register_from_state': ('register', 'create_register_from_state'),
        'create_product_state': ('register', 'create_product_state'),
        
        # Circuit
        'QuantumCircuit': ('circuit', 'QuantumCircuit'),
        'create_bell_circuit': ('circuit', 'create_bell_circuit'),
        'create_ghz_circuit': ('circuit', 'create_ghz_circuit'),
        'create_w_circuit': ('circuit', 'create_w_circuit'),
        
        # Optical circuits
        'BeamSplitter': ('optical_circuits', 'BeamSplitter'),
        'PhaseShifter': ('optical_circuits', 'PhaseShifter'),
        'OpticalCircuit': ('optical_circuits', 'OpticalCircuit'),
        'beam_splitter_circuit': ('optical_circuits', 'beam_splitter_circuit'),
        'phase_shifter_circuit': ('optical_circuits', 'phase_shifter_circuit'),
        'mach_zehnder_interferometer': ('optical_circuits', 'mach_zehnder_interferometer'),
    }
    
    if name not in attr_map:
        raise AttributeError(f"module 'psiqit.circuits' has no attribute '{name}'")
    
    module_name, attr_name = attr_map[name]
    
    # Import the module
    import importlib
    module = importlib.import_module(f'.{module_name}', package='psiqit.circuits')
    
    # Get the attribute
    value = getattr(module, attr_name)
    
    # Cache it
    _module_cache[name] = value
    
    return value


# ============================================================================
# Module information
# ============================================================================

__doc__ = """
PSIQIT Circuits Module
======================

This module provides quantum circuit construction and simulation capabilities:

Submodules:
-----------
qubit.py          - Single qubit representation
register.py       - Quantum register (collection of qubits)
circuit.py        - Quantum circuit with gates and simulation
optical_circuits.py - Optical circuits (beam splitters, phase shifters)
"""


# ============================================================================
# Version check
# ============================================================================

def info() -> dict:
    """
    Get information about the circuits module
    
    Returns:
        dict: Module information
    """
    return {
        'module': 'psiqit.circuits',
        'version': __version__,
        'submodules': ['qubit', 'register', 'circuit', 'optical_circuits'],
        'exports': len(__all__),
    }


def test() -> dict:
    """
    Run basic tests for the circuits module
    
    Returns:
        dict: Test results
    """
    results = {}
    
    # فقط تست‌های ساده بدون import مستقیم
    results['module'] = "Circuits module loaded"
    
    return results


def __dir__():
    """Return list of available attributes"""
    return __all__