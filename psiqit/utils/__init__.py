# psiqit/utils/__init__.py

"""
PSIQIT - Python Scientific Quantum Information Toolkit
Utilities Module

This module provides utility functions and tools for PSIQIT.

Submodules:
-----------
conversion.py   - Convert between different quantum representations
polarization.py - Jones vectors, Jones matrices, and polarization analysis
random.py       - Generate random quantum states, matrices, and operators
validation.py   - Validate quantum states, matrices, and circuits
config.py       - Global configuration settings
logger.py       - Logging utilities
"""

# ============================================================================
# Version and metadata
# ============================================================================

from ..version import __version__

# ============================================================================
# Lazy imports with __getattr__
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
        # Conversion
        'ConversionResult': ('conversion', 'ConversionResult'),
        'ket_to_density': ('conversion', 'ket_to_density'),
        'density_to_ket': ('conversion', 'density_to_ket'),
        'is_pure_state': ('conversion', 'is_pure_state'),
        'change_basis': ('conversion', 'change_basis'),
        'to_computational_basis': ('conversion', 'to_computational_basis'),
        'to_pauli_basis': ('conversion', 'to_pauli_basis'),
        'from_pauli_basis': ('conversion', 'from_pauli_basis'),
        'to_list': ('conversion', 'to_list'),
        'to_operator': ('conversion', 'to_operator'),
        'to_numpy': ('conversion', 'to_numpy'),
        'to_ket': ('conversion', 'to_ket'),
        'to_bra': ('conversion', 'to_bra'),
        'to_matrix': ('conversion', 'to_matrix'),
        'to_bloch_coordinates': ('conversion', 'to_bloch_coordinates'),
        'from_bloch_coordinates': ('conversion', 'from_bloch_coordinates'),
        'vector_to_matrix': ('conversion', 'vector_to_matrix'),
        'matrix_to_vector': ('conversion', 'matrix_to_vector'),
        'to_ket_list': ('conversion', 'to_ket_list'),
        'to_density_list': ('conversion', 'to_density_list'),
        'to_operator_list': ('conversion', 'to_operator_list'),
        'convert_complex_to_real': ('conversion', 'convert_complex_to_real'),
        
        # Polarization
        'horizontal': ('polarization', 'horizontal'),
        'vertical': ('polarization', 'vertical'),
        'diagonal': ('polarization', 'diagonal'),
        'anti_diagonal': ('polarization', 'anti_diagonal'),
        'circular_right': ('polarization', 'circular_right'),
        'circular_left': ('polarization', 'circular_left'),
        'elliptical': ('polarization', 'elliptical'),
        'polarizer': ('polarization', 'polarizer'),
        'waveplate': ('polarization', 'waveplate'),
        'quarter_waveplate': ('polarization', 'quarter_waveplate'),
        'half_waveplate': ('polarization', 'half_waveplate'),
        'full_waveplate': ('polarization', 'full_waveplate'),
        'rotator': ('polarization', 'rotator'),
        'linear_retarder': ('polarization', 'linear_retarder'),
        'circular_dichroism': ('polarization', 'circular_dichroism'),
        'stokes_parameters': ('polarization', 'stokes_parameters'),
        'degree_of_polarization': ('polarization', 'degree_of_polarization'),
        'polarization_angle': ('polarization', 'polarization_angle'),
        'ellipticity': ('polarization', 'ellipticity'),
        'to_poincare': ('polarization', 'to_poincare'),
        'from_poincare': ('polarization', 'from_poincare'),
        'apply_jones': ('polarization', 'apply_jones'),
        'cascade_jones': ('polarization', 'cascade_jones'),
        'is_unitary_jones': ('polarization', 'is_unitary_jones'),
        'jones_to_ket': ('polarization', 'jones_to_ket'),
        'ket_to_jones': ('polarization', 'ket_to_jones'),
        'jones_to_operator': ('polarization', 'jones_to_operator'),
        'operator_to_jones': ('polarization', 'operator_to_jones'),
        
        # Random
        'set_random_seed': ('random', 'set_random_seed'),
        'RandomResult': ('random', 'RandomResult'),
        'random_state': ('random', 'random_state'),
        'random_qubit_state': ('random', 'random_qubit_state'),
        'random_n_qubit_state': ('random', 'random_n_qubit_state'),
        'random_state_result': ('random', 'random_state_result'),
        'random_density_matrix': ('random', 'random_density_matrix'),
        'random_qubit_density_matrix': ('random', 'random_qubit_density_matrix'),
        'random_density_matrix_result': ('random', 'random_density_matrix_result'),
        'random_unitary': ('random', 'random_unitary'),
        'random_hermitian': ('random', 'random_hermitian'),
        'random_positive_operator': ('random', 'random_positive_operator'),
        'random_operator_result': ('random', 'random_operator_result'),
        'random_pauli_rotation': ('random', 'random_pauli_rotation'),
        'random_pauli_string': ('random', 'random_pauli_string'),
        'random_pauli_result': ('random', 'random_pauli_result'),
        'random_ket_from_density_matrix': ('random', 'random_ket_from_density_matrix'),
        'random_mixed_state': ('random', 'random_mixed_state'),
        'random_bloch_vector': ('random', 'random_bloch_vector'),
        
        # Validation
        'ValidationResult': ('validation', 'ValidationResult'),
        'is_unitary': ('validation', 'is_unitary'),
        'is_hermitian': ('validation', 'is_hermitian'),
        'is_positive_semidefinite': ('validation', 'is_positive_semidefinite'),
        'is_density_matrix': ('validation', 'is_density_matrix'),
        'is_normalized': ('validation', 'is_normalized'),
        'are_orthogonal': ('validation', 'are_orthogonal'),
        'are_identical': ('validation', 'are_identical'),
        'fidelity': ('validation', 'fidelity'),
        'validate_circuit': ('validation', 'validate_circuit'),
        'is_valid_qubit_index': ('validation', 'is_valid_qubit_index'),
        'is_power_of_two': ('validation', 'is_power_of_two'),
        'validate_qubits': ('validation', 'validate_qubits'),
        'validate_dimension': ('validation', 'validate_dimension'),
        
        # Config
        'Config': ('config', 'Config'),
        'get_config': ('config', 'get_config'),
        'set_config': ('config', 'set_config'),
        'reset_config': ('config', 'reset_config'),
        'load_from_env': ('config', 'load_from_env'),
        
        # Logger
        'setup_logger': ('logger', 'setup_logger'),
        'get_logger': ('logger', 'get_logger'),
        'get_global_logger': ('logger', 'get_global_logger'),
        'set_log_level': ('logger', 'set_log_level'),
        'set_log_file': ('logger', 'set_log_file'),
        'disable_logging': ('logger', 'disable_logging'),
        'enable_logging': ('logger', 'enable_logging'),
        'debug': ('logger', 'debug'),
        'info': ('logger', 'info'),
        'warning': ('logger', 'warning'),
        'error': ('logger', 'error'),
        'critical': ('logger', 'critical'),
        'exception': ('logger', 'exception'),
        'log_function_call': ('logger', 'log_function_call'),
        'log_exceptions': ('logger', 'log_exceptions'),
        'log_time': ('logger', 'log_time'),
        'logger': ('logger', 'logger'),
    }
    
    if name not in attr_map:
        raise AttributeError(f"module 'psiqit.utils' has no attribute '{name}'")
    
    module_name, attr_name = attr_map[name]
    
    # Import the module
    import importlib
    module = importlib.import_module(f'.{module_name}', package='psiqit.utils')
    
    # Get the attribute
    value = getattr(module, attr_name)
    
    # Cache it
    _module_cache[name] = value
    
    return value


# ============================================================================
# Module information
# ============================================================================

__doc__ = """
PSIQIT Utilities Module
=======================

This module provides utility functions and tools for PSIQIT:

Submodules:
-----------
conversion.py   - Convert between different quantum representations
polarization.py - Jones vectors, Jones matrices, and polarization analysis
random.py       - Generate random quantum states, matrices, and operators
validation.py   - Validate quantum states, matrices, and circuits
config.py       - Global configuration settings
logger.py       - Logging utilities
"""


def info() -> dict:
    """
    Get information about the utils module
    
    Returns:
        dict: Module information
    """
    return {
        'module': 'psiqit.utils',
        'version': __version__,
        'submodules': ['conversion', 'polarization', 'random', 'validation', 'config', 'logger'],
        'exports': len(__all__) if '__all__' in dir() else 0,
    }


def test() -> dict:
    """
    Run basic tests for the utils module
    
    Returns:
        dict: Test results
    """
    results = {}
    
    # Test config
    try:
        from .config import get_config
        config = get_config()
        results['config'] = config.default_shots == 1024
    except Exception as e:
        results['config'] = str(e)
    
    # Test logger
    try:
        from .logger import info as log_info
        results['logger'] = True
    except Exception as e:
        results['logger'] = str(e)
    
    return results


def __dir__():
    """Return list of available attributes"""
    return [
        # Conversion
        'ConversionResult', 'ket_to_density', 'density_to_ket', 'is_pure_state',
        'change_basis', 'to_computational_basis', 'to_pauli_basis', 'from_pauli_basis',
        'to_list', 'to_operator', 'to_numpy', 'to_ket', 'to_bra', 'to_matrix',
        'to_bloch_coordinates', 'from_bloch_coordinates', 'vector_to_matrix',
        'matrix_to_vector', 'to_ket_list', 'to_density_list', 'to_operator_list',
        'convert_complex_to_real',
        
        # Polarization
        'horizontal', 'vertical', 'diagonal', 'anti_diagonal',
        'circular_right', 'circular_left', 'elliptical',
        'polarizer', 'waveplate', 'quarter_waveplate', 'half_waveplate',
        'full_waveplate', 'rotator', 'linear_retarder', 'circular_dichroism',
        'stokes_parameters', 'degree_of_polarization', 'polarization_angle',
        'ellipticity', 'to_poincare', 'from_poincare',
        'apply_jones', 'cascade_jones', 'is_unitary_jones',
        'jones_to_ket', 'ket_to_jones', 'jones_to_operator', 'operator_to_jones',
        
        # Random
        'set_random_seed', 'RandomResult', 'random_state', 'random_qubit_state',
        'random_n_qubit_state', 'random_state_result', 'random_density_matrix',
        'random_qubit_density_matrix', 'random_density_matrix_result',
        'random_unitary', 'random_hermitian', 'random_positive_operator',
        'random_operator_result', 'random_pauli_rotation', 'random_pauli_string',
        'random_pauli_result', 'random_ket_from_density_matrix',
        'random_mixed_state', 'random_bloch_vector',
        
        # Validation
        'ValidationResult', 'is_unitary', 'is_hermitian', 'is_positive_semidefinite',
        'is_density_matrix', 'is_normalized', 'are_orthogonal', 'are_identical',
        'fidelity', 'validate_circuit', 'is_valid_qubit_index', 'is_power_of_two',
        'validate_qubits', 'validate_dimension',
        
        # Config
        'Config', 'get_config', 'set_config', 'reset_config', 'load_from_env',
        
        # Logger
        'setup_logger', 'get_logger', 'get_global_logger', 'set_log_level',
        'set_log_file', 'disable_logging', 'enable_logging',
        'debug', 'info', 'warning', 'error', 'critical', 'exception',
        'log_function_call', 'log_exceptions', 'log_time', 'logger',
    ]