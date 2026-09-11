# psiqit/interface/__init__.py

"""
PSIQIT - Python Scientific Quantum Information Toolkit
Interface Module

This module provides user interface tools for PSIQIT including
command-line interface, LaTeX output, and Jupyter notebook tools.

Submodules:
-----------
cli.py      - Command-line interface for quantum circuits and algorithms
latex.py    - LaTeX code generation for matrices, states, circuits, and reports
notebook.py - Jupyter notebook display tools (circuits, states, histograms, Bloch spheres)
"""

# ============================================================================
# Version and metadata
# ============================================================================

from ..version import __version__

__all__ = [
    # Module information
    '__version__',
    
    # ========================================================================
    # CLI (from cli.py)
    # ========================================================================
    'parse_gates',
    'run_circuit',
    'run_algorithm',
    'main',
    'cli_main',
    
    # ========================================================================
    # LATEX (from latex.py)
    # ========================================================================
    'matrix_to_latex',
    'state_to_latex',
    'circuit_to_latex',
    'equation_to_latex',
    'table_to_latex',
    'generate_report',
    'dirac_notation',
    'pauli_matrices_latex',
    
    # ========================================================================
    # NOTEBOOK (from notebook.py) - فقط توابع موجود
    # ========================================================================
    'init_notebook',
    'display_circuit',
    # 'display_state',  # <-- کامنت شده چون در notebook.py تعریف نشده
    'display_histogram',
    'display_bloch',
    'display_matrix',
    'display_notebook_version',
]

# ============================================================================
# Lazy imports with __getattr__
# ============================================================================

_module_cache = {}


def __getattr__(name):
    """Lazy import for modules to avoid circular imports"""
    if name in _module_cache:
        return _module_cache[name]
    
    attr_map = {
        # CLI
        'parse_gates': ('cli', 'parse_gates'),
        'run_circuit': ('cli', 'run_circuit'),
        'run_algorithm': ('cli', 'run_algorithm'),
        'main': ('cli', 'main'),
        'cli_main': ('cli', 'cli_main'),
        
        # LaTeX
        'matrix_to_latex': ('latex', 'matrix_to_latex'),
        'state_to_latex': ('latex', 'state_to_latex'),
        'circuit_to_latex': ('latex', 'circuit_to_latex'),
        'equation_to_latex': ('latex', 'equation_to_latex'),
        'table_to_latex': ('latex', 'table_to_latex'),
        'generate_report': ('latex', 'generate_report'),
        'dirac_notation': ('latex', 'dirac_notation'),
        'pauli_matrices_latex': ('latex', 'pauli_matrices_latex'),
        
        # Notebook
        'init_notebook': ('notebook', 'init_notebook'),
        'display_circuit': ('notebook', 'display_circuit'),
        'display_histogram': ('notebook', 'display_histogram'),
        'display_bloch': ('notebook', 'display_bloch'),
        'display_matrix': ('notebook', 'display_matrix'),
        'display_notebook_version': ('notebook', 'display_notebook_version'),
    }
    
    if name not in attr_map:
        raise AttributeError(f"module 'psiqit.interface' has no attribute '{name}'")
    
    module_name, attr_name = attr_map[name]
    
    import importlib
    module = importlib.import_module(f'.{module_name}', package='psiqit.interface')
    value = getattr(module, attr_name)
    _module_cache[name] = value
    return value


# ============================================================================
# Module information
# ============================================================================

def info() -> dict:
    """
    Get information about the interface module
    
    Returns:
        dict: Module information
    """
    return {
        'module': 'psiqit.interface',
        'version': __version__,
        'submodules': ['cli', 'latex', 'notebook'],
        'exports': len(__all__),
    }


def test() -> dict:
    """
    Run basic tests for the interface module
    
    Returns:
        dict: Test results
    """
    results = {}
    
    # Test CLI
    try:
        from .cli import parse_gates
        gates = parse_gates("H(0),CNOT(0,1)")
        results['cli'] = len(gates) == 2
    except Exception as e:
        results['cli'] = str(e)
    
    # Test LaTeX
    try:
        from .latex import matrix_to_latex, state_to_latex
        from ..quantum.operator import pauli_x
        from ..quantum.state import Ket
        import numpy as np
        
        latex = matrix_to_latex(pauli_x(), name="X")
        results['latex_matrix'] = "\\begin" in latex
        
        state = Ket([1/np.sqrt(2), 1/np.sqrt(2)])
        latex = state_to_latex(state)
        results['latex_state'] = "\\rangle" in latex
    except Exception as e:
        results['latex'] = str(e)
    
    # Test Notebook
    try:
        from .notebook import init_notebook
        results['notebook'] = True
    except Exception as e:
        results['notebook'] = str(e)
    
    return results


def __dir__():
    """Return list of available attributes"""
    return __all__