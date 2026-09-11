#psiqit/math/__init__.py

"""
PSIQIT - Python Scientific Quantum Information Toolkit
Mathematics Module

This module provides mathematical tools for quantum computing:
- Calculus (derivatives, integrals, quantum derivatives)
- ODE solvers (Euler, Runge-Kutta, adaptive methods)
- PDE solvers (heat, wave, Schrödinger equations)
- Quantum algebra (vectors, matrices, operators)
- Symbolic mathematics (SymPy integration)
"""

# ============================================================================
# Version and metadata
# ============================================================================

from ..version import __version__

__all__ = [
    # Module information
    '__version__',
    
    # ========================================================================
    # CALCULUS (from calculus.py)
    # ========================================================================
    'derivative',
    'partial_derivative',
    'gradient',
    'jacobian',
    'hessian',
    'laplacian',
    'integral',
    'integral_2d',
    'integral_3d',
    'indefinite_integral',
    'derivative_gaussian',
    'derivative_sine_wave',
    'derivative_cosine_wave',
    'derivative_plane_wave',
    'taylor_series',
    'diff_operator',
    'expectation_derivative',
    'kinetic_operator',
    'potential_operator',
    'hamiltonian_1d',
    
    # ========================================================================
    # ODE SOLVERS (from ode_solver.py)
    # ========================================================================
    'ODEResult',
    'euler',
    'euler_system',
    'rk2',
    'rk4',
    'rk4_system',
    'solve_ode',
    'adaptive_rk45',
    'schrodinger_1d_numerical',
    'schrodinger_1d_stationary',
    'lindblad_solver',
    
    # ========================================================================
    # PDE SOLVERS (from pde_solver.py)
    # ========================================================================
    'PDEResult',
    'solve_heat_equation',
    'solve_wave_equation',
    'solve_schrodinger_1d',
    'diffusion_equation_implicit',
    'solve_poisson_equation',
    'solve_helmholtz_equation',
    
    # ========================================================================
    # QUANTUM ALGEBRA (from qalgebra.py)
    # ========================================================================
    'Vector',
    'Matrix',
    'PI', 'TAU', 'EULER', 'INF', 'SQRT2', 'SQRT3', 'PHI',
    'SQRT_PI', 'SQRT_2PI',
    'HBAR', 'H_PLANCK', 'C', 'E_CHARGE', 'M_ELECTRON',
    'M_PROTON', 'K_B', 'EPSILON_0', 'MU_0', 'GRAVITY',
    'is_real', 'phase', 'magnitude', 'cis',
    'from_polar', 'to_polar',
    'eye', 'zeros', 'add', 'mul', 'transpose',
    'dagger', 'trace', 'kron',
    'inner', 'outer', 'norm', 'normalize',
    'determinant', 'inverse', 'eigenvalues',
    'eigenvectors', 'expm', 'sqrt_matrix',
    'is_unitary', 'is_hermitian', 'is_positive',
    'is_distribution', 'normalize_probs', 'sample',
    'entropy', 'born_rule', 'expectation', 'variance',
    'commutator', 'anticommutator', 'dimension',
    'hilbert_space', 'display',
    'eV_to_J', 'J_to_eV', 'nm_to_m', 'm_to_nm',
    'energy_to_frequency', 'frequency_to_energy',
    'wavelength_to_frequency', 'frequency_to_wavelength',
    'wavelength_to_energy', 'energy_to_wavelength',
    
    # ========================================================================
    # SYMBOLIC (from symbolic.py)
    # ========================================================================
    'SYMPY_AVAILABLE',
    'create_symbols',
    'create_matrix',
    'solve_equation',
    'solve_linear_system',
    'differentiate',
    'integrate_expression',
    'simplify_expression',
    'expand_expression',
    'pauli_algebra',
    'commutator_symbolic',
    'anticommutator_symbolic',
    'commutator_algebra_symbolic',
    'hamiltonian_symbolic',
    'eigenvalues_symbolic',
    'eigenvectors_symbolic',
    'schrodinger_1d_symbolic',
    'infinite_well_states',
    'harmonic_oscillator_states',
    'expectation_value_symbolic',
    'time_evolution_symbolic',
    'density_matrix_symbolic',
    'tensor_product_symbolic',
    'partial_trace_symbolic',
    'print_symbolic',
    'to_latex',
    'to_numpy',
]

# ============================================================================
# Lazy imports to avoid circular imports
# ============================================================================

# از import مستقیم ماژول‌ها خودداری می‌کنیم
# توابع به صورت lazy import در دسترس خواهند بود

# ============================================================================
# Module information
# ============================================================================

__doc__ = """
PSIQIT Mathematics Module
=========================

This module provides comprehensive mathematical tools for quantum computing
and quantum information science.

Submodules:
-----------
calculus.py     - Derivatives, integrals, and quantum calculus
ode_solver.py   - ODE solvers (Euler, Runge-Kutta, adaptive)
pde_solver.py   - PDE solvers (heat, wave, Schrödinger)
qalgebra.py     - Quantum algebra (vectors, matrices, operators)
symbolic.py     - Symbolic mathematics with SymPy
"""


# ============================================================================
# Version check
# ============================================================================

def info() -> dict:
    """
    Get information about the mathematics module
    
    Returns:
        dict: Module information
    """
    return {
        'module': 'psiqit.math',
        'version': __version__,
        'submodules': ['calculus', 'ode_solver', 'pde_solver', 'qalgebra', 'symbolic'],
        'exports': len(__all__),
    }


def test() -> dict:
    """
    Run basic tests for the mathematics module
    
    Returns:
        dict: Test results
    """
    results = {}
    
    # Test constants
    try:
        from .qalgebra import PI, HBAR
        results['constants'] = PI > 0 and HBAR > 0
    except Exception as e:
        results['constants'] = str(e)
    
    return results