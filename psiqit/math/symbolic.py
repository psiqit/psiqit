#psiqit/math/symbolic.py

import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any

# Check if SymPy is available
try:
    import sympy as sp
    from sympy import (
        Symbol, symbols, Matrix, Function, Eq, I, pi, E,
        sin, cos, tan, exp, log, sqrt, integrate, diff,
        simplify, expand, factor, collect, solve, dsolve,
        lambdify, pprint, pretty, latex, sympify
    )
    from sympy.physics.quantum import (
        Ket, Bra, Operator, Dagger, Commutator, AntiCommutator,
        TensorProduct, qapply
    )
    from sympy.physics.quantum.pauli import Pauli, SigmaX, SigmaY, SigmaZ, SigmaMinus, SigmaPlus
    from sympy.physics.quantum.spin import Jx, Jy, Jz, J2
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    # Create dummy imports for type hints
    class Symbol: pass
    class Matrix: pass
    class Function: pass
    class Eq: pass
    I = None
    pi = None
    E = None

from ..utils.logger import logger


# ============================================================================
# SYMPY AVAILABILITY
# ============================================================================

if SYMPY_AVAILABLE:
    logger.info("SymPy loaded successfully for symbolic computations")
else:
    logger.warning("SymPy not available. Install with: pip install sympy")


# ============================================================================
# BASIC OPERATIONS
# ============================================================================

def create_symbols(names: Union[str, List[str]], **kwargs) -> Union[Symbol, List[Symbol]]:
    """
    Create SymPy symbols
    
    Args:
        names: Symbol name or list of names
        **kwargs: Additional arguments for Symbol
        
    Returns:
        Symbol or List[Symbol]
        
    Example:
        >>> x = create_symbols('x')
        >>> x, y, z = create_symbols(['x', 'y', 'z'])
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is not available. Install with: pip install sympy")
    
    if isinstance(names, str):
        return Symbol(names, **kwargs)
    else:
        return [Symbol(name, **kwargs) for name in names]


def create_matrix(rows: int, cols: int, elements: List[List[str]], **kwargs) -> Matrix:
    """
    Create a symbolic matrix
    
    Args:
        rows: Number of rows
        cols: Number of columns
        elements: List of lists containing symbolic expressions (as strings)
        **kwargs: Additional arguments for Matrix
        
    Returns:
        Matrix: SymPy matrix
        
    Example:
        >>> M = create_matrix(2, 2, [['a', 'b'], ['c', 'd']])
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is not available. Install with: pip install sympy")
    
    # Convert string expressions to sympy expressions
    expr_elements = []
    for row in elements:
        expr_row = []
        for elem in row:
            if isinstance(elem, str):
                expr_row.append(sympify(elem))
            else:
                expr_row.append(elem)
        expr_elements.append(expr_row)
    
    return Matrix(expr_elements, **kwargs)


def solve_equation(equation, variable: Symbol, **kwargs) -> List:
    """
    Solve a symbolic equation
    
    Args:
        equation: Symbolic equation (Eq object or expression)
        variable: Variable to solve for
        **kwargs: Additional arguments for sp.solve
        
    Returns:
        List: Solutions
        
    Example:
        >>> x = create_symbols('x')
        >>> solve_equation(x**2 - 1, x)  # [-1, 1]
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is not available. Install with: pip install sympy")
    
    return sp.solve(equation, variable, **kwargs)


def solve_linear_system(A: Matrix, b: Matrix, variables: List[Symbol]) -> Dict:
    """
    Solve a linear system A*x = b symbolically
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        variables: List of variables
        
    Returns:
        Dict: Solution dictionary
        
    Example:
        >>> x, y = create_symbols(['x', 'y'])
        >>> A = Matrix([[1, 1], [1, -1]])
        >>> b = Matrix([2, 0])
        >>> solve_linear_system(A, b, [x, y])  # {x: 1, y: 1}
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is not available. Install with: pip install sympy")
    
    # Create augmented matrix and solve
    augmented = A.row_join(b)
    solution = sp.solve_linear_system(augmented, *variables)
    
    return solution


# ============================================================================
# DIFFERENTIATION AND INTEGRATION
# ============================================================================

def differentiate(expr, *variables, **kwargs):
    """
    Differentiate a symbolic expression
    
    Args:
        expr: Symbolic expression
        *variables: Variables to differentiate with respect to
        **kwargs: Additional arguments for sp.diff
        
    Returns:
        Expression: Derivative
        
    Example:
        >>> x = create_symbols('x')
        >>> differentiate(x**3, x)  # 3*x**2
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is not available. Install with: pip install sympy")
    
    return diff(expr, *variables, **kwargs)


def integrate_expression(expr, *variables, **kwargs):
    """
    Integrate a symbolic expression
    
    Args:
        expr: Symbolic expression
        *variables: Variables to integrate with respect to
        **kwargs: Additional arguments for sp.integrate
        
    Returns:
        Expression: Integral
        
    Example:
        >>> x = create_symbols('x')
        >>> integrate_expression(x**2, x)  # x**3/3
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is not available. Install with: pip install sympy")
    
    return integrate(expr, *variables, **kwargs)


def simplify_expression(expr, **kwargs):
    """
    Simplify a symbolic expression
    
    Args:
        expr: Symbolic expression
        **kwargs: Additional arguments for sp.simplify
        
    Returns:
        Expression: Simplified expression
        
    Example:
        >>> x = create_symbols('x')
        >>> simplify_expression(x**2 + 2*x + 1)  # (x+1)**2
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is not available. Install with: pip install sympy")
    
    return simplify(expr, **kwargs)


def expand_expression(expr, **kwargs):
    """
    Expand a symbolic expression
    
    Args:
        expr: Symbolic expression
        **kwargs: Additional arguments for sp.expand
        
    Returns:
        Expression: Expanded expression
        
    Example:
        >>> x = create_symbols('x')
        >>> expand_expression((x+1)**2)  # x**2 + 2*x + 1
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is not available. Install with: pip install sympy")
    
    return expand(expr, **kwargs)


# ============================================================================
# QUANTUM OPERATIONS
# ============================================================================

def pauli_algebra() -> Dict[str, sp.Matrix]:
    """
    Get Pauli matrices symbolically
    
    Returns:
        Dict: {'I': I, 'X': sigma_x, 'Y': sigma_y, 'Z': sigma_z}
        
    Example:
        >>> pauli = pauli_algebra()
        >>> print(pauli['X'])
        Matrix([[0, 1], [1, 0]])
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is not available. Install with: pip install sympy")
    
    sigma_x = Matrix([[0, 1], [1, 0]])
    sigma_y = Matrix([[0, -I], [I, 0]])
    sigma_z = Matrix([[1, 0], [0, -1]])
    identity = Matrix([[1, 0], [0, 1]])
    
    return {
        'I': identity,
        'X': sigma_x,
        'Y': sigma_y,
        'Z': sigma_z,
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'sigma_z': sigma_z,
        'sigma_plus': (sigma_x + I * sigma_y) / 2,
        'sigma_minus': (sigma_x - I * sigma_y) / 2,
    }


def commutator_symbolic(A, B) -> sp.Expr:
    """
    Symbolic commutator: [A, B] = A*B - B*A
    
    Args:
        A: Symbolic operator or expression
        B: Symbolic operator or expression
        
    Returns:
        Expr: Commutator expression
        
    Example:
        >>> x, p = create_symbols(['x', 'p'])
        >>> commutator_symbolic(x, p)  # x*p - p*x
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is not available. Install with: pip install sympy")
    
    return Commutator(A, B).doit()


def anticommutator_symbolic(A, B) -> sp.Expr:
    """
    Symbolic anticommutator: {A, B} = A*B + B*A
    
    Args:
        A: Symbolic operator or expression
        B: Symbolic operator or expression
        
    Returns:
        Expr: Anticommutator expression
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is not available. Install with: pip install sympy")
    
    return AntiCommutator(A, B).doit()


def commutator_algebra_symbolic() -> Dict[str, sp.Expr]:
    """
    Get symbolic commutator algebra relations
    
    Returns:
        Dict: Commutator relations
        
    Example:
        >>> algebra = commutator_algebra_symbolic()
        >>> print(algebra['[x,p]'])  # i*hbar
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is not available. Install with: pip install sympy")
    
    x, p = Symbol('x', real=True), Symbol('p', real=True)
    hbar = Symbol('hbar', real=True, positive=True)
    Lx, Ly, Lz = Symbol('Lx'), Symbol('Ly'), Symbol('Lz')
    
    return {
        '[x,p]': Commutator(x, p).doit(),
        '[Lx,Ly]': I * hbar * Lz,
        '[Ly,Lz]': I * hbar * Lx,
        '[Lz,Lx]': I * hbar * Ly,
        '[L²,Lz]': 0,
        '[x,Lz]': 0,
        '[p,Lz]': 0,
    }


def hamiltonian_symbolic(terms: Dict[str, Any]) -> sp.Expr:
    """
    Build a symbolic Hamiltonian from terms
    
    Args:
        terms: Dictionary of terms {operator: coefficient}
        
    Returns:
        Expr: Hamiltonian expression
        
    Example:
        >>> H = hamiltonian_symbolic({'p**2': '1/(2*m)', 'V(x)': '1'})
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is not available. Install with: pip install sympy")
    
    H = 0
    for op, coeff in terms.items():
        if isinstance(op, str):
            op = sympify(op)
        if isinstance(coeff, str):
            coeff = sympify(coeff)
        H += coeff * op
    
    return H


def eigenvalues_symbolic(matrix: Matrix) -> List[sp.Expr]:
    """
    Compute symbolic eigenvalues of a matrix
    
    Args:
        matrix: SymPy matrix
        
    Returns:
        List: Eigenvalues
        
    Example:
        >>> M = Matrix([[a, b], [b, a]])
        >>> eigenvalues_symbolic(M)  # [a-b, a+b]
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is not available. Install with: pip install sympy")
    
    return matrix.eigenvals()


def eigenvectors_symbolic(matrix: Matrix) -> Tuple[List[sp.Expr], List[Matrix]]:
    """
    Compute symbolic eigenvectors of a matrix
    
    Args:
        matrix: SymPy matrix
        
    Returns:
        Tuple: (eigenvalues, eigenvectors)
        
    Example:
        >>> M = Matrix([[a, b], [b, a]])
        >>> eigvals, eigvecs = eigenvectors_symbolic(M)
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is not available. Install with: pip install sympy")
    
    return matrix.eigenvects()


# ============================================================================
# SCHRÖDINGER EQUATION
# ============================================================================

def schrodinger_1d_symbolic(potential: Union[str, sp.Expr]) -> sp.Expr:
    """
    Build 1D time-independent Schrödinger equation symbolically
    
    -ħ²/(2m) d²ψ/dx² + V(x)ψ = Eψ
    
    Args:
        potential: Potential function V(x)
        
    Returns:
        Expr: Schrödinger equation
        
    Example:
        >>> from sympy import Function, Symbol, Derivative
        >>> x = Symbol('x')
        >>> psi = Function('psi')(x)
        >>> V = x**2  # Harmonic oscillator
        >>> eq = schrodinger_1d_symbolic(V)
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is not available. Install with: pip install sympy")
    
    x = Symbol('x', real=True)
    psi = Function('psi')(x)
    hbar = Symbol('hbar', real=True, positive=True)
    m = Symbol('m', real=True, positive=True)
    E = Symbol('E', real=True)
    
    # Kinetic term: -ħ²/(2m) d²ψ/dx²
    kinetic = -hbar**2 / (2 * m) * diff(psi, x, x)
    
    # Potential term
    if isinstance(potential, str):
        V = sympify(potential)
    else:
        V = potential
    
    # Full equation: Hψ = Eψ
    H_psi = kinetic + V * psi
    equation = Eq(H_psi, E * psi)
    
    return equation


def infinite_well_states(n: int, L: Union[float, sp.Expr]) -> sp.Expr:
    """
    Get infinite square well eigenstates symbolically
    
    ψ_n(x) = √(2/L) sin(nπx/L)
    
    Args:
        n: Quantum number (integer)
        L: Well width
        
    Returns:
        Expr: Wavefunction
        
    Example:
        >>> from sympy import Symbol
        >>> x = Symbol('x')
        >>> psi_1 = infinite_well_states(1, 1)
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is not available. Install with: pip install sympy")
    
    x = Symbol('x', real=True)
    
    # Check if L is a sympy object
    if not hasattr(L, 'is_symbol'):
        L = sp.sympify(L)
    
    norm = sp.sqrt(2 / L)
    psi = norm * sp.sin(n * pi * x / L)
    
    return psi


def harmonic_oscillator_states(
    n: int,
    mass: Union[float, sp.Expr] = 1.0,
    omega: Union[float, sp.Expr] = 1.0,
    hbar: Union[float, sp.Expr] = 1.0
) -> sp.Expr:
    """
    Get harmonic oscillator eigenstates symbolically
    
    ψ_n(x) = (1/√(2^n n!)) (mω/πħ)^(1/4) e^(-mωx²/2ħ) H_n(√(mω/ħ) x)
    
    Args:
        n: Quantum number (integer)
        mass: Mass parameter
        omega: Angular frequency
        hbar: Reduced Planck constant
        
    Returns:
        Expr: Wavefunction
        
    Example:
        >>> from sympy import Symbol
        >>> x = Symbol('x')
        >>> psi_2 = harmonic_oscillator_states(2)
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is not available. Install with: pip install sympy")
    
    x = Symbol('x', real=True)
    
    # Convert to sympy expressions
    if not hasattr(mass, 'is_symbol'):
        mass = sp.sympify(mass)
    if not hasattr(omega, 'is_symbol'):
        omega = sp.sympify(omega)
    if not hasattr(hbar, 'is_symbol'):
        hbar = sp.sympify(hbar)
    
    # Hermite polynomial
    from sympy import hermite, factorial, sqrt
    
    # Prefactor
    alpha = mass * omega / hbar
    prefactor = (alpha / pi)**(sp.Rational(1, 4)) / sp.sqrt(2**n * factorial(n))
    
    # Gaussian part
    gaussian = sp.exp(-alpha * x**2 / 2)
    
    # Hermite polynomial
    hermitian = hermite(n, sp.sqrt(alpha) * x)
    
    psi = prefactor * gaussian * hermitian
    
    return psi


# ============================================================================
# ADDITIONAL SYMBOLIC QUANTUM TOOLS
# ============================================================================

def expectation_value_symbolic(operator: sp.Expr, state: sp.Expr) -> sp.Expr:
    """
    Compute expectation value ⟨ψ|O|ψ⟩ symbolically
    
    Args:
        operator: Operator expression
        state: State expression (wavefunction)
        
    Returns:
        Expr: Expectation value
        
    Example:
        >>> x = Symbol('x')
        >>> psi = exp(-x**2/2)
        >>> expectation_value_symbolic(x**2, psi)
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is not available. Install with: pip install sympy")
    
    from sympy import conjugate
    
    # For real wavefunctions, ⟨ψ|O|ψ⟩ = ∫ ψ* O ψ dx
    x = Symbol('x', real=True)
    integrand = conjugate(state) * operator * state
    
    return integrate(integrand, (x, -sp.oo, sp.oo))


def time_evolution_symbolic(hamiltonian: sp.Expr, psi0: sp.Expr, t: sp.Symbol) -> sp.Expr:
    """
    Compute time evolution U(t) = e^(-iHt/ħ) ψ(0)
    
    Args:
        hamiltonian: Hamiltonian operator
        psi0: Initial state
        t: Time variable
        
    Returns:
        Expr: Time-evolved state
        
    Example:
        >>> t = Symbol('t')
        >>> H = p**2/(2*m)  # Free particle
        >>> psi_t = time_evolution_symbolic(H, exp(-x**2/2), t)
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is not available. Install with: pip install sympy")
    
    hbar = Symbol('hbar', real=True, positive=True)
    
    # For simple cases where H is constant
    # U(t) = exp(-iHt/ħ)
    # For complex cases, this would need to be done differently
    
    # For now, return the formal expression
    from sympy import exp
    
    U = exp(-I * hamiltonian * t / hbar)
    psi_t = U * psi0
    
    return psi_t


def density_matrix_symbolic(psi: sp.Expr) -> sp.Expr:
    """
    Construct density matrix ρ = |ψ⟩⟨ψ| symbolically
    
    Args:
        psi: State vector
        
    Returns:
        Expr: Density matrix
        
    Example:
        >>> from sympy import Matrix
        >>> psi = Matrix([1, 0])
        >>> rho = density_matrix_symbolic(psi)
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is not available. Install with: pip install sympy")
    
    from sympy import conjugate
    
    if isinstance(psi, Matrix):
        # If it's a column vector
        return psi * psi.H
    else:
        # If it's a scalar wavefunction
        return psi * conjugate(psi)


# ============================================================================
# TENSOR PRODUCT AND PARTIAL TRACE
# ============================================================================

def tensor_product_symbolic(A, B) -> sp.Expr:
    """
    Tensor product of two operators/states
    
    Args:
        A: First operator/state
        B: Second operator/state
        
    Returns:
        Expr: Tensor product
        
    Example:
        >>> sigma_x = Matrix([[0, 1], [1, 0]])
        >>> sigma_z = Matrix([[1, 0], [0, -1]])
        >>> tensor_product_symbolic(sigma_x, sigma_z)
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is not available. Install with: pip install sympy")
    
    return TensorProduct(A, B)


def partial_trace_symbolic(rho: Matrix, dims: List[int], keep: List[int]) -> Matrix:
    """
    Compute partial trace symbolically
    
    Args:
        rho: Density matrix
        dims: List of subsystem dimensions
        keep: Indices of subsystems to keep
        
    Returns:
        Matrix: Reduced density matrix
        
    Example:
        >>> rho = Matrix([[1/2, 0], [0, 1/2]])
        >>> partial_trace_symbolic(rho, [2, 2], [0])
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is not available. Install with: pip install sympy")
    
    if len(dims) != 2:
        raise ValueError("Partial trace currently only supports bipartite systems")
    
    dA, dB = dims
    keep_idx = keep[0]
    
    if keep_idx == 0:
        # Trace out B
        rho_reduced = Matrix.zeros(dA, dA)
        for i in range(dB):
            # Projector onto |i⟩_B
            projector = Matrix.zeros(dB, dB)
            projector[i, i] = 1
            rho_reduced += (Matrix.eye(dA) @ projector) * rho * (Matrix.eye(dA) @ projector)
        return rho_reduced
    else:
        # Trace out A
        rho_reduced = Matrix.zeros(dB, dB)
        for i in range(dA):
            projector = Matrix.zeros(dA, dA)
            projector[i, i] = 1
            rho_reduced += (projector @ Matrix.eye(dB)) * rho * (projector @ Matrix.eye(dB))
        return rho_reduced


# ============================================================================
# DISPLAY UTILITIES
# ============================================================================

def print_symbolic(expr, use_latex: bool = False):
    """
    Print a symbolic expression nicely
    
    Args:
        expr: Symbolic expression
        use_latex: Use LaTeX output if True
    """
    if not SYMPY_AVAILABLE:
        print("SymPy is not available")
        return
    
    if use_latex:
        print(latex(expr))
    else:
        pprint(expr, use_unicode=True)


def to_latex(expr) -> str:
    """
    Convert symbolic expression to LaTeX
    
    Args:
        expr: Symbolic expression
        
    Returns:
        str: LaTeX string
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is not available. Install with: pip install sympy")
    
    return latex(expr)


def to_numpy(expr, variables: List[Symbol]) -> callable:
    """
    Convert symbolic expression to numpy function
    
    Args:
        expr: Symbolic expression
        variables: List of variables
        
    Returns:
        callable: Numpy function
        
    Example:
        >>> x = Symbol('x')
        >>> f = to_numpy(x**2, [x])
        >>> f(np.array([1, 2, 3]))  # [1, 4, 9]
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is not available. Install with: pip install sympy")
    
    return lambdify(variables, expr, modules='numpy')


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Availability
    'SYMPY_AVAILABLE',
    
    # Basic operations
    'create_symbols',
    'create_matrix',
    'solve_equation',
    'solve_linear_system',
    
    # Differentiation and integration
    'differentiate',
    'integrate_expression',
    'simplify_expression',
    'expand_expression',
    
    # Quantum operations
    'pauli_algebra',
    'commutator_symbolic',
    'anticommutator_symbolic',
    'commutator_algebra_symbolic',
    'hamiltonian_symbolic',
    'eigenvalues_symbolic',
    'eigenvectors_symbolic',
    
    # Schrödinger equation
    'schrodinger_1d_symbolic',
    'infinite_well_states',
    'harmonic_oscillator_states',
    
    # Additional quantum tools
    'expectation_value_symbolic',
    'time_evolution_symbolic',
    'density_matrix_symbolic',
    'tensor_product_symbolic',
    'partial_trace_symbolic',
    
    # Display utilities
    'print_symbolic',
    'to_latex',
    'to_numpy',
    
    # Exports from sympy (for convenience)
    'Symbol',
    'symbols',
    'Matrix',
    'Function',
    'Eq',
    'I',
    'pi',
    'E',
]

# If SymPy is available, also export common functions
if SYMPY_AVAILABLE:
    __all__.extend([
        'sin', 'cos', 'tan', 'exp', 'log', 'sqrt',
        'integrate', 'diff',
        'simplify', 'expand', 'factor', 'collect',
        'solve', 'dsolve',
        'lambdify', 'pprint', 'pretty', 'latex', 'sympify',
        'Ket', 'Bra', 'Operator', 'Dagger', 'TensorProduct', 'qapply',
        'Pauli', 'SigmaX', 'SigmaY', 'SigmaZ',
        'Jx', 'Jy', 'Jz', 'J2'
    ])