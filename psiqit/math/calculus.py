#psiqit/math/calculus.py

import numpy as np
from typing import Union, List, Callable, Optional, Tuple
from ..utils.logger import logger
from ..utils.validation import validate_qubits


# ============================================================================
# DERIVATIVES
# ============================================================================

def derivative(
    f: Callable[[float], float],
    x: float,
    h: float = 1e-7,
    method: str = 'central'
) -> float:
    """
    Compute numerical derivative of a function at point x
    
    Args:
        f: Function to differentiate
        x: Point at which to evaluate derivative
        h: Step size for finite difference
        method: 'forward', 'backward', or 'central'
        
    Returns:
        float: Derivative value
        
    Example:
        >>> def f(x): return x**2
        >>> derivative(f, 2)  # 4.0
    """
    if method == 'forward':
        return (f(x + h) - f(x)) / h
    elif method == 'backward':
        return (f(x) - f(x - h)) / h
    elif method == 'central':
        return (f(x + h) - f(x - h)) / (2 * h)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'forward', 'backward', or 'central'")


def partial_derivative(
    f: Callable[..., float],
    args: List[float],
    idx: int,
    h: float = 1e-7
) -> float:
    """
    Compute partial derivative of a multi-variable function
    
    Args:
        f: Function that takes a list of arguments
        args: List of argument values
        idx: Index of variable to differentiate with respect to
        h: Step size
        
    Returns:
        float: Partial derivative value
        
    Example:
        >>> def f(x): return x[0]**2 + x[1]**3
        >>> partial_derivative(f, [2, 3], 1)  # 27.0
    """
    args_plus = args.copy()
    args_minus = args.copy()
    args_plus[idx] += h
    args_minus[idx] -= h
    
    return (f(args_plus) - f(args_minus)) / (2 * h)


def gradient(
    f: Callable[..., float],
    x: np.ndarray,
    h: float = 1e-7
) -> np.ndarray:
    """
    Compute gradient (vector of partial derivatives)
    
    Args:
        f: Scalar function of vector input
        x: Point at which to evaluate gradient (numpy array)
        h: Step size
        
    Returns:
        np.ndarray: Gradient vector
        
    Example:
        >>> def f(x): return x[0]**2 + x[1]**2
        >>> gradient(f, np.array([1, 2]))  # [2.0, 4.0]
    """
    grad = np.zeros_like(x, dtype=float)
    
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    
    return grad


def jacobian(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    h: float = 1e-7
) -> np.ndarray:
    """
    Compute Jacobian matrix of a vector-valued function
    
    Args:
        f: Vector-valued function that takes vector input
        x: Point at which to evaluate Jacobian
        h: Step size
        
    Returns:
        np.ndarray: Jacobian matrix (m x n)
        
    Example:
        >>> def f(x): return np.array([x[0]**2 + x[1], x[0] - x[1]**2])
        >>> jacobian(f, np.array([1, 2]))
        # [[2.0, 1.0],
        #  [1.0, -4.0]]
    """
    # Evaluate function at x
    f_x = f(x)
    m = len(f_x)  # number of outputs
    n = len(x)    # number of inputs
    
    J = np.zeros((m, n), dtype=float)
    
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        
        f_plus = f(x_plus)
        f_minus = f(x_minus)
        
        J[:, i] = (f_plus - f_minus) / (2 * h)
    
    return J


def hessian(
    f: Callable[..., float],
    x: np.ndarray,
    h: float = 1e-6
) -> np.ndarray:
    """
    Compute Hessian matrix (matrix of second partial derivatives)
    
    Args:
        f: Scalar function of vector input
        x: Point at which to evaluate Hessian
        h: Step size
        
    Returns:
        np.ndarray: Hessian matrix (n x n)
        
    Example:
        >>> def f(x): return x[0]**2 + 3*x[0]*x[1] + x[1]**2
        >>> hessian(f, np.array([1, 1]))
        # [[2.0, 3.0],
        #  [3.0, 2.0]]
    """
    n = len(x)
    H = np.zeros((n, n), dtype=float)
    
    def partial_gradient_f(xi):
        return gradient(lambda y: f(y), xi, h)
    
    # Compute Hessian row by row using gradient
    grad = partial_gradient_f(x)
    
    for i in range(n):
        # Perturb x[i] by h and compute new gradient
        x_plus = x.copy()
        x_plus[i] += h
        
        grad_plus = partial_gradient_f(x_plus)
        
        # Hessian column i: (grad(x + h*e_i) - grad(x)) / h
        H[:, i] = (grad_plus - grad) / h
    
    return H


def laplacian(
    f: Callable[..., float],
    x: np.ndarray,
    h: float = 1e-6
) -> float:
    """
    Compute Laplacian (sum of second partial derivatives)
    
    Args:
        f: Scalar function of vector input
        x: Point at which to evaluate Laplacian
        h: Step size
        
    Returns:
        float: Laplacian value
        
    Example:
        >>> def f(x): return x[0]**2 + x[1]**2
        >>> laplacian(f, np.array([1, 2]))  # 4.0
    """
    n = len(x)
    lap = 0.0
    
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        
        # Second derivative using central difference
        d2 = (f(x_plus) - 2*f(x) + f(x_minus)) / (h**2)
        lap += d2
    
    return lap


# ============================================================================
# INTEGRALS
# ============================================================================

def integral(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 1000,
    method: str = 'simpson'
) -> float:
    """
    Compute definite integral of a 1D function
    
    Args:
        f: Function to integrate
        a: Lower limit
        b: Upper limit
        n: Number of points
        method: 'trapezoidal', 'simpson', or 'romberg'
        
    Returns:
        float: Integral value
        
    Example:
        >>> def f(x): return x**2
        >>> integral(f, 0, 1)  # 0.333333
    """
    if method == 'trapezoidal':
        x = np.linspace(a, b, n)
        y = f(x)
        return np.trapz(y, x)
    
    elif method == 'simpson':
        x = np.linspace(a, b, n)
        y = f(x)
        return np.trapz(y, x)  # Fallback for now
    
    elif method == 'romberg':
        from scipy.integrate import romberg
        return romberg(f, a, b)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def integral_2d(
    f: Callable[[float, float], float],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    n_x: int = 100,
    n_y: int = 100
) -> float:
    """
    Compute double integral of a 2D function
    
    Args:
        f: Function f(x, y)
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)
        n_x: Number of points in x direction
        n_y: Number of points in y direction
        
    Returns:
        float: Double integral value
        
    Example:
        >>> def f(x, y): return x**2 + y**2
        >>> integral_2d(f, (0, 1), (0, 1))  # 0.6667
    """
    x = np.linspace(x_range[0], x_range[1], n_x)
    y = np.linspace(y_range[0], y_range[1], n_y)
    
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    
    dx = (x_range[1] - x_range[0]) / (n_x - 1)
    dy = (y_range[1] - y_range[0]) / (n_y - 1)
    
    return np.sum(Z) * dx * dy


def integral_3d(
    f: Callable[[float, float, float], float],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
    n_x: int = 50,
    n_y: int = 50,
    n_z: int = 50
) -> float:
    """
    Compute triple integral of a 3D function
    
    Args:
        f: Function f(x, y, z)
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)
        z_range: (z_min, z_max)
        n_x, n_y, n_z: Number of points in each direction
        
    Returns:
        float: Triple integral value
        
    Example:
        >>> def f(x, y, z): return x**2 + y**2 + z**2
        >>> integral_3d(f, (0, 1), (0, 1), (0, 1))  # 1.0
    """
    x = np.linspace(x_range[0], x_range[1], n_x)
    y = np.linspace(y_range[0], y_range[1], n_y)
    z = np.linspace(z_range[0], z_range[1], n_z)
    
    X, Y, Z = np.meshgrid(x, y, z)
    result = f(X, Y, Z)
    
    dx = (x_range[1] - x_range[0]) / (n_x - 1)
    dy = (y_range[1] - y_range[0]) / (n_y - 1)
    dz = (z_range[1] - z_range[0]) / (n_z - 1)
    
    return np.sum(result) * dx * dy * dz


def indefinite_integral(
    f: Callable[[float], float],
    x: np.ndarray,
    constant: float = 0.0
) -> np.ndarray:
    """
    Compute indefinite integral (cumulative integral)
    
    Args:
        f: Function to integrate
        x: Array of points
        constant: Integration constant (initial value)
        
    Returns:
        np.ndarray: Cumulative integral values
        
    Example:
        >>> def f(x): return 2*x
        >>> x = np.array([0, 1, 2, 3])
        >>> indefinite_integral(f, x)  # [0, 1, 4, 9]
    """
    y = f(x)
    result = np.zeros_like(x, dtype=float)
    result[0] = constant
    
    for i in range(1, len(x)):
        dx = x[i] - x[i-1]
        # Trapezoidal rule
        result[i] = result[i-1] + (y[i-1] + y[i]) / 2 * dx
    
    return result


# ============================================================================
# QUANTUM FUNCTION DERIVATIVES
# ============================================================================

def derivative_gaussian(
    x: float,
    x0: float = 0.0,
    sigma: float = 1.0,
    order: int = 1
) -> float:
    """
    Compute derivative of Gaussian wavefunction
    
    ψ(x) = (1/√(2πσ²)) * exp(-(x-x0)²/(2σ²))
    
    Args:
        x: Position
        x0: Mean
        sigma: Standard deviation
        order: Derivative order (1 or 2)
        
    Returns:
        float: Derivative value
        
    Example:
        >>> derivative_gaussian(0, x0=0, sigma=1, order=1)  # 0.0
    """
    from math import exp, pi, sqrt
    
    prefactor = 1.0 / sqrt(2 * pi * sigma**2)
    exponent = exp(-(x - x0)**2 / (2 * sigma**2))
    psi = prefactor * exponent
    
    if order == 1:
        # dψ/dx = -(x-x0)/σ² * ψ
        return - (x - x0) / sigma**2 * psi
    elif order == 2:
        # d²ψ/dx² = ((x-x0)²/σ⁴ - 1/σ²) * ψ
        return ((x - x0)**2 / sigma**4 - 1 / sigma**2) * psi
    else:
        raise ValueError(f"Order must be 1 or 2, got {order}")


def derivative_sine_wave(
    x: float,
    k: float = 1.0,
    phase: float = 0.0,
    order: int = 1
) -> float:
    """
    Compute derivative of sine wave
    
    ψ(x) = sin(kx + φ)
    
    Args:
        x: Position
        k: Wave number
        phase: Phase shift
        order: Derivative order (1 or 2)
        
    Returns:
        float: Derivative value
    """
    from math import sin, cos
    
    if order == 1:
        return k * cos(k * x + phase)
    elif order == 2:
        return -k**2 * sin(k * x + phase)
    else:
        raise ValueError(f"Order must be 1 or 2, got {order}")


def derivative_cosine_wave(
    x: float,
    k: float = 1.0,
    phase: float = 0.0,
    order: int = 1
) -> float:
    """
    Compute derivative of cosine wave
    
    ψ(x) = cos(kx + φ)
    
    Args:
        x: Position
        k: Wave number
        phase: Phase shift
        order: Derivative order (1 or 2)
        
    Returns:
        float: Derivative value
    """
    from math import sin, cos
    
    if order == 1:
        return -k * sin(k * x + phase)
    elif order == 2:
        return -k**2 * cos(k * x + phase)
    else:
        raise ValueError(f"Order must be 1 or 2, got {order}")


def derivative_plane_wave(
    x: float,
    k: float = 1.0,
    phase: float = 0.0,
    hbar: float = 1.0,
    order: int = 1
) -> complex:
    """
    Compute derivative of plane wave (complex exponential)
    
    ψ(x) = e^{i(kx + φ)}
    or ψ(x) = e^{i(kx - ωt)} for time-dependent
    
    Args:
        x: Position
        k: Wave number (momentum/hbar)
        phase: Phase shift
        hbar: Reduced Planck constant
        order: Derivative order
        
    Returns:
        complex: Derivative value
        
    Example:
        >>> derivative_plane_wave(0, k=1.0, order=1)  # i
    """
    from math import exp, pi
    
    amplitude = exp(1j * (k * x + phase))
    
    if order == 1:
        return 1j * k * amplitude
    elif order == 2:
        return -k**2 * amplitude
    elif order == 3:
        return -1j * k**3 * amplitude
    elif order == 4:
        return k**4 * amplitude
    else:
        raise ValueError(f"Order {order} not supported. Use 1-4.")


# ============================================================================
# TOOLS
# ============================================================================

def taylor_series(
    f: Callable[[float], float],
    x0: float,
    n: int = 5,
    h: float = 1e-5
) -> Callable[[float], float]:
    """
    Generate Taylor series approximation of a function
    
    Args:
        f: Function to approximate
        x0: Point around which to expand
        n: Order of Taylor series (number of terms)
        h: Step size for numerical derivatives
        
    Returns:
        Callable: Taylor series function
    
    Example:
        >>> def f(x): return np.exp(x)
        >>> taylor = taylor_series(f, 0, n=4)
        >>> taylor(0.1)  # 1.105... (approx e^0.1)
    """
    # Compute derivatives numerically
    derivatives = []
    for i in range(n):
        # Use central difference for higher accuracy
        if i == 0:
            derivatives.append(f(x0))
        else:
            # Compute ith derivative using finite differences
            coeffs = _finite_difference_coeffs(i, h)
            der = 0.0
            for j, c in enumerate(coeffs):
                x_val = x0 + (j - len(coeffs)//2) * h
                der += c * f(x_val)
            derivatives.append(der / (h**i))
    
    # Create Taylor series function
    def taylor_func(x: float) -> float:
        result = 0.0
        for i, der in enumerate(derivatives):
            result += der * ((x - x0)**i) / np.math.factorial(i)
        return result
    
    return taylor_func


def _finite_difference_coeffs(order: int, h: float) -> List[float]:
    """
    Get finite difference coefficients for numerical differentiation
    
    Args:
        order: Derivative order
        h: Step size
        
    Returns:
        List of coefficients for central difference
    """
    # Simple central difference coefficients
    if order == 1:
        return [-1/(2*h), 0, 1/(2*h)]  # [-1, 0, 1]/(2h)
    elif order == 2:
        return [1/(h**2), -2/(h**2), 1/(h**2)]  # [1, -2, 1]/h²
    else:
        # Fallback: use scipy if available
        try:
            from scipy.misc import derivative
            return derivative
        except ImportError:
            logger.warning(f"Higher order derivatives ({order}) may be inaccurate")
            return [0] * (2*order + 1)


def diff_operator(
    order: int = 1,
    dim: int = 100,
    dx: float = 0.01
) -> np.ndarray:
    """
    Create a finite difference operator matrix
    
    Args:
        order: Derivative order (1 or 2)
        dim: Matrix dimension (number of grid points)
        dx: Grid spacing
        
    Returns:
        np.ndarray: Difference operator matrix
        
    Example:
        >>> D = diff_operator(order=1, dim=5, dx=0.1)
        >>> # Applies derivative to a vector
    """
    if order == 1:
        # First derivative: (ψ_{i+1} - ψ_{i-1}) / (2dx)
        D = np.zeros((dim, dim), dtype=float)
        for i in range(dim):
            if i > 0:
                D[i, i-1] = -1/(2*dx)
            if i < dim-1:
                D[i, i+1] = 1/(2*dx)
        return D
    
    elif order == 2:
        # Second derivative: (ψ_{i+1} - 2ψ_i + ψ_{i-1}) / dx²
        D = np.zeros((dim, dim), dtype=float)
        for i in range(dim):
            D[i, i] = -2/(dx**2)
            if i > 0:
                D[i, i-1] = 1/(dx**2)
            if i < dim-1:
                D[i, i+1] = 1/(dx**2)
        return D
    
    else:
        raise ValueError(f"Order must be 1 or 2, got {order}")


# ============================================================================
# SPECIAL QUANTUM CALCULUS FUNCTIONS
# ============================================================================

def expectation_derivative(
    psi: np.ndarray,
    d_psi: np.ndarray,
    operator: np.ndarray,
    dx: float
) -> float:
    """
    Compute derivative of expectation value: d/dx ⟨ψ|O|ψ⟩
    
    Args:
        psi: Wavefunction (complex)
        d_psi: Derivative of wavefunction
        operator: Operator matrix
        dx: Grid spacing
        
    Returns:
        float: Derivative of expectation value
    """
    # Normalize wavefunction
    norm = np.sum(np.abs(psi)**2) * dx
    if norm == 0:
        return 0.0
    
    psi_norm = psi / np.sqrt(norm)
    d_psi_norm = d_psi / np.sqrt(norm)
    
    # ∂/∂x ⟨ψ|O|ψ⟩ = ⟨∂ψ|O|ψ⟩ + ⟨ψ|O|∂ψ⟩
    term1 = np.vdot(d_psi_norm, operator @ psi_norm)
    term2 = np.vdot(psi_norm, operator @ d_psi_norm)
    
    return float(np.real(term1 + term2))


def kinetic_operator(
    mass: float = 1.0,
    hbar: float = 1.0,
    dim: int = 100,
    dx: float = 0.01
) -> np.ndarray:
    """
    Create kinetic energy operator: T = -ħ²/(2m) d²/dx²
    
    Args:
        mass: Particle mass
        hbar: Reduced Planck constant
        dim: Matrix dimension
        dx: Grid spacing
        
    Returns:
        np.ndarray: Kinetic operator matrix
    """
    D2 = diff_operator(order=2, dim=dim, dx=dx)
    return -hbar**2 / (2 * mass) * D2


def potential_operator(
    potential: Callable[[float], float],
    x: np.ndarray
) -> np.ndarray:
    """
    Create potential energy operator from potential function
    
    Args:
        potential: Potential function V(x)
        x: Grid points
        
    Returns:
        np.ndarray: Diagonal potential operator matrix
    """
    return np.diag(potential(x))


def hamiltonian_1d(
    potential: Callable[[float], float],
    x: np.ndarray,
    mass: float = 1.0,
    hbar: float = 1.0
) -> np.ndarray:
    """
    Create 1D Hamiltonian operator: H = -ħ²/(2m) d²/dx² + V(x)
    
    Args:
        potential: Potential function V(x)
        x: Grid points
        mass: Particle mass
        hbar: Reduced Planck constant
        
    Returns:
        np.ndarray: Hamiltonian operator matrix
    """
    dx = x[1] - x[0]
    dim = len(x)
    
    T = kinetic_operator(mass=mass, hbar=hbar, dim=dim, dx=dx)
    V = potential_operator(potential, x)
    
    return T + V


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Derivatives
    'derivative',
    'partial_derivative',
    'gradient',
    'jacobian',
    'hessian',
    'laplacian',
    
    # Integrals
    'integral',
    'integral_2d',
    'integral_3d',
    'indefinite_integral',
    
    # Quantum derivatives
    'derivative_gaussian',
    'derivative_sine_wave',
    'derivative_cosine_wave',
    'derivative_plane_wave',
    
    # Tools
    'taylor_series',
    'diff_operator',
    
    # Special
    'expectation_derivative',
    'kinetic_operator',
    'potential_operator',
    'hamiltonian_1d',
]