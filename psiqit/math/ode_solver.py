#psiqit/math/ode_solver.py

import numpy as np
from typing import Callable, List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from ..utils.logger import logger
from ..utils.validation import validate_qubits


# ============================================================================
# ODE RESULT CLASS
# ============================================================================

@dataclass
class ODEResult:
    """
    Result container for ODE solutions
    
    Attributes:
        x: Array of time/space points
        y: Array of solution values (for single equation)
        y_list: List of solution arrays (for systems)
        method: Name of the numerical method used
        n_steps: Number of integration steps
        success: Whether integration was successful
        error: Estimated error (for adaptive methods)
        message: Additional information
    """
    x: np.ndarray
    y: Optional[np.ndarray] = None
    y_list: Optional[List[np.ndarray]] = None
    method: str = ""
    n_steps: int = 0
    success: bool = True
    error: Optional[float] = None
    message: str = ""
    
    def __post_init__(self):
        """Validate and set up the result"""
        if self.y is None and self.y_list is None:
            raise ValueError("Either y or y_list must be provided")
        
        if self.y is not None and self.y_list is not None:
            raise ValueError("Provide either y or y_list, not both")
    
    def get_solution(self) -> Union[np.ndarray, List[np.ndarray]]:
        """Get the solution (y or y_list)"""
        return self.y if self.y is not None else self.y_list
    
    def get_last_value(self) -> Union[float, np.ndarray]:
        """Get the last value of the solution"""
        if self.y is not None:
            return self.y[-1]
        return [y_arr[-1] for y_arr in self.y_list]
    
    def __repr__(self) -> str:
        status = "Success" if self.success else "Failed"
        return f"ODEResult(method={self.method}, steps={self.n_steps}, status={status})"


# ============================================================================
# EULER METHODS
# ============================================================================

def euler(
    f: Callable[[float, float], float],
    x0: float,
    y0: float,
    x_end: float,
    n_steps: int = 100
) -> ODEResult:
    """
    Euler method for single ODE: dy/dx = f(x, y)
    
    Args:
        f: Function f(x, y)
        x0: Initial x value
        y0: Initial y value
        x_end: Final x value
        n_steps: Number of integration steps
        
    Returns:
        ODEResult: Solution
        
    Example:
        >>> def f(x, y): return y
        >>> result = euler(f, 0, 1, 1, n_steps=10)
        >>> print(result.y[-1])  # Approx e^1
    """
    x = np.linspace(x0, x_end, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0
    
    dx = (x_end - x0) / n_steps
    
    for i in range(n_steps):
        y[i + 1] = y[i] + dx * f(x[i], y[i])
    
    logger.info(f"Euler method completed with {n_steps} steps")
    
    return ODEResult(
        x=x,
        y=y,
        method="Euler",
        n_steps=n_steps,
        success=True
    )


def euler_system(
    f: Callable[[float, List[float]], List[float]],
    x0: float,
    y0: List[float],
    x_end: float,
    n_steps: int = 100
) -> ODEResult:
    """
    Euler method for system of ODEs: dy/dx = f(x, y)
    
    Args:
        f: Function f(x, y) returning list of derivatives
        x0: Initial x value
        y0: Initial y vector
        x_end: Final x value
        n_steps: Number of integration steps
        
    Returns:
        ODEResult: Solution
        
    Example:
        >>> def f(x, y): return [y[1], -y[0]]  # Simple harmonic oscillator
        >>> result = euler_system(f, 0, [1, 0], 10, n_steps=100)
    """
    n_equations = len(y0)
    x = np.linspace(x0, x_end, n_steps + 1)
    y_list = [np.zeros(n_steps + 1) for _ in range(n_equations)]
    
    for i in range(n_equations):
        y_list[i][0] = y0[i]
    
    dx = (x_end - x0) / n_steps
    y_current = y0.copy()
    
    for i in range(n_steps):
        derivatives = f(x[i], y_current)
        
        for j in range(n_equations):
            y_list[j][i + 1] = y_list[j][i] + dx * derivatives[j]
        
        y_current = [y_list[j][i + 1] for j in range(n_equations)]
    
    logger.info(f"Euler system completed with {n_steps} steps")
    
    return ODEResult(
        x=x,
        y_list=y_list,
        method="Euler (System)",
        n_steps=n_steps,
        success=True
    )


# ============================================================================
# RUNGE-KUTTA METHODS
# ============================================================================

def rk2(
    f: Callable[[float, float], float],
    x0: float,
    y0: float,
    x_end: float,
    n_steps: int = 100
) -> ODEResult:
    """
    Runge-Kutta 2nd order (Midpoint method) for single ODE
    
    Args:
        f: Function f(x, y)
        x0: Initial x value
        y0: Initial y value
        x_end: Final x value
        n_steps: Number of integration steps
        
    Returns:
        ODEResult: Solution
    """
    x = np.linspace(x0, x_end, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0
    
    dx = (x_end - x0) / n_steps
    dx_half = dx / 2
    
    for i in range(n_steps):
        x_i = x[i]
        y_i = y[i]
        
        # Midpoint method
        k1 = f(x_i, y_i)
        k2 = f(x_i + dx_half, y_i + dx_half * k1)
        
        y[i + 1] = y[i] + dx * k2
    
    logger.info(f"RK2 method completed with {n_steps} steps")
    
    return ODEResult(
        x=x,
        y=y,
        method="RK2 (Midpoint)",
        n_steps=n_steps,
        success=True
    )


def rk4(
    f: Callable[[float, float], float],
    x0: float,
    y0: float,
    x_end: float,
    n_steps: int = 100
) -> ODEResult:
    """
    Runge-Kutta 4th order (Classical RK4) for single ODE
    
    Args:
        f: Function f(x, y)
        x0: Initial x value
        y0: Initial y value
        x_end: Final x value
        n_steps: Number of integration steps
        
    Returns:
        ODEResult: Solution
        
    Example:
        >>> def f(x, y): return y
        >>> result = rk4(f, 0, 1, 1, n_steps=10)
        >>> print(result.y[-1])  # More accurate approximation of e^1
    """
    x = np.linspace(x0, x_end, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0
    
    dx = (x_end - x0) / n_steps
    
    for i in range(n_steps):
        x_i = x[i]
        y_i = y[i]
        
        k1 = f(x_i, y_i)
        k2 = f(x_i + dx/2, y_i + dx/2 * k1)
        k3 = f(x_i + dx/2, y_i + dx/2 * k2)
        k4 = f(x_i + dx, y_i + dx * k3)
        
        y[i + 1] = y[i] + dx/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    logger.info(f"RK4 method completed with {n_steps} steps")
    
    return ODEResult(
        x=x,
        y=y,
        method="RK4 (Classical)",
        n_steps=n_steps,
        success=True
    )


def rk4_system(
    f: Callable[[float, List[float]], List[float]],
    x0: float,
    y0: List[float],
    x_end: float,
    n_steps: int = 100
) -> ODEResult:
    """
    Runge-Kutta 4th order for system of ODEs: dy/dx = f(x, y)
    
    Args:
        f: Function f(x, y) returning list of derivatives
        x0: Initial x value
        y0: Initial y vector
        x_end: Final x value
        n_steps: Number of integration steps
        
    Returns:
        ODEResult: Solution
        
    Example:
        >>> def harmonic(x, y): return [y[1], -y[0]]
        >>> result = rk4_system(harmonic, 0, [1, 0], 10, n_steps=100)
    """
    n_equations = len(y0)
    x = np.linspace(x0, x_end, n_steps + 1)
    y_list = [np.zeros(n_steps + 1) for _ in range(n_equations)]
    
    for i in range(n_equations):
        y_list[i][0] = y0[i]
    
    dx = (x_end - x0) / n_steps
    y_current = y0.copy()
    
    for i in range(n_steps):
        x_i = x[i]
        
        # k1
        k1 = f(x_i, y_current)
        
        # k2
        y_temp = [y_current[j] + dx/2 * k1[j] for j in range(n_equations)]
        k2 = f(x_i + dx/2, y_temp)
        
        # k3
        y_temp = [y_current[j] + dx/2 * k2[j] for j in range(n_equations)]
        k3 = f(x_i + dx/2, y_temp)
        
        # k4
        y_temp = [y_current[j] + dx * k3[j] for j in range(n_equations)]
        k4 = f(x_i + dx, y_temp)
        
        # Update
        for j in range(n_equations):
            y_list[j][i + 1] = y_current[j] + dx/6 * (k1[j] + 2*k2[j] + 2*k3[j] + k4[j])
        
        y_current = [y_list[j][i + 1] for j in range(n_equations)]
    
    logger.info(f"RK4 system completed with {n_steps} steps")
    
    return ODEResult(
        x=x,
        y_list=y_list,
        method="RK4 (System)",
        n_steps=n_steps,
        success=True
    )


# ============================================================================
# GENERAL ODE SOLVER
# ============================================================================

def solve_ode(
    f: Callable,
    x0: float,
    y0: Union[float, List[float]],
    x_end: float,
    method: str = 'rk4',
    n_steps: int = 100,
    **kwargs
) -> ODEResult:
    """
    General ODE solver with multiple methods
    
    Args:
        f: Function f(x, y)
        x0: Initial x value
        y0: Initial y value (float or list)
        x_end: Final x value
        method: 'euler', 'euler_system', 'rk2', 'rk4', 'rk4_system'
        n_steps: Number of integration steps
        **kwargs: Additional arguments for specific methods
        
    Returns:
        ODEResult: Solution
        
    Example:
        >>> def f(x, y): return y
        >>> result = solve_ode(f, 0, 1, 1, method='rk4', n_steps=10)
    """
    # Determine if system or single equation
    is_system = isinstance(y0, list)
    
    method_map = {
        'euler': euler,
        'euler_system': euler_system,
        'rk2': rk2,
        'rk4': rk4,
        'rk4_system': rk4_system,
    }
    
    # Auto-select method if needed
    if method not in method_map:
        raise ValueError(f"Unknown method: {method}. Available: {list(method_map.keys())}")
    
    if is_system:
        if method in ['euler', 'rk2', 'rk4']:
            method = f"{method}_system"
    
    solver = method_map[method]
    
    if is_system:
        return solver(f, x0, y0, x_end, n_steps, **kwargs)
    else:
        return solver(f, x0, y0, x_end, n_steps, **kwargs)


# ============================================================================
# ADAPTIVE RK45 (Runge-Kutta-Fehlberg)
# ============================================================================

def adaptive_rk45(
    f: Callable[[float, float], float],
    x0: float,
    y0: float,
    x_end: float,
    tol: float = 1e-6,
    h_init: float = 0.01,
    h_min: float = 1e-8,
    h_max: float = 0.1,
    max_steps: int = 10000
) -> ODEResult:
    """
    Adaptive Runge-Kutta-Fehlberg (RK45) method
    
    Args:
        f: Function f(x, y)
        x0: Initial x value
        y0: Initial y value
        x_end: Final x value
        tol: Error tolerance
        h_init: Initial step size
        h_min: Minimum step size
        h_max: Maximum step size
        max_steps: Maximum number of steps
        
    Returns:
        ODEResult: Solution with estimated error
        
    Example:
        >>> def f(x, y): return y
        >>> result = adaptive_rk45(f, 0, 1, 1, tol=1e-8)
    """
    x_vals = [x0]
    y_vals = [y0]
    
    x = x0
    y = y0
    h = h_init
    
    step_count = 0
    
    while x < x_end and step_count < max_steps:
        # Make sure we don't overshoot
        if x + h > x_end:
            h = x_end - x
        
        # RK4 coefficients
        k1 = f(x, y)
        k2 = f(x + h/4, y + h/4 * k1)
        k3 = f(x + 3*h/8, y + 3*h/32 * k1 + 9*h/32 * k2)
        k4 = f(x + 12*h/13, y + 1932*h/2197 * k1 - 7200*h/2197 * k2 + 7296*h/2197 * k3)
        k5 = f(x + h, y + 439*h/216 * k1 - 8*h * k2 + 3680*h/513 * k3 - 845*h/4104 * k4)
        k6 = f(x + h/2, y - 8*h/27 * k1 + 2*h * k2 - 3544*h/2565 * k3 + 1859*h/4104 * k4 - 11*h/40 * k5)
        
        # RK4 approximation (4th order)
        y_rk4 = y + h * (25/216 * k1 + 1408/2565 * k3 + 2197/4104 * k4 - 1/5 * k5)
        
        # RK5 approximation (5th order - more accurate)
        y_rk5 = y + h * (16/135 * k1 + 6656/12825 * k3 + 28561/56430 * k4 - 9/50 * k5 + 2/55 * k6)
        
        # Estimate error
        error = abs(y_rk5 - y_rk4)
        
        # Accept or reject step
        if error <= tol:
            x = x + h
            y = y_rk5
            x_vals.append(x)
            y_vals.append(y)
            step_count += 1
            logger.debug(f"Step {step_count}: x={x:.6f}, h={h:.6f}, error={error:.2e}")
        
        # Adjust step size
        if error == 0:
            h_new = h * 2
        else:
            h_new = 0.84 * h * (tol / error) ** 0.2
        
        # Bound step size
        h_new = max(h_min, min(h_new, h_max))
        h = h_new
        
        # Prevent infinite loop
        if h < h_min:
            logger.warning(f"Step size too small: {h} < {h_min}")
            break
    
    logger.info(f"Adaptive RK45 completed with {step_count} steps")
    
    return ODEResult(
        x=np.array(x_vals),
        y=np.array(y_vals),
        method="Adaptive RK45 (Fehlberg)",
        n_steps=step_count,
        success=step_count < max_steps,
        error=error if 'error' in locals() else None,
        message="Integration completed" if step_count < max_steps else "Max steps reached"
    )


# ============================================================================
# NUMERICAL SCHRÖDINGER 1D
# ============================================================================

def schrodinger_1d_numerical(
    potential: Callable[[float, float], float],
    psi0: np.ndarray,
    x: np.ndarray,
    t_max: float,
    dt: float = 0.001,
    method: str = 'rk4',
    mass: float = 1.0,
    hbar: float = 1.0
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Numerical solution of 1D Schrödinger equation
    
    iħ ∂ψ/∂t = -ħ²/(2m) ∂²ψ/∂x² + V(x)ψ
    
    Args:
        potential: Potential function V(x, t)
        psi0: Initial wavefunction
        x: Spatial grid
        t_max: Maximum time
        dt: Time step
        method: Integration method ('euler', 'rk4')
        mass: Particle mass
        hbar: Reduced Planck constant
        
    Returns:
        Tuple: (times, wavefunctions)
        
    Example:
        >>> import numpy as np
        >>> x = np.linspace(-10, 10, 200)
        >>> psi0 = np.exp(-x**2/2) / np.sqrt(np.sqrt(np.pi))
        >>> def harmonic(x, t): return 0.5 * x**2
        >>> times, psi = schrodinger_1d_numerical(harmonic, psi0, x, 5, dt=0.01)
    """
    dx = x[1] - x[0]
    n_points = len(x)
    
    # Create kinetic operator matrix (finite difference)
    from .calculus import kinetic_operator, potential_operator
    
    T = kinetic_operator(mass=mass, hbar=hbar, dim=n_points, dx=dx)
    
    # Normalize initial state
    norm = np.sum(np.abs(psi0)**2) * dx
    if norm == 0:
        raise ValueError("Initial wavefunction is zero")
    psi = psi0 / np.sqrt(norm)
    
    # Time evolution
    times = np.arange(0, t_max + dt, dt)
    wavefunctions = [psi.copy()]
    
    # Define time derivative function for ODE solver
    def schrodinger_derivative(t: float, psi_vec: np.ndarray) -> np.ndarray:
        """Compute dψ/dt = -i/ħ H ψ"""
        V = potential_operator(lambda x: potential(x, t), x)
        H = T + V
        return -1j / hbar * H @ psi_vec
    
    # Integrate using RK4
    if method == 'rk4':
        for i in range(len(times) - 1):
            t = times[i]
            psi_current = wavefunctions[-1]
            dt_current = times[i+1] - times[i]
            
            # RK4 for complex vector
            k1 = schrodinger_derivative(t, psi_current)
            k2 = schrodinger_derivative(t + dt_current/2, psi_current + dt_current/2 * k1)
            k3 = schrodinger_derivative(t + dt_current/2, psi_current + dt_current/2 * k2)
            k4 = schrodinger_derivative(t + dt_current, psi_current + dt_current * k3)
            
            psi_new = psi_current + dt_current/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            # Normalize
            norm = np.sum(np.abs(psi_new)**2) * dx
            psi_new = psi_new / np.sqrt(norm)
            
            wavefunctions.append(psi_new)
    
    elif method == 'euler':
        for i in range(len(times) - 1):
            t = times[i]
            psi_current = wavefunctions[-1]
            dt_current = times[i+1] - times[i]
            
            psi_new = psi_current + dt_current * schrodinger_derivative(t, psi_current)
            
            # Normalize
            norm = np.sum(np.abs(psi_new)**2) * dx
            psi_new = psi_new / np.sqrt(norm)
            
            wavefunctions.append(psi_new)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'euler' or 'rk4'")
    
    logger.info(f"Schrödinger 1D simulation completed: {len(times)} time steps")
    
    return np.array(times), wavefunctions


# ============================================================================
# SPECIAL ODE SOLVERS FOR QUANTUM SYSTEMS
# ============================================================================

def schrodinger_1d_stationary(
    potential: Callable[[float], float],
    x: np.ndarray,
    n_states: int = 5,
    mass: float = 1.0,
    hbar: float = 1.0
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Solve stationary 1D Schrödinger equation
    
    -ħ²/(2m) d²ψ/dx² + V(x)ψ = Eψ
    
    Args:
        potential: Time-independent potential V(x)
        x: Spatial grid
        n_states: Number of eigenstates to compute
        mass: Particle mass
        hbar: Reduced Planck constant
        
    Returns:
        Tuple: (eigenvalues, eigenstates)
        
    Example:
        >>> x = np.linspace(-10, 10, 200)
        >>> def harmonic(x): return 0.5 * x**2
        >>> energies, states = schrodinger_1d_stationary(harmonic, x, n_states=5)
    """
    from .calculus import hamiltonian_1d
    
    # Build Hamiltonian
    H = hamiltonian_1d(potential, x, mass=mass, hbar=hbar)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    
    # Normalize eigenvectors
    dx = x[1] - x[0]
    for i in range(len(eigenvectors)):
        norm = np.sum(np.abs(eigenvectors[:, i])**2) * dx
        if norm > 0:
            eigenvectors[:, i] = eigenvectors[:, i] / np.sqrt(norm)
    
    logger.info(f"Found {min(n_states, len(eigenvalues))} eigenstates")
    
    return eigenvalues[:n_states], [eigenvectors[:, i] for i in range(min(n_states, len(eigenvectors)))]


def lindblad_solver(
    rho0: np.ndarray,
    H: np.ndarray,
    collapse_ops: List[np.ndarray],
    t_max: float,
    dt: float = 0.001,
    method: str = 'rk4'
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Solve Lindblad master equation
    
    dρ/dt = -i/ħ [H, ρ] + Σ_k (L_k ρ L_k† - 1/2{L_k†L_k, ρ})
    
    Args:
        rho0: Initial density matrix
        H: Hamiltonian
        collapse_ops: List of collapse operators
        t_max: Maximum time
        dt: Time step
        method: Integration method ('euler', 'rk4')
        
    Returns:
        Tuple: (times, density_matrices)
    """
    n = len(rho0)
    times = np.arange(0, t_max + dt, dt)
    matrices = [rho0.copy()]
    
    # Compute Lindblad superoperator
    def lindblad_superoperator(rho: np.ndarray) -> np.ndarray:
        # Unitary part: -i[H, ρ]
        result = -1j * (H @ rho - rho @ H)
        
        # Dissipative part
        for L in collapse_ops:
            L_dag = L.conj().T
            result += L @ rho @ L_dag - 0.5 * (L_dag @ L @ rho + rho @ L_dag @ L)
        
        return result
    
    # Flatten matrix for ODE solver
    def flattened_derivative(t: float, rho_flat: np.ndarray) -> np.ndarray:
        rho = rho_flat.reshape(n, n)
        drho = lindblad_superoperator(rho)
        return drho.flatten()
    
    for i in range(len(times) - 1):
        t = times[i]
        rho_current = matrices[-1].flatten()
        dt_current = times[i+1] - times[i]
        
        if method == 'rk4':
            k1 = flattened_derivative(t, rho_current)
            k2 = flattened_derivative(t + dt_current/2, rho_current + dt_current/2 * k1)
            k3 = flattened_derivative(t + dt_current/2, rho_current + dt_current/2 * k2)
            k4 = flattened_derivative(t + dt_current, rho_current + dt_current * k3)
            
            rho_new = rho_current + dt_current/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        elif method == 'euler':
            rho_new = rho_current + dt_current * flattened_derivative(t, rho_current)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Reshape and ensure Hermitian
        rho_new = rho_new.reshape(n, n)
        rho_new = (rho_new + rho_new.conj().T) / 2
        
        # Normalize trace
        trace = np.trace(rho_new).real
        if trace > 0:
            rho_new = rho_new / trace
        
        matrices.append(rho_new)
    
    logger.info(f"Lindblad simulation completed: {len(times)} time steps")
    
    return times, matrices


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Result class
    'ODEResult',
    
    # Euler methods
    'euler',
    'euler_system',
    
    # Runge-Kutta methods
    'rk2',
    'rk4',
    'rk4_system',
    
    # General solver
    'solve_ode',
    
    # Adaptive methods
    'adaptive_rk45',
    
    # Quantum-specific
    'schrodinger_1d_numerical',
    'schrodinger_1d_stationary',
    'lindblad_solver',
]