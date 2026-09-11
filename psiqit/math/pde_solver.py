#psiqit/math/pde_solver.py

import numpy as np
from typing import Callable, Tuple, Optional, List, Union
from dataclasses import dataclass, field
from ..utils.logger import logger
from ..utils.validation import validate_qubits


# ============================================================================
# PDE RESULT CLASS
# ============================================================================

@dataclass
class PDEResult:
    """
    Result container for PDE solutions
    
    Attributes:
        x: Spatial grid points
        t: Time grid points
        u: Solution array (x_points x t_points)
        method: Name of the numerical method used
        success: Whether integration was successful
        error: Estimated error (if available)
        message: Additional information
    """
    x: np.ndarray
    t: np.ndarray
    u: np.ndarray
    method: str = ""
    success: bool = True
    error: Optional[float] = None
    message: str = ""
    
    def __post_init__(self):
        """Validate the result"""
        if self.u.shape != (len(self.x), len(self.t)):
            raise ValueError(f"u shape {self.u.shape} does not match x ({len(self.x)}) x t ({len(self.t)})")
    
    def get_solution_at_time(self, time_index: int) -> np.ndarray:
        """Get solution at specific time index"""
        if time_index < 0 or time_index >= len(self.t):
            raise ValueError(f"Time index {time_index} out of range [0, {len(self.t)-1}]")
        return self.u[:, time_index]
    
    def get_solution_at_position(self, position_index: int) -> np.ndarray:
        """Get solution at specific position index"""
        if position_index < 0 or position_index >= len(self.x):
            raise ValueError(f"Position index {position_index} out of range [0, {len(self.x)-1}]")
        return self.u[position_index, :]
    
    def __repr__(self) -> str:
        status = "Success" if self.success else "Failed"
        return f"PDEResult(x={len(self.x)}, t={len(self.t)}, method={self.method}, status={status})"


# ============================================================================
# HEAT EQUATION SOLVER
# ============================================================================

def solve_heat_equation(
    initial_condition: Callable[[float], float],
    boundary_left: Callable[[float], float],
    boundary_right: Callable[[float], float],
    x_range: Tuple[float, float],
    t_range: Tuple[float, float],
    n_x: int = 100,
    n_t: int = 100,
    alpha: float = 1.0,
    method: str = 'explicit'
) -> PDEResult:
    """
    Solve 1D heat equation: ∂u/∂t = α ∂²u/∂x²
    
    Args:
        initial_condition: Initial condition u(x, 0)
        boundary_left: Left boundary condition u(0, t)
        boundary_right: Right boundary condition u(L, t)
        x_range: (x_min, x_max)
        t_range: (t_min, t_max)
        n_x: Number of spatial points
        n_t: Number of time points
        alpha: Thermal diffusivity
        method: 'explicit', 'implicit', or 'crank_nicolson'
        
    Returns:
        PDEResult: Solution
        
    Example:
        >>> def ic(x): return np.sin(np.pi * x)
        >>> def bc_left(t): return 0
        >>> def bc_right(t): return 0
        >>> result = solve_heat_equation(ic, bc_left, bc_right, (0, 1), (0, 1), n_x=50, n_t=50)
    """
    x = np.linspace(x_range[0], x_range[1], n_x)
    t = np.linspace(t_range[0], t_range[1], n_t)
    
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    # Stability condition for explicit method
    r = alpha * dt / (dx ** 2)
    
    if method == 'explicit':
        if r > 0.5:
            logger.warning(f"Explicit method may be unstable: r={r:.3f} > 0.5")
        
        u = np.zeros((n_x, n_t))
        
        # Initial condition
        u[:, 0] = initial_condition(x)
        
        # Boundary conditions
        u[0, :] = boundary_left(t)
        u[-1, :] = boundary_right(t)
        
        # Explicit finite difference
        for j in range(n_t - 1):
            for i in range(1, n_x - 1):
                u[i, j + 1] = u[i, j] + r * (u[i + 1, j] - 2*u[i, j] + u[i - 1, j])
        
        logger.info(f"Heat equation solved with explicit method (r={r:.3f})")
        
        return PDEResult(
            x=x,
            t=t,
            u=u,
            method=f"Explicit (r={r:.3f})",
            success=True
        )
    
    elif method == 'implicit':
        # Build tridiagonal matrix
        A = np.zeros((n_x - 2, n_x - 2))
        for i in range(n_x - 2):
            A[i, i] = 1 + 2*r
            if i > 0:
                A[i, i-1] = -r
            if i < n_x - 3:
                A[i, i+1] = -r
        
        u = np.zeros((n_x, n_t))
        u[:, 0] = initial_condition(x)
        u[0, :] = boundary_left(t)
        u[-1, :] = boundary_right(t)
        
        # Implicit method (Crank-Nicolson)
        for j in range(n_t - 1):
            b = u[1:-1, j].copy()
            b[0] += r * u[0, j + 1]
            b[-1] += r * u[-1, j + 1]
            
            u[1:-1, j + 1] = np.linalg.solve(A, b)
        
        logger.info(f"Heat equation solved with implicit method")
        
        return PDEResult(
            x=x,
            t=t,
            u=u,
            method="Implicit",
            success=True
        )
    
    elif method == 'crank_nicolson':
        # Crank-Nicolson method (second order in time)
        r_half = r / 2
        
        # Build tridiagonal matrices
        n_inner = n_x - 2
        
        A = np.zeros((n_inner, n_inner))
        B = np.zeros((n_inner, n_inner))
        
        for i in range(n_inner):
            A[i, i] = 1 + r
            B[i, i] = 1 - r
            if i > 0:
                A[i, i-1] = -r_half
                B[i, i-1] = r_half
            if i < n_inner - 1:
                A[i, i+1] = -r_half
                B[i, i+1] = r_half
        
        u = np.zeros((n_x, n_t))
        u[:, 0] = initial_condition(x)
        u[0, :] = boundary_left(t)
        u[-1, :] = boundary_right(t)
        
        for j in range(n_t - 1):
            # Right-hand side
            b = B @ u[1:-1, j]
            
            # Add boundary contributions
            b[0] += r_half * (u[0, j] + u[0, j + 1])
            b[-1] += r_half * (u[-1, j] + u[-1, j + 1])
            
            u[1:-1, j + 1] = np.linalg.solve(A, b)
        
        logger.info(f"Heat equation solved with Crank-Nicolson method")
        
        return PDEResult(
            x=x,
            t=t,
            u=u,
            method="Crank-Nicolson",
            success=True
        )
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'explicit', 'implicit', or 'crank_nicolson'")


# ============================================================================
# WAVE EQUATION SOLVER
# ============================================================================

def solve_wave_equation(
    initial_position: Callable[[float], float],
    initial_velocity: Callable[[float], float],
    boundary_left: Callable[[float], float],
    boundary_right: Callable[[float], float],
    x_range: Tuple[float, float],
    t_range: Tuple[float, float],
    n_x: int = 100,
    n_t: int = 100,
    c: float = 1.0
) -> PDEResult:
    """
    Solve 1D wave equation: ∂²u/∂t² = c² ∂²u/∂x²
    
    Args:
        initial_position: Initial position u(x, 0)
        initial_velocity: Initial velocity ∂u/∂t(x, 0)
        boundary_left: Left boundary condition u(0, t)
        boundary_right: Right boundary condition u(L, t)
        x_range: (x_min, x_max)
        t_range: (t_min, t_max)
        n_x: Number of spatial points
        n_t: Number of time points
        c: Wave speed
        
    Returns:
        PDEResult: Solution
        
    Example:
        >>> def pos(x): return np.sin(np.pi * x)
        >>> def vel(x): return 0
        >>> result = solve_wave_equation(pos, vel, (0, 0), (0, 0), (0, 1), (0, 2), n_x=50, n_t=100)
    """
    x = np.linspace(x_range[0], x_range[1], n_x)
    t = np.linspace(t_range[0], t_range[1], n_t)
    
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    r = (c * dt / dx) ** 2
    
    # Stability condition
    if r > 1:
        logger.warning(f"CFL condition violated: r={r:.3f} > 1. Solution may be unstable.")
    
    u = np.zeros((n_x, n_t))
    
    # Initial condition: position
    u[:, 0] = initial_position(x)
    
    # Initial condition: velocity (using first time step)
    u[:, 1] = u[:, 0] + dt * initial_velocity(x)
    
    # Boundary conditions
    u[0, :] = boundary_left(t)
    u[-1, :] = boundary_right(t)
    
    # Finite difference method
    for j in range(1, n_t - 1):
        for i in range(1, n_x - 1):
            u[i, j + 1] = 2 * (1 - r) * u[i, j] - u[i, j - 1] + r * (u[i + 1, j] + u[i - 1, j])
    
    logger.info(f"Wave equation solved with CFL parameter r={r:.3f}")
    
    return PDEResult(
        x=x,
        t=t,
        u=u,
        method=f"Explicit (CFL={r:.3f})",
        success=True
    )


# ============================================================================
# SCHRÖDINGER 1D PDE SOLVER
# ============================================================================

def solve_schrodinger_1d(
    psi0: np.ndarray,
    x: np.ndarray,
    potential: Callable[[float, float], float],
    t_range: Tuple[float, float],
    n_t: int = 100,
    method: str = 'crank_nicolson',
    mass: float = 1.0,
    hbar: float = 1.0,
    boundary_type: str = 'infinite'
) -> PDEResult:
    """
    Solve 1D time-dependent Schrödinger equation
    
    iħ ∂ψ/∂t = -ħ²/(2m) ∂²ψ/∂x² + V(x,t)ψ
    
    Args:
        psi0: Initial wavefunction (complex)
        x: Spatial grid
        potential: Potential function V(x, t)
        t_range: (t_min, t_max)
        n_t: Number of time points
        method: 'crank_nicolson', 'split_step', or 'explicit'
        mass: Particle mass
        hbar: Reduced Planck constant
        boundary_type: 'infinite' (zero at boundaries) or 'periodic'
        
    Returns:
        PDEResult: Solution (complex)
        
    Example:
        >>> import numpy as np
        >>> x = np.linspace(-10, 10, 200)
        >>> psi0 = np.exp(-x**2/2) / np.sqrt(np.sqrt(np.pi))
        >>> def harmonic(x, t): return 0.5 * x**2
        >>> result = solve_schrodinger_1d(psi0, x, harmonic, (0, 5), n_t=100)
    """
    n_x = len(x)
    dx = x[1] - x[0]
    t = np.linspace(t_range[0], t_range[1], n_t)
    dt = t[1] - t[0]
    
    # Initialize wavefunction array
    psi = np.zeros((n_x, n_t), dtype=complex)
    psi[:, 0] = psi0
    
    # Normalize initial state
    norm = np.sum(np.abs(psi0)**2) * dx
    if norm == 0:
        raise ValueError("Initial wavefunction is zero")
    psi[:, 0] = psi[:, 0] / np.sqrt(norm)
    
    logger.info(f"Solving Schrödinger equation with {method} method...")
    
    if method == 'crank_nicolson':
        # Crank-Nicolson method (unitary)
        # Build tridiagonal matrices
        
        r = hbar**2 / (2 * mass * dx**2)
        gamma = 1j * dt / (4 * r)  # For Crank-Nicolson
        
        # Crank-Nicolson coefficients
        a = -r * dt / 2
        b = 1 + r * dt
        c = -r * dt / 2
        
        n_inner = n_x - 2
        
        A = np.zeros((n_inner, n_inner), dtype=complex)
        B = np.zeros((n_inner, n_inner), dtype=complex)
        
        for i in range(n_inner):
            if boundary_type == 'infinite':
                A[i, i] = 1 + r * dt
                B[i, i] = 1 - r * dt
                if i > 0:
                    A[i, i-1] = -r * dt / 2
                    B[i, i-1] = r * dt / 2
                if i < n_inner - 1:
                    A[i, i+1] = -r * dt / 2
                    B[i, i+1] = r * dt / 2
        
        for j in range(n_t - 1):
            # Add potential contribution (using trapezoidal rule)
            V_j = np.array([potential(x[i], t[j]) for i in range(n_x)])
            V_jp1 = np.array([potential(x[i], t[j + 1]) for i in range(n_x)])
            
            # Right-hand side including potential
            psi_inner = psi[1:-1, j]
            rhs = B @ psi_inner
            
            # Add potential contribution (Crank-Nicolson)
            rhs += -1j * dt / (2 * hbar) * V_j[1:-1] * psi_inner
            rhs += -1j * dt / (2 * hbar) * V_jp1[1:-1] * psi_inner
            
            # Solve tridiagonal system
            psi[1:-1, j + 1] = np.linalg.solve(A, rhs)
            
            # Apply boundary conditions
            if boundary_type == 'infinite':
                psi[0, j + 1] = 0
                psi[-1, j + 1] = 0
            # For periodic, boundaries wrap (not implemented in simple tridiagonal)
            
            # Normalize
            norm = np.sum(np.abs(psi[:, j + 1])**2) * dx
            if norm > 0:
                psi[:, j + 1] = psi[:, j + 1] / np.sqrt(norm)
    
    elif method == 'split_step':
        # Split-step Fourier method (fastest)
        # Involves FFT, so we need to use scipy if available
        try:
            from scipy.fft import fft, ifft, fftfreq
        except ImportError:
            raise ImportError("scipy.fft is required for split-step method")
        
        # k-space grid
        k = 2 * np.pi * fftfreq(n_x, dx)
        
        # Kinetic propagator in k-space
        T_prop = np.exp(-1j * hbar * k**2 * dt / (2 * mass))
        
        for j in range(n_t - 1):
            # Kinetic step (in k-space)
            psi_k = fft(psi[:, j])
            psi_k = psi_k * T_prop
            psi_new = ifft(psi_k)
            
            # Potential step (in x-space)
            V = np.array([potential(x[i], t[j] + dt/2) for i in range(n_x)])
            psi_new = np.exp(-1j * V * dt / hbar) * psi_new
            
            # Normalize
            norm = np.sum(np.abs(psi_new)**2) * dx
            if norm > 0:
                psi_new = psi_new / np.sqrt(norm)
            
            psi[:, j + 1] = psi_new
    
    elif method == 'explicit':
        # Explicit finite difference (simplest, but not unitary)
        r = hbar / (2 * mass * dx**2)
        
        for j in range(n_t - 1):
            V = np.array([potential(x[i], t[j]) for i in range(n_x)])
            
            # Explicit time evolution
            for i in range(1, n_x - 1):
                psi[i, j + 1] = psi[i, j] - 1j * dt / hbar * (
                    V[i] * psi[i, j] - r * (psi[i+1, j] - 2*psi[i, j] + psi[i-1, j])
                )
            
            # Boundary conditions
            if boundary_type == 'infinite':
                psi[0, j + 1] = 0
                psi[-1, j + 1] = 0
            
            # Normalize
            norm = np.sum(np.abs(psi[:, j + 1])**2) * dx
            if norm > 0:
                psi[:, j + 1] = psi[:, j + 1] / np.sqrt(norm)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'crank_nicolson', 'split_step', or 'explicit'")
    
    logger.info(f"Schrödinger equation solved with {method} method")
    
    return PDEResult(
        x=x,
        t=t,
        u=psi,
        method=f"Schrödinger ({method})",
        success=True
    )


# ============================================================================
# DIFFUSION EQUATION (IMPLICIT)
# ============================================================================

def diffusion_equation_implicit(
    initial_condition: Callable[[float], float],
    boundary_left: Callable[[float], float],
    boundary_right: Callable[[float], float],
    x_range: Tuple[float, float],
    t_range: Tuple[float, float],
    n_x: int = 100,
    n_t: int = 100,
    D: float = 1.0
) -> PDEResult:
    """
    Solve 1D diffusion equation with implicit method (unconditionally stable)
    
    ∂u/∂t = D ∂²u/∂x²
    
    Args:
        initial_condition: Initial condition u(x, 0)
        boundary_left: Left boundary condition u(0, t)
        boundary_right: Right boundary condition u(L, t)
        x_range: (x_min, x_max)
        t_range: (t_min, t_max)
        n_x: Number of spatial points
        n_t: Number of time points
        D: Diffusion coefficient
        
    Returns:
        PDEResult: Solution
        
    Example:
        >>> def ic(x): return np.exp(-x**2/2)
        >>> def bc_left(t): return 0
        >>> def bc_right(t): return 0
        >>> result = diffusion_equation_implicit(ic, bc_left, bc_right, (-5, 5), (0, 1))
    """
    x = np.linspace(x_range[0], x_range[1], n_x)
    t = np.linspace(t_range[0], t_range[1], n_t)
    
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    r = D * dt / (dx ** 2)
    
    logger.info(f"Solving diffusion equation with implicit method (r={r:.3f})")
    
    # Build tridiagonal matrix for implicit scheme
    n_inner = n_x - 2
    A = np.zeros((n_inner, n_inner))
    
    for i in range(n_inner):
        A[i, i] = 1 + 2 * r
        if i > 0:
            A[i, i-1] = -r
        if i < n_inner - 1:
            A[i, i+1] = -r
    
    u = np.zeros((n_x, n_t))
    u[:, 0] = initial_condition(x)
    u[0, :] = boundary_left(t)
    u[-1, :] = boundary_right(t)
    
    # Implicit time stepping
    for j in range(n_t - 1):
        # Right-hand side (explicit part)
        b = u[1:-1, j].copy()
        
        # Boundary contributions
        b[0] += r * u[0, j + 1]
        b[-1] += r * u[-1, j + 1]
        
        # Solve tridiagonal system
        u[1:-1, j + 1] = np.linalg.solve(A, b)
    
    logger.info(f"Diffusion equation solved with implicit method")
    
    return PDEResult(
        x=x,
        t=t,
        u=u,
        method=f"Implicit (r={r:.3f})",
        success=True
    )


# ============================================================================
# ADDITIONAL PDE SOLVERS
# ============================================================================

def solve_poisson_equation(
    source: Callable[[float, float], float],
    boundary: Callable[[float, float], float],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    n_x: int = 50,
    n_y: int = 50,
    tol: float = 1e-6,
    max_iter: int = 10000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve 2D Poisson equation: ∇²u = f(x,y) with boundary conditions
    
    Args:
        source: Source function f(x, y)
        boundary: Boundary condition function u(x, y)
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)
        n_x: Number of points in x direction
        n_y: Number of points in y direction
        tol: Convergence tolerance
        max_iter: Maximum iterations
        
    Returns:
        Tuple: (x, y, u)
        
    Example:
        >>> def f(x, y): return np.exp(-x**2 - y**2)
        >>> def b(x, y): return 0
        >>> x, y, u = solve_poisson_equation(f, b, (-1, 1), (-1, 1))
    """
    x = np.linspace(x_range[0], x_range[1], n_x)
    y = np.linspace(y_range[0], y_range[1], n_y)
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # Initialize solution
    u = np.zeros((n_x, n_y))
    
    # Set boundary conditions
    X, Y = np.meshgrid(x, y)
    for i in range(n_x):
        for j in range(n_y):
            if i == 0 or i == n_x - 1 or j == 0 or j == n_y - 1:
                u[i, j] = boundary(x[i], y[j])
    
    # Source function
    f = source(X, Y)
    
    # Gauss-Seidel iteration
    for iteration in range(max_iter):
        u_old = u.copy()
        
        for i in range(1, n_x - 1):
            for j in range(1, n_y - 1):
                u[i, j] = (1 / (2 * (1/dx**2 + 1/dy**2))) * (
                    (u[i+1, j] + u[i-1, j]) / dx**2 +
                    (u[i, j+1] + u[i, j-1]) / dy**2 -
                    f[i, j]
                )
        
        # Check convergence
        if np.max(np.abs(u - u_old)) < tol:
            logger.info(f"Poisson equation converged in {iteration + 1} iterations")
            return x, y, u
    
    logger.warning(f"Poisson equation did not converge in {max_iter} iterations")
    return x, y, u


def solve_helmholtz_equation(
    source: Callable[[float, float], float],
    k: float,
    boundary: Callable[[float, float], float],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    n_x: int = 50,
    n_y: int = 50,
    tol: float = 1e-6,
    max_iter: int = 10000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve 2D Helmholtz equation: (∇² + k²)u = f(x,y)
    
    Args:
        source: Source function f(x, y)
        k: Wave number
        boundary: Boundary condition function u(x, y)
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)
        n_x: Number of points in x direction
        n_y: Number of points in y direction
        tol: Convergence tolerance
        max_iter: Maximum iterations
        
    Returns:
        Tuple: (x, y, u)
    """
    x = np.linspace(x_range[0], x_range[1], n_x)
    y = np.linspace(y_range[0], y_range[1], n_y)
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # Initialize solution
    u = np.zeros((n_x, n_y))
    
    # Set boundary conditions
    X, Y = np.meshgrid(x, y)
    for i in range(n_x):
        for j in range(n_y):
            if i == 0 or i == n_x - 1 or j == 0 or j == n_y - 1:
                u[i, j] = boundary(x[i], y[j])
    
    # Source function
    f = source(X, Y)
    
    # Gauss-Seidel iteration with relaxation
    for iteration in range(max_iter):
        u_old = u.copy()
        
        for i in range(1, n_x - 1):
            for j in range(1, n_y - 1):
                u[i, j] = (1 / (2 * (1/dx**2 + 1/dy**2) - k**2)) * (
                    (u[i+1, j] + u[i-1, j]) / dx**2 +
                    (u[i, j+1] + u[i, j-1]) / dy**2 -
                    f[i, j]
                )
        
        # Check convergence
        if np.max(np.abs(u - u_old)) < tol:
            logger.info(f"Helmholtz equation converged in {iteration + 1} iterations")
            return x, y, u
    
    logger.warning(f"Helmholtz equation did not converge in {max_iter} iterations")
    return x, y, u


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Result class
    'PDEResult',
    
    # Main PDE solvers
    'solve_heat_equation',
    'solve_wave_equation',
    'solve_schrodinger_1d',
    'diffusion_equation_implicit',
    
    # Additional PDE solvers
    'solve_poisson_equation',
    'solve_helmholtz_equation',
]