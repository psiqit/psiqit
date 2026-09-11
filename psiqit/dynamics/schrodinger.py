# psiqit/dynamics/schrodinger.py


import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from ..quantum.state import Ket
from ..quantum.operator import Operator
from ..math.qalgebra import PI, SQRT2, SQRT_PI
from ..math.calculus import derivative, integral, gradient, laplacian
from ..utils.logger import logger
from ..utils.validation import validate_qubits


# ============================================================================
# WAVEFUNCTION CLASS
# ============================================================================

class WaveFunction:
    """
    Wavefunction representation for quantum mechanics
    
    Represents a wavefunction ψ(x, t) with methods for computing
    expectation values, uncertainties, and probability densities.
    
    Example:
        >>> import numpy as np
        >>> x = np.linspace(-5, 5, 100)
        >>> psi = np.exp(-x**2/2) / np.sqrt(np.sqrt(np.pi))
        >>> wf = WaveFunction(x, psi)
        >>> print(wf.expectation_x())  # 0.0
        >>> print(wf.uncertainty_x())  # 0.707
    """
    
    def __init__(self, x: np.ndarray, psi: np.ndarray, t: float = 0.0):
        """
        Initialize a wavefunction
        
        Args:
            x: Spatial grid
            psi: Wavefunction values (complex)
            t: Time (default: 0.0)
        """
        self.x = np.array(x, dtype=float)
        self.psi = np.array(psi, dtype=complex)
        self.t = t
        self.dx = self.x[1] - self.x[0] if len(self.x) > 1 else 0.01
        
        # Ensure normalization
        self.normalize()
        
        logger.debug(f"WaveFunction initialized: {len(x)} points, t={t:.3f}")
    
    def probability_density(self) -> np.ndarray:
        """
        Compute the probability density |ψ(x)|²
        
        Returns:
            np.ndarray: Probability density
        """
        return np.abs(self.psi) ** 2
    
    def normalize(self) -> 'WaveFunction':
        """
        Normalize the wavefunction so that ∫|ψ|² dx = 1
        
        Returns:
            WaveFunction: Self for chaining
        """
        norm = np.sqrt(np.sum(np.abs(self.psi) ** 2) * self.dx)
        if norm > 0:
            self.psi = self.psi / norm
        return self
    
    def expectation_x(self) -> float:
        """
        Compute expectation value of position ⟨x⟩
        
        Returns:
            float: ⟨x⟩
        """
        prob = self.probability_density()
        return float(np.sum(self.x * prob) * self.dx)
    
    def expectation_p(self, hbar: float = 1.0) -> float:
        """
        Compute expectation value of momentum ⟨p⟩
        
        ⟨p⟩ = -iħ ∫ ψ* ∂ψ/∂x dx
        
        Args:
            hbar: Reduced Planck constant
            
        Returns:
            float: ⟨p⟩
        """
        # Compute derivative using finite differences
        dpsi = np.gradient(self.psi, self.dx)
        
        # ⟨p⟩ = -iħ ∫ ψ* ∂ψ/∂x dx
        integrand = np.conj(self.psi) * dpsi
        return float(-1j * hbar * np.sum(integrand) * self.dx).real
    
    def expectation_x2(self) -> float:
        """
        Compute expectation value of x² ⟨x²⟩
        
        Returns:
            float: ⟨x²⟩
        """
        prob = self.probability_density()
        return float(np.sum(self.x**2 * prob) * self.dx)
    
    def expectation_p2(self, hbar: float = 1.0) -> float:
        """
        Compute expectation value of p² ⟨p²⟩
        
        ⟨p²⟩ = -ħ² ∫ ψ* ∂²ψ/∂x² dx
        
        Args:
            hbar: Reduced Planck constant
            
        Returns:
            float: ⟨p²⟩
        """
        # Compute second derivative
        d2psi = np.gradient(np.gradient(self.psi, self.dx), self.dx)
        
        # ⟨p²⟩ = -ħ² ∫ ψ* ∂²ψ/∂x² dx
        integrand = np.conj(self.psi) * d2psi
        return float(-hbar**2 * np.sum(integrand) * self.dx).real
    
    def uncertainty_x(self) -> float:
        """
        Compute uncertainty in position Δx = √(⟨x²⟩ - ⟨x⟩²)
        
        Returns:
            float: Δx
        """
        ex = self.expectation_x()
        ex2 = self.expectation_x2()
        return np.sqrt(max(0, ex2 - ex**2))
    
    def uncertainty_p(self, hbar: float = 1.0) -> float:
        """
        Compute uncertainty in momentum Δp = √(⟨p²⟩ - ⟨p⟩²)
        
        Args:
            hbar: Reduced Planck constant
            
        Returns:
            float: Δp
        """
        ep = self.expectation_p(hbar)
        ep2 = self.expectation_p2(hbar)
        return np.sqrt(max(0, ep2 - ep**2))
    
    def uncertainty_product(self, hbar: float = 1.0) -> float:
        """
        Compute the product of uncertainties Δx * Δp
        
        Args:
            hbar: Reduced Planck constant
            
        Returns:
            float: Δx * Δp (should be >= ħ/2)
        """
        dx = self.uncertainty_x()
        dp = self.uncertainty_p(hbar)
        return dx * dp
    
    def kinetic_energy(self, mass: float = 1.0, hbar: float = 1.0) -> float:
        """
        Compute kinetic energy ⟨p²⟩/(2m)
        
        Args:
            mass: Particle mass
            hbar: Reduced Planck constant
            
        Returns:
            float: Kinetic energy
        """
        ep2 = self.expectation_p2(hbar)
        return ep2 / (2 * mass)
    
    def potential_energy(self, potential: Callable[[float], float]) -> float:
        """
        Compute potential energy ⟨V(x)⟩
        
        Args:
            potential: Potential function V(x)
            
        Returns:
            float: Potential energy
        """
        prob = self.probability_density()
        V = potential(self.x)
        return float(np.sum(V * prob) * self.dx)
    
    def total_energy(self, potential: Callable[[float], float],
                     mass: float = 1.0, hbar: float = 1.0) -> float:
        """
        Compute total energy ⟨H⟩ = ⟨p²⟩/(2m) + ⟨V(x)⟩
        
        Args:
            potential: Potential function V(x)
            mass: Particle mass
            hbar: Reduced Planck constant
            
        Returns:
            float: Total energy
        """
        return self.kinetic_energy(mass, hbar) + self.potential_energy(potential)
    
    def expectation_value(self, operator: Callable[[np.ndarray], np.ndarray]) -> float:
        """
        Compute expectation value of a general operator
        
        ⟨O⟩ = ∫ ψ* O ψ dx
        
        Args:
            operator: Operator function acting on wavefunction
            
        Returns:
            float: Expectation value
        """
        O_psi = operator(self.psi)
        return float(np.sum(np.conj(self.psi) * O_psi) * self.dx).real
    
    def copy(self) -> 'WaveFunction':
        """Create a copy of the wavefunction"""
        return WaveFunction(self.x.copy(), self.psi.copy(), self.t)
    
    def __repr__(self) -> str:
        return f"WaveFunction(n_points={len(self.x)}, t={self.t:.3f})"
    
    def __str__(self) -> str:
        return f"WaveFunction at t={self.t:.3f}, norm={np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx):.6f}"


# ============================================================================
# TIME-INDEPENDENT SCHRÖDINGER EQUATION
# ============================================================================

def solve_time_independent(
    potential: Callable[[float], float],
    x_range: Tuple[float, float],
    n_points: int = 1000,
    n_states: int = 5,
    mass: float = 1.0,
    hbar: float = 1.0
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Solve the time-independent Schrödinger equation numerically
    
    -ħ²/(2m) d²ψ/dx² + V(x)ψ = Eψ
    
    Args:
        potential: Potential function V(x)
        x_range: (x_min, x_max)
        n_points: Number of grid points
        n_states: Number of eigenstates to compute
        mass: Particle mass
        hbar: Reduced Planck constant
        
    Returns:
        Tuple: (eigenvalues, eigenstates)
        
    Example:
        >>> def harmonic(x): return 0.5 * x**2
        >>> energies, states = solve_time_independent(harmonic, (-5, 5), n_points=500, n_states=5)
        >>> print(energies)  # [0.5, 1.5, 2.5, ...]
    """
    logger.info(f"Solving TISE: n_points={n_points}, n_states={n_states}")
    
    x = np.linspace(x_range[0], x_range[1], n_points)
    dx = x[1] - x[0]
    
    # Build Hamiltonian matrix using finite differences
    # H = -ħ²/(2m) d²/dx² + V(x)
    dim = n_points
    H = np.zeros((dim, dim), dtype=float)
    
    # Kinetic energy (second derivative)
    for i in range(dim):
        H[i, i] = -2 * hbar**2 / (2 * mass * dx**2)
        if i > 0:
            H[i, i-1] = hbar**2 / (2 * mass * dx**2)
        if i < dim - 1:
            H[i, i+1] = hbar**2 / (2 * mass * dx**2)
    
    # Potential energy (diagonal)
    for i in range(dim):
        H[i, i] += potential(x[i])
    
    # Solve eigenvalue problem
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    
    # Normalize eigenstates
    for i in range(min(n_states, len(eigenvalues))):
        norm = np.sqrt(np.sum(np.abs(eigenvectors[:, i])**2) * dx)
        if norm > 0:
            eigenvectors[:, i] = eigenvectors[:, i] / norm
    
    logger.info(f"Found {min(n_states, len(eigenvalues))} eigenstates")
    
    return eigenvalues[:n_states], [eigenvectors[:, i] for i in range(min(n_states, len(eigenvalues)))]


# ============================================================================
# TIME-DEPENDENT SCHRÖDINGER EQUATION
# ============================================================================

def solve_time_dependent(
    psi0: np.ndarray,
    x: np.ndarray,
    potential: Callable[[float, float], float],
    t_max: float,
    dt: float = 0.001,
    mass: float = 1.0,
    hbar: float = 1.0,
    method: str = 'split',
    save_every: int = 10
) -> List[WaveFunction]:
    """
    Solve the time-dependent Schrödinger equation
    
    iħ ∂ψ/∂t = -ħ²/(2m) ∂²ψ/∂x² + V(x,t)ψ
    
    Args:
        psi0: Initial wavefunction
        x: Spatial grid
        potential: Potential function V(x, t)
        t_max: Maximum time
        dt: Time step
        mass: Particle mass
        hbar: Reduced Planck constant
        method: 'split' (split-step Fourier) or 'crank' (Crank-Nicolson)
        save_every: Save state every N steps
        
    Returns:
        List[WaveFunction]: Time evolution results
        
    Example:
        >>> def harmonic(x, t): return 0.5 * x**2
        >>> psi0 = np.exp(-x**2/2) / np.sqrt(np.sqrt(np.pi))
        >>> results = solve_time_dependent(psi0, x, harmonic, t_max=5.0, dt=0.01)
        >>> print(len(results))  # Number of saved states
    """
    logger.info(f"Solving TDSE: t_max={t_max:.3f}, dt={dt:.4f}, method={method}")
    
    n_points = len(x)
    dx = x[1] - x[0]
    
    # Normalize initial state
    norm = np.sqrt(np.sum(np.abs(psi0)**2) * dx)
    if norm > 0:
        psi = psi0 / norm
    else:
        raise ValueError("Initial wavefunction is zero")
    
    # Initialize
    results = [WaveFunction(x, psi.copy(), 0.0)]
    
    if method == 'split':
        # Split-step Fourier method
        try:
            from scipy.fft import fft, ifft, fftfreq
        except ImportError:
            raise ImportError("scipy.fft is required for split-step method")
        
        # k-space grid
        k = 2 * np.pi * fftfreq(n_points, dx)
        
        # Number of steps
        n_steps = int(np.ceil(t_max / dt))
        dt_actual = t_max / n_steps
        
        for step in range(n_steps):
            t = step * dt_actual
            
            # Kinetic step (in k-space)
            psi_k = fft(psi)
            psi_k = psi_k * np.exp(-1j * hbar * k**2 * dt_actual / (2 * mass))
            psi = ifft(psi_k)
            
            # Potential step (in x-space)
            V = potential(x, t + dt_actual/2)
            psi = np.exp(-1j * V * dt_actual / hbar) * psi
            
            # Normalize
            norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
            if norm > 0:
                psi = psi / norm
            
            # Save state
            if (step + 1) % save_every == 0 or step == n_steps - 1:
                results.append(WaveFunction(x, psi.copy(), (step + 1) * dt_actual))
    
    elif method == 'crank':
        # Crank-Nicolson method
        r = hbar**2 * dt / (4 * mass * dx**2)
        
        # Build tridiagonal matrices
        A = np.zeros((n_points, n_points), dtype=complex)
        B = np.zeros((n_points, n_points), dtype=complex)
        
        for i in range(n_points):
            A[i, i] = 1 + r
            B[i, i] = 1 - r
            if i > 0:
                A[i, i-1] = -r/2
                B[i, i-1] = r/2
            if i < n_points - 1:
                A[i, i+1] = -r/2
                B[i, i+1] = r/2
        
        n_steps = int(np.ceil(t_max / dt))
        dt_actual = t_max / n_steps
        
        for step in range(n_steps):
            t = step * dt_actual
            
            # Update with potential
            V = potential(x, t + dt_actual/2)
            psi = psi * np.exp(-1j * V * dt_actual / (2 * hbar))
            
            # Crank-Nicolson step
            rhs = B @ psi
            psi = np.linalg.solve(A, rhs)
            
            # Potential update
            psi = psi * np.exp(-1j * V * dt_actual / (2 * hbar))
            
            # Normalize
            norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
            if norm > 0:
                psi = psi / norm
            
            # Save state
            if (step + 1) % save_every == 0 or step == n_steps - 1:
                results.append(WaveFunction(x, psi.copy(), (step + 1) * dt_actual))
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'split' or 'crank'")
    
    logger.info(f"TDSE solved: {len(results)} states saved")
    return results


# ============================================================================
# TIME EVOLUTION OPERATOR
# ============================================================================

def time_evolution_operator(
    H: np.ndarray,
    t: float,
    hbar: float = 1.0
) -> np.ndarray:
    """
    Compute the time evolution operator U(t) = e^{-iHt/ħ}
    
    Args:
        H: Hamiltonian matrix
        t: Time
        hbar: Reduced Planck constant
        
    Returns:
        np.ndarray: Time evolution operator
        
    Example:
        >>> H = np.array([[1, 0], [0, -1]])
        >>> U = time_evolution_operator(H, np.pi/2)
        >>> print(U)  # [[e^{-iπ/2}, 0], [0, e^{iπ/2}]]
    """
    return np.linalg.expm(-1j * H * t / hbar)


def propagate_state(
    psi0: np.ndarray,
    H: np.ndarray,
    t: float,
    hbar: float = 1.0
) -> np.ndarray:
    """
    Propagate a state using the time evolution operator
    
    |ψ(t)⟩ = e^{-iHt/ħ} |ψ(0)⟩
    
    Args:
        psi0: Initial state vector
        H: Hamiltonian matrix
        t: Time
        hbar: Reduced Planck constant
        
    Returns:
        np.ndarray: Propagated state
        
    Example:
        >>> psi0 = np.array([1, 0])
        >>> H = np.array([[0, 1], [1, 0]])
        >>> psi = propagate_state(psi0, H, np.pi/2)
        >>> print(psi)  # Rotated state
    """
    U = time_evolution_operator(H, t, hbar)
    return U @ psi0


# ============================================================================
# EXPECTATION VALUES AND UNCERTAINTIES
# ============================================================================

def expectation_value(
    psi: np.ndarray,
    operator: np.ndarray,
    dx: float
) -> complex:
    """
    Compute expectation value ⟨ψ|O|ψ⟩
    
    Args:
        psi: Wavefunction
        operator: Operator matrix
        dx: Grid spacing
        
    Returns:
        complex: Expectation value
    """
    return np.sum(np.conj(psi) * (operator @ psi)) * dx


def uncertainty_relation(
    psi: np.ndarray,
    x_operator: np.ndarray,
    p_operator: np.ndarray,
    hbar: float = 1.0,
    dx: float = 0.01
) -> Tuple[float, float, float]:
    """
    Compute uncertainties and verify Heisenberg uncertainty relation
    
    Args:
        psi: Wavefunction
        x_operator: Position operator
        p_operator: Momentum operator
        hbar: Reduced Planck constant
        dx: Grid spacing
        
    Returns:
        Tuple: (Δx, Δp, Δx*Δp)
        
    Example:
        >>> from psiqit.dynamics import gaussian_wavepacket
        >>> x = np.linspace(-5, 5, 100)
        >>> psi = gaussian_wavepacket(x, sigma=1.0)
        >>> dx, dp, product = uncertainty_relation(psi, x_op, p_op)
        >>> print(product)  # >= ħ/2
    """
    # Normalize
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
    if norm > 0:
        psi = psi / norm
    
    # ⟨x⟩
    ex = np.sum(x_operator * np.abs(psi)**2) * dx
    
    # ⟨x²⟩
    ex2 = np.sum(x_operator**2 * np.abs(psi)**2) * dx
    
    # ⟨p⟩
    ep = expectation_value(psi, p_operator, dx).real
    
    # ⟨p²⟩
    ep2 = expectation_value(psi, p_operator @ p_operator, dx).real
    
    # Uncertainties
    delta_x = np.sqrt(max(0, ex2 - ex**2))
    delta_p = np.sqrt(max(0, ep2 - ep**2))
    
    return float(delta_x), float(delta_p), float(delta_x * delta_p)


# ============================================================================
# ANALYTICAL WAVEFUNCTIONS
# ============================================================================

def gaussian_wavepacket(
    x: np.ndarray,
    x0: float = 0.0,
    sigma: float = 1.0,
    k0: float = 0.0,
    phase: float = 0.0
) -> np.ndarray:
    """
    Create a Gaussian wavepacket
    
    ψ(x) = (1/(2πσ²))^(1/4) * exp(-(x-x0)²/(4σ²)) * e^{ik0x}
    
    Args:
        x: Spatial grid
        x0: Center position
        sigma: Width parameter
        k0: Momentum (wave number)
        phase: Global phase
        
    Returns:
        np.ndarray: Wavefunction
        
    Example:
        >>> x = np.linspace(-10, 10, 100)
        >>> psi = gaussian_wavepacket(x, x0=0, sigma=1.0, k0=2.0)
    """
    norm = 1 / (2 * np.pi * sigma**2)**(0.25)
    envelope = np.exp(-(x - x0)**2 / (4 * sigma**2))
    phase_factor = np.exp(1j * (k0 * x + phase))
    return norm * envelope * phase_factor


def plane_wave(
    x: np.ndarray,
    k: float,
    phase: float = 0.0
) -> np.ndarray:
    """
    Create a plane wave
    
    ψ(x) = e^{i(kx + φ)}
    
    Args:
        x: Spatial grid
        k: Wave number
        phase: Global phase
        
    Returns:
        np.ndarray: Wavefunction
        
    Example:
        >>> x = np.linspace(-10, 10, 100)
        >>> psi = plane_wave(x, k=1.0)
    """
    return np.exp(1j * (k * x + phase))


def square_well_ground_state(
    x: np.ndarray,
    L: float
) -> np.ndarray:
    """
    Create the ground state wavefunction of an infinite square well
    
    ψ(x) = √(2/L) sin(πx/L) for 0 < x < L
    
    Args:
        x: Spatial grid
        L: Well width
        
    Returns:
        np.ndarray: Wavefunction
        
    Example:
        >>> x = np.linspace(0, 1, 100)
        >>> psi = square_well_ground_state(x, L=1.0)
    """
    psi = np.zeros_like(x, dtype=complex)
    
    # Domain: 0 < x < L
    mask = (x > 0) & (x < L)
    psi[mask] = np.sqrt(2 / L) * np.sin(PI * x[mask] / L)
    
    return psi


def harmonic_oscillator_state(
    x: np.ndarray,
    n: int,
    mass: float = 1.0,
    omega: float = 1.0,
    hbar: float = 1.0
) -> np.ndarray:
    """
    Create harmonic oscillator eigenstate
    
    ψ_n(x) = (1/√(2^n n!)) (mω/πħ)^(1/4) 
             * e^(-mωx²/2ħ) * H_n(√(mω/ħ) x)
    
    Args:
        x: Spatial grid
        n: Quantum number
        mass: Particle mass
        omega: Angular frequency
        hbar: Reduced Planck constant
        
    Returns:
        np.ndarray: Wavefunction
        
    Example:
        >>> x = np.linspace(-5, 5, 100)
        >>> psi_2 = harmonic_oscillator_state(x, n=2)
    """
    from math import factorial, sqrt
    
    alpha = mass * omega / hbar
    sqrt_alpha = np.sqrt(alpha)
    
    # Prefactor
    prefactor = (alpha / PI)**(0.25) / sqrt(2**n * factorial(n))
    
    # Gaussian envelope
    gaussian = np.exp(-alpha * x**2 / 2)
    
    # Hermite polynomial (using recurrence relation)
    if n == 0:
        H_n = np.ones_like(x)
    elif n == 1:
        H_n = 2 * sqrt_alpha * x
    else:
        # Recurrence: H_{n+1}(y) = 2y H_n(y) - 2n H_{n-1}(y)
        y = sqrt_alpha * x
        H_prev = np.ones_like(x)
        H_curr = 2 * y
        for _ in range(2, n + 1):
            H_next = 2 * y * H_curr - 2 * (_ - 1) * H_prev
            H_prev, H_curr = H_curr, H_next
        H_n = H_curr
    
    return prefactor * gaussian * H_n


def infinite_well_state(
    x: np.ndarray,
    n: int,
    L: float
) -> np.ndarray:
    """
    Create the n-th state of an infinite square well
    
    ψ_n(x) = √(2/L) sin(nπx/L) for 0 < x < L
    
    Args:
        x: Spatial grid
        n: Quantum number (1, 2, 3, ...)
        L: Well width
        
    Returns:
        np.ndarray: Wavefunction
        
    Example:
        >>> x = np.linspace(0, 1, 100)
        >>> psi_2 = infinite_well_state(x, n=2, L=1.0)
    """
    psi = np.zeros_like(x, dtype=complex)
    
    mask = (x > 0) & (x < L)
    psi[mask] = np.sqrt(2 / L) * np.sin(n * PI * x[mask] / L)
    
    return psi


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # WaveFunction class
    'WaveFunction',
    
    # Time-independent Schrödinger
    'solve_time_independent',
    
    # Time-dependent Schrödinger
    'solve_time_dependent',
    
    # Time evolution
    'time_evolution_operator',
    'propagate_state',
    
    # Expectation values and uncertainties
    'expectation_value',
    'uncertainty_relation',
    
    # Analytical wavefunctions
    'gaussian_wavepacket',
    'plane_wave',
    'square_well_ground_state',
    'harmonic_oscillator_state',
    'infinite_well_state',
]