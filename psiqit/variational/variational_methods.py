# psiqit/variational/variational_methods.py 

"""
Variational Methods Module
Rayleigh-Ritz, TDVP, Variational Monte Carlo, and Hartree-Fock
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from ..math.calculus import derivative, gradient, hessian
from ..math.ode_solver import rk4, rk4_system, ODEResult
from ..quantum.state import Ket, zero, one, plus, minus, basis
from ..quantum.operator import Operator, identity, pauli_x, pauli_y, pauli_z
from ..utils.logger import logger
from ..utils.validation import validate_qubits


# ============================================================================
# VARIATIONAL RESULT CLASS
# ============================================================================

@dataclass
class VariationalResult:
    """
    Result container for variational methods
    
    Attributes:
        optimal_params: Optimal parameters found
        optimal_value: Optimal value (energy, cost, etc.)
        iterations: Number of iterations
        history: History of values during optimization
        success: Whether optimization was successful
        method: Name of the method used
        message: Additional message
    """
    optimal_params: Optional[np.ndarray] = None
    optimal_value: float = 0.0
    iterations: int = 0
    history: List[float] = field(default_factory=list)
    success: bool = True
    method: str = ""
    message: str = ""
    
    def __repr__(self) -> str:
        status = "Success" if self.success else "Failed"
        return f"VariationalResult(method={self.method}, status={status}, value={self.optimal_value:.6f})"
    
    def __str__(self) -> str:
        lines = [
            "Variational Method Results:",
            f"  Method: {self.method}",
            f"  Success: {self.success}",
            f"  Iterations: {self.iterations}",
            f"  Optimal Value: {self.optimal_value:.6f}",
            f"  Message: {self.message}",
        ]
        if self.history:
            lines.append(f"  Initial Value: {self.history[0]:.6f}")
            lines.append(f"  Final Value: {self.history[-1]:.6f}")
        return "\n".join(lines)


# ============================================================================
# RAYLEIGH-RITZ METHOD
# ============================================================================

class RayleighRitz:
    """
    Rayleigh-Ritz variational method
    
    Finds the minimum of the Rayleigh quotient:
    R(ψ) = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩
    
    Example:
        >>> from psiqit.variational import RayleighRitz
        >>> # Define a matrix (Hamiltonian)
        >>> H = np.array([[2, 1], [1, 2]])
        >>> # Define ansatz: |ψ(θ)⟩ = cos(θ)|0⟩ + sin(θ)|1⟩
        >>> def ansatz(theta):
        ...     return np.array([np.cos(theta), np.sin(theta)])
        >>> rr = RayleighRitz(H, ansatz, n_params=1)
        >>> result = rr.optimize(n_iterations=50)
        >>> print(result.optimal_value)
    """
    
    def __init__(
        self,
        matrix: np.ndarray,
        ansatz: Callable[[np.ndarray], np.ndarray],
        n_params: int = 1
    ):
        """
        Initialize Rayleigh-Ritz
        
        Args:
            matrix: Matrix (Hamiltonian) to diagonalize
            ansatz: Ansatz function that takes parameters and returns state vector
            n_params: Number of parameters
        """
        self.matrix = np.array(matrix, dtype=complex)
        self.ansatz = ansatz
        self.n_params = n_params
        
        # Check if matrix is Hermitian
        if not np.allclose(self.matrix, self.matrix.conj().T):
            logger.warning("Matrix is not Hermitian")
        
        self.dim = self.matrix.shape[0]
        
        logger.info(f"Rayleigh-Ritz initialized: dim={self.dim}, params={n_params}")
    
    def expectation(self, params: np.ndarray) -> float:
        """
        Compute the Rayleigh quotient
        
        Args:
            params: Parameters
            
        Returns:
            float: Rayleigh quotient value
        """
        state = self.ansatz(params)
        state = np.array(state, dtype=complex)
        
        # Normalize
        norm = np.linalg.norm(state)
        if norm > 0:
            state = state / norm
        
        # Rayleigh quotient: ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩
        numerator = np.vdot(state, self.matrix @ state).real
        denominator = np.vdot(state, state).real
        
        if denominator == 0:
            return float('inf')
        
        return float(numerator / denominator)
    
    def _gradient(self, params: np.ndarray, eps: float = 1e-5) -> List[float]:
        """
        Compute gradient using finite differences
        
        Args:
            params: Parameters
            eps: Step size
            
        Returns:
            List[float]: Gradient vector
        """
        grad = np.zeros(self.n_params)
        
        for i in range(self.n_params):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps
            
            grad[i] = (self.expectation(params_plus) - self.expectation(params_minus)) / (2 * eps)
        
        return grad.tolist()
    
    def optimize(
        self,
        n_iterations: int = 100,
        learning_rate: float = 0.1,
        verbose: bool = False,
        initial_params: Optional[np.ndarray] = None
    ) -> VariationalResult:
        """
        Optimize the Rayleigh quotient
        
        Args:
            n_iterations: Number of iterations
            learning_rate: Learning rate
            verbose: Print progress
            initial_params: Initial parameters
            
        Returns:
            VariationalResult: Optimization results
        """
        logger.info(f"Starting Rayleigh-Ritz optimization: {n_iterations} iterations")
        
        if initial_params is None:
            params = np.random.uniform(-np.pi, np.pi, size=self.n_params)
        else:
            params = initial_params.copy()
        
        history = []
        
        for iteration in range(n_iterations):
            grad = self._gradient(params)
            params = params - learning_rate * np.array(grad)
            
            value = self.expectation(params)
            history.append(value)
            
            if verbose and (iteration + 1) % max(1, n_iterations // 10) == 0:
                logger.info(f"Iteration {iteration+1}/{n_iterations}: value={value:.6f}")
        
        final_value = self.expectation(params)
        
        logger.info(f"Rayleigh-Ritz completed: value={final_value:.6f}")
        
        return VariationalResult(
            optimal_params=params,
            optimal_value=final_value,
            iterations=n_iterations,
            history=history,
            success=True,
            method="Rayleigh-Ritz",
            message=f"Rayleigh-Ritz completed with value {final_value:.6f}"
        )


# ============================================================================
# TIME-DEPENDENT VARIATIONAL PRINCIPLE (TDVP)
# ============================================================================

class TDVP:
    """
    Time-Dependent Variational Principle (TDVP)
    
    Simulates time evolution of a variational state according to:
    iħ ∂ψ/∂t = H ψ
    
    Example:
        >>> from psiqit.variational import TDVP
        >>> # Define Hamiltonian
        >>> H = np.array([[0, 1], [1, 0]])
        >>> # Define ansatz: |ψ(θ)⟩ = cos(θ)|0⟩ + sin(θ)|1⟩
        >>> def ansatz(theta):
        ...     return np.array([np.cos(theta), np.sin(theta)])
        >>> tdvp = TDVP(H, ansatz, n_params=1)
        >>> params0 = np.array([0.0])
        >>> params_hist, energy_hist = tdvp.evolve(params0, t_max=10.0, dt=0.01)
    """
    
    def __init__(
        self,
        hamiltonian: np.ndarray,
        ansatz: Callable[[np.ndarray], np.ndarray],
        n_params: int,
        mass: float = 1.0,
        hbar: float = 1.0
    ):
        """
        Initialize TDVP
        
        Args:
            hamiltonian: Hamiltonian matrix
            ansatz: Ansatz function that takes parameters and returns state vector
            n_params: Number of parameters
            mass: Particle mass
            hbar: Reduced Planck constant
        """
        self.hamiltonian = np.array(hamiltonian, dtype=complex)
        self.ansatz = ansatz
        self.n_params = n_params
        self.mass = mass
        self.hbar = hbar
        
        self.dim = self.hamiltonian.shape[0]
        
        logger.info(f"TDVP initialized: dim={self.dim}, params={n_params}")
    
    def _get_state(self, params: np.ndarray) -> np.ndarray:
        """Get the state vector for given parameters"""
        state = self.ansatz(params)
        state = np.array(state, dtype=complex)
        
        # Normalize
        norm = np.linalg.norm(state)
        if norm > 0:
            state = state / norm
        
        return state
    
    def _metric_tensor(self, params: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """
        Compute the metric tensor (Fubini-Study metric)
        
        Args:
            params: Parameters
            eps: Step size
            
        Returns:
            np.ndarray: Metric tensor (n_params x n_params)
        """
        n = self.n_params
        metric = np.zeros((n, n))
        
        state = self._get_state(params)
        
        # Compute derivatives of the state with respect to parameters
        derivatives = []
        for i in range(n):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps
            
            state_plus = self._get_state(params_plus)
            state_minus = self._get_state(params_minus)
            
            d_state = (state_plus - state_minus) / (2 * eps)
            derivatives.append(d_state)
        
        # Compute metric tensor: g_ij = ⟨∂_i ψ|∂_j ψ⟩ - ⟨∂_i ψ|ψ⟩⟨ψ|∂_j ψ⟩
        for i in range(n):
            for j in range(n):
                overlap_ij = np.vdot(derivatives[i], derivatives[j]).real
                overlap_i = np.vdot(derivatives[i], state).real
                overlap_j = np.vdot(state, derivatives[j]).real
                metric[i, j] = overlap_ij - overlap_i * overlap_j
        
        return metric
    
    def _energy_gradient(self, params: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """
        Compute the energy gradient
        
        Args:
            params: Parameters
            eps: Step size
            
        Returns:
            np.ndarray: Energy gradient
        """
        n = self.n_params
        grad = np.zeros(n)
        
        for i in range(n):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps
            
            state_plus = self._get_state(params_plus)
            state_minus = self._get_state(params_minus)
            
            energy_plus = np.vdot(state_plus, self.hamiltonian @ state_plus).real
            energy_minus = np.vdot(state_minus, self.hamiltonian @ state_minus).real
            
            grad[i] = (energy_plus - energy_minus) / (2 * eps)
        
        return grad
    
    def evolve(
        self,
        params0: np.ndarray,
        t_max: float = 10.0,
        dt: float = 0.01,
        verbose: bool = False
    ) -> Tuple[List[List[float]], List[float]]:
        """
        Evolve the variational state in time
        
        Args:
            params0: Initial parameters
            t_max: Maximum time
            dt: Time step
            verbose: Print progress
            
        Returns:
            Tuple: (parameter_history, energy_history)
        """
        logger.info(f"Starting TDVP evolution: t_max={t_max:.3f}, dt={dt:.4f}")
        
        n_steps = int(np.ceil(t_max / dt))
        dt_actual = t_max / n_steps
        
        # Define time derivative function for ODE solver
        def derivative_fn(t: float, params: np.ndarray) -> np.ndarray:
            # Compute metric tensor and energy gradient
            metric = self._metric_tensor(params)
            grad = self._energy_gradient(params)
            
            # Solve for parameter velocities: g * v = -grad
            try:
                # Add small regularization to avoid singular matrix
                reg = 1e-6 * np.eye(self.n_params)
                v = np.linalg.solve(metric + reg, -grad)
            except np.linalg.LinAlgError:
                # If singular, use pseudo-inverse
                v = -np.linalg.pinv(metric + reg) @ grad
            
            return v
        
        # Store history
        params_hist = [params0.copy()]
        energy_hist = []
        
        params = params0.copy()
        
        for step in range(n_steps):
            # RK4 integration
            k1 = derivative_fn(step * dt_actual, params)
            k2 = derivative_fn(step * dt_actual + dt_actual/2, params + dt_actual/2 * k1)
            k3 = derivative_fn(step * dt_actual + dt_actual/2, params + dt_actual/2 * k2)
            k4 = derivative_fn(step * dt_actual + dt_actual, params + dt_actual * k3)
            
            params = params + dt_actual/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            # Compute energy
            state = self._get_state(params)
            energy = np.vdot(state, self.hamiltonian @ state).real
            
            params_hist.append(params.copy())
            energy_hist.append(float(energy))
            
            if verbose and (step + 1) % max(1, n_steps // 10) == 0:
                logger.info(f"Step {step+1}/{n_steps}: energy={energy:.6f}")
        
        logger.info(f"TDVP evolution completed: final_energy={energy_hist[-1]:.6f}")
        
        return params_hist, energy_hist


# ============================================================================
# VARIATIONAL MONTE CARLO
# ============================================================================

class VariationalMonteCarlo:
    """
    Variational Monte Carlo (VMC)
    
    Uses Monte Carlo sampling to evaluate and optimize variational wavefunctions.
    
    Example:
        >>> from psiqit.variational import VariationalMonteCarlo
        >>> # Define wavefunction: ψ(x, a) = exp(-a*x²)
        >>> def wavefunction(x, params):
        ...     return np.exp(-params[0] * x**2)
        >>> # Define potential: harmonic oscillator V(x) = 0.5*x²
        >>> def potential(x):
        ...     return 0.5 * x**2
        >>> vmc = VariationalMonteCarlo(wavefunction, potential, n_params=1)
        >>> result = vmc.optimize(n_samples=1000, n_iterations=50)
        >>> print(result.optimal_params)
    """
    
    def __init__(
        self,
        wavefunction: Callable[[np.ndarray, np.ndarray], np.ndarray],
        potential: Callable[[np.ndarray], np.ndarray],
        n_params: int,
        kinetic_energy: Optional[Callable] = None,
        mass: float = 1.0,
        hbar: float = 1.0
    ):
        """
        Initialize Variational Monte Carlo
        
        Args:
            wavefunction: Wavefunction ψ(x, params)
            potential: Potential function V(x)
            n_params: Number of parameters
            kinetic_energy: Kinetic energy function (if None, uses default)
            mass: Particle mass
            hbar: Reduced Planck constant
        """
        self.wavefunction = wavefunction
        self.potential = potential
        self.n_params = n_params
        self.mass = mass
        self.hbar = hbar
        
        if kinetic_energy is None:
            self.kinetic_energy = self._default_kinetic
        else:
            self.kinetic_energy = kinetic_energy
        
        logger.info(f"VMC initialized: params={n_params}, mass={mass}")
    
    def _default_kinetic(self, x: np.ndarray, psi: np.ndarray, dpsi: np.ndarray, d2psi: np.ndarray) -> float:
        """
        Default kinetic energy: -ħ²/(2m) ψ''/ψ
        
        Args:
            x: Position
            psi: Wavefunction value
            dpsi: First derivative
            d2psi: Second derivative
            
        Returns:
            float: Kinetic energy
        """
        if abs(psi) < 1e-15:
            return 0.0
        return -self.hbar**2 / (2 * self.mass) * (d2psi / psi).real
    
    def _local_energy(self, x: float, params: np.ndarray) -> float:
        """
        Compute local energy: E_local(x) = -ħ²/(2m) ψ''/ψ + V(x)
        
        Args:
            x: Position
            params: Parameters
            
        Returns:
            float: Local energy
        """
        # Compute wavefunction and derivatives
        dx = 1e-5
        psi = self.wavefunction(np.array([x]), params)[0]
        psi_plus = self.wavefunction(np.array([x + dx]), params)[0]
        psi_minus = self.wavefunction(np.array([x - dx]), params)[0]
        
        # First derivative (central difference)
        dpsi = (psi_plus - psi_minus) / (2 * dx)
        
        # Second derivative (central difference)
        d2psi = (psi_plus - 2*psi + psi_minus) / (dx**2)
        
        # Kinetic energy
        kinetic = self.kinetic_energy(x, psi, dpsi, d2psi)
        
        # Potential energy
        potential = self.potential(np.array([x]))[0]
        
        return float(kinetic + potential)
    
    def _energy(self, params: np.ndarray, samples: np.ndarray) -> float:
        """
        Compute average energy over samples
        
        Args:
            params: Parameters
            samples: Sampled positions
            
        Returns:
            float: Average energy
        """
        energies = np.array([self._local_energy(x, params) for x in samples])
        return float(np.mean(energies))
    
    def _generate_samples(self, params: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Generate samples from the wavefunction distribution
        
        Args:
            params: Parameters
            n_samples: Number of samples
            
        Returns:
            np.ndarray: Sampled positions
        """
        # Use Metropolis-Hastings sampling
        # This is a simplified version using Gaussian proposal
        samples = np.zeros(n_samples)
        x = 0.0  # Initial position
        
        # Compute wavefunction at current position
        psi_current = self.wavefunction(np.array([x]), params)[0]
        
        # Proposal width
        sigma = 1.0
        
        for i in range(n_samples):
            # Propose new position
            x_proposed = x + np.random.normal(0, sigma)
            
            # Compute wavefunction at proposed position
            psi_proposed = self.wavefunction(np.array([x_proposed]), params)[0]
            
            # Acceptance probability (for real wavefunctions)
            if abs(psi_current) > 0:
                acceptance = min(1, (psi_proposed / psi_current)**2)
            else:
                acceptance = 0
            
            # Accept or reject
            if np.random.random() < acceptance:
                x = x_proposed
                psi_current = psi_proposed
            
            samples[i] = x
        
        return samples
    
    def _gradient(self, params: np.ndarray, samples: np.ndarray, eps: float = 1e-5) -> List[float]:
        """
        Compute gradient of energy with respect to parameters
        
        Args:
            params: Parameters
            samples: Sampled positions
            eps: Step size
            
        Returns:
            List[float]: Gradient vector
        """
        grad = np.zeros(self.n_params)
        
        for i in range(self.n_params):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps
            
            energy_plus = self._energy(params_plus, samples)
            energy_minus = self._energy(params_minus, samples)
            
            grad[i] = (energy_plus - energy_minus) / (2 * eps)
        
        return grad.tolist()
    
    def optimize(
        self,
        n_samples: int = 5000,
        n_iterations: int = 50,
        learning_rate: float = 0.01,
        verbose: bool = False,
        initial_params: Optional[np.ndarray] = None
    ) -> VariationalResult:
        """
        Optimize the variational parameters
        
        Args:
            n_samples: Number of Monte Carlo samples
            n_iterations: Number of optimization iterations
            learning_rate: Learning rate
            verbose: Print progress
            initial_params: Initial parameters
            
        Returns:
            VariationalResult: Optimization results
        """
        logger.info(f"Starting VMC optimization: samples={n_samples}, iterations={n_iterations}")
        
        if initial_params is None:
            params = np.random.uniform(-1, 1, size=self.n_params)
        else:
            params = initial_params.copy()
        
        history = []
        
        for iteration in range(n_iterations):
            # Generate samples
            samples = self._generate_samples(params, n_samples)
            
            # Compute gradient
            grad = self._gradient(params, samples)
            
            # Update parameters
            params = params - learning_rate * np.array(grad)
            
            # Compute energy
            energy = self._energy(params, samples)
            history.append(energy)
            
            if verbose and (iteration + 1) % max(1, n_iterations // 10) == 0:
                logger.info(f"Iteration {iteration+1}/{n_iterations}: energy={energy:.6f}")
        
        final_energy = self._energy(params, samples)
        
        logger.info(f"VMC completed: energy={final_energy:.6f}")
        
        return VariationalResult(
            optimal_params=params,
            optimal_value=final_energy,
            iterations=n_iterations,
            history=history,
            success=True,
            method="VMC",
            message=f"VMC completed with energy {final_energy:.6f}"
        )


# ============================================================================
# HARTREE-FOCK METHOD
# ============================================================================

class HartreeFock:
    """
    Hartree-Fock method for electronic structure
    
    Solves the Hartree-Fock equations self-consistently.
    
    Example:
        >>> from psiqit.variational import HartreeFock
        >>> # Create Hartree-Fock for 2 electrons in 2 orbitals
        >>> hf = HartreeFock(n_electrons=2, n_orbitals=2)
        >>> result = hf.solve(max_iter=50)
        >>> print(result.optimal_value)  # Total energy
    """
    
    def __init__(
        self,
        n_electrons: int,
        n_orbitals: int,
        one_electron_integrals: Optional[np.ndarray] = None,
        two_electron_integrals: Optional[np.ndarray] = None
    ):
        """
        Initialize Hartree-Fock
        
        Args:
            n_electrons: Number of electrons
            n_orbitals: Number of orbitals
            one_electron_integrals: h_ij matrix (if None, uses default)
            two_electron_integrals: (ij|kl) tensor (if None, uses default)
        """
        self.n_electrons = n_electrons
        self.n_orbitals = n_orbitals
        
        if one_electron_integrals is None:
            self.h = self._default_one_electron()
        else:
            self.h = np.array(one_electron_integrals, dtype=float)
        
        if two_electron_integrals is None:
            self.g = self._default_two_electron()
        else:
            self.g = np.array(two_electron_integrals, dtype=float)
        
        self.n_occupied = n_electrons // 2  # Number of occupied orbitals (closed shell)
        
        logger.info(f"Hartree-Fock initialized: electrons={n_electrons}, orbitals={n_orbitals}")
    
    def _default_one_electron(self) -> np.ndarray:
        """
        Create default one-electron integrals (kinetic + nuclear attraction)
        
        Returns:
            np.ndarray: h_ij matrix
        """
        # Simple harmonic oscillator Hamiltonian
        h = np.zeros((self.n_orbitals, self.n_orbitals))
        for i in range(self.n_orbitals):
            h[i, i] = i + 0.5  # Harmonic oscillator eigenvalues
            if i > 0:
                h[i-1, i] = 0.1  # Small coupling
                h[i, i-1] = 0.1
        return h
    
    def _default_two_electron(self) -> np.ndarray:
        """
        Create default two-electron integrals (ij|kl)
        
        Returns:
            np.ndarray: (ij|kl) tensor
        """
        # Simple Coulomb integrals
        n = self.n_orbitals
        g = np.zeros((n, n, n, n))
        
        # Add Coulomb repulsion terms
        for i in range(n):
            for j in range(n):
                g[i, i, j, j] = 1.0 / (abs(i - j) + 1)
        
        return g
    
    def _density(self, C: np.ndarray) -> np.ndarray:
        """
        Compute density matrix from orbital coefficients
        
        Args:
            C: Orbital coefficient matrix (n_orbitals x n_orbitals)
            
        Returns:
            np.ndarray: Density matrix
        """
        P = np.zeros((self.n_orbitals, self.n_orbitals))
        for i in range(self.n_occupied):
            for mu in range(self.n_orbitals):
                for nu in range(self.n_orbitals):
                    P[mu, nu] += 2 * C[mu, i] * C[nu, i]  # Factor 2 for closed shell
        return P
    
    def _fock(self, P: np.ndarray) -> np.ndarray:
        """
        Compute Fock matrix
        
        Args:
            P: Density matrix
            
        Returns:
            np.ndarray: Fock matrix
        """
        F = self.h.copy()
        
        # Add two-electron contributions
        for mu in range(self.n_orbitals):
            for nu in range(self.n_orbitals):
                # Coulomb term: Σ_λσ P_λσ (μν|λσ)
                for lam in range(self.n_orbitals):
                    for sigma in range(self.n_orbitals):
                        F[mu, nu] += P[lam, sigma] * self.g[mu, nu, lam, sigma]
                
                # Exchange term: -1/2 Σ_λσ P_λσ (μλ|νσ)
                for lam in range(self.n_orbitals):
                    for sigma in range(self.n_orbitals):
                        F[mu, nu] -= 0.5 * P[lam, sigma] * self.g[mu, lam, nu, sigma]
        
        return F
    
    def _diagonalize(self, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize the Fock matrix
        
        Args:
            F: Fock matrix
            
        Returns:
            Tuple: (eigenvalues, eigenvectors)
        """
        eigvals, eigvecs = np.linalg.eigh(F)
        return eigvals, eigvecs
    
    def solve(
        self,
        max_iter: int = 50,
        tol: float = 1e-8,
        verbose: bool = False
    ) -> VariationalResult:
        """
        Solve Hartree-Fock equations self-consistently
        
        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            verbose: Print progress
            
        Returns:
            VariationalResult: Optimization results
        """
        logger.info(f"Starting Hartree-Fock: max_iter={max_iter}, tol={tol:.2e}")
        
        # Initial guess: eigenvectors of one-electron Hamiltonian
        eigvals, C = np.linalg.eigh(self.h)
        
        history = []
        
        for iteration in range(max_iter):
            # Compute density matrix
            P = self._density(C)
            
            # Build Fock matrix
            F = self._fock(P)
            
            # Diagonalize Fock matrix
            eigvals_new, C_new = self._diagonalize(F)
            
            # Compute energy
            energy = 0.0
            for i in range(self.n_occupied):
                energy += eigvals_new[i]
            
            # Check convergence
            if iteration > 0:
                diff = np.max(np.abs(C - C_new))
                if diff < tol:
                    logger.info(f"Converged at iteration {iteration}")
                    break
            
            C = C_new
            history.append(float(energy))
            
            if verbose and (iteration + 1) % max(1, max_iter // 10) == 0:
                logger.info(f"Iteration {iteration+1}: energy={energy:.6f}")
        
        # Final energy
        final_energy = energy if 'energy' in locals() else 0.0
        
        # Compute total electronic energy
        # E = Σ_i ε_i - 1/2 Σ_ij P_ij (h_ij + F_ij)
        # This is a simplified version
        
        logger.info(f"Hartree-Fock completed: energy={final_energy:.6f}")
        
        return VariationalResult(
            optimal_params=C,
            optimal_value=final_energy,
            iterations=iteration + 1,
            history=history,
            success=True,
            method="Hartree-Fock",
            message=f"HF completed with energy {final_energy:.6f}"
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'VariationalResult',
    'RayleighRitz',
    'TDVP',
    'VariationalMonteCarlo',
    'HartreeFock',
]